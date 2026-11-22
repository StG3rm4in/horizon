//! The server thread's own CPU (`HORIZON_TICK_CPU`, Linux): the thread that ticks keeps one CPU
//! to itself, and the other threads leave it and its SMT siblings. The tick then never migrates
//! nor shares its core with a chunk worker or a helper. Nothing changes in what the tick computes,
//! only where the threads run.

/// What [`reserve_core_for_current`] set: the CPU the calling thread keeps, its SMT siblings (left
/// idle), and the CPUs every other thread of the process runs on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reserved {
    pub cpu: usize,
    pub siblings: Vec<usize>,
    pub others: Vec<usize>,
}

/// `HORIZON_TICK_CPU`: None when unset or empty, else the CPU number, or the value that isn't one.
pub fn tick_cpu() -> Option<Result<usize, String>> {
    let value = std::env::var("HORIZON_TICK_CPU").ok()?;
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    Some(value.parse().map_err(|_| value.to_owned()))
}

/// The CPUs of a sysfs CPU list (`0-3,8,10-11`), in order and once each; a part that isn't a
/// number or a range is left out.
pub fn parse_cpu_list(list: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in list
        .trim()
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
    {
        match part.split_once('-') {
            Some((first, last)) => {
                if let (Ok(first), Ok(last)) = (first.parse::<usize>(), last.parse::<usize>()) {
                    cpus.extend(first..=last);
                }
            }
            None => cpus.extend(part.parse::<usize>().ok()),
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    cpus
}

/// The CPUs of `allowed` outside `core` (the kept CPU and its SMT siblings), in `allowed`'s order.
pub fn others_of(allowed: &[usize], core: &[usize]) -> Vec<usize> {
    allowed
        .iter()
        .copied()
        .filter(|c| !core.contains(c))
        .collect()
}

/// Makes the calling thread keep `cpu` to itself: it runs there only, and every other thread of
/// the process runs on the process's other CPUs but `cpu`'s SMT siblings. A thread started later
/// inherits its starter's CPUs, so one the calling thread starts is moved off within a second (at
/// once for a pool whose threads call [`leave_server_core`] as they start). [`crate::machine_threads`]
/// and [`crate::worker_count`] are fixed first: afterwards `available_parallelism` answers one on
/// the calling thread. Once per process; Linux only.
pub fn reserve_core_for_current(cpu: usize) -> Result<Reserved, String> {
    let _ = crate::machine_threads();
    let _ = crate::worker_count();
    imp::reserve(cpu)
}

/// For a thread that may start on the kept CPU (one the server thread starts inherits its CPU):
/// it moves to the other threads' CPUs. Nothing while no CPU is kept.
pub fn leave_server_core() {
    imp::leave();
}

/// While a CPU is kept, what [`crate::available_parallelism`] answers: [`crate::machine_threads`],
/// fixed before the pin (the calling thread's own mask is that one CPU, the others' the rest).
pub(crate) fn whole_set() -> Option<std::num::NonZeroUsize> {
    if imp::kept() {
        std::num::NonZeroUsize::new(crate::machine_threads())
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::{Reserved, others_of, parse_cpu_list};
    use std::sync::OnceLock;
    use std::time::Duration;

    /// The thread that keeps its CPU, and the other threads' CPUs, once a CPU is kept.
    static RESERVED: OnceLock<(libc::pid_t, Vec<usize>)> = OnceLock::new();

    fn set_of(cpus: &[usize]) -> libc::cpu_set_t {
        // SAFETY: `cpu_set_t` is a plain bit array, and all zeros is the empty set.
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        for &cpu in cpus {
            // SAFETY: sets one bit of `set`, which lives here; a CPU past its size is ignored.
            unsafe { libc::CPU_SET(cpu, &mut set) };
        }
        set
    }

    fn cpus_of(set: &libc::cpu_set_t) -> Vec<usize> {
        let bits = 8 * std::mem::size_of::<libc::cpu_set_t>();
        // SAFETY: reads bits of `set`, a borrowed plain value, below its size.
        (0..bits)
            .filter(|&c| unsafe { libc::CPU_ISSET(c, set) })
            .collect()
    }

    /// The CPUs thread `tid` may run on (0: the calling thread), None when it is gone.
    fn affinity(tid: libc::pid_t) -> Option<Vec<usize>> {
        // SAFETY: as in `set_of`.
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::cpu_set_t>();
        // SAFETY: the kernel writes at most `size` bytes into `set`, which lives across the call.
        let read = unsafe { libc::sched_getaffinity(tid, size, &mut set) } == 0;
        read.then(|| cpus_of(&set))
    }

    /// Thread `tid` (0: the calling thread) onto `cpus`; false when it is gone or refused.
    fn set_affinity(tid: libc::pid_t, cpus: &[usize]) -> bool {
        let set = set_of(cpus);
        let size = std::mem::size_of::<libc::cpu_set_t>();
        // SAFETY: the kernel reads `size` bytes of `set`, which lives across the call.
        unsafe { libc::sched_setaffinity(tid, size, &set) == 0 }
    }

    fn current_tid() -> libc::pid_t {
        // SAFETY: gettid takes no argument and cannot fail.
        unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t }
    }

    /// Every thread of the process but `keeper` onto `others`, where it isn't there already.
    fn move_others(keeper: libc::pid_t, others: &[usize]) {
        let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
            return;
        };
        for task in tasks.flatten() {
            let tid = task
                .file_name()
                .to_str()
                .and_then(|t| t.parse::<libc::pid_t>().ok());
            if let Some(tid) = tid
                && tid != keeper
                && affinity(tid).is_some_and(|now| now != others)
            {
                set_affinity(tid, others);
            }
        }
    }

    pub(super) fn reserve(cpu: usize) -> Result<Reserved, String> {
        if RESERVED.get().is_some() {
            return Err("a CPU is already kept".to_owned());
        }
        let allowed = affinity(0).ok_or("this thread's CPUs could not be read")?;
        if !allowed.contains(&cpu) {
            return Err(format!(
                "CPU {cpu} is not one of this process's CPUs {allowed:?}"
            ));
        }
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let mut core = std::fs::read_to_string(path)
            .map(|list| parse_cpu_list(&list))
            .unwrap_or_default();
        if !core.contains(&cpu) {
            core = vec![cpu];
        }
        let others = others_of(&allowed, &core);
        if others.is_empty() {
            return Err(format!(
                "no CPU would be left for the other threads ({allowed:?})"
            ));
        }
        if !set_affinity(0, &[cpu]) {
            return Err(format!("this thread could not be moved to CPU {cpu}"));
        }
        let keeper = current_tid();
        let _ = RESERVED.set((keeper, others.clone()));
        move_others(keeper, &others);
        // The threads this one starts from now on inherit its single CPU: moved off every second.
        std::thread::Builder::new()
            .name("tick-cpu-keeper".into())
            .spawn(|| {
                leave();
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    if let Some((keeper, others)) = RESERVED.get() {
                        move_others(*keeper, others);
                    }
                }
            })
            .map_err(|e| format!("the thread that moves the others off did not start: {e}"))?;
        let siblings = core.into_iter().filter(|&c| c != cpu).collect();
        Ok(Reserved {
            cpu,
            siblings,
            others,
        })
    }

    pub(super) fn leave() {
        if let Some((_, others)) = RESERVED.get() {
            set_affinity(0, others);
        }
    }

    pub(super) fn kept() -> bool {
        RESERVED.get().is_some()
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::Reserved;

    pub(super) fn reserve(_cpu: usize) -> Result<Reserved, String> {
        Err("a CPU can be kept on Linux only".to_owned())
    }

    pub(super) fn leave() {}

    pub(super) fn kept() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists_read_as_sysfs_writes_them() {
        assert_eq!(parse_cpu_list("2,8\n"), vec![2, 8]);
        assert_eq!(parse_cpu_list("0-3,8,10-11"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpu_list("5"), vec![5]);
        assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
        assert_eq!(parse_cpu_list("3,x,1,1"), vec![1, 3]);
    }

    #[test]
    fn the_other_threads_leave_the_kept_core() {
        let allowed: Vec<usize> = (0..12).collect();
        assert_eq!(
            others_of(&allowed, &[2, 8]),
            vec![0, 1, 3, 4, 5, 6, 7, 9, 10, 11]
        );
        assert_eq!(others_of(&[2, 8], &[2, 8]), Vec::<usize>::new());
    }
}
