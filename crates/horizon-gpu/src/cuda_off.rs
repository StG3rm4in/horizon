//! What this crate uses of cudarc, for a build without the `cuda` feature: the same names and
//! signatures with no GPU behind them. The library checks report CUDA as absent and
//! `CudaContext::new` fails, so every GPU stage stays on the CPU and nothing else here is reached.
#![allow(
    dead_code,
    unused_variables,
    non_camel_case_types,
    non_snake_case,
    clippy::all
)]

pub mod driver {
    use std::marker::PhantomData;
    use std::ops::RangeBounds;
    use std::sync::Arc;

    /// The only answer of this build.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DriverError;
    impl std::fmt::Display for DriverError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("built without CUDA (the cuda feature of horizon-gpu)")
        }
    }
    impl std::error::Error for DriverError {}
    fn absent<T>() -> Result<T, DriverError> {
        Err(DriverError)
    }

    pub trait DeviceRepr {}
    impl<T> DeviceRepr for T {}
    pub trait ValidAsZeroBits {}
    impl<T> ValidAsZeroBits for T {}
    pub trait HostSlice<T> {}
    impl<T> HostSlice<T> for [T] {}
    impl<T> HostSlice<T> for Vec<T> {}
    impl<T, const N: usize> HostSlice<T> for [T; N] {}
    impl<T> HostSlice<T> for PinnedHostSlice<T> {}
    pub trait DevicePtr<T> {}
    pub trait DevicePtrMut<T> {}
    impl<T> DevicePtr<T> for CudaSlice<T> {}
    impl<T> DevicePtrMut<T> for CudaSlice<T> {}
    impl<T> DevicePtr<T> for CudaView<'_, T> {}
    impl<T> DevicePtr<T> for CudaViewMut<'_, T> {}
    impl<T> DevicePtrMut<T> for CudaViewMut<'_, T> {}

    #[derive(Debug)]
    pub struct CudaContext(());
    impl CudaContext {
        pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
            absent()
        }
        pub fn device_count() -> Result<i32, DriverError> {
            absent()
        }
        pub fn name(&self) -> Result<String, DriverError> {
            absent()
        }
        pub fn compute_capability(&self) -> Result<(i32, i32), DriverError> {
            absent()
        }
        pub fn total_mem(&self) -> Result<usize, DriverError> {
            absent()
        }
        pub fn mem_get_info(&self) -> Result<(usize, usize), DriverError> {
            absent()
        }
        pub fn bind_to_thread(&self) -> Result<(), DriverError> {
            absent()
        }
        pub fn synchronize(&self) -> Result<(), DriverError> {
            absent()
        }
        pub fn set_flags(&self, flags: sys::CUctx_flags) -> Result<(), DriverError> {
            absent()
        }
        pub unsafe fn disable_event_tracking(&self) {}
        pub fn new_stream(self: &Arc<Self>) -> Result<Arc<CudaStream>, DriverError> {
            absent()
        }
        pub fn new_stream_with_priority(
            self: &Arc<Self>,
            priority: i32,
        ) -> Result<Arc<CudaStream>, DriverError> {
            absent()
        }
        pub fn new_event(
            self: &Arc<Self>,
            flags: Option<sys::CUevent_flags>,
        ) -> Result<CudaEvent, DriverError> {
            absent()
        }
        pub fn load_module(
            self: &Arc<Self>,
            ptx: super::nvrtc::Ptx,
        ) -> Result<Arc<CudaModule>, DriverError> {
            absent()
        }
        pub unsafe fn alloc_pinned<T>(
            self: &Arc<Self>,
            len: usize,
        ) -> Result<PinnedHostSlice<T>, DriverError> {
            absent()
        }
        pub unsafe fn alloc_pinned_with_flags<T>(
            self: &Arc<Self>,
            len: usize,
            flags: u32,
        ) -> Result<PinnedHostSlice<T>, DriverError> {
            absent()
        }
    }

    #[derive(Debug)]
    pub struct CudaStream(());
    impl CudaStream {
        pub fn context(&self) -> &Arc<CudaContext> {
            unreachable!("no CUDA stream exists in a build without CUDA")
        }
        pub fn synchronize(&self) -> Result<(), DriverError> {
            absent()
        }
        pub fn record_event(
            &self,
            flags: Option<sys::CUevent_flags>,
        ) -> Result<CudaEvent, DriverError> {
            absent()
        }
        pub fn wait(&self, event: &CudaEvent) -> Result<(), DriverError> {
            absent()
        }
        pub fn join(&self, other: &CudaStream) -> Result<(), DriverError> {
            absent()
        }
        pub fn launch_builder<'a>(&'a self, func: &'a CudaFunction) -> LaunchArgs<'a> {
            LaunchArgs(PhantomData)
        }
        pub fn alloc_zeros<T>(&self, len: usize) -> Result<CudaSlice<T>, DriverError> {
            absent()
        }
        pub fn memset_zeros<T, Dst: DevicePtrMut<T>>(
            &self,
            dst: &mut Dst,
        ) -> Result<(), DriverError> {
            absent()
        }
        pub fn clone_htod<T, Src: HostSlice<T> + ?Sized>(
            &self,
            src: &Src,
        ) -> Result<CudaSlice<T>, DriverError> {
            absent()
        }
        pub fn memcpy_htod<T, Src: HostSlice<T> + ?Sized, Dst: DevicePtrMut<T>>(
            &self,
            src: &Src,
            dst: &mut Dst,
        ) -> Result<(), DriverError> {
            absent()
        }
        pub fn memcpy_dtod<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
            &self,
            src: &Src,
            dst: &mut Dst,
        ) -> Result<(), DriverError> {
            absent()
        }
        pub fn clone_dtoh<T, Src: DevicePtr<T>>(&self, src: &Src) -> Result<Vec<T>, DriverError> {
            absent()
        }
        pub fn memcpy_dtoh<T, Src: DevicePtr<T>, Dst: HostSlice<T> + ?Sized>(
            &self,
            src: &Src,
            dst: &mut Dst,
        ) -> Result<(), DriverError> {
            absent()
        }
    }

    #[derive(Debug)]
    pub struct CudaSlice<T>(PhantomData<T>);
    /// cudarc frees the device memory here; kept so that code dropping a slice reads the same.
    impl<T> Drop for CudaSlice<T> {
        fn drop(&mut self) {}
    }
    impl<T> CudaSlice<T> {
        pub fn len(&self) -> usize {
            0
        }
        pub fn is_empty(&self) -> bool {
            true
        }
        pub fn slice(&self, bounds: impl RangeBounds<usize>) -> CudaView<'_, T> {
            CudaView(PhantomData)
        }
        pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> CudaViewMut<'_, T> {
            CudaViewMut(PhantomData)
        }
    }
    #[derive(Debug)]
    pub struct CudaView<'a, T>(PhantomData<&'a T>);
    impl<T> CudaView<'_, T> {
        pub fn len(&self) -> usize {
            0
        }
        pub fn slice(&self, bounds: impl RangeBounds<usize>) -> Self {
            CudaView(PhantomData)
        }
    }
    #[derive(Debug)]
    pub struct CudaViewMut<'a, T>(PhantomData<&'a mut T>);
    impl<T> CudaViewMut<'_, T> {
        pub fn len(&self) -> usize {
            0
        }
        pub fn slice<'b>(&'b self, bounds: impl RangeBounds<usize>) -> CudaView<'b, T> {
            CudaView(PhantomData)
        }
        pub fn slice_mut<'b>(&'b mut self, bounds: impl RangeBounds<usize>) -> CudaViewMut<'b, T> {
            CudaViewMut(PhantomData)
        }
    }

    #[derive(Debug)]
    pub struct PinnedHostSlice<T>(PhantomData<T>);
    impl<T> PinnedHostSlice<T> {
        pub fn len(&self) -> usize {
            0
        }
        pub fn as_slice(&self) -> Result<&[T], DriverError> {
            absent()
        }
        pub fn as_mut_slice(&mut self) -> Result<&mut [T], DriverError> {
            absent()
        }
    }

    #[derive(Clone, Debug)]
    pub struct CudaFunction(());
    #[derive(Debug)]
    pub struct CudaModule(());
    impl CudaModule {
        pub fn load_function(self: &Arc<Self>, fn_name: &str) -> Result<CudaFunction, DriverError> {
            absent()
        }
    }
    #[derive(Debug)]
    pub struct CudaEvent(());
    impl CudaEvent {
        pub fn record(&self, stream: &CudaStream) -> Result<(), DriverError> {
            absent()
        }
        pub fn synchronize(&self) -> Result<(), DriverError> {
            absent()
        }
        pub fn elapsed_ms(&self, end: &Self) -> Result<f32, DriverError> {
            absent()
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct LaunchConfig {
        pub grid_dim: (u32, u32, u32),
        pub block_dim: (u32, u32, u32),
        pub shared_mem_bytes: u32,
    }
    pub struct LaunchArgs<'a>(PhantomData<&'a ()>);
    impl LaunchArgs<'_> {
        pub unsafe fn launch(
            &mut self,
            cfg: LaunchConfig,
        ) -> Result<Option<(CudaEvent, CudaEvent)>, DriverError> {
            absent()
        }
    }
    pub trait PushKernelArg<T> {
        fn arg(&mut self, arg: T) -> &mut Self;
    }
    impl<T> PushKernelArg<T> for LaunchArgs<'_> {
        fn arg(&mut self, arg: T) -> &mut Self {
            self
        }
    }

    pub mod sys {
        /// No driver library in this build.
        pub unsafe fn is_culib_present() -> bool {
            false
        }
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum CUctx_flags {
            CU_CTX_SCHED_AUTO,
            CU_CTX_SCHED_SPIN,
            CU_CTX_SCHED_YIELD,
            CU_CTX_SCHED_BLOCKING_SYNC,
        }
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum CUevent_flags {
            CU_EVENT_DEFAULT,
            CU_EVENT_BLOCKING_SYNC,
            CU_EVENT_DISABLE_TIMING,
            CU_EVENT_INTERPROCESS,
        }
    }

    pub mod result {
        use super::DriverError;
        pub fn mem_get_info() -> Result<(usize, usize), DriverError> {
            Err(DriverError)
        }
        pub mod stream {
            use super::super::DriverError;
            pub fn get_priority_range() -> Result<(i32, i32), DriverError> {
                Err(DriverError)
            }
        }
    }
}

pub mod nvrtc {
    /// A compiled program; never produced in this build.
    #[derive(Clone, Debug)]
    pub struct Ptx(String);
    impl Ptx {
        pub fn from_src<S: Into<String>>(src: S) -> Self {
            Ptx(src.into())
        }
        pub fn to_src(&self) -> String {
            self.0.clone()
        }
    }
    /// cudarc's fields, so that code filling some of them reads the same.
    #[derive(Clone, Debug, Default)]
    pub struct CompileOptions {
        pub ftz: Option<bool>,
        pub prec_sqrt: Option<bool>,
        pub prec_div: Option<bool>,
        pub fmad: Option<bool>,
        pub use_fast_math: Option<bool>,
        pub maxrregcount: Option<usize>,
        pub include_paths: Vec<String>,
        pub arch: Option<&'static str>,
        pub name: Option<String>,
        pub options: Vec<String>,
    }
    #[derive(Debug)]
    pub struct CompileError;
    impl std::fmt::Display for CompileError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("built without CUDA (the cuda feature of horizon-gpu)")
        }
    }
    impl std::error::Error for CompileError {}
    pub fn compile_ptx_with_opts<S: AsRef<str>>(
        src: S,
        opts: CompileOptions,
    ) -> Result<Ptx, CompileError> {
        Err(CompileError)
    }
    pub mod sys {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum nvrtcResult {
            NVRTC_SUCCESS,
            NVRTC_ERROR_INTERNAL_ERROR,
        }
        /// No NVRTC library in this build.
        pub unsafe fn is_culib_present() -> bool {
            false
        }
        pub unsafe fn nvrtcVersion(
            major: *mut ::core::ffi::c_int,
            minor: *mut ::core::ffi::c_int,
        ) -> nvrtcResult {
            nvrtcResult::NVRTC_ERROR_INTERNAL_ERROR
        }
    }
}
