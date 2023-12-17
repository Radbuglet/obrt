use core::fmt;
use std::{
    cell::Cell,
    marker::PhantomData,
    sync::atomic::{AtomicBool, Ordering},
};

use derive_where::derive_where;

// === Token === //

pub unsafe trait ExclusiveToken<N: ?Sized> {}

// === MainThreadToken === //

static HAS_MAIN_THREAD: AtomicBool = AtomicBool::new(false);

thread_local! {
    static IS_MAIN_THREAD: Cell<bool> = const { Cell::new(false) };
}

#[inline]
pub fn is_main_thread() -> bool {
    IS_MAIN_THREAD.with(|v| v.get())
}

#[inline]
#[must_use]
fn try_become_main_thread() -> bool {
    if is_main_thread() {
        return true;
    }

    if HAS_MAIN_THREAD
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        IS_MAIN_THREAD.with(|v| v.set(true));
        true
    } else {
        false
    }
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct MainThreadToken {
    _no_send_or_sync: PhantomData<*const ()>,
}

impl fmt::Debug for MainThreadToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MainThreadToken").finish_non_exhaustive()
    }
}

impl MainThreadToken {
    #[inline]
    pub fn try_acquire() -> Option<Self> {
        if try_become_main_thread() {
            Some(Self {
                _no_send_or_sync: PhantomData,
            })
        } else {
            None
        }
    }

    #[inline]
    pub fn acquire_fmt(attempted_verb: impl fmt::Display) -> Self {
        assert!(
            try_become_main_thread(),
            "Attempted to {attempted_verb} on non-main thread. See the \"multi-threading\"
             section of the module documentation for details.",
        );
        Self {
            _no_send_or_sync: PhantomData,
        }
    }

    #[inline]
    pub fn acquire() -> Self {
        Self::acquire_fmt("perform a main-thread action")
    }

    #[inline]
    pub fn make_ref(self) -> &'static Self {
        &Self {
            _no_send_or_sync: PhantomData,
        }
    }

    pub fn parallelize(self) {
        todo!();
    }
}

unsafe impl<N: ?Sized> ExclusiveToken<N> for MainThreadToken {}

// === TokenCell === //

#[derive_where(Debug; T: Sized + fmt::Debug)]
#[derive_where(Default; T: Sized + Default)]
pub struct TokenCell<T: ?Sized, N: ?Sized>(PhantomData<N>, T);

impl<T: ?Sized, N: ?Sized> TokenCell<T, N> {
    pub const fn new(value: T) -> Self
    where
        T: Sized,
    {
        Self(PhantomData, value)
    }

    pub fn get_exclusive<'a>(&'a self, token: impl ExclusiveToken<N> + 'a) -> T::ExclusiveView
    where
        T: 'a + AcquireExclusiveFor<'a>,
    {
        let _ = token;
        unsafe { self.1.borrow_exclusive() }
    }
}

// === AcquireExclusive === //

pub trait AcquireExclusive: for<'a> AcquireExclusiveFor<'a> {}

impl<T: ?Sized + for<'a> AcquireExclusiveFor<'a>> AcquireExclusive for T {}

pub trait AcquireExclusiveFor<'a, WhereAOutlivesSelf = &'a Self> {
    type ExclusiveView;

    unsafe fn borrow_exclusive(&'a self) -> Self::ExclusiveView;

    fn borrow_exclusive_mut(&'a mut self) -> Self::ExclusiveView {
        unsafe { self.borrow_exclusive() }
    }
}
