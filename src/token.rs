pub trait AcquireExclusive: for<'a> AcquireExclusiveFor<'a> {}

impl<T: ?Sized + for<'a> AcquireExclusiveFor<'a>> AcquireExclusive for T {}

pub trait AcquireExclusiveFor<'a, WhereAOutlivesSelf = &'a Self> {
    type ExclusiveView;

    unsafe fn borrow_exclusive(&'a self) -> Self::ExclusiveView;

    fn borrow_exclusive_mut(&'a mut self) -> Self::ExclusiveView {
        unsafe { self.borrow_exclusive() }
    }
}
