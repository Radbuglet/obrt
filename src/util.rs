use std::ptr::NonNull;

pub trait PtrExt {
    unsafe fn add_addr(self, offset: usize) -> Self;
}

impl<T> PtrExt for *const T {
    unsafe fn add_addr(self, offset: usize) -> Self {
        self.cast::<u8>().add(offset).cast()
    }
}

impl<T> PtrExt for *mut T {
    unsafe fn add_addr(self, offset: usize) -> Self {
        self.cast::<u8>().add(offset).cast()
    }
}

impl<T> PtrExt for NonNull<T> {
    unsafe fn add_addr(self, offset: usize) -> Self {
        NonNull::new_unchecked(self.as_ptr().add_addr(offset))
    }
}
