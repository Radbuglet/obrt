use std::{cell::Cell, mem::transmute, ptr::NonNull};

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

pub fn cell_u64_ne_i32(v: &Cell<u64>) -> &[Cell<i32>; 2] {
    unsafe { transmute(v) }
}

#[cfg(target_endian = "little")]
pub fn cell_u64_ms_i32(v: &Cell<u64>) -> &Cell<i32> {
    &cell_u64_ne_i32(v)[1]
}

#[cfg(target_endian = "big")]
pub fn cell_u64_ms_i32(v: &Cell<u64>) -> &Cell<i32> {
    &cell_u64_ne_i32(v)[0]
}
