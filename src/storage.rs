use std::{
    cell::{Cell, UnsafeCell},
    fmt,
    marker::PhantomData,
    mem::{size_of, MaybeUninit},
    num::NonZeroU32,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use derive_where::derive_where;

use crate::{token::AcquireExclusiveFor, util::PtrExt};

// === Storage === //

// Data structures
pub struct Storage<T>(UnsafeCell<StorageInner<T>>);

struct StorageInner<T> {
    // The index of the first hammered block in the hammered block linked list or `u16::MAX` if
    // there is no hammered block available. This block is guaranteed to have at least one free
    // slot.
    hammered: u16,

    // We use raw pointers rather than pointers to slices because doing so ensures that the structure
    // has a power-of-two size, which makes computing byte offsets into the array much more efficient.
    block_ptrs: Vec<NonNull<Slot<T>>>,
    block_states: Vec<BlockState>,
}

struct BlockState {
    // The byte offset into the block of the first `Slot<T>` instance in the free list. The state
    // of this field is undefined if there is no linked list head.
    free_list_head: u16,

    // The index of the next hammered block or `u16::MAX` if there is no next block.
    next_hammered: u16,

    // The number of slots which are currently allocated.
    non_free_count: u16,
}

struct Slot<T> {
    // This is secretly two u32s in disguise. From LSB to MSB...
    //
    // - The first 32 bits indicate the generation. When the slot is dead, this is set to the
    //   generation of the next object to take its place. As such, this value is initialized to one
    //   instead of zero.
    // - The last 32 bits indicate the borrow state if the slot is alive, or the byte-offset
    //   of the next allocation in the free list if the slot is zero. If there is no next slot
    //   in the linked list, this value will be `u32::MAX`.
    //
    // Borrow state format:
    //
    // - 0 means unborrowed
    // - TODO
    //
    state: Cell<u64>,
    value: UnsafeCell<MaybeUninit<T>>,
}

#[derive_where(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct Obj<T> {
    _ty: PhantomData<fn() -> T>,
    block_idx: u16,
    slot_offset: u16,
    generation: NonZeroU32,
}

// Lifecycle
impl<T> Storage<T> {
    pub const fn new() -> Self {
        Self(UnsafeCell::new(StorageInner {
            hammered: u16::MAX,
            block_ptrs: Vec::new(),
            block_states: Vec::new(),
        }))
    }
}

impl<T> Drop for Storage<T> {
    fn drop(&mut self) {
        // TODO
    }
}

impl<T> fmt::Debug for Storage<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Storage").finish_non_exhaustive()
    }
}

impl<T> Default for Storage<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Threading
#[derive_where(Debug, Copy, Clone)]
pub struct StorageViewMut<'a, T> {
    _no_sync: PhantomData<UnsafeCell<()>>,
    inner: &'a Storage<T>,
}

unsafe impl<T: Send> Sync for Storage<T> {}
unsafe impl<T: Send> Send for Storage<T> {}

impl<'a, T> AcquireExclusiveFor<'a> for Storage<T> {
    type ExclusiveView = StorageViewMut<'a, T>;

    unsafe fn borrow_exclusive(&'a self) -> Self::ExclusiveView {
        StorageViewMut {
            _no_sync: PhantomData,
            inner: self,
        }
    }
}

// ViewMut
impl<'a, T: 'a> StorageViewMut<'a, T> {
    const SLOT_SIZE: usize = size_of::<Slot<T>>();
    const MAX_COUNT: u16 = (u16::MAX as usize / Self::SLOT_SIZE) as u16;

    pub fn alloc(self, value: T) -> Obj<T> {
        let inner = unsafe { &mut *self.inner.0.get() };

        // Ensure that we have a block to hammer.
        if inner.hammered == u16::MAX {
            Self::expand_capacity(inner);
        }

        // Fetch the next slot in the block.
        let block_idx = inner.hammered;

        debug_assert_eq!(inner.block_ptrs.len(), inner.block_states.len());
        debug_assert_ne!(block_idx, u16::MAX);
        debug_assert!((block_idx as usize) < inner.block_states.len());

        let block_ptr = *unsafe { inner.block_ptrs.get_unchecked(block_idx as usize) };
        let block_state = unsafe { inner.block_states.get_unchecked_mut(block_idx as usize) };
        let slot_offset = block_state.free_list_head;
        debug_assert!(Self::is_valid_slot_offset(block_idx, slot_offset));

        let slot = unsafe { block_ptr.add_addr(slot_offset as usize).as_ref() };

        // Pop the slot from the free list.
        let next_slot = (slot.state.get() >> 32) as u16;
        block_state.free_list_head = next_slot;
        block_state.non_free_count += 1;

        if next_slot == u16::MAX {
            inner.hammered = block_state.next_hammered;
        } else {
            debug_assert!(Self::is_valid_slot_offset(block_idx, next_slot));
        }

        // Zero the slot state to mark it as borrowable and fetch the generation.
        let slot_state = slot.state.get();
        let generation = slot_state as u32;

        debug_assert_eq!(generation, 0);
        slot.state.set(generation as u64);

        let generation = unsafe { NonZeroU32::new_unchecked(generation) };

        // Write its initial value.
        unsafe { &mut *slot.value.get() }.write(value);

        // Create an Obj handle for users to access it.
        Obj {
            _ty: PhantomData,
            block_idx,
            slot_offset,
            generation,
        }
    }

    #[cold]
    fn expand_capacity(inner: &mut StorageInner<T>) {
        // Ensure that we haven't created too many blocks yet.
        assert!(inner.block_ptrs.len() < u16::MAX as usize - 1);

        // Determine an index and length for this block. The call to `block_len` validates the
        // layout for this block.
        let block_idx = inner.block_ptrs.len() as u16;
        let block_len = Self::block_len(block_idx);

        // Allocate the block's data.
        let alloc = Box::from_iter((0..block_len).map(|i| {
            let generation = 1;
            let next_free = if i == block_len - 1 {
                u32::MAX
            } else {
                Self::SLOT_SIZE as u32 * (i as u32 + 1)
            };

            Slot {
                state: Cell::new(generation + ((next_free as u64) << 32)),
                value: UnsafeCell::new(MaybeUninit::<T>::uninit()),
            }
        }));
        let alloc = NonNull::from(Box::leak(alloc)).cast::<Slot<T>>();

        // Register the block and push it to the front of the hammered block list.
        inner.block_ptrs.push(alloc);
        inner.block_states.push(BlockState {
            free_list_head: 0,
            next_hammered: inner.hammered,
            non_free_count: 0,
        });
        inner.hammered = block_idx;
    }

    fn is_valid_slot_offset(block: u16, offset: u16) -> bool {
        offset as usize % Self::SLOT_SIZE == 0
            && offset < Self::block_len(block) * Self::SLOT_SIZE as u16
    }

    fn block_len(idx: u16) -> u16 {
        debug_assert_ne!(idx, u16::MAX);
        u16::MAX.min(Self::MAX_COUNT)
    }

    pub fn dealloc(self) {
        todo!();
    }

    pub fn get(self) {
        todo!();
    }

    #[inline]
    pub fn get_mut(self, obj: Obj<T>) -> ObjMut<'a, T> {
        let inner = unsafe { &*self.inner.0.get() };

        let block = unsafe { inner.block_ptrs.get_unchecked(obj.block_idx as usize) };

        debug_assert!(Self::is_valid_slot_offset(obj.block_idx, obj.slot_offset));
        let slot = unsafe { block.add_addr(obj.slot_offset as usize).as_ref() };

        if slot.state.get() != obj.generation.get() as u64 {
            Self::borrow_or_generation_err(inner, obj);
        }

        slot.state.set(slot.state.get() | ((u32::MAX as u64) << 32));

        ObjMut {
            _variance: PhantomData,
            state: &slot.state,
            value: NonNull::from(unsafe { (*slot.value.get()).assume_init_mut() }),
        }
    }

    #[cold]
    fn borrow_or_generation_err(inner: &StorageInner<T>, obj: Obj<T>) -> ! {
        let _ = (inner, obj);
        panic!("a big bad has occurred");
    }
}

pub struct ObjMut<'a, T> {
    _variance: PhantomData<&'a mut T>,
    state: &'a Cell<u64>,
    value: NonNull<T>,
}

impl<T> Deref for ObjMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.value.as_ref() }
    }
}

impl<T> DerefMut for ObjMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.value.as_mut() }
    }
}

impl<T> Drop for ObjMut<'_, T> {
    fn drop(&mut self) {
        self.state.set(self.state.get() & (u32::MAX as u64));
    }
}
