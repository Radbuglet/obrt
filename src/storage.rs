use std::{
    cell::{Cell, UnsafeCell},
    error::Error,
    fmt,
    marker::PhantomData,
    mem::{size_of, MaybeUninit},
    num::NonZeroU32,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use derive_where::derive_where;

use crate::{
    token::AcquireExclusiveFor,
    util::{cell_u64_ms_i32, cell_u64_ms_u32, PtrExt},
};

// === Borrow tracker === //

cfgenius::define!(pub tracks_borrow_location = cfg(debug_assertions));

cfgenius::cond! {
    if macro(tracks_borrow_location) {
        use std::panic::Location;

        #[derive(Debug, Clone)]
        struct BorrowTracker(Cell<Option<&'static Location<'static>>>);

        impl BorrowTracker {
            pub const fn new() -> Self {
                Self(Cell::new(None))
            }

            #[inline(always)]
            #[track_caller]
            pub fn set(&self) {
                self.0.set(Some(Location::caller()));
            }
        }
    } else {
        #[derive(Debug, Clone)]
        struct BorrowTracker(());

        impl BorrowTracker {
            pub const fn new() -> Self {
                Self(())
            }

            #[inline(always)]
            pub fn set(&self) {}
        }
    }
}

// === Consistency Checks === //

cfgenius::define!(pub checks_consistency = false());

cfgenius::cond! {
    if macro(checks_consistency) {
        macro_rules! sound_assert {
            ($($args:tt)*) => {
                assert!($($args)*);
            };
        }
    } else {
        macro_rules! sound_assert {
            ($($args:tt)*) => {
                if false {
                    assert!($($args)*);
                }
            };
        }
    }
}

// === BorrowGuard === //

type BorrowGuardMut<'a> = BorrowGuard<'a, true>;
type BorrowGuardRef<'a> = BorrowGuard<'a, false>;

struct BorrowGuard<'a, const MUT: bool>(&'a Cell<i32>);

impl<const MUT: bool> Clone for BorrowGuard<'_, MUT> {
    fn clone(&self) -> Self {
        if MUT {
            self.0.set(
                self.0
                    .get()
                    .checked_sub(1)
                    .expect("too many mutable borrows"),
            );
        } else {
            self.0.set(
                self.0
                    .get()
                    .checked_add(1)
                    .expect("too many immutable borrows"),
            );
        }

        Self(self.0)
    }
}

impl<const MUT: bool> Drop for BorrowGuard<'_, MUT> {
    fn drop(&mut self) {
        if MUT {
            self.0.set(self.0.get() + 1);
        } else {
            self.0.set(self.0.get() - 1);
        }
    }
}

// === AccessError === //

fn fmt_error_common_prefix(
    f: &mut fmt::Formatter<'_>,
    slot_state: i32,
    mutable: bool,
) -> fmt::Result {
    write!(
        f,
        "failed to borrow obj {}: ",
        if mutable { "mutably" } else { "immutably" }
    )?;

    let confounding = slot_state.unsigned_abs();
    write!(
        f,
        "cell is borrowed by {confounding} {}{}",
        if mutable { "reader" } else { "writer" },
        if confounding == 1 { "" } else { "s" }
    )?;

    Ok(())
}

cfgenius::cond! {
    if macro(tracks_borrow_location) {
        #[derive(Debug, Clone)]
        struct CommonBorrowError<const MUT: bool> {
            location: Option<&'static Location<'static>>,
            slot_state: i32,
        }

        impl<const MUT: bool> CommonBorrowError<MUT> {
            fn new<T>(slot: &Slot<T>) -> Self {
                Self {
                    location: slot.borrow_location.0.get(),
                    slot_state: cell_u64_ms_i32(&slot.state).get(),
                }
            }
        }

        impl<const MUT: bool> fmt::Display for CommonBorrowError<MUT> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt_error_common_prefix(f, self.slot_state, MUT)?;

                if let Some(location) = self.location {
                    write!(
                        f,
                        " (first borrow location: {} at {}:{})",
                        location.file(),
                        location.line(),
                        location.column(),
                    )?;
                }

                Ok(())
            }
        }
    } else {
        #[derive(Debug, Clone)]
        struct CommonBorrowError<const MUT: bool> {
            slot_state: i32,
        }

        impl<const MUT: bool> CommonBorrowError<MUT> {
            fn new<T>(slot: &Slot<T>) -> Self {
                Self {
                    slot_state: cell_u64_ms_i32(&slot.state).get(),
                }
            }
        }

        impl<const MUT: bool> fmt::Display for CommonBorrowError<MUT> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt_error_common_prefix(f, self.slot_state, MUT)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum AccessMutError {
    Dead(ObjDeadError),
    Borrow(BorrowMutError),
}

impl fmt::Display for AccessMutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccessMutError::Dead(err) => fmt::Display::fmt(err, f),
            AccessMutError::Borrow(err) => fmt::Display::fmt(err, f),
        }
    }
}

impl Error for AccessMutError {}

#[derive(Debug, Clone)]
pub enum AccessRefError {
    Dead(ObjDeadError),
    Borrow(BorrowRefError),
}

impl fmt::Display for AccessRefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccessRefError::Dead(err) => fmt::Display::fmt(err, f),
            AccessRefError::Borrow(err) => fmt::Display::fmt(err, f),
        }
    }
}

impl Error for AccessRefError {}

#[derive(Debug, Clone)]
pub struct ObjDeadError {
    slot_gen: u32,
    handle_gen: u32,
}

impl fmt::Display for ObjDeadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "obj is dead: slot has generation {} but handle has generation {}",
            self.slot_gen, self.handle_gen,
        )
    }
}

impl Error for ObjDeadError {}

#[derive(Debug, Clone)]
pub struct BorrowMutError(CommonBorrowError<true>);

impl fmt::Display for BorrowMutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Error for BorrowMutError {}

#[derive(Debug, Clone)]
pub struct BorrowRefError(CommonBorrowError<false>);

impl fmt::Display for BorrowRefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Error for BorrowRefError {}

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
    borrow_location: BorrowTracker,

    // This is secretly two 32 bit numbers in disguise. From LSB to MSB...
    //
    // - The first 32 bits indicate the generation. When the slot is dead, this is set to the
    //   generation of the next object to take its place. As such, this value is initialized to one
    //   instead of zero.
    // - The last 32 bits indicate the borrow state (as an i32) if the slot is alive, or the
    //   byte-offset of the next allocation in the free list (as a u32) if the slot is zero. If
    //   there is no next slot in the linked list, this value will be `StorageViewMut::<T>::SLOT_SENTINEL`.
    //
    // Borrow state format:
    //
    // - 0 means unborrowed
    // - positive means immutably borrowed
    // - negative means mutably borrowed
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
    // The size of Slot<T> in bytes.
    const SLOT_SIZE: usize = size_of::<Slot<T>>();

    // The maximum number of slots we can put in a block and still have all the slots be addressable.
    const MAX_COUNT: u16 = (u16::MAX as usize / Self::SLOT_SIZE) as u16;

    // A sentinel value for a slot which is guaranteed to never be a valid offset.
    const SLOT_SENTINEL: u16 = {
        if u16::MAX as usize % Self::SLOT_SIZE == 0 {
            // A slot is at least 8 bytes in size so this is guaranteed to not be a valid offset.
            u16::MAX - 1
        } else {
            u16::MAX
        }
    };

    pub fn alloc(self, value: T) -> Obj<T> {
        let inner = unsafe { &mut *self.inner.0.get() };

        // Ensure that we have a block to hammer.
        if inner.hammered == u16::MAX {
            Self::expand_capacity(inner);
        }

        // Fetch the hammered block.
        let block_idx = inner.hammered;

        sound_assert!(inner.block_ptrs.len() == inner.block_states.len()); // size invariant met?
        sound_assert!(block_idx != u16::MAX); // block actually initialized?
        sound_assert!((block_idx as usize) < inner.block_states.len()); // index is valid?

        let block_ptr = *unsafe { inner.block_ptrs.get_unchecked(block_idx as usize) };
        let block_state = unsafe { inner.block_states.get_unchecked_mut(block_idx as usize) };

        // Fetch the slot in the block.
        let slot_offset = block_state.free_list_head;
        sound_assert!(Self::is_valid_slot_offset(block_idx, slot_offset));

        let slot = unsafe { block_ptr.add_addr(slot_offset as usize).as_ref() };

        // Pop the slot from the free list.
        let next_slot = cell_u64_ms_u32(&slot.state).get() as u16;
        block_state.free_list_head = next_slot;
        block_state.non_free_count += 1;

        // If the slot is now empty, move on to the next hammered block.
        if next_slot == Self::SLOT_SENTINEL {
            inner.hammered = block_state.next_hammered;
        } else {
            sound_assert!(Self::is_valid_slot_offset(block_idx, next_slot));
        }

        // Zero the slot state to mark it as borrowable and fetch the generation.
        let slot_state = slot.state.get();
        let generation = slot_state as u32;

        sound_assert!(generation != 0);
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
        assert!(
            inner.block_ptrs.len() < u16::MAX as usize - 1,
            "cannot allocate more than ~4.29 gigabytes of storage for a given object type"
        );

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
                borrow_location: BorrowTracker::new(),
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
        offset as usize % Self::SLOT_SIZE == 0  // aligned? (and not sentinel)
            && offset < Self::block_len(block) * Self::SLOT_SIZE as u16 // in bounds?
    }

    fn block_len(idx: u16) -> u16 {
        sound_assert!(idx != u16::MAX);
        u16::MAX.min(Self::MAX_COUNT)
    }

    pub fn dealloc(self) {
        todo!();
    }

    #[inline]
    #[track_caller]
    pub fn get(self, obj: Obj<T>) -> ObjRef<'a, T> {
        let inner = unsafe { &*self.inner.0.get() };

        // Fetch the requested block
        let Some(block) = inner.block_ptrs.get(obj.block_idx as usize) else {
            Self::raise_trivial_generation_err(obj);
        };

        // Fetch the requested slot
        sound_assert!(Self::is_valid_slot_offset(obj.block_idx, obj.slot_offset));
        let slot = unsafe { block.add_addr(obj.slot_offset as usize).as_ref() };

        // Ensure that it is borrowable and that doing so will not overflow the counter
        const MASK: u64 = (1 << 63) | (1 << 62) | (u32::MAX as u64);
        if slot.state.get() & MASK != obj.generation.get() as u64 {
            Self::raise_borrow_ref_or_generation_err(slot, obj);
        }

        // If this is the first immutable borrow, set the location.
        if slot.state.get() == obj.generation.get() as u64 {
            slot.borrow_location.set();
        }

        // Increment the immutable borrow counter
        slot.state.set(slot.state.get().wrapping_add(1 << 32));

        // Construct a guard
        let cell_state = cell_u64_ms_i32(&slot.state);
        sound_assert!(cell_state.get() > 0);

        ObjRef {
            _borrow: BorrowGuard(cell_u64_ms_i32(&slot.state)),
            value: NonNull::from(unsafe { (*slot.value.get()).assume_init_ref() }),
        }
    }

    #[inline]
    #[track_caller]
    pub fn get_mut(self, obj: Obj<T>) -> ObjRefMut<'a, T> {
        let inner = unsafe { &*self.inner.0.get() };

        // Fetch the requested block
        let Some(block) = inner.block_ptrs.get(obj.block_idx as usize) else {
            Self::raise_trivial_generation_err(obj);
        };

        // Fetch the requested slot
        sound_assert!(Self::is_valid_slot_offset(obj.block_idx, obj.slot_offset));
        let slot = unsafe { block.add_addr(obj.slot_offset as usize).as_ref() };

        // Ensure that it is borrowable
        if slot.state.get() != obj.generation.get() as u64 {
            Self::raise_borrow_mut_or_generation_err(slot, obj);
        }

        // Since this is the first mutable borrow, set the location.
        slot.borrow_location.set();

        // Set the borrow counter
        let cell_state = cell_u64_ms_i32(&slot.state);
        sound_assert!(cell_state.get() == 0);
        cell_state.set(-1);

        ObjRefMut {
            _variance: PhantomData,
            _borrow: BorrowGuard(cell_state),
            value: NonNull::from(unsafe { (*slot.value.get()).assume_init_mut() }),
        }
    }

    #[cold]
    fn raise_trivial_generation_err(obj: Obj<T>) -> ! {
        panic!(
            "{}",
            ObjDeadError {
                handle_gen: obj.generation.get(),
                slot_gen: 0
            }
        );
    }

    #[cold]
    fn raise_borrow_ref_or_generation_err(slot: &Slot<T>, obj: Obj<T>) -> ! {
        panic!("{}", Self::make_ref_or_generation_err(slot, obj));
    }

    #[cold]
    fn raise_borrow_mut_or_generation_err(slot: &Slot<T>, obj: Obj<T>) -> ! {
        panic!("{}", Self::make_mut_or_generation_err(slot, obj));
    }

    fn make_ref_or_generation_err(slot: &Slot<T>, obj: Obj<T>) -> AccessRefError {
        let slot_gen = slot.state.get() as u32;
        let handle_gen = obj.generation.get();
        if slot_gen != handle_gen {
            AccessRefError::Dead(ObjDeadError {
                slot_gen,
                handle_gen,
            })
        } else {
            AccessRefError::Borrow(BorrowRefError(CommonBorrowError::new(slot)))
        }
    }

    fn make_mut_or_generation_err(slot: &Slot<T>, obj: Obj<T>) -> AccessMutError {
        let slot_gen = slot.state.get() as u32;
        let handle_gen = obj.generation.get();
        if slot_gen != handle_gen {
            AccessMutError::Dead(ObjDeadError {
                slot_gen,
                handle_gen,
            })
        } else {
            AccessMutError::Borrow(BorrowMutError(CommonBorrowError::new(slot)))
        }
    }
}

// ObjRef
pub struct ObjRef<'a, T> {
    _borrow: BorrowGuardRef<'a>,
    value: NonNull<T>,
}

impl<T> Deref for ObjRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.value.as_ref() }
    }
}

// ObjRefMut
pub struct ObjRefMut<'a, T> {
    _variance: PhantomData<&'a mut T>,
    _borrow: BorrowGuardMut<'a>,
    value: NonNull<T>,
}

impl<T> Deref for ObjRefMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.value.as_ref() }
    }
}

impl<T> DerefMut for ObjRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.value.as_mut() }
    }
}

// === Tests === //

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn linked_list_traversal_works() {
        struct Item {
            next: Option<Obj<Self>>,
            value: u64,
        }

        let mut storage = Storage::<Item>::new();
        let storage = storage.borrow_exclusive_mut();
        let items = (0..100_000)
            .map(|i| {
                storage.alloc(Item {
                    next: None,
                    value: i,
                })
            })
            .collect::<Vec<_>>();

        let (start, targets) = generate_permuted_chain(100_000);

        for (src, target) in targets.into_iter().enumerate() {
            if target != usize::MAX {
                storage.get_mut(items[src]).next = Some(items[target]);
            }
        }

        let start = items[start];

        let mut cursor = Some(start);
        let mut accum = 0;

        while let Some(curr) = cursor {
            let curr = storage.get_mut(curr);
            accum += (*curr).value;
            cursor = curr.next;
        }

        assert_eq!(4999950000, accum);
    }

    #[test]
    #[ignore = "expensive"]
    fn allocation_capacity_on_right_order() {
        type Value = [u64; 64];
        let mut storage = Storage::<Value>::new();
        let storage = storage.borrow_exclusive_mut();

        let mut bytes_allocated = 0;
        let alloc_size = std::mem::size_of::<(u64, Value)>();

        assert_panics(|| loop {
            bytes_allocated += alloc_size;
            storage.alloc(std::array::from_fn(|_| 0));
        });

        println!(
            "Failed after allocating {bytes_allocated} bytes ({} objects); delta from ideal is {}",
            bytes_allocated / alloc_size,
            u32::MAX as usize - bytes_allocated,
        );

        assert!(u32::MAX as usize - bytes_allocated < 2_000_000); // ~2 mb of loss are permitted.
    }

    #[test]
    fn storage_ref_celled_properly() {
        let mut storage = Storage::new();
        let storage = storage.borrow_exclusive_mut();
        let obj = storage.alloc(1);

        let a = storage.get(obj);
        let b = storage.get(obj);
        assert_panics(|| storage.get_mut(obj));
        drop(a);
        assert_panics(|| storage.get_mut(obj));
        drop(b);
        let c = storage.get_mut(obj);
        assert_panics(|| storage.get(obj));
        assert_panics(|| storage.get_mut(obj));
        drop(c);
        let _d = storage.get(obj);
    }

    #[test]
    #[ignore = "expensive"]
    fn storage_ref_cell_does_not_overflow() {
        let mut storage = Storage::new();
        let storage = storage.borrow_exclusive_mut();
        let obj = storage.alloc(1);

        for _ in 0..(1 << 30) {
            std::mem::forget(storage.get(obj));
        }

        assert_panics(|| {
            storage.get(obj);
        });
    }

    // === Helpers === //

    fn generate_permuted_chain(n: usize) -> (usize, Vec<usize>) {
        fastrand::seed(4);

        let mut remaining = (0..n).collect::<Vec<_>>();
        let mut chain = (0..n).map(|_| usize::MAX).collect::<Vec<_>>();

        let start = remaining.swap_remove(fastrand::usize(0..remaining.len()));
        let mut cursor = start;

        while !remaining.is_empty() {
            let target = remaining.swap_remove(fastrand::usize(0..remaining.len()));

            chain[cursor] = target;
            cursor = target;
        }

        (start, chain)
    }

    fn assert_panics<R>(f: impl FnOnce() -> R) {
        use std::panic::*;

        let hook = take_hook();
        set_hook(Box::new(|_info| {}));

        let res = catch_unwind(AssertUnwindSafe(|| {
            f();
        }));

        set_hook(hook);

        assert!(res.is_err());
    }
}
