#![allow(clippy::empty_loop)]

use std::{
    cell::{Cell, RefCell},
    time::Duration,
};

use bort::{
    storage::{Obj, Storage},
    token::AcquireExclusiveFor,
};
use criterion::{criterion_main, Criterion};

criterion_main!(bench);

fn bench() {
    let mut c = Criterion::default()
        .configure_from_args()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_millis(1500));

    c.bench_function("sequential/contiguous/regular", |c| {
        let mut items = Vec::new();

        struct Item {
            next: usize,
            value: u64,
        }

        for i in 0..100_000 {
            if i != 100_000 - 2 {
                items.push(Item {
                    next: i + 1,
                    value: 100 + i as u64,
                });
            } else {
                items.push(Item {
                    next: usize::MAX,
                    value: 2,
                });
            }
        }

        c.iter(|| {
            let mut cursor = 0;
            let mut accum = 0;

            while cursor < items.len() {
                accum += items[cursor].value;
                cursor = items[cursor].next;
            }

            accum
        });
    });

    c.bench_function("sequential/contiguous/byte_offset", |c| {
        let mut items = Vec::new();

        struct Item {
            next: usize,
            value: u64,
        }

        for i in 0..100_000 {
            if i != 100_000 - 2 {
                items.push(Item {
                    next: (i + 1) * std::mem::size_of::<Item>(),
                    value: 100 + i as u64,
                });
            } else {
                items.push(Item {
                    next: usize::MAX,
                    value: 2,
                });
            }
        }

        let items_ptr = items.as_ptr().cast::<u8>();

        c.iter(|| {
            let mut cursor = 0;
            let mut accum = 0;

            while cursor != usize::MAX {
                let item = unsafe { &*items_ptr.add(cursor).cast::<Item>() };
                accum += item.value;
                cursor = item.next;
            }

            accum
        });
    });

    c.bench_function("sequential/contiguous/ref_cell", |c| {
        let mut items = Vec::new();

        struct Item {
            next: usize,
            value: u64,
        }

        for i in 0..100_000 {
            if i != 100_000 - 2 {
                items.push(RefCell::new(Item {
                    next: i + 1,
                    value: 100 + i as u64,
                }));
            } else {
                items.push(RefCell::new(Item {
                    next: usize::MAX,
                    value: 2,
                }));
            }
        }

        c.iter(|| {
            let mut cursor = 0;
            let mut accum = 0;

            while cursor < items.len() {
                let item = items[cursor].borrow();
                accum += item.value;
                cursor = item.next;
            }

            accum
        });
    });

    c.bench_function("sequential/contiguous/ref_cell_mut", |c| {
        let mut items = Vec::new();

        struct Item {
            next: usize,
            value: u64,
        }

        for i in 0..100_000 {
            if i != 100_000 - 2 {
                items.push(RefCell::new(Item {
                    next: i + 1,
                    value: 100 + i as u64,
                }));
            } else {
                items.push(RefCell::new(Item {
                    next: usize::MAX,
                    value: 2,
                }));
            }
        }

        c.iter(|| {
            let mut cursor = 0;
            let mut accum = 0;

            while cursor < items.len() {
                let item = items[cursor].borrow_mut();
                accum += item.value;
                cursor = item.next;
            }

            accum
        });
    });

    c.bench_function("sequential/contiguous_u32/ref_cell", |c| {
        let mut items = Vec::new();

        struct Item {
            next: u32,
            value: u64,
        }

        for i in 0..100_000 {
            if i != 100_000 - 2 {
                items.push(RefCell::new(Item {
                    next: i + 1,
                    value: 100 + i as u64,
                }));
            } else {
                items.push(RefCell::new(Item {
                    next: u32::MAX,
                    value: 2,
                }));
            }
        }

        c.iter(|| {
            let mut cursor = 0u32;
            let mut accum = 0;

            while (cursor as usize) < items.len() {
                let item = items[cursor as usize].borrow();
                accum += item.value;
                cursor = item.next;
            }

            accum
        });
    });

    c.bench_function("sequential/contiguous_u32/ref_cell_mut", |c| {
        let mut items = Vec::new();

        struct Item {
            next: u32,
            value: u64,
        }

        for i in 0..100_000 {
            if i != 100_000 - 2 {
                items.push(RefCell::new(Item {
                    next: i + 1,
                    value: 100 + i as u64,
                }));
            } else {
                items.push(RefCell::new(Item {
                    next: u32::MAX,
                    value: 2,
                }));
            }
        }

        c.iter(|| {
            let mut cursor = 0u32;
            let mut accum = 0;

            while (cursor as usize) < items.len() {
                let item = items[cursor as usize].borrow_mut();
                accum += item.value;
                cursor = item.next;
            }

            accum
        });
    });

    c.bench_function("sequential/blocky/regular", |c| {
        let mut items = Vec::<Box<[Item]>>::new();

        #[derive(Default)]
        struct Item {
            next_block: u16,
            next_slot: u16,
            value: u64,
        }

        let ipb = 5000;

        for _ in 0..(100_000 / ipb) {
            items.push(Box::from_iter((0..ipb).map(|_| Item::default())));
        }

        let index_of = |idx: usize| (idx / ipb, idx % ipb);

        for i in 0..100_000 {
            let (block, slot) = index_of(i);

            if i != 100_000 - 2 {
                let (next_block, next_slot) = index_of(i + 1);

                items[block][slot] = Item {
                    next_block: next_block as u16,
                    next_slot: next_slot as u16,
                    value: 100 + i as u64,
                };
            } else {
                items[block][slot] = Item {
                    next_block: u16::MAX,
                    next_slot: u16::MAX,
                    value: 2,
                };
            }
        }

        c.iter(|| {
            let mut cursor = (0u16, 0u16);
            let mut accum = 0;

            while (cursor.0 as usize) < items.len() {
                let item = unsafe { items[cursor.0 as usize].get_unchecked(cursor.1 as usize) };
                accum += item.value;
                cursor = (item.next_block, item.next_slot);
            }

            accum
        });
    });

    c.bench_function("sequential/blocky/checked_v1", |c| {
        fn split_cell(v: &Cell<u16>) -> &[Cell<u8>; 2] {
            unsafe { std::mem::transmute(v) }
        }

        let mut items = Vec::<Box<[Item]>>::new();

        #[derive(Default)]
        struct Item {
            gen_and_state: Cell<u16>,
            next_block: u16,
            next_slot: u16,
            next_gen: u16,
            value: u64,
        }

        let ipb = 5000;

        for _ in 0..(100_000 / ipb) {
            items.push(Box::from_iter((0..ipb).map(|_| Default::default())));
        }

        let index_of = |idx: usize| (idx / ipb, idx % ipb);

        for i in 0..100_000 {
            let (block, slot) = index_of(i);

            if i != 100_000 - 2 {
                let (next_block, next_slot) = index_of(i + 1);

                items[block][slot] = Item {
                    next_block: next_block as u16,
                    next_slot: next_slot as u16,
                    next_gen: 1,
                    gen_and_state: Cell::new(1),
                    value: 100 + i as u64,
                };
            } else {
                items[block][slot] = Item {
                    next_block: u16::MAX,
                    next_slot: u16::MAX,
                    next_gen: 1,
                    gen_and_state: Cell::new(1),
                    value: 2,
                };
            }
        }

        c.iter(|| {
            let mut cursor = (0u16, 0u16, 1u16);
            let mut accum = 0;

            while (cursor.0 as usize) < items.len() {
                let item = unsafe { items[cursor.0 as usize].get_unchecked(cursor.1 as usize) };

                if item.gen_and_state.get() != cursor.2 {
                    loop {}
                }
                split_cell(&item.gen_and_state)[1].set(u8::MAX);

                accum += item.value;

                cursor = (item.next_block, item.next_slot, item.next_gen);
                split_cell(&item.gen_and_state)[1].set(0);
            }

            accum
        });
    });

    c.bench_function("sequential/blocky/checked_v2", |c| {
        let mut items = Vec::<Box<[Item]>>::new();

        #[derive(Default)]
        struct Item {
            gen_and_state: Cell<u16>,
            next_block: u16,
            next_slot: u16,
            next_gen: u16,
            value: u64,
        }

        let ipb = 5000;

        for _ in 0..(100_000 / ipb) {
            items.push(Box::from_iter((0..ipb).map(|_| Default::default())));
        }

        let index_of = |idx: usize| (idx / ipb, idx % ipb);

        for i in 0..100_000 {
            let (block, slot) = index_of(i);

            if i != 100_000 - 2 {
                let (next_block, next_slot) = index_of(i + 1);

                items[block][slot] = Item {
                    next_block: next_block as u16,
                    next_slot: next_slot as u16,
                    next_gen: 1,
                    gen_and_state: Cell::new(1),
                    value: 100 + i as u64,
                };
            } else {
                items[block][slot] = Item {
                    next_block: u16::MAX,
                    next_slot: u16::MAX,
                    next_gen: 1,
                    gen_and_state: Cell::new(1),
                    value: 2,
                };
            }
        }

        c.iter(|| {
            let mut cursor = (0u16, 0u16, 1u16);
            let mut accum = 0;

            while (cursor.0 as usize) < items.len() {
                let item = unsafe { items[cursor.0 as usize].get_unchecked(cursor.1 as usize) };

                if item.gen_and_state.get() != cursor.2 {
                    loop {}
                }
                let old_state = item.gen_and_state.get();
                item.gen_and_state.set(u16::MAX);

                accum += item.value;

                cursor = (item.next_block, item.next_slot, item.next_gen);
                item.gen_and_state.set(old_state);
            }

            accum
        });
    });

    c.bench_function("sequential/blocky/checked_v3", |c| {
        let mut items = Vec::<Box<[Item]>>::new();

        #[derive(Default)]
        struct Item {
            gen_and_state: Cell<u16>,
            next_block: u16,
            next_slot: u16,
            next_gen: u16,
            value: u64,
        }

        let ipb = 1000;

        for _ in 0..(100_000 / ipb) {
            items.push(Box::from_iter((0..ipb).map(|_| Default::default())));
        }

        let index_of = |idx: usize| (idx / ipb, idx % ipb);

        for i in 0..100_000 {
            let (block, slot) = index_of(i);

            if i != 100_000 - 2 {
                let (next_block, next_slot) = index_of(i + 1);

                items[block][slot] = Item {
                    next_block: u16::try_from(next_block * std::mem::size_of::<Box<[Item]>>())
                        .unwrap(),
                    next_slot: u16::try_from(next_slot * std::mem::size_of::<Item>()).unwrap(),
                    next_gen: 1,
                    gen_and_state: Cell::new(1),
                    value: 100 + i as u64,
                };
            } else {
                items[block][slot] = Item {
                    next_block: u16::MAX,
                    next_slot: u16::MAX,
                    next_gen: 1,
                    gen_and_state: Cell::new(1),
                    value: 2,
                };
            }
        }

        c.iter(|| {
            let mut cursor = (0u16, 0u16, 1u16);
            let mut accum = 0;

            while cursor.0 != u16::MAX {
                let item = unsafe {
                    &*(*items
                        .as_ptr()
                        .cast::<u8>()
                        .add(cursor.0 as usize)
                        .cast::<Box<[Item]>>())
                    .as_ptr()
                    .cast::<u8>()
                    .add(cursor.1 as usize)
                    .cast::<Item>()
                };

                if item.gen_and_state.get() != cursor.2 {
                    loop {}
                }
                let old_state = item.gen_and_state.get();
                item.gen_and_state.set(u16::MAX);

                accum += item.value;

                cursor = (item.next_block, item.next_slot, item.next_gen);
                item.gen_and_state.set(old_state);
            }

            accum
        });
    });

    c.bench_function("sequential/beap/checked", |c| {
        #[derive(Default)]
        struct Item {
            gen_and_state: Cell<u32>,
            next_ptr: u32,
            next_gen: u32,
            value: u64,
        }

        let heap = beap::reserve(beap::round_up_to_page_size(u32::MAX as usize)).unwrap();
        unsafe {
            beap::commit(
                heap,
                beap::round_up_to_page_size(100_000 * std::mem::size_of::<Item>()),
            )
        }
        .unwrap();

        let heap = heap.cast::<Item>().as_ptr();

        for i in 0..100_000 {
            let value = if i != 100_000 - 2 {
                Item {
                    value: 100 + i as u64,
                    gen_and_state: Cell::new(1),
                    next_ptr: (i + 1) * std::mem::size_of::<Item>() as u32,
                    next_gen: 1,
                }
            } else {
                Item {
                    value: 2,
                    gen_and_state: Cell::new(1),
                    next_ptr: u32::MAX,
                    next_gen: 1,
                }
            };

            unsafe { heap.add(i as usize).write(value) };
        }

        let heap = heap.cast::<u8>();

        c.iter(|| {
            let mut cursor = (0u32, 1u32);
            let mut accum = 0;

            while cursor.0 != u32::MAX {
                let item = unsafe { &*heap.add(cursor.0 as usize).cast::<Item>() };
                if item.gen_and_state.get() != cursor.1 {
                    loop {}
                }

                let old_state = cursor.1;
                item.gen_and_state.set(u32::MAX);

                accum += item.value;
                cursor = (item.next_ptr, item.next_gen);
                item.gen_and_state.set(old_state);
            }

            accum
        });
    });

    c.bench_function("sequential/obrt/mut", |c| {
        #[derive(Default)]
        struct Item {
            next: Option<Obj<Self>>,
            value: u64,
        }

        let mut storage = Storage::<Item>::new();
        let storage = storage.borrow_exclusive_mut();

        let head;
        {
            let mut curr = storage.alloc(Item {
                next: None,
                value: 100,
            });
            head = curr;

            for i in 1..100_000 {
                let next = storage.alloc(Item {
                    next: None,
                    value: i + 100,
                });
                storage.get_mut(curr).next = Some(next);
                curr = next;
            }
        }

        c.iter(|| {
            let mut cursor = Some(head);
            let mut accum = 0;

            while let Some(curr) = cursor {
                let curr = storage.get_mut(curr);
                accum += curr.value;
                cursor = curr.next;
            }

            accum
        });
    });

    c.bench_function("random/contiguous/byte_offset", |c| {
        struct Item {
            next: usize,
            value: u64,
        }

        let mut items = (0..100_000)
            .map(|i| Item {
                next: usize::MAX,
                value: i + 100,
            })
            .collect::<Vec<_>>();

        let (start, targets) = generate_permuted_chain(100_000);

        for (src, target) in targets.into_iter().enumerate() {
            if target != usize::MAX {
                items[src].next = target * std::mem::size_of::<Item>();
            }
        }

        let start = start * std::mem::size_of::<Item>();
        let items_ptr = items.as_ptr().cast::<u8>();

        c.iter(|| {
            let mut cursor = start;
            let mut accum = 0;

            while cursor != usize::MAX {
                let item = unsafe { &*items_ptr.add(cursor).cast::<Item>() };
                accum += item.value;
                cursor = item.next;
            }

            accum
        });
    });

    c.bench_function("random/obrt/mut", |c| {
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
                    value: i + 100,
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

        c.iter(|| {
            let mut cursor = Some(start);
            let mut accum = 0;

            while let Some(curr) = cursor {
                let curr = storage.get_mut(curr);
                accum += curr.value;
                cursor = curr.next;
            }

            accum
        });
    });
}

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
