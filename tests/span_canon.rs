//! The span canonical-key equivalence gate.
//!
//! `canon_key(a) == canon_key(b)` must hold exactly when `a` and `b` have equal
//! boundary labels and some bijection of their apexes commutes with both legs.
//! The oracle below decides the right-hand side by backtracking search over
//! apex bijections; it never sorts or otherwise canonicalises pairs.
//!
//! The generated space is small (at most four boundary nodes per side, at most
//! five middle pairs, labels from a two-value alphabet), and half of the
//! generated pairs `(a, b)` take `b` as a shuffle of `a`'s middle pairs, so both
//! sides of the "if and only if" are exercised; the run asserts that they were.

use std::cell::Cell;

use catgraph::span::Span;
use catgraph_surreal::span::{canon_key, encode};
use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};

/// A span's constructor arguments, which — unlike `Span` — are `Debug`.
#[derive(Debug, Clone)]
struct Raw {
    left: Vec<usize>,
    right: Vec<usize>,
    pairs: Vec<(usize, usize)>,
}

impl Raw {
    fn build(&self) -> Span<usize> {
        Span::new(self.left.clone(), self.right.clone(), self.pairs.clone())
            .expect("the generator emits only in-bounds, label-agreeing pairs")
    }
}

/// Up to five label-agreeing middle pairs over the given boundaries.
fn pairs_over(left: Vec<usize>, right: Vec<usize>) -> BoxedStrategy<Vec<(usize, usize)>> {
    if left.is_empty() || right.is_empty() {
        return Just(Vec::new()).boxed();
    }
    prop::collection::vec((0usize..4, 0usize..4), 0..=5)
        .prop_map(move |raw| {
            raw.into_iter()
                .map(|(d, c)| (d % left.len(), c % right.len()))
                .filter(|&(d, c)| left[d] == right[c])
                .collect()
        })
        .boxed()
}

/// A small span: at most four boundary nodes per side, labels in `{0, 1}`.
fn raw_span() -> BoxedStrategy<Raw> {
    (
        prop::collection::vec(0usize..2, 0..=4),
        prop::collection::vec(0usize..2, 0..=4),
    )
        .prop_flat_map(|(left, right)| {
            let pairs = pairs_over(left.clone(), right.clone());
            (Just(left), Just(right), pairs)
        })
        .prop_map(|(left, right, pairs)| Raw { left, right, pairs })
        .boxed()
}

/// `(a, b)`: half the time `b` is `a` with its middle pairs shuffled; a quarter
/// of the time `b` shares `a`'s boundaries with independently drawn pairs; a
/// quarter of the time `b` is independent.
fn span_pair() -> impl Strategy<Value = (Raw, Raw)> {
    prop_oneof![
        2 => raw_span().prop_flat_map(|a| {
            let shuffled = Just(a.pairs.clone()).prop_shuffle();
            (Just(a), shuffled).prop_map(|(a, pairs)| {
                let b = Raw { pairs, ..a.clone() };
                (a, b)
            })
        }),
        1 => raw_span().prop_flat_map(|a| {
            let pairs = pairs_over(a.left.clone(), a.right.clone());
            (Just(a), pairs).prop_map(|(a, pairs)| {
                let b = Raw { pairs, ..a.clone() };
                (a, b)
            })
        }),
        1 => (raw_span(), raw_span()),
    ]
}

/// Whether some bijection `σ` of apex elements satisfies
/// `b.pairs[σ(i)] == a.pairs[i]` for every `i`, with equal boundary labels.
fn apex_isomorphic(a: &Span<usize>, b: &Span<usize>) -> bool {
    if a.left() != b.left() || a.right() != b.right() {
        return false;
    }
    let (from, to) = (a.middle_pairs(), b.middle_pairs());
    if from.len() != to.len() {
        return false;
    }
    let mut used = vec![false; to.len()];
    extend_bijection(0, from, to, &mut used)
}

/// Try every unused target for apex element `i`, backtracking on failure.
fn extend_bijection(
    i: usize,
    from: &[(usize, usize)],
    to: &[(usize, usize)],
    used: &mut [bool],
) -> bool {
    if i == from.len() {
        return true;
    }
    for j in 0..to.len() {
        if !used[j] && to[j].0 == from[i].0 && to[j].1 == from[i].1 {
            used[j] = true;
            if extend_bijection(i + 1, from, to, used) {
                return true;
            }
            used[j] = false;
        }
    }
    false
}

#[test]
fn key_equality_is_exactly_apex_isomorphism() {
    let equal = Cell::new(0usize);
    let unequal = Cell::new(0usize);
    let unequal_same_boundaries = Cell::new(0usize);
    let equal_different_presentation = Cell::new(0usize);

    let mut runner = TestRunner::new(Config {
        cases: 1024,
        failure_persistence: None,
        ..Config::default()
    });
    runner
        .run(&span_pair(), |(a, b)| {
            let (sa, sb) = (a.build(), b.build());
            let keys_agree = canon_key(&sa).expect("a well-formed span has a key")
                == canon_key(&sb).expect("a well-formed span has a key");
            let isomorphic = apex_isomorphic(&sa, &sb);
            prop_assert_eq!(keys_agree, isomorphic, "{:?} vs {:?}", a, b);

            if isomorphic {
                equal.set(equal.get() + 1);
                let addr_a = encode(&sa).expect("encodes").addr().clone();
                let addr_b = encode(&sb).expect("encodes").addr().clone();
                if addr_a != addr_b {
                    equal_different_presentation.set(equal_different_presentation.get() + 1);
                }
            } else {
                unequal.set(unequal.get() + 1);
                if a.left == b.left && a.right == b.right {
                    unequal_same_boundaries.set(unequal_same_boundaries.get() + 1);
                }
            }
            Ok(())
        })
        .expect("key equality is exactly apex isomorphism");

    assert!(equal.get() > 0, "no apex-isomorphic pair was generated");
    assert!(unequal.get() > 0, "no non-isomorphic pair was generated");
    assert!(
        unequal_same_boundaries.get() > 0,
        "no non-isomorphic pair over equal boundaries was generated"
    );
    assert!(
        equal_different_presentation.get() > 0,
        "no isomorphic pair with distinct presentations was generated"
    );
}

proptest! {
    /// The key is reproducible: every load compares a stored key against a
    /// freshly derived one.
    #[test]
    fn the_key_is_deterministic(raw in raw_span()) {
        let span = raw.build();
        prop_assert_eq!(
            canon_key(&span).expect("a well-formed span has a key"),
            canon_key(&span).expect("a well-formed span has a key")
        );
    }
}

/// The oracle itself: a reordering is isomorphic, the braid is not the
/// identity, and a repeated pair is not a single one.
#[test]
fn the_oracle_decides_known_cases() {
    let span = |pairs: Vec<(usize, usize)>| {
        Span::new(vec![7usize, 7], vec![7, 7], pairs).expect("valid pairs")
    };
    let id2 = span(vec![(0, 0), (1, 1)]);
    assert!(apex_isomorphic(&id2, &span(vec![(1, 1), (0, 0)])));
    assert!(!apex_isomorphic(&id2, &span(vec![(0, 1), (1, 0)])));
    assert!(!apex_isomorphic(
        &span(vec![(0, 0)]),
        &span(vec![(0, 0), (0, 0)])
    ));
    assert!(!apex_isomorphic(
        &span(vec![(0, 0), (0, 0)]),
        &span(vec![(0, 0), (1, 1)])
    ));
}
