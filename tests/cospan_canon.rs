//! The canonical-key equivalence gate.
//!
//! The store does not persist `CospanCanon` — it cannot: the type carries no
//! serializable form, its `classes` field is private, and standard-library hash
//! output is not stable across processes. It recomputes the same equivalence
//! data from the public accessors and hashes its own deterministic encoding
//! instead.
//!
//! That re-derivation is only safe if it keys *exactly* the equivalence catgraph
//! decides, in **both** directions:
//!
//! - `a.canonical_form() == b.canonical_form()` ⟹ equal keys. Otherwise the
//!   unique index would admit two rows for one morphism, and the store would
//!   silently hold duplicates it promised not to.
//! - equal keys ⟹ `a.canonical_form() == b.canonical_form()`. Otherwise the
//!   index would *refuse* a genuinely new morphism, and a key lookup would hand
//!   back the wrong cospan entirely.
//!
//! Two properties, because they fail differently. The first walks independently
//! generated pairs over a deliberately small space, so equal and unequal cases
//! both occur often. The second constructs the interesting positive case
//! directly — a cospan and the same cospan with its apex vertices permuted,
//! which is precisely the relabelling the canonical form is invariant under and
//! the one an ordinary structural comparison would get wrong.

use catgraph::cospan::Cospan;
use catgraph_surreal::cospan::canon_key;
use proptest::prelude::*;

/// Small cospans over `usize` labels.
///
/// The space is deliberately tiny — at most three apex vertices, two labels,
/// legs of at most three boundary nodes. A large space would make independently
/// generated pairs almost always unequal, and the property would then only ever
/// test one direction of the "if and only if".
fn any_cospan() -> impl Strategy<Value = Cospan<usize>> {
    prop::collection::vec(0usize..2, 0..4usize).prop_flat_map(|middle| {
        let apex = middle.len();
        (Just(middle), leg(apex), leg(apex)).prop_map(|(middle, left, right)| {
            Cospan::new(left, right, middle).expect("`leg` generates in-bounds entries")
        })
    })
}

/// A leg into an apex of `apex` vertices. An empty apex admits only an empty
/// leg — there is nowhere for a boundary node to land.
fn leg(apex: usize) -> BoxedStrategy<Vec<usize>> {
    if apex == 0 {
        Just(Vec::new()).boxed()
    } else {
        prop::collection::vec(0..apex, 0..4usize).boxed()
    }
}

/// A cospan together with a permutation of its apex vertices, as an
/// `old index → new index` map.
fn cospan_with_apex_permutation() -> impl Strategy<Value = (Cospan<usize>, Vec<usize>)> {
    any_cospan().prop_flat_map(|cospan| {
        let apex = cospan.middle().len();
        (Just(cospan), permutation_of(apex))
    })
}

/// A uniform-ish permutation of `0..n`, derived by sorting random keys.
fn permutation_of(n: usize) -> BoxedStrategy<Vec<usize>> {
    if n == 0 {
        return Just(Vec::new()).boxed();
    }
    prop::collection::vec(any::<u64>(), n)
        .prop_map(move |keys| {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by_key(|&i| (keys[i], i));
            let mut permutation = vec![0; n];
            for (new, &old) in order.iter().enumerate() {
                permutation[old] = new;
            }
            permutation
        })
        .boxed()
}

/// Relabel a cospan's apex vertices, changing nothing about the morphism.
fn permute_apex(cospan: &Cospan<usize>, permutation: &[usize]) -> Cospan<usize> {
    let mut middle = vec![0usize; cospan.middle().len()];
    for (old, &label) in cospan.middle().iter().enumerate() {
        middle[permutation[old]] = label;
    }
    let relabel = |leg: &[usize]| leg.iter().map(|&apex| permutation[apex]).collect();
    Cospan::new(
        relabel(cospan.left_to_middle()),
        relabel(cospan.right_to_middle()),
        middle,
    )
    .expect("a permutation of the apex keeps every leg entry in bounds")
}

proptest! {
    /// The gate itself, in both directions at once.
    #[test]
    fn key_equality_is_exactly_canonical_equality(a in any_cospan(), b in any_cospan()) {
        let keys_agree = canon_key(&a).expect("a well-formed cospan has a key")
            == canon_key(&b).expect("a well-formed cospan has a key");
        let forms_agree = a.canonical_form() == b.canonical_form();
        prop_assert_eq!(keys_agree, forms_agree, "{:?} vs {:?}", a, b);
    }

    /// The positive direction, constructed rather than stumbled upon: permuting
    /// apex vertices is the relabelling the canonical form quotients by, so the
    /// key must not move — while the *presentation* generally does, which is
    /// what makes the two identities different columns.
    #[test]
    fn an_apex_permutation_leaves_the_key_alone(
        (cospan, permutation) in cospan_with_apex_permutation()
    ) {
        let permuted = permute_apex(&cospan, &permutation);
        prop_assert_eq!(cospan.canonical_form(), permuted.canonical_form());
        prop_assert_eq!(
            canon_key(&cospan).expect("a well-formed cospan has a key"),
            canon_key(&permuted).expect("a well-formed cospan has a key")
        );
    }

    /// A key is worthless if it is not reproducible: the store compares stored
    /// keys against freshly derived ones on every load.
    #[test]
    fn the_key_is_deterministic(cospan in any_cospan()) {
        prop_assert_eq!(
            canon_key(&cospan).expect("a well-formed cospan has a key"),
            canon_key(&cospan).expect("a well-formed cospan has a key")
        );
    }
}

/// The permutation generator has to actually permute, or the second property
/// passes vacuously by only ever testing the identity.
#[test]
fn permuting_an_apex_moves_the_presentation() {
    let cospan =
        Cospan::new(vec![0, 1], vec![0, 1], vec![7usize, 7]).expect("id₂'s legs are in bounds");
    let swapped = permute_apex(&cospan, &[1, 0]);
    assert_eq!(swapped.left_to_middle(), [1, 0]);
    assert_eq!(swapped.right_to_middle(), [1, 0]);
    assert_ne!(cospan, swapped);
    assert_eq!(cospan.canonical_form(), swapped.canonical_form());
}
