//! Golden digest vectors: the on-disk identity of all three tiers, pinned to
//! exact bytes.
//!
//! Every hash pre-image in this crate is `serde_json` output, so every content
//! address and canonical key ultimately rests on serde_json's byte-level
//! rendering and on the field order of the structs being serialized. Nothing
//! else in the test suite notices if either shifts: a rendering change would
//! give the same value a *new* address and a *new* canonical key together, so
//! round-trip tests keep passing while old rows start failing revalidation as
//! corrupt and re-writes of already-stored data quietly duplicate.
//!
//! These vectors are the tripwire. If one fails and the change was
//! intentional, bump the affected codec tag (`TERM_CODEC` / `COSPAN_CODEC` /
//! `WEIGHT_CODEC`) and re-pin the vector **in the same commit** — that is what
//! the tags exist for. If the change was not intentional, a dependency just
//! moved the on-disk format out from under the store.
//!
//! Engine-free on purpose: identity must not depend on which engine feature is
//! compiled in, and these run on every lane.

use std::borrow::Cow;

use catgraph::cospan::Cospan;
use catgraph_applied::prop::colored::ColoredExpr;
use catgraph_applied::prop::{Free, PropSignature};
use catgraph_surreal::{cospan, term, weight};
use serde::{Deserialize, Serialize};

/// The fixture signature: four unit variants, derived serde, one color.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
enum Gen {
    /// `Δ : 1 → 2`
    Copy,
    /// `μ : 2 → 1`
    Add,
}

impl PropSignature for Gen {
    type Color = ();

    fn source_word(&self) -> Cow<'_, [()]> {
        Cow::Owned(match self {
            Self::Copy => vec![()],
            Self::Add => vec![(), ()],
        })
    }

    fn target_word(&self) -> Cow<'_, [()]> {
        Cow::Owned(match self {
            Self::Copy => vec![(), ()],
            Self::Add => vec![()],
        })
    }
}

/// `Δ ; μ : 1 → 1` — the fixture term.
fn fixture_term() -> ColoredExpr<Gen> {
    let expr = Free::compose(Free::generator(Gen::Copy), Free::<Gen>::generator(Gen::Add))
        .expect("Δ ; μ composes");
    ColoredExpr::new(vec![()], expr).expect("Δ ; μ type-checks")
}

#[test]
fn term_identity_is_pinned_to_exact_bytes() {
    let record = term::encode(&fixture_term()).expect("the fixture encodes");
    assert_eq!(record.codec(), "cgj2");
    assert_eq!(
        (
            record.addr().as_str(),
            record.nf_class(),
            record.signature()
        ),
        (
            "b3_a0fe125c31779cf32abf1c2b87fc94f832002c284c7139bedd0d7723fcde2e49",
            "37bbf8fc3705c21799e68ced0210855b83edb5e6830d05b9ad9029cfc2b99d35",
            "[[null],[null]]",
        )
    );
}

#[test]
fn cospan_identity_is_pinned_to_exact_bytes() {
    // μ-shape with a scalar: two domain wires onto one apex vertex, one
    // codomain wire, one untouched vertex.
    let fixture = Cospan::new(vec![0, 0], vec![0], vec![3usize, 9]);
    let record = cospan::encode(&fixture).expect("the fixture encodes");
    assert_eq!(record.codec(), "cgc1");
    assert_eq!(
        (record.addr().as_str(), record.canon_key()),
        (
            "b3_c1693380253896f7f36ecefe6232e659971ada00756cd8775268973f9492f6fc",
            "b3_9f5e9818e8971f77630b264110ddb989330cc0b1ce5561b91d42815efe81c4cd",
        )
    );
}

#[test]
fn weight_key_is_pinned_to_exact_bytes() {
    let key = weight::record_key("genome", "layer-0").expect("a pair has a key");
    assert_eq!(
        key,
        "b3_aa84f12f952605d8d87eaa56b38e96ede8ce7d9f921282d50446b38d6b694d31"
    );
}
