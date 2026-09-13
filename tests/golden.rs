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
//! intentional, bump the affected codec tag (`TERM_CODEC`, `COSPAN_CODEC`,
//! `WEIGHT_CODEC`, `RULE_SET_CODEC`, `REWRITE_RUN_CODEC`, `DERIVATION_CODEC`,
//! `DOCUMENT_CODEC`, `BUS_CODEC`) and re-pin the vector **in the same commit** —
//! that is what the tags exist for. If the change was not intentional, a
//! dependency just moved the on-disk format out from under the store.
//!
//! The pre-image is not only the *encoding* but the choice of what goes into it,
//! so widening an address is a codec bump too: `cge2` folded the derivation
//! edge's denormalized `rule_set` and `cost_model` into its digest, which moved
//! every edge address and was re-pinned here alongside the tag.
//!
//! Engine-free on purpose: identity must not depend on which engine feature is
//! compiled in, and these run on every lane.

use std::borrow::Cow;

use catgraph::cospan::Cospan;
use catgraph_applied::prop::colored::ColoredExpr;
use catgraph_applied::prop::presentation::rewrite::{RewriteRule, optimize};
use catgraph_applied::prop::{Free, PropSignature};
use catgraph_surreal::{bus, cospan, doc, lineage, term, weight};
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
    let fixture =
        Cospan::new(vec![0, 0], vec![0], vec![3usize, 9]).expect("μ's legs are in bounds");
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

/// `id₁ : 1 → 1` — the right-hand side of the fixture rule.
fn fixture_identity() -> ColoredExpr<Gen> {
    ColoredExpr::new(vec![()], Free::<Gen>::identity(1)).expect("id₁ type-checks")
}

/// `Δ ; μ ⇒ id₁` — the fixture rule set: parallel sides, a non-empty left-hand
/// side, and a mono interface.
fn fixture_rules() -> Vec<(ColoredExpr<Gen>, ColoredExpr<Gen>)> {
    vec![(fixture_term(), fixture_identity())]
}

#[test]
fn rule_set_identity_is_pinned_to_exact_bytes() {
    let record = lineage::encode_rule_set(&fixture_rules()).expect("the fixture encodes");
    assert_eq!(record.codec(), "cgs1");
    assert_eq!(
        record.addr().as_str(),
        "b3_8ba60ef5b2b03b88c1953ec42c456b13c96fd806238aaaae27a5cc86a9098c14"
    );
}

/// A run's address covers everything about it that is reproducible, so pinning
/// it pins the whole column set — and the trace with it, since the steps are
/// part of the pre-image.
#[test]
fn run_and_derivation_identity_are_pinned_to_exact_bytes() {
    let rule_set = lineage::encode_rule_set(&fixture_rules()).expect("the fixture encodes");
    let compiled: Vec<RewriteRule<Gen>> = rule_set.revalidate().expect("it revalidates");
    let start = fixture_term();
    let outcome = optimize(&start, &compiled, 16, |_| 1).expect("the search runs");

    let run = lineage::encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");
    assert_eq!(run.codec(), "cgt1");
    assert_eq!(
        run.addr().as_str(),
        "b3_94386ff918a31a91c67ce4da92b73741edab0a2162928a30760ebe0b3ac082bc"
    );

    let edge = lineage::encode_derivation(&run).expect("the endpoints imply an edge");
    assert_eq!(edge.codec(), "cge2");
    assert_eq!(
        edge.addr().as_str(),
        "b3_7a57ab5622cd95676da7b20968ff27839ed569edec05e448dbd5ac582a9a8f93"
    );
}

#[test]
fn document_digest_is_pinned_to_exact_bytes() {
    let record = doc::encode(
        "state-1",
        "solver",
        &serde_json::json!({ "beliefs": [0.25, 0.5, 0.25], "boundary": 7 }),
    )
    .expect("the fixture encodes");
    assert_eq!(record.codec(), "cgd1");
    assert_eq!(
        record.digest(),
        "b3_0ea375d42d7ab4468bbf9fd9b8058e3029632f2502cf800da5c04a94a770dde1"
    );
}

#[test]
fn bus_addresses_are_pinned_to_exact_bytes() {
    assert_eq!(bus::BUS_CODEC, "cgb1");
    assert_eq!(
        bus::event_address("goals", 3).as_str(),
        "b3_ed5f43239887a901a1bc676d11cdbbbb142446fb2e0e358d6ed8fb4900634424"
    );
    assert_eq!(
        bus::allocator_key("goals"),
        "b3_4809bd221996a318c60622010051244b744064a3ecb209902f57ee75ea6f1165"
    );
    assert_eq!(
        bus::cursor_key("worker-1"),
        "b3_1125adaa8b569a204ef86fe171e9dc6c3851ad8d5f91f0b102fac1efc30b5035"
    );
}
