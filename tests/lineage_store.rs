//! End-to-end lineage behaviour against the in-memory engine.
//!
//! These need a real engine compiled in, so the whole file is gated on `mem`.
//!
//! Three things are pinned here that no unit test can reach. The **derivation
//! edge** is one: a relation table's `in`/`out` are defined by `DEFINE TABLE`
//! rather than by the DDL, they are written by `RELATE` rather than by a field
//! write, and whether re-relating an existing edge is idempotent is a property
//! of the engine, not of the encoding. The **trace round trip** is the second:
//! a `SCHEMAFULL` object column drops nested keys nobody declared, so a step's
//! `rule` and `matched_edges` surviving is a schema question. And the third is
//! the **shared transaction**: a run's trace and its edge either both land or
//! neither does.

#![cfg(feature = "mem")]

use std::borrow::Cow;
use std::marker::PhantomData;

use catgraph_applied::prop::colored::ColoredExpr;
use catgraph_applied::prop::presentation::rewrite::{RewriteOutcome, RewriteRule, optimize};
use catgraph_applied::prop::{Free, PropSignature};
use catgraph_surreal::{
    LineageStore, RuleSetAddr, Store, StoreBuilder, StoreError, TermStore, lineage, schema,
};
use serde::{Deserialize, Serialize};
use surrealdb::types::{RecordId, RecordIdKey, SurrealValue};

/// Two generators that can cancel: `Δ ; μ ⇒ id₁` is a rule with a non-empty
/// left-hand side, a mono interface, and parallel sides.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
enum Gen {
    /// `Δ : 1 → 2`
    Copy,
    /// `μ : 2 → 1`
    Add,
}

/// One variant byte: `Copy` 0, `Add` 1.
impl catgraph::CanonicalEncode for Gen {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        out.push(match self {
            Self::Copy => 0,
            Self::Add => 1,
        });
    }
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

fn copy_then_add() -> ColoredExpr<Gen> {
    let expr = Free::compose(Free::generator(Gen::Copy), Free::<Gen>::generator(Gen::Add))
        .expect("Δ ; μ composes");
    ColoredExpr::new(vec![()], expr).expect("Δ ; μ type-checks")
}

fn identity() -> ColoredExpr<Gen> {
    ColoredExpr::new(vec![()], Free::<Gen>::identity(1)).expect("id₁ type-checks")
}

fn rules() -> Vec<(ColoredExpr<Gen>, ColoredExpr<Gen>)> {
    vec![(copy_then_add(), identity())]
}

/// `μ ; Δ : 2 → 2`, whose left-hand side cannot fire on `Δ ; μ`.
fn add_then_copy() -> ColoredExpr<Gen> {
    let expr = Free::compose(Free::generator(Gen::Add), Free::<Gen>::generator(Gen::Copy))
        .expect("μ ; Δ composes");
    ColoredExpr::new(vec![(), ()], expr).expect("μ ; Δ type-checks")
}

/// `id₂ : 2 → 2`.
fn identity_two() -> ColoredExpr<Gen> {
    ColoredExpr::new(vec![(), ()], Free::<Gen>::identity(2)).expect("id₂ type-checks")
}

/// Two rules, so that the order they are rebuilt in is observable.
fn two_rules() -> Vec<(ColoredExpr<Gen>, ColoredExpr<Gen>)> {
    vec![
        (copy_then_add(), identity()),
        (add_then_copy(), identity_two()),
    ]
}

/// `μ ; Δ ; μ ; Δ : 2 → 2` — overlapping reducible sites for both rules.
fn two_reducible_sites() -> ColoredExpr<Gen> {
    let add_copy = || {
        Free::compose(Free::generator(Gen::Add), Free::<Gen>::generator(Gen::Copy))
            .expect("μ ; Δ composes")
    };
    let twice = Free::compose(add_copy(), add_copy()).expect("μ ; Δ ; μ ; Δ composes");
    ColoredExpr::new(vec![(), ()], twice).expect("μ ; Δ ; μ ; Δ type-checks")
}

/// A signature type that is itself neither `Send` nor `Sync`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
struct NotSend(PhantomData<*const ()>);

/// The type has one value, which encodes to no bytes.
impl catgraph::CanonicalEncode for NotSend {
    fn encode_canonical(&self, _out: &mut Vec<u8>) {}
}

impl PropSignature for NotSend {
    type Color = ();

    fn source_word(&self) -> Cow<'_, [()]> {
        Cow::Owned(vec![()])
    }

    fn target_word(&self) -> Cow<'_, [()]> {
        Cow::Owned(vec![()])
    }
}

async fn connect(database: &str) -> Store {
    StoreBuilder::new("memory")
        .namespace("catgraph_test")
        .database(database)
        .connect()
        .await
        .expect("connecting to the in-memory engine")
}

async fn bootstrapped(database: &str) -> (Store, LineageStore<Gen>) {
    let store = connect(database).await;
    let lineage = LineageStore::open(store.clone())
        .await
        .expect("opening the lineage store bootstraps and verifies");
    (store, lineage)
}

/// Store a rule set, compile it, and run the optimizer over it.
async fn a_run(
    lineage: &LineageStore<Gen>,
) -> (RuleSetAddr, ColoredExpr<Gen>, RewriteOutcome<Gen>) {
    let addr = lineage
        .put_rule_set(&rules())
        .await
        .expect("storing a rule set");
    let compiled: Vec<RewriteRule<Gen>> = lineage
        .get_rule_set(&addr)
        .await
        .expect("loading it back")
        .expect("it is present");
    let start = copy_then_add();
    let outcome = optimize(&start, &compiled, 16, |_| 1).expect("the search runs");
    (addr, start, outcome)
}

/// [`a_run`] over two rules and a start with a site for each, so the trace is
/// two *distinct* steps and the stored rule order is observable.
async fn a_two_rule_run(
    lineage: &LineageStore<Gen>,
) -> (RuleSetAddr, ColoredExpr<Gen>, RewriteOutcome<Gen>) {
    let addr = lineage
        .put_rule_set(&two_rules())
        .await
        .expect("storing a rule set");
    let compiled: Vec<RewriteRule<Gen>> = lineage
        .get_rule_set(&addr)
        .await
        .expect("loading it back")
        .expect("it is present");
    let start = two_reducible_sites();
    let outcome = optimize(&start, &compiled, 64, |_| 1).expect("the search runs");
    (addr, start, outcome)
}

// ----------------------------------------------------------------- rule sets

#[tokio::test]
async fn a_rule_set_comes_back_as_compiled_rules() {
    let (_store, lineage) = bootstrapped("lineage_rule_set").await;
    let addr = lineage.put_rule_set(&rules()).await.expect("storing");
    let loaded = lineage
        .get_rule_set(&addr)
        .await
        .expect("loading")
        .expect("present");
    assert_eq!(loaded.len(), 1);

    // The rules actually work, which is the only thing "compiled" can mean.
    let outcome = optimize(&copy_then_add(), &loaded, 16, |_| 1).expect("the search runs");
    assert!(!outcome.steps().is_empty());
}

#[tokio::test]
async fn storing_the_same_rule_set_twice_is_idempotent() {
    let (store, lineage) = bootstrapped("lineage_rule_set_idempotent").await;
    let first = lineage.put_rule_set(&rules()).await.expect("first write");
    let second = lineage.put_rule_set(&rules()).await.expect("second write");
    assert_eq!(first, second);
    assert_eq!(row_count(&store, schema::RULE_SET_TABLE).await, 1);
}

#[tokio::test]
async fn an_absent_rule_set_reads_as_none() {
    let (_store, lineage) = bootstrapped("lineage_rule_set_absent").await;
    let missing = RuleSetAddr::from_digest(&"a".repeat(64)).expect("a valid digest");
    assert!(
        lineage
            .get_rule_set(&missing)
            .await
            .expect("querying")
            .is_none()
    );
}

/// A pair that is not a rewrite rule must be refused **before** anything is
/// written, and refused with upstream's own diagnosis rather than a vaguer one.
#[tokio::test]
async fn a_rule_set_that_is_not_rules_never_reaches_the_database() {
    let (store, lineage) = bootstrapped("lineage_rule_set_rejected").await;
    let copy = ColoredExpr::new(vec![()], Free::<Gen>::generator(Gen::Copy))
        .expect("Δ type-checks at one wire");
    let err = lineage
        .put_rule_set(&[(copy, identity())])
        .await
        .expect_err("non-parallel sides are not a rewrite rule");
    assert!(matches!(err, StoreError::Catgraph(_)), "{err}");
    assert_eq!(row_count(&store, schema::RULE_SET_TABLE).await, 0);
}

// --------------------------------------------------------------- rewrite runs

#[tokio::test]
async fn a_run_records_its_trace_and_its_edge() {
    let (store, lineage) = bootstrapped("lineage_run").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;

    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording the run");

    let loaded = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");
    assert_eq!(loaded.rule_set(), &rule_set);
    assert_eq!(loaded.cost_model(), "unit");
    assert!(loaded.best_cost() < loaded.initial_cost());
    assert!(loaded.replayable());

    // The trace survived the object column, which is the schema question this
    // test exists to answer: a non-FLEXIBLE object drops undeclared keys, so a
    // step whose shape were not declared would come back as `{}`.
    assert!(!loaded.steps().is_empty());
    assert_eq!(loaded.steps().len(), outcome.steps().len());
    for (stored, original) in loaded.steps().iter().zip(outcome.steps()) {
        assert_eq!(stored.rule() as usize, original.rule());
        let edges: Vec<usize> = stored
            .matched_edges()
            .iter()
            .map(|edge| usize::try_from(*edge).expect("a stored edge index is non-negative"))
            .collect();
        assert_eq!(edges, original.matched_edges());
    }

    // And the edge landed in the same transaction.
    let edges = lineage.edges_of_run(&run).await.expect("reading the edges");
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].parent(), loaded.start());
    assert_eq!(edges[0].child(), loaded.best());
    assert_eq!(edges[0].run(), &run);
    assert_eq!(edges[0].cost_model(), "unit");

    // Both endpoints were stored as terms, outside the transaction.
    assert_eq!(row_count(&store, schema::TERM_TABLE).await, 2);
}

/// The round trip a trace exists for: what was recorded replays, out of the
/// database, to the morphism the run filed as its best. Both halves of the
/// reconstruction are exercised — the step rows crossing back into upstream's
/// own type, and the rules being rebuilt from the stored set in stored order.
#[tokio::test]
async fn a_recorded_run_replays_to_its_stored_best() {
    let (_store, lineage) = bootstrapped("lineage_replay").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording the run");

    let loaded = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");
    assert!(
        !loaded.steps().is_empty(),
        "an empty trace replays vacuously"
    );

    let replayed = lineage
        .replay_run(&loaded)
        .await
        .expect("a recorded run replays");
    let encoded = catgraph_surreal::term::encode(&replayed).expect("the endpoint encodes");
    assert_eq!(encoded.addr(), loaded.best());
    assert_eq!(&replayed, outcome.best());
}

/// A run whose rule set was never stored cannot be replayed, and says which
/// address is missing rather than failing somewhere inside the rewrite engine.
#[tokio::test]
async fn a_run_whose_rule_set_is_absent_does_not_replay() {
    let (_store, lineage) = bootstrapped("lineage_replay_absent_rules").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording the run");
    let loaded = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");

    // A second store on a fresh database, holding the run's start term and not
    // its rule set: the guard before this one passes, so the branch this test
    // names is the one that fires.
    let (other_store, empty) = bootstrapped("lineage_replay_empty").await;
    let terms: TermStore<Gen> = TermStore::open(other_store)
        .await
        .expect("opening the term store");
    terms.put(&start).await.expect("storing the start term");

    let err = empty
        .replay_run(&loaded)
        .await
        .expect_err("the run's rule set is not stored there");
    let StoreError::Corrupt { detail, .. } = &err else {
        panic!("expected a corrupt document, got {err:?}");
    };
    assert!(detail.contains("rule set"), "{detail}");
    assert!(detail.contains(rule_set.as_str()), "{detail}");
}

/// And the guard before it: a replay starts from the run's `start` term, so a
/// database without it is refused there rather than at the rule set.
#[tokio::test]
async fn a_run_whose_start_term_is_absent_does_not_replay() {
    let (_store, lineage) = bootstrapped("lineage_replay_absent_start").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording the run");
    let loaded = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");

    // The mirror image: the rule set is stored there, the start term is not.
    let (_other_store, empty) = bootstrapped("lineage_replay_no_start").await;
    empty
        .put_rule_set(&rules())
        .await
        .expect("storing the same rule set");

    let err = empty
        .replay_run(&loaded)
        .await
        .expect_err("the run's start term is not stored there");
    let StoreError::Corrupt { detail, .. } = &err else {
        panic!("expected a corrupt document, got {err:?}");
    };
    assert!(detail.contains("`start` term"), "{detail}");
}

/// A trace's rule indices are relative to the order the rules were stored in,
/// and this is that claim out of the database: two rules, two distinct steps,
/// loaded and replayed to the run's own `best`.
#[tokio::test]
async fn a_two_rule_run_replays_out_of_the_store_to_its_stored_best() {
    let (_store, lineage) = bootstrapped("lineage_replay_two_rules").await;
    let (rule_set, start, outcome) = a_two_rule_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording the run");

    let loaded = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");
    assert_eq!(
        loaded.steps().len(),
        2,
        "the order pin needs two steps, got {:?}",
        loaded.steps()
    );
    assert_ne!(
        loaded.steps()[0],
        loaded.steps()[1],
        "two identical steps reverse to themselves, pinning nothing"
    );

    let replayed = lineage
        .replay_run(&loaded)
        .await
        .expect("a recorded run replays");
    let encoded = catgraph_surreal::term::encode(&replayed).expect("the endpoint encodes");
    assert_eq!(encoded.addr(), loaded.best());
    assert_eq!(&replayed, outcome.best());
}

/// A rule set whose pairs were swapped inside the stored encoding no longer
/// digests to the address it is filed under, so a replay reaching for it is
/// refused by revalidation rather than replaying the other order.
#[tokio::test]
async fn a_run_whose_rule_set_was_reordered_in_place_does_not_replay() {
    let (store, lineage) = bootstrapped("lineage_replay_swapped_rules").await;
    let (rule_set, start, outcome) = a_two_rule_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording the run");
    let loaded = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");

    let rid = RecordId::new(schema::RULE_SET_TABLE, rule_set.as_str());
    let mut response = store
        .client()
        .query("SELECT VALUE rules_json FROM $rid")
        .bind(("rid", rid.clone()))
        .await
        .expect("reading the stored encoding");
    let stored: Option<String> = response.take(0).expect("the column reads back");
    let stored = stored.expect("the rule set is stored");
    let mut pairs: Vec<serde_json::Value> =
        serde_json::from_str(&stored).expect("the encoding is a list of pairs");
    assert_eq!(pairs.len(), 2, "the swap below needs two rules");
    pairs.swap(0, 1);
    let swapped = serde_json::to_string(&pairs).expect("the swapped encoding serializes");
    assert_ne!(swapped, stored, "the swap has to change the encoding");

    // `OPTION IMPORT;` disables READONLY for the statement, which is how an
    // edited encoding reaches a write-once row at all.
    store
        .client()
        .query("OPTION IMPORT; UPDATE $rid SET rules_json = $json RETURN NONE")
        .bind(("rid", rid))
        .bind(("json", swapped))
        .await
        .expect("the edit runs")
        .check()
        .expect("and is accepted under OPTION IMPORT");

    let err = lineage
        .replay_run(&loaded)
        .await
        .expect_err("a reordered rule set does not verify");
    let StoreError::Corrupt { context, detail } = &err else {
        panic!("expected a corrupt document, got {err:?}");
    };
    assert_eq!(
        context,
        &format!("{}:{}", schema::RULE_SET_TABLE, rule_set.as_str())
    );
    assert!(
        detail.contains("content address of the stored encoding is"),
        "{detail}"
    );
}

/// The edge write is a content-addressed `RELATE OR UPDATE`, so recording the
/// same run twice targets the same edge row with the same values. Neither the
/// engine's silent-overwrite path nor a read-only refusal should be reachable.
#[tokio::test]
async fn recording_the_same_run_twice_is_idempotent() {
    let (store, lineage) = bootstrapped("lineage_run_idempotent").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;

    let first = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("first write");
    let second = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("second write");

    assert_eq!(first, second);
    assert_eq!(row_count(&store, schema::REWRITE_RUN_TABLE).await, 1);
    assert_eq!(row_count(&store, schema::DERIVES_TABLE).await, 1);
    assert_eq!(
        lineage
            .edges_of_run(&first)
            .await
            .expect("reading the edges")
            .len(),
        1
    );
}

/// `replayable` is `READONLY` and outside the run's content address, so a row
/// stored with `false` under it is a row re-recording the same run has to
/// write `true` over. The refused write is retried with `false`: the call
/// succeeds and the stored flag stands.
///
/// **And both halves of the transaction land.** The edge is deleted before the
/// re-record, so a retry that rolled back — or a refusal absorbed without one —
/// leaves the run reachable in the table and unreachable in the graph, which is
/// what `edges_of_run` is read for here.
#[tokio::test]
async fn recording_a_run_over_a_row_that_is_not_replayable_is_idempotent() {
    let (store, lineage) = bootstrapped("lineage_run_replayable_false").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("first write");

    // `OPTION IMPORT;` disables READONLY for the statement, which is how a row
    // written before the flag existed reaches this database at all.
    store
        .client()
        .query("OPTION IMPORT; UPDATE $rid SET replayable = false RETURN NONE")
        .bind((
            "rid",
            RecordId::new(schema::REWRITE_RUN_TABLE, run.as_str()),
        ))
        .await
        .expect("the edit runs")
        .check()
        .expect("and is accepted under OPTION IMPORT");

    store
        .client()
        .query(format!("DELETE {}", schema::DERIVES_TABLE))
        .await
        .expect("the delete runs")
        .check()
        .expect("and the edge is gone");
    assert_eq!(
        lineage
            .edges_of_run(&run)
            .await
            .expect("reading the edges")
            .len(),
        0,
        "the deleted edge has to be gone, or the assertion below passes vacuously"
    );

    let again = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("re-recording the same run over a row flagged false");
    assert_eq!(again, run);

    assert_eq!(
        lineage
            .edges_of_run(&run)
            .await
            .expect("reading the edges")
            .len(),
        1,
        "the retry's edge half has to land with its run half"
    );

    let loaded = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");
    assert!(!loaded.replayable(), "the stored flag stands");
    assert_eq!(row_count(&store, schema::REWRITE_RUN_TABLE).await, 1);
}

/// And that tolerance is `replayable` alone: a row differing in `best`, a
/// column the run's content address covers, is the collision alarm.
#[tokio::test]
async fn recording_a_run_over_a_row_with_a_different_best_is_refused() {
    let (store, lineage) = bootstrapped("lineage_run_tampered_best").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("first write");

    let planted = format!("b3_{}", "d".repeat(64));
    store
        .client()
        .query("OPTION IMPORT; UPDATE $rid SET best = $best RETURN NONE")
        .bind((
            "rid",
            RecordId::new(schema::REWRITE_RUN_TABLE, run.as_str()),
        ))
        .bind(("best", planted))
        .await
        .expect("the edit runs")
        .check()
        .expect("and is accepted under OPTION IMPORT");

    let err = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect_err("a changed `best` under one address is the alarm");
    let StoreError::ReadOnly { table, field } = &err else {
        panic!("expected a read-only refusal, got {err:?}");
    };
    assert_eq!(table, schema::REWRITE_RUN_TABLE);
    assert_eq!(field, "best");
}

/// A column that sorts *after* `replayable`, differing alone: the row's flag
/// still reads `true`, so the write is refused on `rule_set` and no retry runs.
///
/// This is the row that separates the retry's `field == "replayable"` guard
/// from an unguarded one. A retry that fired on any read-only refusal would
/// write `replayable = false` over this row's `true` and be refused on
/// `replayable`, the earlier column — measured: `field` reads `"replayable"`
/// with the guard removed, `"rule_set"` with it.
#[tokio::test]
async fn recording_a_run_over_a_row_with_a_different_rule_set_is_refused() {
    let (store, lineage) = bootstrapped("lineage_run_tampered_rule_set_only").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("first write");

    let planted = format!("b3_{}", "e".repeat(64));
    assert_ne!(
        planted,
        rule_set.as_str(),
        "the plant has to change the column"
    );
    store
        .client()
        .query("OPTION IMPORT; UPDATE $rid SET rule_set = $planted RETURN NONE")
        .bind((
            "rid",
            RecordId::new(schema::REWRITE_RUN_TABLE, run.as_str()),
        ))
        .bind(("planted", planted))
        .await
        .expect("the edit runs")
        .check()
        .expect("and is accepted under OPTION IMPORT");

    // The flag a retry would overwrite is untouched, which is what makes this
    // row separate the two. Read raw: the planted `rule_set` is exactly what a
    // revalidating read refuses.
    let mut response = store
        .client()
        .query("SELECT VALUE replayable FROM $rid")
        .bind((
            "rid",
            RecordId::new(schema::REWRITE_RUN_TABLE, run.as_str()),
        ))
        .await
        .expect("reading the flag");
    let flag: Option<bool> = response.take(0).expect("the column reads back");
    assert_eq!(flag, Some(true), "the stored flag has to still be `true`");

    let err = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect_err("a changed `rule_set` under one address is the alarm");
    let StoreError::ReadOnly { table, field } = &err else {
        panic!("expected a read-only refusal, got {err:?}");
    };
    assert_eq!(table, schema::REWRITE_RUN_TABLE);
    assert_eq!(field, "rule_set");
}

/// And the alarm survives a row that *also* carries the stale flag: a row
/// differing in `rule_set` and in `replayable` is refused on `replayable`
/// first — the engine names the alphabetically earlier column — and the retry,
/// writing the `false` that row already holds, surfaces `rule_set`.
#[tokio::test]
async fn recording_a_run_over_a_row_with_a_different_rule_set_and_a_stale_flag_is_refused() {
    let (store, lineage) = bootstrapped("lineage_run_tampered_rule_set").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("first write");

    let planted = format!("b3_{}", "e".repeat(64));
    assert_ne!(
        planted,
        rule_set.as_str(),
        "the plant has to change the column"
    );
    store
        .client()
        .query("OPTION IMPORT; UPDATE $rid SET rule_set = $planted, replayable = false RETURN NONE")
        .bind((
            "rid",
            RecordId::new(schema::REWRITE_RUN_TABLE, run.as_str()),
        ))
        .bind(("planted", planted))
        .await
        .expect("the edit runs")
        .check()
        .expect("and is accepted under OPTION IMPORT");

    let err = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect_err("a changed `rule_set` under one address is the alarm");
    let StoreError::ReadOnly { table, field } = &err else {
        panic!("expected a read-only refusal, got {err:?}");
    };
    assert_eq!(table, schema::REWRITE_RUN_TABLE);
    assert_eq!(field, "rule_set");
}

/// Two runs that differ only in the weighting are different records, because
/// their costs are not comparable — and they share endpoints, so their edges
/// are different rows too.
#[tokio::test]
async fn the_cost_model_separates_runs_and_their_edges() {
    let (store, lineage) = bootstrapped("lineage_cost_model").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;

    let unit = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("the unit-cost run");
    let weighted = lineage
        .record_run(&rule_set, &start, &outcome, "weighted")
        .await
        .expect("the weighted run");

    assert_ne!(unit, weighted);
    assert_eq!(row_count(&store, schema::REWRITE_RUN_TABLE).await, 2);
    assert_eq!(row_count(&store, schema::DERIVES_TABLE).await, 2);
}

/// A run that does not say which weighting produced its numbers is refused
/// before the transaction opens.
#[tokio::test]
async fn a_run_without_a_cost_model_never_reaches_the_database() {
    let (store, lineage) = bootstrapped("lineage_no_cost_model").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let err = lineage
        .record_run(&rule_set, &start, &outcome, "")
        .await
        .expect_err("a run must name its cost model");
    assert!(matches!(err, StoreError::Revalidation { .. }), "{err}");
    assert_eq!(row_count(&store, schema::REWRITE_RUN_TABLE).await, 0);
    assert_eq!(row_count(&store, schema::DERIVES_TABLE).await, 0);
}

#[tokio::test]
async fn an_absent_run_reads_as_none_and_has_no_edges() {
    let (_store, lineage) = bootstrapped("lineage_run_absent").await;
    let missing = catgraph_surreal::RunAddr::from_digest(&"b".repeat(64)).expect("a valid digest");
    assert!(lineage.get_run(&missing).await.expect("querying").is_none());
    assert!(
        lineage
            .edges_of_run(&missing)
            .await
            .expect("querying")
            .is_empty()
    );
}

// ------------------------------------------------------------ the edge itself

/// The traversal a lineage graph exists for: from a term, to the morphisms
/// some run derived from it.
#[tokio::test]
async fn edges_are_traversable_from_their_parent_term() {
    let (_store, lineage) = bootstrapped("lineage_traversal").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording");

    let stored = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");
    let out = lineage
        .edges_from(stored.start())
        .await
        .expect("traversing from the starting term");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].child(), stored.best());

    // Nothing leads out of the endpoint.
    assert!(
        lineage
            .edges_from(stored.best())
            .await
            .expect("traversing from the best term")
            .is_empty()
    );
}

/// An edge's pointers are defined by `DEFINE TABLE … TYPE RELATION`, not by the
/// DDL, and a read has to project them explicitly — this pins that they arrive
/// and that they point where they should.
#[tokio::test]
async fn edge_pointers_are_stored_and_projected() {
    let (store, lineage) = bootstrapped("lineage_edge_pointers").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording");
    let stored = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");

    let mut response = store
        .client()
        .query(format!(
            "SELECT in, out FROM {} WHERE run = $run",
            schema::DERIVES_TABLE
        ))
        .bind(("run", run.as_str().to_owned()))
        .await
        .expect("reading the raw edge");
    #[derive(Debug, SurrealValue)]
    struct Pointers {
        r#in: RecordId,
        out: RecordId,
    }
    let rows: Vec<Pointers> = response.take(0).expect("the pointers project");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].r#in.table.as_str(), schema::TERM_TABLE);
    assert_eq!(rows[0].out.table.as_str(), schema::TERM_TABLE);
    assert_eq!(
        rows[0].r#in.key,
        RecordIdKey::String(stored.start().as_str().to_owned())
    );
    assert_eq!(
        rows[0].out.key,
        RecordIdKey::String(stored.best().as_str().to_owned())
    );
}

/// A `SurrealValue`-derived field named `r#in` binds to the `in` column with no
/// rename attribute: the derive strips the raw-identifier prefix. Pinned
/// because getting it wrong produces a *missing field*, not a compile error.
#[tokio::test]
async fn the_raw_identifier_field_binds_to_the_in_column() {
    let (store, lineage) = bootstrapped("lineage_raw_identifier").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording");

    #[derive(Debug, SurrealValue)]
    struct JustIn {
        r#in: RecordId,
    }
    let mut response = store
        .client()
        .query(format!("SELECT in FROM {}", schema::DERIVES_TABLE))
        .await
        .expect("reading");
    let rows: Vec<JustIn> = response
        .take(0)
        .expect("`r#in` must deserialize from the `in` column with no attribute");
    assert_eq!(rows.len(), 1);
}

/// **The traversal actually uses the index**, not merely "an index exists".
///
/// The two are different claims and only the second is free: the planner builds
/// its candidate set from defined indexes, so a `WHERE in = …` with no index on
/// `in` is a full scan of the edge table on every hop — correct, and the tier's
/// dominant cost the moment the graph is large. Asserting the definition alone
/// would leave the reason for the definition unchecked, so this reads the plan.
#[tokio::test]
async fn the_forward_traversal_is_served_by_an_index() {
    let (store, lineage) = bootstrapped("lineage_edge_plan").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording");
    let stored = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");

    let mut response = store
        .client()
        .query(format!(
            "SELECT id FROM {} WHERE in = $rid EXPLAIN",
            schema::DERIVES_TABLE
        ))
        .bind((
            "rid",
            RecordId::new(schema::TERM_TABLE, stored.start().as_str()),
        ))
        .await
        .expect("explaining the traversal");
    let plan: Vec<surrealdb::types::Value> = response.take(0).expect("the plan reads back");
    let rendered = format!("{plan:?}");
    assert!(
        rendered.contains(schema::DERIVES_IN_INDEX),
        "the forward traversal must be served by `{}`, not by a table scan: {rendered}",
        schema::DERIVES_IN_INDEX
    );
}

/// The declared indexes have to match the engine's own rendering, which is what
/// the drift guard compares against. Pinned here because the rendering is the
/// engine's to choose, not this crate's — `in` is a keyword, and whether it
/// comes back bare or escaped is not something to guess at.
#[tokio::test]
async fn the_edge_indexes_are_the_engines_renderings() {
    let (store, _lineage) = bootstrapped("lineage_edge_indexes").await;
    let mut response = store
        .client()
        .query(format!(
            "RETURN (INFO FOR TABLE {}).indexes",
            schema::DERIVES_TABLE
        ))
        .await
        .expect("reading the index definitions");
    let live: Option<std::collections::BTreeMap<String, String>> =
        response.take(0).expect("the definitions read back");
    let live = live.unwrap_or_default();
    for (name, definition) in schema::DERIVES_INDEX_DEFINITIONS {
        assert_eq!(
            live.get(name).map(String::as_str),
            Some(definition),
            "{name}"
        );
    }
}

// ------------------------------------------------------------ corrupt reads

/// A run whose costs were edited in place does not match its own content
/// address, and a load says so rather than handing back numbers nobody
/// measured.
#[tokio::test]
async fn a_tampered_run_is_corrupt_on_load() {
    let (store, lineage) = bootstrapped("lineage_tampered_run").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording");

    // `OPTION IMPORT;` disables READONLY for the statement, which is exactly how
    // a hostile or careless edit would reach a write-once row — and exactly why
    // the store re-derives rather than trusting what it reads.
    //
    // The edited value has to be one the row does not already hold: a write that
    // changes nothing is not a modification, so it would neither trip a guard
    // nor change the content address, and the test would pass vacuously.
    let stored = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");
    assert_ne!(stored.initial_cost(), 99);
    store
        .client()
        .query("OPTION IMPORT; UPDATE $rid SET initial_cost = 99 RETURN NONE")
        .bind((
            "rid",
            RecordId::new(schema::REWRITE_RUN_TABLE, run.as_str()),
        ))
        .await
        .expect("the edit runs")
        .check()
        .expect("and is accepted under OPTION IMPORT");

    let err = lineage
        .get_run(&run)
        .await
        .expect_err("an edited run is corrupt");
    assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
}

/// **The edge's denormalized weighting is inside its content address**, so an
/// edge re-labelled in place does not verify.
///
/// `rule_set` and `cost_model` are copies of columns the run already carries,
/// kept on the edge so a traversal can read what a derivation's numbers mean
/// without following it to the run. Under the `cge1` address — which digested
/// only `(parent, child, run)` — they were the one pair of stored values in the
/// tier that revalidation could not re-derive: an edge whose weighting had been
/// swapped verified cleanly and misattributed every derivation it described.
#[tokio::test]
async fn an_edge_with_a_tampered_weighting_is_corrupt_on_load() {
    let (store, lineage) = bootstrapped("lineage_tampered_edge").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording");
    let stored = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");
    let edge = lineage::encode_derivation(&stored).expect("the endpoints imply an edge");

    // `OPTION IMPORT;` disables READONLY for the statement, which is how a
    // hostile edit or a doctored dump would reach the column at all.
    store
        .client()
        .query("OPTION IMPORT; UPDATE $rid SET cost_model = 'weighted' RETURN NONE")
        .bind((
            "rid",
            RecordId::new(schema::DERIVES_TABLE, edge.addr().as_str()),
        ))
        .await
        .expect("the edit runs")
        .check()
        .expect("and is accepted under OPTION IMPORT");

    let err = lineage
        .edges_of_run(&run)
        .await
        .expect_err("a re-labelled edge is corrupt");
    assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
}

/// A step object whose keys were emptied disagrees with the run's own
/// `step_count`, which is what that column is beside the steps for.
#[tokio::test]
async fn a_truncated_trace_disagrees_with_its_own_count() {
    let (store, lineage) = bootstrapped("lineage_truncated_trace").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording");

    store
        .client()
        .query("OPTION IMPORT; UPDATE $rid SET steps = [] RETURN NONE")
        .bind((
            "rid",
            RecordId::new(schema::REWRITE_RUN_TABLE, run.as_str()),
        ))
        .await
        .expect("the edit runs")
        .check()
        .expect("and is accepted under OPTION IMPORT");

    match lineage.get_run(&run).await {
        Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "step_count"),
        other => panic!("expected a step-count mismatch, got {other:?}"),
    }
}

// ------------------------------------------------------------------- schema

/// Opening twice must be a no-op, and the drift guard must agree with the DDL
/// on a freshly bootstrapped database — the whole point of comparing rendered
/// definition strings is that they are the engine's, not ours.
#[tokio::test]
async fn opening_twice_is_idempotent_and_verifies() {
    let store = connect("lineage_bootstrap").await;
    let first = LineageStore::<Gen>::open(store.clone())
        .await
        .expect("the first open bootstraps");
    let second = LineageStore::<Gen>::open(store.clone())
        .await
        .expect("the second open is a no-op");
    first.assert_schema().await.expect("still verified");
    second.assert_schema().await.expect("still verified");
}

/// A dropped `READONLY` clause keeps the column's name, which is why the guard
/// compares definitions rather than names.
#[tokio::test]
async fn a_relaxed_column_is_drift() {
    let (store, lineage) = bootstrapped("lineage_drift").await;
    store
        .client()
        .query(format!(
            "DEFINE FIELD OVERWRITE cost_model ON {} TYPE string",
            schema::REWRITE_RUN_TABLE
        ))
        .await
        .expect("the redefinition runs")
        .check()
        .expect("and is accepted");
    let err = lineage
        .assert_schema()
        .await
        .expect_err("a dropped READONLY is drift");
    let StoreError::Schema { table, detail } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert_eq!(table, schema::REWRITE_RUN_TABLE);
    assert!(detail.contains("cost_model"), "{detail}");
}

/// The engine defines the nested and element fields the DDL does not write, and
/// the guard's expected set has to include exactly those — a database that
/// bootstrapped cleanly must verify.
#[tokio::test]
async fn the_engine_created_field_definitions_are_the_expected_ones() {
    let (store, _lineage) = bootstrapped("lineage_implicit_fields").await;
    let mut response = store
        .client()
        .query(format!(
            "RETURN (INFO FOR TABLE {}).fields",
            schema::REWRITE_RUN_TABLE
        ))
        .await
        .expect("reading the field definitions");
    let live: Option<std::collections::BTreeMap<String, String>> =
        response.take(0).expect("the definitions read back");
    let live = live.unwrap_or_default();
    for (name, definition) in schema::REWRITE_RUN_ELEMENT_DEFINITIONS {
        assert_eq!(live.get(name).map(String::as_str), Some(definition));
    }
    for (name, definition) in schema::REWRITE_RUN_NESTED_DEFINITIONS {
        assert_eq!(live.get(name).map(String::as_str), Some(definition));
    }
}

/// A relation's `in`/`out` are created by `DEFINE TABLE`, so the `READONLY` the
/// DDL would have put on them never lands. This pins the definitions the guard
/// actually expects, and the reason the crate does not try to declare them.
#[tokio::test]
async fn a_relations_pointers_carry_no_readonly_clause() {
    let (store, _lineage) = bootstrapped("lineage_pointer_definitions").await;
    let mut response = store
        .client()
        .query(format!(
            "RETURN (INFO FOR TABLE {}).fields",
            schema::DERIVES_TABLE
        ))
        .await
        .expect("reading the field definitions");
    let live: Option<std::collections::BTreeMap<String, String>> =
        response.take(0).expect("the definitions read back");
    let live = live.unwrap_or_default();
    for (name, definition) in schema::DERIVES_POINTER_DEFINITIONS {
        assert_eq!(live.get(name).map(String::as_str), Some(definition));
        assert!(!definition.contains("READONLY"), "{name}");
    }
}

/// Reads absorb a vanished table as absence; a write must be loud, because it
/// is about to re-create the table `SCHEMALESS`.
#[tokio::test]
async fn a_removed_table_is_absent_to_reads_and_loud_to_writes() {
    let (store, lineage) = bootstrapped("lineage_removed_table").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;

    store
        .client()
        .query(format!("REMOVE TABLE {}", schema::REWRITE_RUN_TABLE))
        .await
        .expect("removing the table")
        .check()
        .expect("and it is removed");

    let missing = catgraph_surreal::RunAddr::from_digest(&"c".repeat(64)).expect("a valid digest");
    assert!(
        lineage
            .get_run(&missing)
            .await
            .expect("reads absorb")
            .is_none()
    );

    let err = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect_err("a write against an undefined table must be loud");
    let StoreError::Schema { table, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert_eq!(table, schema::REWRITE_RUN_TABLE);
}

/// **The other table a recorded run writes.** The statement writes the trace
/// *and* `RELATE`s the derivation edge, so removing the edge table has to be as
/// loud as removing the trace table.
///
/// This is the hole a single-table guard leaves, and on a relation table it is
/// the expensive one: the auto-created impostor is `TYPE ANY SCHEMALESS`, so the
/// `IN term OUT term` typing, the `READONLY` columns and the id `ASSERT` all
/// vanish at once while `assert_schema` on `rewrite_run` stays green.
#[tokio::test]
async fn a_removed_edge_table_is_loud_to_writes() {
    let (store, lineage) = bootstrapped("lineage_removed_edge_table").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;

    store
        .client()
        .query(format!("REMOVE TABLE {}", schema::DERIVES_TABLE))
        .await
        .expect("removing the edge table")
        .check()
        .expect("and it is removed");

    let err = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect_err("a RELATE against an undefined table must be loud");
    let StoreError::Schema { table, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert_eq!(table, schema::DERIVES_TABLE);

    // And the guard refused rather than repaired: the table stays undefined.
    let mut response = store
        .client()
        .query(format!(
            "RETURN (INFO FOR DB).tables.{}",
            schema::DERIVES_TABLE
        ))
        .await
        .expect("reading the table definition");
    let definition: Option<String> = response.take(0).expect("the definition slot");
    assert_eq!(definition, None, "the table must remain undefined");
}

/// The nested step fields carry no `READONLY` of their own, and that is not a
/// hole for the same reason an array element's absent clause is not: writing
/// `steps[0].rule` changes the value of the `steps` column, and `steps` is
/// `READONLY`. Pinned because it is invisible in the definitions — the parallel
/// of the cospan tier's element-write test.
#[tokio::test]
async fn a_nested_step_write_is_refused_by_the_parent_column() {
    let (store, lineage) = bootstrapped("lineage_nested_write").await;
    let (rule_set, start, outcome) = a_run(&lineage).await;
    let run = lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording");
    let before = lineage
        .get_run(&run)
        .await
        .expect("loading")
        .expect("present");
    assert!(!before.steps().is_empty(), "the fixture run rewrites");
    assert_ne!(before.steps()[0].rule(), 9);

    let outcome = store
        .client()
        .query("UPDATE $rid SET steps[0].rule = 9 RETURN NONE")
        .bind((
            "rid",
            RecordId::new(schema::REWRITE_RUN_TABLE, run.as_str()),
        ))
        .await
        .expect("the statement runs")
        .check();
    assert!(
        outcome.is_err(),
        "the parent `steps` column must refuse a nested write"
    );

    // And the trace is untouched — it still loads, and still revalidates against
    // the address it is filed under.
    let after = lineage
        .get_run(&run)
        .await
        .expect("reading back")
        .expect("still present");
    assert_eq!(after.steps(), before.steps());
}

// ------------------------------------------------------------------ helpers

async fn row_count(store: &Store, table: &str) -> usize {
    let mut response = store
        .client()
        .query(format!("SELECT VALUE id FROM {table}"))
        .await
        .expect("counting the stored rows");
    let ids: Vec<RecordId> = response.take(0).expect("reading the ids");
    ids.len()
}

/// Both stores are `Send + Sync` over a generator type that is neither: `G`
/// reaches them only as `PhantomData<fn() -> G>`.
#[test]
fn the_stores_auto_traits_do_not_depend_on_the_generator() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TermStore<NotSend>>();
    assert_send_sync::<LineageStore<NotSend>>();
}

/// The encoding constants are part of the on-disk format, so a change to one is
/// a migration. Pinned here beside the tier that writes them.
#[test]
fn the_lineage_codecs_are_pinned() {
    assert_eq!(lineage::RULE_SET_CODEC, "cgs1");
    assert_eq!(lineage::REWRITE_RUN_CODEC, "cgt1");
    assert_eq!(lineage::DERIVATION_CODEC, "cge2");
}
