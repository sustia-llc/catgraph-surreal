//! End-to-end document and manifest behaviour against the in-memory engine.
//!
//! These need a real engine compiled in, so the whole file is gated on `mem`.
//!
//! The two things pinned here are both properties of the *database*, not of the
//! encoding. The first is the **`FLEXIBLE` payload column**: a `SCHEMAFULL`
//! object column without it keeps only the keys the schema declares and discards
//! the rest silently, so the round trip is asserted on the whole key set rather
//! than on a value or two — a test that checked one field would pass on a schema
//! that had already lost everything else.
//!
//! The second is the **write-once stack**. Which layer refuses a given write is
//! an ordering fact about the engine's document pipeline: field processing runs
//! before events, so an *update* is refused by `READONLY` while a *delete*,
//! which processes no fields, is refused by the event. Both are pinned, because
//! a store that expected the other one would report the wrong thing at exactly
//! the moment it mattered.

#![cfg(feature = "mem")]

use std::collections::HashMap;

use catgraph_surreal::{DocStore, ManifestStore, Store, StoreBuilder, StoreError, schema};
use serde::{Deserialize, Serialize};
use surrealdb::types::RecordId;

/// A consumer-defined type the store knows nothing about: plain numbers, a
/// vector, and a map whose iteration order is unspecified.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SolverState {
    beliefs: Vec<f64>,
    beta: f64,
    trial_boundary: usize,
    labels: HashMap<String, i32>,
}

fn state(boundary: usize) -> SolverState {
    let mut labels = HashMap::new();
    labels.insert("gamma".to_owned(), 3);
    labels.insert("alpha".to_owned(), 1);
    labels.insert("beta".to_owned(), 2);
    SolverState {
        beliefs: vec![0.25, 0.5, 0.25],
        beta: 1.5,
        trial_boundary: boundary,
        labels,
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

async fn documents(database: &str) -> (Store, DocStore<SolverState>) {
    let store = connect(database).await;
    let docs = DocStore::open(store.clone())
        .await
        .expect("opening the document store bootstraps and verifies");
    (store, docs)
}

async fn manifests(database: &str) -> (Store, ManifestStore<serde_json::Value>) {
    let store = connect(database).await;
    let manifests = ManifestStore::open(store.clone())
        .await
        .expect("opening the manifest store bootstraps and verifies");
    (store, manifests)
}

// ------------------------------------------------------- the mutable tier

#[tokio::test]
async fn a_consumer_type_round_trips_through_the_database() {
    let (_store, docs) = documents("doc_round_trip").await;
    docs.put("state-1", "solver", &state(7))
        .await
        .expect("storing");
    let loaded = docs
        .get("state-1")
        .await
        .expect("loading")
        .expect("present");
    assert_eq!(loaded, state(7));
}

#[tokio::test]
async fn a_document_can_be_replaced() {
    let (_store, docs) = documents("doc_replace").await;
    docs.put("state-1", "solver", &state(7))
        .await
        .expect("first");
    docs.put("state-1", "solver", &state(9))
        .await
        .expect("second");
    let loaded = docs
        .get("state-1")
        .await
        .expect("loading")
        .expect("present");
    assert_eq!(loaded.trial_boundary, 9);
}

#[tokio::test]
async fn documents_are_listed_by_their_class() {
    let (_store, docs) = documents("doc_list").await;
    docs.put("a", "solver", &state(1)).await.expect("storing");
    docs.put("b", "solver", &state(2)).await.expect("storing");
    docs.put("c", "archive", &state(3)).await.expect("storing");

    let mut solvers: Vec<usize> = docs
        .list("solver")
        .await
        .expect("listing")
        .into_iter()
        .map(|s| s.trial_boundary)
        .collect();
    solvers.sort_unstable();
    assert_eq!(solvers, vec![1, 2]);
    assert_eq!(docs.list("archive").await.expect("listing").len(), 1);
    assert!(docs.list("nothing").await.expect("listing").is_empty());
}

#[tokio::test]
async fn deleting_is_idempotent_and_absence_is_not_an_error() {
    let (_store, docs) = documents("doc_delete").await;
    docs.put("a", "solver", &state(1)).await.expect("storing");
    assert!(docs.contains("a").await.expect("existence"));

    docs.delete("a").await.expect("deleting");
    assert!(!docs.contains("a").await.expect("existence"));
    docs.delete("a").await.expect("deleting again is a no-op");
    assert!(docs.get("a").await.expect("querying").is_none());
}

/// Caller-supplied ids are opaque strings the store places no constraints on
/// beyond being non-empty, which means they are exactly the values that must
/// never reach query text.
#[tokio::test]
async fn awkward_caller_supplied_ids_round_trip() {
    let (_store, docs) = documents("doc_awkward_ids").await;
    for id in [
        "12345",
        "has space",
        "has-dash",
        "has:colon",
        "has`backtick",
        "has'quote",
        "⟨brackets⟩",
        "a/b/c",
    ] {
        docs.put(id, "solver", &state(1))
            .await
            .unwrap_or_else(|e| panic!("storing under `{id}`: {e}"));
        assert!(
            docs.get(id)
                .await
                .unwrap_or_else(|e| panic!("loading `{id}`: {e}"))
                .is_some(),
            "`{id}` did not come back"
        );
    }
}

#[tokio::test]
async fn an_empty_id_never_reaches_the_database() {
    let (store, docs) = documents("doc_empty_id").await;
    let err = docs
        .put("", "solver", &state(1))
        .await
        .expect_err("an empty id is not an id");
    assert!(matches!(err, StoreError::Revalidation { .. }), "{err}");
    assert_eq!(row_count(&store, schema::DOCUMENT_TABLE).await, 0);
}

// ---------------------------------------------- the FLEXIBLE payload hazard

/// **The corrupt-suite case for this tier.** The declared schema must not lose
/// keys, over a payload whose shape the store cannot possibly know — nested
/// objects, arrays of objects, empty containers, and keys that collide with the
/// table's own column names. The assertion is on the **whole key set**, because
/// a payload that had been flattened or truncated would still return the one or
/// two fields a spot check happened to name.
#[tokio::test]
async fn the_payload_column_keeps_every_nested_key() {
    let (_store, docs) = documents("doc_flexible").await;
    let store = docs.store().clone();
    let generic: DocStore<serde_json::Value> = DocStore::open(store)
        .await
        .expect("the same table serves any payload type");

    let payload = serde_json::json!({
        "top": 1,
        "nested": { "a": { "b": { "c": [1, 2, { "deep": "value" }] } } },
        "list_of_objects": [{ "x": 1 }, { "y": 2 }],
        "empty_object": {},
        "empty_list": [],
        "null_value": null,
        "bool": true,
        "float": 1.5,
        "string": "text",
        // A key named like one of the table's own columns, which a schema that
        // flattened the payload would collide with rather than nest.
        "digest": "not the column",
        "kind": "not the column",
    });

    generic
        .put("flexible-1", "probe", &payload)
        .await
        .expect("storing an unknown shape");
    let loaded: serde_json::Value = generic
        .get("flexible-1")
        .await
        .expect("loading")
        .expect("present");

    // The whole key set, not a field or two: a schema that had lost the payload
    // would still return the keys a spot check happened to name.
    let expected_keys: Vec<&String> = payload
        .as_object()
        .expect("the fixture is an object")
        .keys()
        .collect();
    let actual_keys: Vec<&String> = loaded
        .as_object()
        .expect("the payload comes back as an object")
        .keys()
        .collect();
    assert_eq!(actual_keys, expected_keys, "the payload lost keys");
    assert_eq!(loaded, payload, "the payload changed shape");

    // And the column really is declared FLEXIBLE, which is what makes the above
    // a property of the schema rather than a coincidence of this engine.
    assert!(
        schema::DOCUMENT_FIELD_DEFINITIONS
            .iter()
            .any(|(name, definition)| *name == "payload"
                && definition.contains("TYPE object FLEXIBLE")),
        "the payload column must be declared FLEXIBLE"
    );
}

/// The control for the test above: an object column *without* `FLEXIBLE` cannot
/// hold an unknown shape at all. The failure is loud rather than silent —
/// verified here rather than assumed, because "silently drops the extra keys"
/// is the plausible-sounding alternative and would be far more dangerous.
///
/// Either way the tier's column has to carry `FLEXIBLE`; this pins which
/// failure a schema that lost it would produce.
#[tokio::test]
async fn an_object_column_without_flexible_refuses_unknown_keys() {
    let store = connect("doc_flexible_control").await;
    store
        .client()
        .query(
            "DEFINE TABLE control SCHEMAFULL TYPE NORMAL;
             DEFINE FIELD payload ON control TYPE object;",
        )
        .await
        .expect("defining the control table")
        .check()
        .expect("and it is defined");

    let mut response = store
        .client()
        .query("UPSERT $rid CONTENT { payload: $payload } RETURN NONE")
        .bind(("rid", RecordId::new("control", "probe")))
        .bind(("payload", serde_json::json!({ "kept": 1, "dropped": 2 })))
        .await
        .expect("the write runs");
    let errors = response.take_errors();
    assert_eq!(errors.len(), 1, "an undeclared nested key must be refused");
    let (_, err) = errors.into_iter().next().expect("one error");
    assert!(
        err.message().contains("no such field exists"),
        "the refusal must name the undeclared field: {}",
        err.message()
    );

    // And nothing landed — not a truncated row.
    let mut response = store
        .client()
        .query("SELECT VALUE id FROM control")
        .await
        .expect("reading");
    let ids: Vec<RecordId> = response.take(0).expect("the ids read back");
    assert!(ids.is_empty(), "a refused write must leave no row");
}

// -------------------------------------------------------- the write-once tier

#[tokio::test]
async fn a_registered_manifest_reads_back() {
    let (_store, manifests) = manifests("manifest_round_trip").await;
    let manifest = serde_json::json!({ "bar": 0.05, "declared_at": "before the run" });
    manifests
        .register("run-1", "prereg", &manifest)
        .await
        .expect("registering");
    let loaded = manifests
        .get("run-1")
        .await
        .expect("loading")
        .expect("present");
    assert_eq!(loaded, manifest);
    assert!(manifests.contains("run-1").await.expect("existence"));
}

/// Registration is create-only: re-registering an id is refused **even when the
/// contents are identical**.
///
/// This is also where the `Immutable` classifier is pinned against a real
/// engine refusal through the real public API. No column's value changed, so
/// `READONLY` has nothing to object to; the update event is the layer that
/// fires, and its wrapped rendering is what the classifier reads.
///
/// The engine rule underneath is exact and easy to guess wrong: `UPSERT …
/// CONTENT` replaces the document wholesale, and the replacement counts as a
/// modification even when the bytes match — unlike `UPDATE … SET x = <the value
/// it already holds>`, which does not. Both halves are pinned, one here and one
/// below.
#[tokio::test]
async fn re_registering_a_manifest_is_refused_even_when_identical() {
    let (store, manifests) = manifests("manifest_idempotent").await;
    let manifest = serde_json::json!({ "bar": 0.05 });
    manifests
        .register("run-1", "prereg", &manifest)
        .await
        .expect("the first registration");

    let err = manifests
        .register("run-1", "prereg", &manifest)
        .await
        .expect_err("a manifest is registered once");
    let StoreError::Immutable { table, event } = &err else {
        panic!("expected an immutability refusal, got {err:?}");
    };
    assert_eq!(table, schema::MANIFEST_TABLE);
    assert_eq!(event, schema::MANIFEST_NO_UPDATE_EVENT);
    assert!(!err.is_conflict());
    assert!(!err.is_shutdown());

    // Rolled back, and still exactly one row.
    assert_eq!(row_count(&store, schema::MANIFEST_TABLE).await, 1);
    let loaded = manifests
        .get("run-1")
        .await
        .expect("loading")
        .expect("present");
    assert_eq!(loaded, manifest);
}

/// The other half of the data-clause rule, isolated: `SET` with an unchanged
/// value is genuinely not a modification, so no event fires. Pinned because it
/// is the half that *is* true, and a store that assumed it held for `CONTENT`
/// too would write an immutability test that passes vacuously.
#[tokio::test]
async fn a_set_of_an_unchanged_value_is_not_a_modification() {
    let store = connect("manifest_set_no_op").await;
    store
        .client()
        .query(
            "DEFINE TABLE probe SCHEMAFULL TYPE NORMAL;
             DEFINE FIELD n ON probe TYPE int;
             DEFINE EVENT probe_no_update ON probe WHEN $event = \"UPDATE\"
                 THEN { THROW \"changed\"; };",
        )
        .await
        .expect("defining the probe table")
        .check()
        .expect("and it is defined");

    let rid = RecordId::new("probe", "x");
    store
        .client()
        .query("CREATE $rid SET n = 1 RETURN NONE")
        .bind(("rid", rid.clone()))
        .await
        .expect("creating")
        .check()
        .expect("and it is created");

    let mut unchanged = store
        .client()
        .query("UPDATE $rid SET n = 1 RETURN NONE")
        .bind(("rid", rid.clone()))
        .await
        .expect("the unchanged write runs");
    assert!(
        unchanged.take_errors().is_empty(),
        "SET with an unchanged value must not fire the event"
    );

    let mut changed = store
        .client()
        .query("UPDATE $rid SET n = 2 RETURN NONE")
        .bind(("rid", rid))
        .await
        .expect("the changed write runs");
    assert_eq!(
        changed.take_errors().len(),
        1,
        "SET with a changed value must fire the event"
    );
}

/// The alarm. A *changed* manifest under an existing id is refused, and it is
/// refused by the `READONLY` column rather than by the update event: field
/// processing runs before events in the engine's document pipeline, so the
/// column is what names the problem.
#[tokio::test]
async fn a_changed_manifest_is_refused_by_the_read_only_column() {
    let (_store, manifests) = manifests("manifest_write_once").await;
    manifests
        .register("run-1", "prereg", &serde_json::json!({ "bar": 0.05 }))
        .await
        .expect("the first registration");

    let err = manifests
        .register("run-1", "prereg", &serde_json::json!({ "bar": 0.5 }))
        .await
        .expect_err("a manifest cannot be moved after the fact");
    let StoreError::ReadOnly { table, field } = &err else {
        panic!("expected a write-once refusal, got {err:?}");
    };
    assert_eq!(table, schema::MANIFEST_TABLE);
    assert_eq!(field, "digest");
    // Not retryable: the bar will not become movable.
    assert!(!err.is_conflict());
    assert!(!err.is_shutdown());

    // And the registered manifest is untouched.
    let loaded: serde_json::Value = manifests
        .get("run-1")
        .await
        .expect("loading")
        .expect("present");
    assert_eq!(loaded, serde_json::json!({ "bar": 0.05 }));
}

/// The delete twin, and the path that pins the `Immutable` classifier against a
/// real engine refusal. `READONLY` does not stop a delete — no fields are
/// processed — so the event is the layer that fires here, and its synchronous
/// `THROW` rolls the transaction back rather than merely reporting.
#[tokio::test]
async fn deleting_a_manifest_is_refused_by_the_event_and_rolls_back() {
    let (store, manifests) = manifests("manifest_no_delete").await;
    manifests
        .register("run-1", "prereg", &serde_json::json!({ "bar": 0.05 }))
        .await
        .expect("registering");

    // There is no delete on the handle — the API is the primary guard — so this
    // reaches the event the way anything else would have to: raw.
    let mut response = store
        .client()
        .query("DELETE $rid")
        .bind(("rid", RecordId::new(schema::MANIFEST_TABLE, "run-1")))
        .await
        .expect("the delete runs");
    let errors = response.take_errors();
    assert_eq!(errors.len(), 1, "the delete must be refused");
    let (_, raw) = errors.into_iter().next().expect("one error");
    let message = raw.message();
    assert!(
        message.contains(&format!(
            "Error while processing event {}:",
            schema::MANIFEST_NO_DELETE_EVENT
        )),
        "the refusal must name the event: {message}"
    );
    assert!(
        message.contains(schema::MANIFEST_IMMUTABLE_MESSAGE),
        "the refusal must carry this crate's own message: {message}"
    );

    // Rolled back: the record is still there.
    assert!(manifests.contains("run-1").await.expect("existence"));
    assert!(manifests.get("run-1").await.expect("loading").is_some());
}

/// **The `Immutable` classifier, pinned against a real engine refusal through
/// the real public API.**
///
/// It also demonstrates the claim defence in depth is *for*: with the
/// `READONLY` clauses and the write-once `ASSERT`s stripped off — the drift a
/// careless migration produces — a changed registration is no longer refused by
/// a column, and the update event is what catches it. The store classifies that
/// refusal from the engine's wrapped rendering, keyed on the event's name.
#[tokio::test]
async fn the_update_event_catches_what_a_dropped_read_only_clause_would_let_through() {
    let (store, manifests) = manifests("manifest_immutable_classifier").await;
    manifests
        .register("run-1", "prereg", &serde_json::json!({ "bar": 0.05 }))
        .await
        .expect("registering");

    // Strip the two layers that would otherwise fire first. This is drift, and
    // the store's own guard reports it as such — which is checked below, so the
    // test cannot quietly become a test of a schema nobody declared.
    store
        .client()
        .query(format!(
            "DEFINE FIELD OVERWRITE codec ON {table} TYPE string;
             DEFINE FIELD OVERWRITE kind ON {table} TYPE string;
             DEFINE FIELD OVERWRITE digest ON {table} TYPE string;
             DEFINE FIELD OVERWRITE payload ON {table} TYPE object FLEXIBLE;",
            table = schema::MANIFEST_TABLE
        ))
        .await
        .expect("the redefinition runs")
        .check()
        .expect("and is accepted");
    assert!(
        manifests.assert_schema().await.is_err(),
        "relaxing the columns is drift, and the guard has to say so"
    );

    let err = manifests
        .register("run-1", "prereg", &serde_json::json!({ "bar": 0.5 }))
        .await
        .expect_err("the event refuses what the columns no longer do");
    let StoreError::Immutable { table, event } = &err else {
        panic!("expected an immutability refusal, got {err:?}");
    };
    assert_eq!(table, schema::MANIFEST_TABLE);
    assert_eq!(event, schema::MANIFEST_NO_UPDATE_EVENT);
    assert!(!err.is_conflict());
    assert!(!err.is_shutdown());

    // The sync THROW rolled the write back: the registered manifest is intact.
    let loaded = manifests
        .get("run-1")
        .await
        .expect("loading")
        .expect("present");
    assert_eq!(loaded, serde_json::json!({ "bar": 0.05 }));
}

/// The post-restore check. Every database-side guard is void under
/// `OPTION IMPORT`, so a replayed dump proves nothing about the rows it carries
/// — what it cannot forge unnoticed is a payload agreeing with its own digest.
#[tokio::test]
async fn verification_finds_the_manifests_a_privileged_write_moved() {
    let (store, manifests) = manifests("manifest_verify").await;
    manifests
        .register("run-1", "prereg", &serde_json::json!({ "bar": 0.05 }))
        .await
        .expect("registering");
    manifests
        .register("run-2", "prereg", &serde_json::json!({ "bar": 0.01 }))
        .await
        .expect("registering");

    assert!(
        manifests.verify_all().await.expect("verifying").is_empty(),
        "freshly registered manifests verify"
    );

    // The privileged path an import takes: guards off, payload rewritten, digest
    // left behind.
    store
        .client()
        .query("OPTION IMPORT; UPDATE $rid SET payload = { bar: 0.5 } RETURN NONE")
        .bind(("rid", RecordId::new(schema::MANIFEST_TABLE, "run-2")))
        .await
        .expect("the edit runs")
        .check()
        .expect("and is accepted under OPTION IMPORT");

    let failed = manifests.verify_all().await.expect("verifying");
    assert_eq!(failed, vec!["run-2".to_owned()]);

    // And a load of that manifest says so rather than handing back the payload.
    let err = manifests
        .get("run-2")
        .await
        .expect_err("a moved manifest is corrupt");
    assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    // The untouched one still loads.
    assert!(manifests.get("run-1").await.expect("loading").is_some());
}

// ------------------------------------------------------------------- schema

#[tokio::test]
async fn opening_twice_is_idempotent_and_verifies() {
    let store = connect("doc_bootstrap").await;
    let first = DocStore::<SolverState>::open(store.clone())
        .await
        .expect("the first open bootstraps");
    let second = ManifestStore::<SolverState>::open(store.clone())
        .await
        .expect("the manifest table too");
    DocStore::<SolverState>::open(store.clone())
        .await
        .expect("the second open is a no-op");
    first.assert_schema().await.expect("still verified");
    second.assert_schema().await.expect("still verified");
}

/// A removed refusal event is a write-once table that is no longer write-once,
/// with every name still looking right — which is why the drift guard compares
/// event definitions and treats an unexpected one as drift too.
#[tokio::test]
async fn a_removed_refusal_event_is_drift() {
    let (store, manifests) = manifests("manifest_event_drift").await;
    store
        .client()
        .query(format!(
            "REMOVE EVENT {} ON {}",
            schema::MANIFEST_NO_DELETE_EVENT,
            schema::MANIFEST_TABLE
        ))
        .await
        .expect("removing the event")
        .check()
        .expect("and it is removed");

    let err = manifests
        .assert_schema()
        .await
        .expect_err("a missing refusal event is drift");
    let StoreError::Schema { table, detail } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert_eq!(table, schema::MANIFEST_TABLE);
    assert!(
        detail.contains(schema::MANIFEST_NO_DELETE_EVENT),
        "{detail}"
    );
}

/// An event nobody declared can rewrite or refuse writes, so it is drift rather
/// than an operator's harmless addition — unlike an extra index.
#[tokio::test]
async fn an_undeclared_event_is_drift() {
    let (store, docs) = documents("doc_extra_event").await;
    store
        .client()
        .query(format!(
            "DEFINE EVENT intruder ON {} WHEN $event = \"CREATE\" THEN {{ RETURN NONE; }}",
            schema::DOCUMENT_TABLE
        ))
        .await
        .expect("defining the event")
        .check()
        .expect("and it is defined");

    let err = docs
        .assert_schema()
        .await
        .expect_err("an undeclared event is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("intruder"), "{detail}");
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
