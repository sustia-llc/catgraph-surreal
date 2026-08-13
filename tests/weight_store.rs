//! End-to-end weight store behaviour against the in-memory engine.
//!
//! These need a real engine compiled in, so the whole file is gated on `mem`.
//!
//! Two things are being pinned here that no unit test can reach. The first is
//! **bit-exactness through a real engine**: the byte lane exists because native
//! float columns lose `NaN` payloads and signed zeros on the export path, and
//! the only way to know the bytes survived a round trip through SurrealDB is to
//! send them through one. The second is the **write-once contract** — a changed
//! vector under an existing key must be refused, and refused with a typed error
//! rather than an opaque database complaint.

#![cfg(feature = "mem")]

use catgraph_dl::para::RModule;
use catgraph_surreal::{Store, StoreBuilder, StoreError, WeightStore, schema, weight};
use surrealdb::types::{Bytes, RecordId, SurrealValue};

/// Coordinates chosen so that a lossy lane would be caught: a `NaN` carrying a
/// payload, both zeros, both infinities, and one ordinary value.
fn awkward() -> RModule<f64> {
    RModule::new(vec![
        f64::from_bits(0x7FF8_0000_DEAD_BEEF),
        -0.0,
        0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.5,
    ])
}

fn bits(module: &RModule<f64>) -> Vec<u64> {
    module.as_slice().iter().map(|c| c.to_bits()).collect()
}

async fn connect(database: &str) -> Store {
    StoreBuilder::new("memory")
        .namespace("catgraph_test")
        .database(database)
        .connect()
        .await
        .expect("connecting to the in-memory engine")
}

async fn bootstrapped(database: &str) -> (Store, WeightStore) {
    let store = connect(database).await;
    let weights = WeightStore::open(store.clone())
        .await
        .expect("opening the weight store bootstraps and verifies");
    (store, weights)
}

// --------------------------------------------------------------- happy paths

#[tokio::test]
async fn a_stored_vector_comes_back_equal() {
    let (_store, weights) = bootstrapped("weight_round_trip").await;
    let vector = RModule::new(vec![1.0, -2.5, 3.25]);

    weights
        .put("genome-a", "layer-0", &vector)
        .await
        .expect("storing a weight vector");
    let loaded = weights
        .get("genome-a", "layer-0")
        .await
        .expect("loading it back")
        .expect("it is present");

    assert_eq!(loaded.as_slice(), vector.as_slice());
    assert_eq!(loaded.dim(), 3);
}

/// The reason the coordinates are bytes. Through the value layer *and* through
/// an export, these bit patterns are what a float column cannot promise: every
/// `NaN` renders as the bare literal `NaN` on the export path, sign and payload
/// gone.
#[tokio::test]
async fn non_finite_coordinates_survive_bit_exactly() {
    let (_store, weights) = bootstrapped("weight_bit_exact").await;
    let vector = awkward();

    weights
        .put("genome-a", "step-1", &vector)
        .await
        .expect("storing a non-finite vector");
    let loaded = weights
        .get("genome-a", "step-1")
        .await
        .expect("loading it back")
        .expect("it is present");

    assert_eq!(bits(&loaded), bits(&vector));
    // Spelled out, because `==` cannot see either of these differences.
    assert_eq!(loaded.as_slice()[0].to_bits(), 0x7FF8_0000_DEAD_BEEF);
    assert_eq!(loaded.as_slice()[1].to_bits(), (-0.0f64).to_bits());
    assert_ne!(
        loaded.as_slice()[1].to_bits(),
        loaded.as_slice()[2].to_bits(),
        "the two zeros must not have merged"
    );
}

#[tokio::test]
async fn the_zero_dimensional_module_round_trips() {
    let (_store, weights) = bootstrapped("weight_zero_dim").await;
    weights
        .put("genome-a", "empty", &RModule::new(Vec::new()))
        .await
        .expect("storing the empty module");
    let loaded = weights
        .get("genome-a", "empty")
        .await
        .expect("loading it back")
        .expect("it is present");
    assert_eq!(loaded.dim(), 0);
}

/// Storing the same vector under the same key twice changes no value, so the
/// read-only columns raise nothing and the unique key sees the same row.
#[tokio::test]
async fn re_storing_an_identical_vector_is_a_no_op() {
    let (store, weights) = bootstrapped("weight_idempotent").await;
    let vector = awkward();

    weights.put("g", "k", &vector).await.expect("first write");
    weights.put("g", "k", &vector).await.expect("second write");

    assert_eq!(row_count(&store).await, 1, "one key pair, one row");
    let loaded = weights
        .get("g", "k")
        .await
        .expect("loading")
        .expect("present");
    assert_eq!(bits(&loaded), bits(&vector));
}

/// The two halves of the key are independent: neither alone identifies a row.
#[tokio::test]
async fn the_key_pair_separates_rows() {
    let (store, weights) = bootstrapped("weight_key_pair").await;
    for (genome, gen_key, value) in [
        ("g1", "k1", 1.0),
        ("g1", "k2", 2.0),
        ("g2", "k1", 3.0),
        ("g2", "k2", 4.0),
    ] {
        weights
            .put(genome, gen_key, &RModule::new(vec![value]))
            .await
            .expect("storing");
    }
    assert_eq!(row_count(&store).await, 4);

    let loaded = weights
        .get("g2", "k1")
        .await
        .expect("loading")
        .expect("present");
    assert_eq!(loaded.as_slice(), [3.0]);
}

#[tokio::test]
async fn an_absent_key_reads_as_none_rather_than_an_error() {
    let (_store, weights) = bootstrapped("weight_absent").await;
    assert!(
        weights
            .get("nobody", "nothing")
            .await
            .expect("querying")
            .is_none()
    );
    assert!(
        !weights
            .contains("nobody", "nothing")
            .await
            .expect("existence check")
    );
}

#[tokio::test]
async fn existence_tracks_what_was_written() {
    let (_store, weights) = bootstrapped("weight_contains").await;
    weights
        .put("g", "k", &RModule::new(vec![1.0]))
        .await
        .expect("storing");
    assert!(weights.contains("g", "k").await.expect("existence check"));
    assert!(!weights.contains("g", "other").await.expect("existence"));
}

// ---------------------------------------------------------- the write-once contract

/// The alarm. A *changed* vector under an existing key is refused — and it has
/// to be a changed one: a no-op write fires nothing, so a test that re-wrote the
/// same value would pass vacuously.
#[tokio::test]
async fn a_changed_vector_under_an_existing_key_is_refused() {
    let (_store, weights) = bootstrapped("weight_write_once").await;
    weights
        .put("g", "k", &RModule::new(vec![1.0, 2.0]))
        .await
        .expect("the first vector");

    let err = weights
        .put("g", "k", &RModule::new(vec![1.0, 3.0]))
        .await
        .expect_err("weights are write-once under their key");
    let StoreError::ReadOnly { table, field } = &err else {
        panic!("expected a write-once refusal, got {err:?}");
    };
    assert_eq!(table, schema::WEIGHT_TABLE);
    assert_eq!(field, "coordinates");
    // Not retryable: the vector will not become writable.
    assert!(!err.is_conflict());
    assert!(!err.is_shutdown());

    // And the stored vector is untouched.
    let loaded = weights
        .get("g", "k")
        .await
        .expect("loading")
        .expect("present");
    assert_eq!(loaded.as_slice(), [1.0, 2.0]);
}

/// A vector of a different length under the same key trips the same guard —
/// through `dim` rather than `coordinates`, depending on which column the engine
/// reaches first. Either is the same refusal.
#[tokio::test]
async fn a_vector_of_a_different_dimension_is_refused_too() {
    let (_store, weights) = bootstrapped("weight_write_once_dim").await;
    weights
        .put("g", "k", &RModule::new(vec![1.0]))
        .await
        .expect("the first vector");

    let err = weights
        .put("g", "k", &RModule::new(vec![1.0, 1.0]))
        .await
        .expect_err("weights are write-once under their key");
    let StoreError::ReadOnly { table, field } = &err else {
        panic!("expected a write-once refusal, got {err:?}");
    };
    assert_eq!(table, schema::WEIGHT_TABLE);
    assert!(
        field == "dim" || field == "coordinates",
        "unexpected column: {field}"
    );
}

/// The unique index, doing what the derived record id cannot: a row this store
/// did not write, claiming the same key pair under a different id.
#[tokio::test]
async fn a_foreign_row_claiming_the_key_pair_is_refused() {
    let (store, weights) = bootstrapped("weight_duplicate").await;
    let mut row = good_row("g", "k");
    row.id = RecordId::new(schema::WEIGHT_TABLE, format!("b3_{}", "c".repeat(64)));
    write_raw(&store, &row).await;

    let err = weights
        .put("g", "k", &RModule::new(vec![1.0]))
        .await
        .expect_err("the key pair is taken");
    let StoreError::Duplicate { table, index } = &err else {
        panic!("expected a duplicate-key failure, got {err:?}");
    };
    assert_eq!(table, schema::WEIGHT_TABLE);
    assert_eq!(index, schema::WEIGHT_KEY_INDEX);
    assert!(!err.is_conflict());
}

// -------------------------------------------------------------- schema drift

#[tokio::test]
async fn bootstrapping_twice_leaves_the_schema_intact() {
    let (_store, weights) = bootstrapped("weight_bootstrap_twice").await;
    weights.bootstrap().await.expect("second bootstrap");
    weights.assert_schema().await.expect("the schema matches");

    weights
        .put("g", "k", &RModule::new(vec![1.0]))
        .await
        .expect("storing after re-bootstrap");
    assert!(weights.contains("g", "k").await.expect("existence check"));
}

#[tokio::test]
async fn the_bootstrapped_schema_declares_exactly_the_expected_columns() {
    let (store, weights) = bootstrapped("weight_schema_columns").await;
    weights.assert_schema().await.expect("the schema matches");

    let mut response = store
        .client()
        .query("RETURN object::keys((INFO FOR TABLE weight).fields)")
        .await
        .expect("reading the live column names");
    let mut live: Vec<String> = response.take(0).expect("reading the key set");
    live.sort();

    let mut expected: Vec<String> = schema::WEIGHT_FIELDS
        .iter()
        .map(|f| (*f).to_owned())
        .collect();
    expected.sort();
    assert_eq!(live, expected);
}

#[tokio::test]
async fn a_dropped_column_is_detected() {
    let (store, weights) = bootstrapped("weight_dropped_column").await;
    run(&store, "REMOVE FIELD finite ON weight").await;

    let err = weights
        .assert_schema()
        .await
        .expect_err("a dropped column is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("finite"), "{detail}");
}

/// The sharpest case for comparing definitions rather than names: the index
/// keeps its name, its table, and both columns, and loses only the word
/// `UNIQUE` — which is the entire guarantee it exists to provide.
#[tokio::test]
async fn an_index_that_loses_unique_is_detected_as_drift() {
    let (store, weights) = bootstrapped("weight_index_not_unique").await;
    run(
        &store,
        "DEFINE INDEX OVERWRITE weight_key ON weight FIELDS genome, gen_key",
    )
    .await;

    let err = weights
        .assert_schema()
        .await
        .expect_err("an index without UNIQUE is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("UNIQUE"), "{detail}");
}

/// `ALTER TABLE … SCHEMALESS` keeps every column defined and every name in
/// place while disarming the schema entirely.
#[tokio::test]
async fn a_schemaless_alteration_is_detected_as_drift() {
    let (store, weights) = bootstrapped("weight_altered_schemaless").await;
    run(&store, "ALTER TABLE weight SCHEMALESS").await;

    let err = weights
        .assert_schema()
        .await
        .expect_err("a schemaless weight table is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("SCHEMALESS"), "{detail}");
}

/// Field-level version of the same blindness: re-defining the coordinate column
/// without `READONLY` keeps its name while disarming the write-once contract.
#[tokio::test]
async fn a_disarmed_readonly_clause_is_detected_as_drift() {
    let (store, weights) = bootstrapped("weight_disarmed_readonly").await;
    run(
        &store,
        "DEFINE FIELD OVERWRITE coordinates ON weight TYPE bytes",
    )
    .await;

    let err = weights
        .assert_schema()
        .await
        .expect_err("a field without READONLY is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("coordinates"), "{detail}");
}

#[tokio::test]
async fn an_undefined_table_is_detected_as_drift() {
    let store = connect("weight_no_bootstrap").await;
    let err = schema::assert_weight_schema(store.client())
        .await
        .expect_err("nothing has been defined");
    assert!(matches!(err, StoreError::Schema { .. }), "{err:?}");
}

/// The auto-create hole, closed and pinned: a raw write against an undefined
/// table makes the engine create it `SCHEMALESS`, and a later bootstrap would
/// bless the impostor rather than replace it.
#[tokio::test]
async fn an_implicitly_created_table_is_refused_at_open() {
    let store = connect("weight_implicit_table").await;
    run(&store, "CREATE weight:sneak SET smuggled = true").await;

    let err = WeightStore::open(store)
        .await
        .expect_err("an implicitly created table must not verify");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("SCHEMALESS"), "{detail}");
}

/// A table that vanishes after open answers "absent" from both read methods —
/// the same condition must not read as `false` from one and as an opaque query
/// error from the other.
#[tokio::test]
async fn a_vanished_table_reads_as_absent_from_both_read_methods() {
    let (store, weights) = bootstrapped("weight_vanished_table").await;
    weights
        .put("g", "k", &RModule::new(vec![1.0]))
        .await
        .expect("storing");
    run(&store, "REMOVE TABLE weight").await;

    assert!(
        weights
            .get("g", "k")
            .await
            .expect("get absorbs it")
            .is_none()
    );
    assert!(
        !weights
            .contains("g", "k")
            .await
            .expect("contains absorbs it")
    );
}

// ------------------------------------------------------- corrupt documents

/// A weight row, written by hand.
#[derive(Debug, Clone, SurrealValue)]
struct RawRow {
    id: RecordId,
    codec: String,
    genome: String,
    gen_key: String,
    dim: i64,
    coordinates: Bytes,
    finite: bool,
}

/// A faithful row, ready to be corrupted.
fn good_row(genome: &str, gen_key: &str) -> RawRow {
    let record = weight::encode(genome, gen_key, &awkward()).expect("a module encodes");
    RawRow {
        id: RecordId::new(schema::WEIGHT_TABLE, record.key()),
        codec: record.codec().to_owned(),
        genome: record.genome().to_owned(),
        gen_key: record.gen_key().to_owned(),
        dim: record.dim(),
        coordinates: Bytes::from(record.coordinates().to_vec()),
        finite: record.finite(),
    }
}

/// Re-file a row under the key its `(genome, gen_key)` pair derives, so the key
/// check passes and a *later* stage fires.
fn refile(mut row: RawRow) -> RawRow {
    let key = weight::record_key(&row.genome, &row.gen_key).expect("a pair always has a key");
    row.id = RecordId::new(schema::WEIGHT_TABLE, key);
    row
}

async fn write_raw(store: &Store, row: &RawRow) {
    store
        .client()
        .query("CREATE $row.id CONTENT $row RETURN NONE")
        .bind(("row", row.clone()))
        .await
        .expect("writing a raw row")
        .check()
        .expect("the raw write is accepted");
}

/// Load a hand-written row and return whatever the store makes of it.
async fn load_raw(database: &str, row: RawRow) -> Result<Option<RModule<f64>>, StoreError> {
    let (store, weights) = bootstrapped(database).await;
    let genome = row.genome.clone();
    let gen_key = row.gen_key.clone();
    write_raw(&store, &row).await;
    weights.get(&genome, &gen_key).await
}

/// The cross-column invariant no schema type can express: the byte width has to
/// be exactly `dim × 8`.
#[tokio::test]
async fn a_row_whose_byte_width_disagrees_with_dim_is_rejected() {
    let mut row = good_row("g", "k");
    let mut bytes = row.coordinates.to_vec();
    bytes.truncate(bytes.len() - 1);
    row.coordinates = Bytes::from(bytes);
    match load_raw("weight_corrupt_width", row).await {
        Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "coordinates"),
        other => panic!("expected a coordinate width mismatch, got {other:?}"),
    }
}

#[tokio::test]
async fn a_row_whose_dim_column_lies_is_rejected() {
    let mut row = good_row("g", "k");
    row.dim = 99;
    match load_raw("weight_corrupt_dim", row).await {
        Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "coordinates"),
        other => panic!("expected a coordinate width mismatch, got {other:?}"),
    }
}

#[tokio::test]
async fn a_row_with_a_negative_dim_is_rejected() {
    let mut row = good_row("g", "k");
    row.dim = -1;
    match load_raw("weight_corrupt_negative_dim", row).await {
        Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "dim"),
        other => panic!("expected a dimension mismatch, got {other:?}"),
    }
}

/// The flag is a claim, and a claim nobody checks is worth nothing. These
/// coordinates are not finite.
#[tokio::test]
async fn a_row_whose_finite_flag_lies_is_rejected() {
    let mut row = good_row("g", "k");
    row.finite = true;
    match load_raw("weight_corrupt_finite", row).await {
        Err(StoreError::Corrupt { .. }) => {}
        other => panic!("expected a corrupt-document failure, got {other:?}"),
    }
}

/// A row filed under a key that is not its own pair's — moved by hand, or
/// written by something that derives keys differently.
#[tokio::test]
async fn a_row_filed_under_the_wrong_key_is_rejected() {
    let mut row = good_row("g", "k");
    row.id = RecordId::new(schema::WEIGHT_TABLE, format!("b3_{}", "c".repeat(64)));
    match load_raw("weight_corrupt_key", row).await {
        Err(StoreError::Corrupt { .. }) => {}
        other => panic!("expected a corrupt-document failure, got {other:?}"),
    }
}

#[tokio::test]
async fn a_row_written_under_an_unknown_codec_is_rejected() {
    let mut row = good_row("g", "k");
    row.codec = "cgw99".to_owned();
    match load_raw("weight_corrupt_codec", refile(row)).await {
        Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "codec"),
        other => panic!("expected a type mismatch on the codec column, got {other:?}"),
    }
}

// ------------------------------------------------- database-side defences

/// Defence in depth, pinned so it is noticed if it stops working.
#[tokio::test]
async fn the_database_refuses_an_id_that_is_not_a_derived_key() {
    let (store, _weights) = bootstrapped("weight_bad_id").await;
    let mut row = good_row("g", "k");
    row.id = RecordId::new(schema::WEIGHT_TABLE, "not-a-key");

    let outcome = store
        .client()
        .query("CREATE $row.id CONTENT $row RETURN NONE")
        .bind(("row", row))
        .await
        .expect("the statement runs")
        .check();
    assert!(outcome.is_err(), "the id ASSERT must reject this key");
}

// ------------------------------------------------------------------ helpers

async fn run(store: &Store, statement: &str) {
    store
        .client()
        .query(statement)
        .await
        .expect("the statement runs")
        .check()
        .expect("the statement succeeds");
}

async fn row_count(store: &Store) -> usize {
    let mut response = store
        .client()
        .query("SELECT VALUE id FROM weight")
        .await
        .expect("counting the stored rows");
    let ids: Vec<RecordId> = response.take(0).expect("reading the ids");
    ids.len()
}
