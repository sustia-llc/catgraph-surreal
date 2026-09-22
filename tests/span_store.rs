//! End-to-end span store behaviour against the in-memory engine.
//!
//! Gated on `mem`. The corrupt-document suite writes rows the store would never
//! produce straight through the connection; every case must come back as a
//! typed error.

#![cfg(feature = "mem")]

use catgraph::span::Span;
use catgraph_surreal::{SpanAddr, SpanStore, Store, StoreBuilder, StoreError, schema, span};
use surrealdb::types::{RecordId, SurrealValue};

/// A span's observable parts — `Span` implements neither `PartialEq` nor
/// `Debug`.
type Parts = (Vec<usize>, Vec<usize>, Vec<(usize, usize)>);

fn parts(span: &Span<usize>) -> Parts {
    (
        span.left().to_vec(),
        span.right().to_vec(),
        span.middle_pairs().to_vec(),
    )
}

/// `id₂`: two apex elements, each linking node `i` to node `i`.
fn id2() -> Span<usize> {
    Span::new(vec![7, 7], vec![7, 7], vec![(0, 0), (1, 1)]).expect("id₂'s pairs are valid")
}

/// The same morphism, middle pairs in the other order — a different
/// presentation.
fn id2_swapped() -> Span<usize> {
    Span::new(vec![7, 7], vec![7, 7], vec![(1, 1), (0, 0)]).expect("id₂'s pairs are valid")
}

/// The braid on two wires: a different morphism.
fn braid() -> Span<usize> {
    Span::new(vec![7, 7], vec![7, 7], vec![(0, 1), (1, 0)]).expect("the braid's pairs are valid")
}

/// Two domain nodes onto one codomain node.
fn merge() -> Span<usize> {
    Span::new(vec![7, 7], vec![7], vec![(0, 0), (1, 0)]).expect("the merge's pairs are valid")
}

/// `1 → 1` with the single pair repeated `k` times.
fn repeated(k: usize) -> Span<usize> {
    Span::new(vec![7], vec![7], vec![(0, 0); k]).expect("repeated pairs are valid")
}

async fn connect(database: &str) -> Store {
    StoreBuilder::new("memory")
        .namespace("catgraph_test")
        .database(database)
        .connect()
        .await
        .expect("connecting to the in-memory engine")
}

async fn bootstrapped(database: &str) -> (Store, SpanStore<usize>) {
    let store = connect(database).await;
    let spans = SpanStore::open(store.clone())
        .await
        .expect("opening the span store bootstraps and verifies");
    (store, spans)
}

/// Load and reduce to comparable parts.
async fn get_parts(spans: &SpanStore<usize>, addr: &SpanAddr) -> Option<Parts> {
    spans
        .get(addr)
        .await
        .expect("loading a span the store wrote")
        .as_ref()
        .map(parts)
}

// --------------------------------------------------------------- happy paths

#[tokio::test]
async fn a_stored_span_comes_back_equal() {
    let (_store, spans) = bootstrapped("span_round_trip").await;
    let addr = spans.put(&merge()).await.expect("storing a span");
    assert_eq!(get_parts(&spans, &addr).await, Some(parts(&merge())));
}

#[tokio::test]
async fn re_storing_a_presentation_is_a_no_op() {
    let (store, spans) = bootstrapped("span_idempotent").await;
    let first = spans.put(&id2()).await.expect("first write");
    let second = spans.put(&id2()).await.expect("second write");
    assert_eq!(first, second);
    assert_eq!(row_count(&store).await, 1, "one presentation, one row");
}

#[tokio::test]
async fn an_absent_span_reads_as_none_rather_than_an_error() {
    let (_store, spans) = bootstrapped("span_absent").await;
    let addr = SpanAddr::from_digest(&"a".repeat(64)).expect("64 lowercase hex chars");
    assert!(spans.get(&addr).await.expect("querying").is_none());
    assert!(!spans.contains(&addr).await.expect("existence check"));
}

#[tokio::test]
async fn existence_tracks_what_was_written() {
    let (_store, spans) = bootstrapped("span_contains").await;
    let addr = spans.put(&merge()).await.expect("storing a span");
    assert!(spans.contains(&addr).await.expect("existence check"));
}

#[tokio::test]
async fn a_batch_stores_every_span_and_returns_addresses_in_order() {
    let (_store, spans) = bootstrapped("span_batch").await;
    let batch = [id2(), braid(), merge(), repeated(2)];

    let addrs = spans.put_many(&batch).await.expect("storing a batch");
    assert_eq!(addrs.len(), batch.len());
    for (addr, span) in addrs.iter().zip(batch.iter()) {
        assert_eq!(get_parts(&spans, addr).await, Some(parts(span)));
    }
}

#[tokio::test]
async fn an_empty_batch_touches_nothing() {
    let (_store, spans) = bootstrapped("span_empty_batch").await;
    let addrs = spans.put_many(&[]).await.expect("an empty batch succeeds");
    assert!(addrs.is_empty());
}

#[tokio::test]
async fn a_batch_may_repeat_a_presentation() {
    let (store, spans) = bootstrapped("span_batch_repeat").await;
    let addrs = spans
        .put_many(&[merge(), merge()])
        .await
        .expect("storing a batch");
    assert_eq!(addrs[0], addrs[1]);
    assert_eq!(row_count(&store).await, 1);
}

/// Pair multiplicity is morphism data: zero, one, and two copies of one pair
/// are three morphisms, and all three coexist as rows.
#[tokio::test]
async fn pair_multiplicities_are_distinct_morphisms() {
    let (store, spans) = bootstrapped("span_multiplicity").await;
    spans
        .put_many(&[repeated(0), repeated(1), repeated(2)])
        .await
        .expect("three distinct morphisms");
    assert_eq!(row_count(&store).await, 3);
}

// ------------------------------------------------------- morphism identity

#[tokio::test]
async fn a_second_presentation_of_one_morphism_is_refused() {
    let (store, spans) = bootstrapped("span_duplicate").await;
    let first = spans.put(&id2()).await.expect("the first presentation");

    let err = spans
        .put(&id2_swapped())
        .await
        .expect_err("the morphism is already stored");
    let StoreError::Duplicate { table, index } = &err else {
        panic!("expected a duplicate-key failure, got {err:?}");
    };
    assert_eq!(table, schema::SPAN_TABLE);
    assert_eq!(index, schema::SPAN_CANON_INDEX);
    assert!(!err.is_conflict());
    assert!(!err.is_shutdown());

    assert_eq!(row_count(&store).await, 1);
    let found = spans
        .find_by_canon(&id2_swapped())
        .await
        .expect("looking the morphism up")
        .expect("it is stored");
    assert_eq!(found, first);
}

#[tokio::test]
async fn a_batch_holding_two_presentations_of_one_morphism_lands_nothing() {
    let (store, spans) = bootstrapped("span_batch_duplicate").await;
    let err = spans
        .put_many(&[merge(), id2(), id2_swapped()])
        .await
        .expect_err("the batch names one morphism twice");
    assert!(matches!(err, StoreError::Duplicate { .. }), "{err:?}");
    assert_eq!(row_count(&store).await, 0, "a rejected batch lands nothing");
}

#[tokio::test]
async fn an_unstored_morphism_is_not_found() {
    let (_store, spans) = bootstrapped("span_not_found").await;
    spans.put(&id2()).await.expect("storing the identity");
    assert_eq!(
        spans.find_by_canon(&braid()).await.expect("looking up"),
        None
    );
}

/// A span whose middle pairs are out of bounds is refused before it reaches
/// the database. `Cargo.toml` turns `debug-assertions` off for `catgraph` in the
/// test profile, so `Span::new_unchecked` builds it.
#[tokio::test]
async fn an_ill_formed_span_is_refused_on_write_and_lands_nothing() {
    let (store, spans) = bootstrapped("span_write_refusal").await;
    let out_of_bounds = Span::new_unchecked(vec![7usize], vec![7], vec![(0, 3)]);
    let err = spans
        .put_many(&[merge(), out_of_bounds])
        .await
        .expect_err("an out-of-bounds pair must not become a row");
    assert!(matches!(err, StoreError::Corrupt { .. }), "{err:?}");
    assert_eq!(row_count(&store).await, 0);
}

// -------------------------------------------------------------- schema drift

#[tokio::test]
async fn bootstrapping_twice_leaves_the_schema_intact() {
    let (_store, spans) = bootstrapped("span_bootstrap_twice").await;
    spans.bootstrap().await.expect("second bootstrap");
    spans.assert_schema().await.expect("the schema matches");

    let addr = spans
        .put(&merge())
        .await
        .expect("storing after re-bootstrap");
    assert!(spans.contains(&addr).await.expect("existence check"));
}

#[tokio::test]
async fn the_bootstrapped_schema_declares_exactly_the_expected_columns() {
    let (store, spans) = bootstrapped("span_schema_columns").await;
    spans.assert_schema().await.expect("the schema matches");

    let mut response = store
        .client()
        .query("RETURN object::keys((INFO FOR TABLE span).fields)")
        .await
        .expect("reading the live column names");
    let mut live: Vec<String> = response.take(0).expect("reading the key set");
    live.sort();

    let mut expected: Vec<String> = schema::SPAN_FIELDS
        .iter()
        .map(|f| (*f).to_owned())
        .chain(
            schema::SPAN_ELEMENT_DEFINITIONS
                .iter()
                .map(|(f, _)| (*f).to_owned()),
        )
        .collect();
    expected.sort();
    assert_eq!(live, expected);
}

#[tokio::test]
async fn a_dropped_column_is_detected() {
    let (store, spans) = bootstrapped("span_dropped_column").await;
    run(&store, "REMOVE FIELD canon_key ON span").await;
    let detail = expect_drift(spans.assert_schema().await);
    assert!(detail.contains("canon_key"), "{detail}");
}

#[tokio::test]
async fn a_dropped_element_definition_is_detected() {
    let (store, spans) = bootstrapped("span_dropped_element").await;
    run(&store, "REMOVE FIELD mid_dom[*] ON span").await;
    let detail = expect_drift(spans.assert_schema().await);
    assert!(detail.contains("mid_dom"), "{detail}");
}

#[tokio::test]
async fn a_dropped_index_is_detected() {
    let (store, spans) = bootstrapped("span_dropped_index").await;
    run(&store, "REMOVE INDEX span_canon ON span").await;
    let detail = expect_drift(spans.assert_schema().await);
    assert!(detail.contains("span_canon"), "{detail}");
}

#[tokio::test]
async fn an_index_that_loses_unique_is_detected_as_drift() {
    let (store, spans) = bootstrapped("span_index_not_unique").await;
    run(
        &store,
        "DEFINE INDEX OVERWRITE span_canon ON span FIELDS canon_key",
    )
    .await;
    let detail = expect_drift(spans.assert_schema().await);
    assert!(detail.contains("UNIQUE"), "{detail}");
}

#[tokio::test]
async fn an_unexpected_column_is_detected() {
    let (store, spans) = bootstrapped("span_extra_column").await;
    run(&store, "DEFINE FIELD provenance ON span TYPE string").await;
    let detail = expect_drift(spans.assert_schema().await);
    assert!(detail.contains("provenance"), "{detail}");
}

#[tokio::test]
async fn an_undefined_table_is_detected_as_drift() {
    let store = connect("span_no_bootstrap").await;
    let err = schema::assert_span_schema(store.client())
        .await
        .expect_err("nothing has been defined");
    assert!(matches!(err, StoreError::Schema { .. }), "{err:?}");
}

#[tokio::test]
async fn an_implicitly_created_table_is_refused_at_open() {
    let store = connect("span_implicit_table").await;
    run(&store, "CREATE span:sneak SET smuggled = true").await;
    let detail = expect_drift(SpanStore::<usize>::open(store).await.map(|_| ()));
    assert!(detail.contains("SCHEMALESS"), "{detail}");
}

#[tokio::test]
async fn a_schemaless_alteration_is_detected_as_drift() {
    let (store, spans) = bootstrapped("span_altered_schemaless").await;
    run(&store, "ALTER TABLE span SCHEMALESS").await;
    let detail = expect_drift(spans.assert_schema().await);
    assert!(detail.contains("SCHEMALESS"), "{detail}");
}

#[tokio::test]
async fn a_disarmed_readonly_clause_is_detected_as_drift() {
    let (store, spans) = bootstrapped("span_disarmed_readonly").await;
    run(
        &store,
        "DEFINE FIELD OVERWRITE mid_dom ON span TYPE array<int>",
    )
    .await;
    let detail = expect_drift(spans.assert_schema().await);
    assert!(detail.contains("mid_dom"), "{detail}");
}

#[tokio::test]
async fn a_vanished_table_reads_as_absent_from_every_read_method() {
    let (store, spans) = bootstrapped("span_vanished_table").await;
    let addr = spans.put(&merge()).await.expect("storing a span");
    run(&store, "REMOVE TABLE span").await;

    assert!(spans.get(&addr).await.expect("get absorbs it").is_none());
    assert!(!spans.contains(&addr).await.expect("contains absorbs it"));
    assert_eq!(
        spans
            .find_by_canon(&merge())
            .await
            .expect("find absorbs it"),
        None
    );
}

#[tokio::test]
async fn a_write_after_the_table_vanishes_is_refused_loudly() {
    let (store, spans) = bootstrapped("span_write_after_drop").await;
    spans.put(&merge()).await.expect("the first write lands");
    run(&store, "REMOVE TABLE span").await;

    let err = spans
        .put(&id2())
        .await
        .expect_err("a write must not silently re-create the table");
    assert!(matches!(err, StoreError::Schema { .. }), "{err:?}");

    let mut response = store
        .client()
        .query("RETURN (INFO FOR DB).tables.span")
        .await
        .expect("reading the table definition");
    let definition: Option<String> = response.take(0).expect("the definition slot");
    assert_eq!(definition, None, "the table must remain undefined");
}

// ------------------------------------------------------- corrupt documents

/// A span row, written by hand.
#[derive(Debug, Clone, SurrealValue)]
struct RawRow {
    id: RecordId,
    codec: String,
    dom: Vec<String>,
    cod: Vec<String>,
    mid_dom: Vec<i64>,
    mid_cod: Vec<i64>,
    dom_len: i64,
    cod_len: i64,
    apex_len: i64,
    canon_key: String,
}

fn row_of(span: &Span<usize>) -> RawRow {
    let record = span::encode(span).expect("a well-formed span encodes");
    RawRow {
        id: RecordId::new(schema::SPAN_TABLE, record.addr().as_str()),
        codec: record.codec().to_owned(),
        dom: record.dom().to_vec(),
        cod: record.cod().to_vec(),
        mid_dom: record.mid_dom().to_vec(),
        mid_cod: record.mid_cod().to_vec(),
        dom_len: record.dom_len(),
        cod_len: record.cod_len(),
        apex_len: record.apex_len(),
        canon_key: record.canon_key().to_owned(),
    }
}

/// A faithful row for the merge span, ready to be corrupted.
fn good_row() -> RawRow {
    row_of(&merge())
}

/// Re-file a tampered row under the content address of the presentation it now
/// holds, so the address check passes and a later stage fires.
fn refile(mut row: RawRow) -> RawRow {
    let addr = span::address_of(&row.dom, &row.cod, &row.mid_dom, &row.mid_cod)
        .expect("a presentation is always addressable");
    row.id = RecordId::new(schema::SPAN_TABLE, addr.as_str());
    row
}

/// Write a row straight through the connection, bypassing the store.
async fn write_raw(store: &Store, row: &RawRow) -> SpanAddr {
    store
        .client()
        .query("CREATE $row.id CONTENT $row RETURN NONE")
        .bind(("row", row.clone()))
        .await
        .expect("writing a raw row")
        .check()
        .expect("the raw write is accepted");
    let surrealdb::types::RecordIdKey::String(key) = &row.id.key else {
        panic!("the fixture always uses a string key");
    };
    SpanAddr::parse(key).expect("the fixture always uses a well-formed address")
}

/// Load a hand-written row and return whatever the store makes of it.
async fn load_raw(database: &str, row: RawRow) -> Result<Option<Parts>, StoreError> {
    let (store, spans) = bootstrapped(database).await;
    let addr = write_raw(&store, &row).await;
    spans.get(&addr).await.map(|span| span.as_ref().map(parts))
}

/// Assert a corrupt-document failure whose detail contains `needle`.
fn expect_corrupt(result: Result<Option<Parts>, StoreError>, needle: &str) {
    match result {
        Err(StoreError::Corrupt { detail, .. }) => {
            assert!(detail.contains(needle), "`{needle}` not in: {detail}");
        }
        other => panic!("expected a corrupt-document failure, got {other:?}"),
    }
}

#[tokio::test]
async fn a_row_whose_pair_points_outside_the_boundary_is_rejected() {
    let mut row = good_row();
    row.mid_cod = vec![0, 5];
    expect_corrupt(
        load_raw("span_corrupt_bounds", refile(row)).await,
        "codomain node 5",
    );
}

#[tokio::test]
async fn a_row_whose_pair_links_different_labels_is_rejected() {
    let mut row = good_row();
    row.cod = vec!["8".to_owned()];
    expect_corrupt(
        load_raw("span_corrupt_labels", refile(row)).await,
        "labels differ",
    );
}

/// A forged row spelled `"007"` carrying the canonical spelling's key is
/// refused; `usize::decode` refuses the spelling at the decode stage.
#[tokio::test]
async fn a_non_canonical_label_spelling_is_rejected_on_load() {
    let mut row = good_row();
    row.cod = vec!["007".to_owned()];
    row.canon_key = span::canon_key(&merge()).expect("the canonical key");
    expect_corrupt(
        load_raw("span_squat_load", refile(row)).await,
        "`007`, which is not a label of this type",
    );
}

/// `find_by_canon` revalidates the matched row rather than trusting the
/// stored key.
#[tokio::test]
async fn find_by_canon_rejects_a_row_squatting_a_key_it_does_not_derive() {
    let (store, spans) = bootstrapped("span_squat_find").await;
    let mut row = row_of(&braid());
    row.canon_key = span::canon_key(&id2()).expect("the identity's key");
    write_raw(&store, &refile(row)).await;

    let err = spans
        .find_by_canon(&id2())
        .await
        .expect_err("a squatting row must not be served as a confident yes");
    assert!(matches!(err, StoreError::Corrupt { .. }), "{err:?}");
}

#[tokio::test]
async fn a_row_with_a_negative_pair_index_is_rejected() {
    let mut row = good_row();
    row.mid_dom = vec![0, -1];
    expect_corrupt(
        load_raw("span_corrupt_negative", refile(row)).await,
        "domain index -1",
    );
}

#[tokio::test]
async fn a_row_with_mismatched_middle_columns_is_rejected() {
    let mut row = good_row();
    row.mid_cod = vec![0];
    expect_corrupt(
        load_raw("span_corrupt_mid_lengths", refile(row)).await,
        "mid_dom has 2 entries but mid_cod has 1",
    );
}

#[tokio::test]
async fn a_row_whose_canonical_key_lies_is_rejected() {
    let mut row = good_row();
    row.canon_key = span::canon_key(&braid()).expect("the braid has a key");
    expect_corrupt(load_raw("span_corrupt_canon", row).await, "canon_key");
}

#[tokio::test]
async fn a_row_whose_size_columns_lie_is_rejected() {
    let mut row = good_row();
    row.apex_len = 9;
    expect_corrupt(load_raw("span_corrupt_apex_len", row).await, "apex_len");

    let mut row = good_row();
    row.dom_len = 4;
    expect_corrupt(load_raw("span_corrupt_dom_len", row).await, "dom_len");
}

#[tokio::test]
async fn a_row_filed_under_the_wrong_address_is_rejected() {
    let mut row = good_row();
    row.id = RecordId::new(schema::SPAN_TABLE, format!("b3_{}", "c".repeat(64)));
    expect_corrupt(
        load_raw("span_corrupt_address", row).await,
        "content address",
    );
}

#[tokio::test]
async fn a_row_with_an_undecodable_label_is_rejected() {
    let mut row = good_row();
    row.dom = vec!["not-a-number".to_owned(), "7".to_owned()];
    expect_corrupt(
        load_raw("span_corrupt_label", refile(row)).await,
        "not a label of this type",
    );
}

#[tokio::test]
async fn a_row_written_under_an_unknown_codec_is_rejected() {
    let mut row = good_row();
    row.codec = "cgs99".to_owned();
    match load_raw("span_corrupt_codec", row).await {
        Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "codec"),
        other => panic!("expected a type mismatch on the codec column, got {other:?}"),
    }
}

// ------------------------------------------------- database-side defences

#[tokio::test]
async fn the_database_refuses_an_id_that_is_not_a_content_address() {
    let (store, _spans) = bootstrapped("span_bad_id").await;
    let mut row = good_row();
    row.id = RecordId::new(schema::SPAN_TABLE, "not-an-address");

    let outcome = store
        .client()
        .query("CREATE $row.id CONTENT $row RETURN NONE")
        .bind(("row", row))
        .await
        .expect("the statement runs")
        .check();
    assert!(outcome.is_err(), "the id ASSERT must reject this key");
}

/// An element write changes the parent array column's value, and the parent
/// is `READONLY`.
#[tokio::test]
async fn an_element_level_write_is_refused_by_the_parent_column() {
    let (store, spans) = bootstrapped("span_element_write").await;
    let addr = spans.put(&id2()).await.expect("storing a span");

    let outcome = store
        .client()
        .query("UPDATE $rid SET mid_dom[0] = 9 RETURN NONE")
        .bind(("rid", RecordId::new(schema::SPAN_TABLE, addr.as_str())))
        .await
        .expect("the statement runs")
        .check();
    assert!(
        outcome.is_err(),
        "the parent array column must refuse an element write"
    );
    assert_eq!(get_parts(&spans, &addr).await, Some(parts(&id2())));
}

// ------------------------------------------------------------------ helpers

/// Run a statement for its effect, failing loudly if it does not take.
async fn run(store: &Store, statement: &str) {
    store
        .client()
        .query(statement)
        .await
        .expect("the statement runs")
        .check()
        .expect("the statement succeeds");
}

/// Assert schema drift and return its detail.
fn expect_drift(result: Result<(), StoreError>) -> String {
    match result {
        Err(StoreError::Schema { detail, .. }) => detail,
        other => panic!("expected schema drift, got {other:?}"),
    }
}

/// How many rows the span table holds.
async fn row_count(store: &Store) -> usize {
    let mut response = store
        .client()
        .query("SELECT VALUE id FROM span")
        .await
        .expect("counting the stored rows");
    let ids: Vec<RecordId> = response.take(0).expect("reading the ids");
    ids.len()
}
