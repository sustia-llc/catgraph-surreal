//! End-to-end cospan store behaviour against the in-memory engine.
//!
//! These need a real engine compiled in, so the whole file is gated on `mem`.
//!
//! The corrupt-document suite below writes rows the store itself would never
//! produce, straight through the connection. That is the point: the store's
//! guarantee is not "documents this store wrote are safe", it is "no document
//! reaches a caller unchecked". Every case here must come back as a typed error
//! and never as a panic — an out-of-bounds leg column included, which is the
//! class the store's own bounds check refuses.

#![cfg(feature = "mem")]

use catgraph::cospan::Cospan;
use catgraph_surreal::{CospanAddr, CospanStore, Store, StoreBuilder, StoreError, cospan, schema};
use surrealdb::types::{RecordId, SurrealValue};

/// `id₂`: two wires, each on its own apex vertex.
fn id2() -> Cospan<usize> {
    Cospan::new(vec![0, 1], vec![0, 1], vec![7, 7]).expect("id₂'s legs are in bounds")
}

/// The same morphism, apex vertices swapped — a different presentation.
fn id2_swapped() -> Cospan<usize> {
    Cospan::new(vec![1, 0], vec![1, 0], vec![7, 7]).expect("id₂'s legs are in bounds")
}

/// The braid on two wires: a genuinely different morphism.
fn braid() -> Cospan<usize> {
    Cospan::new(vec![0, 1], vec![1, 0], vec![7, 7]).expect("the braid's legs are in bounds")
}

/// `μ`: two wires in, one out, all on one apex vertex.
fn merge() -> Cospan<usize> {
    Cospan::new(vec![0, 0], vec![0], vec![7]).expect("μ's legs are in bounds")
}

/// A `0 → 0` cospan whose single apex vertex is hit by neither leg — a scalar.
fn bubble() -> Cospan<usize> {
    Cospan::new(vec![], vec![], vec![7]).expect("no legs to check")
}

async fn connect(database: &str) -> Store {
    StoreBuilder::new("memory")
        .namespace("catgraph_test")
        .database(database)
        .connect()
        .await
        .expect("connecting to the in-memory engine")
}

async fn bootstrapped(database: &str) -> (Store, CospanStore<usize>) {
    let store = connect(database).await;
    let cospans = CospanStore::open(store.clone())
        .await
        .expect("opening the cospan store bootstraps and verifies");
    (store, cospans)
}

// --------------------------------------------------------------- happy paths

/// The contract in one test: what goes in comes back out, revalidated, with its
/// presentation intact — apex ordering included, which is exactly what a
/// canonicalising store would have lost.
#[tokio::test]
async fn a_stored_cospan_comes_back_equal() {
    let (_store, cospans) = bootstrapped("cospan_round_trip").await;
    let cospan = merge();

    let addr = cospans.put(&cospan).await.expect("storing a cospan");
    let loaded = cospans
        .get(&addr)
        .await
        .expect("loading a cospan the store just wrote")
        .expect("the cospan is present");

    assert_eq!(loaded, cospan);
}

/// Content addressing makes writing idempotent. Re-storing the same
/// presentation must be a success and must not create a second row.
#[tokio::test]
async fn re_storing_a_presentation_is_a_no_op() {
    let (store, cospans) = bootstrapped("cospan_idempotent").await;

    let first = cospans.put(&id2()).await.expect("first write");
    let second = cospans.put(&id2()).await.expect("second write");
    assert_eq!(first, second);

    assert_eq!(row_count(&store).await, 1, "one presentation, one row");
}

#[tokio::test]
async fn an_absent_cospan_reads_as_none_rather_than_an_error() {
    let (_store, cospans) = bootstrapped("cospan_absent").await;
    let addr = CospanAddr::from_digest(&"a".repeat(64)).expect("64 lowercase hex chars");
    assert!(cospans.get(&addr).await.expect("querying").is_none());
    assert!(!cospans.contains(&addr).await.expect("existence check"));
}

#[tokio::test]
async fn existence_tracks_what_was_written() {
    let (_store, cospans) = bootstrapped("cospan_contains").await;
    let addr = cospans.put(&merge()).await.expect("storing a cospan");
    assert!(cospans.contains(&addr).await.expect("existence check"));
}

#[tokio::test]
async fn a_batch_stores_every_cospan_and_returns_addresses_in_order() {
    let (_store, cospans) = bootstrapped("cospan_batch").await;
    let batch = [id2(), braid(), merge(), bubble()];

    let addrs = cospans.put_many(&batch).await.expect("storing a batch");
    assert_eq!(addrs.len(), batch.len());

    for (addr, cospan) in addrs.iter().zip(batch.iter()) {
        let loaded = cospans
            .get(addr)
            .await
            .expect("loading a batched cospan")
            .expect("the cospan is present");
        assert_eq!(&loaded, cospan);
    }
}

#[tokio::test]
async fn an_empty_batch_touches_nothing() {
    let (_store, cospans) = bootstrapped("cospan_empty_batch").await;
    let addrs = cospans
        .put_many(&[])
        .await
        .expect("an empty batch succeeds");
    assert!(addrs.is_empty());
}

/// A batch containing the same presentation twice is still idempotent — the
/// second write changes no value, so neither the read-only columns nor the
/// unique key raise anything.
#[tokio::test]
async fn a_batch_may_repeat_a_presentation() {
    let (store, cospans) = bootstrapped("cospan_batch_repeat").await;
    let addrs = cospans
        .put_many(&[merge(), merge()])
        .await
        .expect("storing a batch");
    assert_eq!(addrs[0], addrs[1]);
    assert_eq!(row_count(&store).await, 1);
}

/// Scalars are morphism data, not noise: a closed bubble is not the empty
/// cospan, and two bubbles are not one. All three must coexist as rows.
#[tokio::test]
async fn scalars_are_distinct_morphisms() {
    let (store, cospans) = bootstrapped("cospan_scalars").await;
    let none = Cospan::<usize>::new(vec![], vec![], vec![]).expect("no legs to check");
    let two = Cospan::new(vec![], vec![], vec![7usize, 7]).expect("no legs to check");

    cospans
        .put_many(&[none, bubble(), two])
        .await
        .expect("three distinct morphisms");
    assert_eq!(row_count(&store).await, 3);
}

// ------------------------------------------------------- morphism identity

/// The unique key doing its job. Two presentations, one morphism: the second is
/// refused, and the refusal is a typed error rather than a raw database
/// complaint — because it is an expected outcome, not a fault.
#[tokio::test]
async fn a_second_presentation_of_one_morphism_is_refused() {
    let (store, cospans) = bootstrapped("cospan_duplicate").await;
    let first = cospans.put(&id2()).await.expect("the first presentation");

    let err = cospans
        .put(&id2_swapped())
        .await
        .expect_err("the morphism is already stored");
    let StoreError::Duplicate { table, index } = &err else {
        panic!("expected a duplicate-key failure, got {err:?}");
    };
    assert_eq!(table, schema::COSPAN_TABLE);
    assert_eq!(index, schema::COSPAN_CANON_INDEX);
    // Not a conflict: retrying changes nothing, and treating it as one would
    // spin.
    assert!(!err.is_conflict());
    assert!(!err.is_shutdown());

    // Nothing landed, and the stored presentation is findable.
    assert_eq!(row_count(&store).await, 1);
    let found = cospans
        .find_by_canon(&id2_swapped())
        .await
        .expect("looking the morphism up")
        .expect("it is stored");
    assert_eq!(found, first);
}

/// A batch is one unit, so a batch that contains two presentations of one
/// morphism writes nothing at all rather than a prefix.
#[tokio::test]
async fn a_batch_holding_two_presentations_of_one_morphism_lands_nothing() {
    let (store, cospans) = bootstrapped("cospan_batch_duplicate").await;
    let err = cospans
        .put_many(&[merge(), id2(), id2_swapped()])
        .await
        .expect_err("the batch names one morphism twice");
    assert!(matches!(err, StoreError::Duplicate { .. }), "{err:?}");
    assert_eq!(row_count(&store).await, 0, "a rejected batch lands nothing");
}

#[tokio::test]
async fn an_unstored_morphism_is_not_found() {
    let (_store, cospans) = bootstrapped("cospan_not_found").await;
    cospans.put(&id2()).await.expect("storing the identity");
    assert_eq!(
        cospans.find_by_canon(&braid()).await.expect("looking up"),
        None
    );
}

// -------------------------------------------------------------- schema drift

/// Bootstrapping is idempotent, which is what lets it run on every open. The
/// interesting half is the second call: `DEFINE TABLE` drops a table's fields,
/// so a bootstrap that re-defined the table would leave the schema empty.
#[tokio::test]
async fn bootstrapping_twice_leaves_the_schema_intact() {
    let (_store, cospans) = bootstrapped("cospan_bootstrap_twice").await;
    cospans.bootstrap().await.expect("second bootstrap");
    cospans.assert_schema().await.expect("the schema matches");

    let addr = cospans
        .put(&merge())
        .await
        .expect("storing after re-bootstrap");
    assert!(cospans.contains(&addr).await.expect("existence check"));
}

/// The schema the engine actually holds includes the element definitions that
/// the `array<T>` columns brought with them. If this ever stops matching, a
/// bootstrap fails on a column nobody wrote.
#[tokio::test]
async fn the_bootstrapped_schema_declares_exactly_the_expected_columns() {
    let (store, cospans) = bootstrapped("cospan_schema_columns").await;
    cospans.assert_schema().await.expect("the schema matches");

    let mut response = store
        .client()
        .query("RETURN object::keys((INFO FOR TABLE cospan).fields)")
        .await
        .expect("reading the live column names");
    let mut live: Vec<String> = response.take(0).expect("reading the key set");
    live.sort();

    let mut expected: Vec<String> = schema::COSPAN_FIELDS
        .iter()
        .map(|f| (*f).to_owned())
        .chain(
            schema::COSPAN_ELEMENT_DEFINITIONS
                .iter()
                .map(|(f, _)| (*f).to_owned()),
        )
        .collect();
    expected.sort();
    assert_eq!(live, expected);
}

#[tokio::test]
async fn a_dropped_column_is_detected() {
    let (store, cospans) = bootstrapped("cospan_dropped_column").await;
    run(&store, "REMOVE FIELD canon_key ON cospan").await;

    let err = cospans
        .assert_schema()
        .await
        .expect_err("a dropped column is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("canon_key"), "{detail}");
}

/// The element definition an array column brings with it is part of the live
/// schema, so removing it is drift too — and a name-level check that only knew
/// about the declared columns would miss it entirely.
#[tokio::test]
async fn a_dropped_element_definition_is_detected() {
    let (store, cospans) = bootstrapped("cospan_dropped_element").await;
    run(&store, "REMOVE FIELD dom_leg[*] ON cospan").await;

    let err = cospans
        .assert_schema()
        .await
        .expect_err("a dropped element definition is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("dom_leg"), "{detail}");
}

#[tokio::test]
async fn a_dropped_index_is_detected() {
    let (store, cospans) = bootstrapped("cospan_dropped_index").await;
    run(&store, "REMOVE INDEX cospan_canon ON cospan").await;

    let err = cospans
        .assert_schema()
        .await
        .expect_err("a dropped index is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("cospan_canon"), "{detail}");
}

/// The sharpest case for comparing definitions rather than names: an index that
/// keeps its name, its table, and its column, and loses only the word `UNIQUE` —
/// which is the entire guarantee it exists to provide.
#[tokio::test]
async fn an_index_that_loses_unique_is_detected_as_drift() {
    let (store, cospans) = bootstrapped("cospan_index_not_unique").await;
    run(
        &store,
        "DEFINE INDEX OVERWRITE cospan_canon ON cospan FIELDS canon_key",
    )
    .await;

    let err = cospans
        .assert_schema()
        .await
        .expect_err("an index without UNIQUE is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("UNIQUE"), "{detail}");
}

#[tokio::test]
async fn an_unexpected_column_is_detected() {
    let (store, cospans) = bootstrapped("cospan_extra_column").await;
    run(&store, "DEFINE FIELD provenance ON cospan TYPE string").await;

    let err = cospans
        .assert_schema()
        .await
        .expect_err("an unknown column is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("provenance"), "{detail}");
}

/// A table that was never defined reports no definition, which lands as drift
/// rather than as an opaque table-not-found. Checked through the free function
/// on purpose: a `CospanStore` value cannot exist without a verified schema.
#[tokio::test]
async fn an_undefined_table_is_detected_as_drift() {
    let store = connect("cospan_no_bootstrap").await;
    let err = schema::assert_cospan_schema(store.client())
        .await
        .expect_err("nothing has been defined");
    assert!(matches!(err, StoreError::Schema { .. }), "{err:?}");
}

/// The auto-create hole, closed and pinned. A raw write against an undefined
/// table makes the engine create it `TYPE ANY SCHEMALESS`; a later bootstrap's
/// `DEFINE TABLE IF NOT EXISTS` then blesses the impostor. The definition-level
/// guard is what refuses to serve it.
#[tokio::test]
async fn an_implicitly_created_table_is_refused_at_open() {
    let store = connect("cospan_implicit_table").await;
    run(&store, "CREATE cospan:sneak SET smuggled = true").await;

    let err = CospanStore::<usize>::open(store)
        .await
        .expect_err("an implicitly created table must not verify");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("SCHEMALESS"), "{detail}");
}

/// `ALTER TABLE … SCHEMALESS` keeps every column defined and every name in
/// place while disarming the schema entirely. A name-level check passes this;
/// the definition check must not.
#[tokio::test]
async fn a_schemaless_alteration_is_detected_as_drift() {
    let (store, cospans) = bootstrapped("cospan_altered_schemaless").await;
    run(&store, "ALTER TABLE cospan SCHEMALESS").await;

    let err = cospans
        .assert_schema()
        .await
        .expect_err("a schemaless cospan table is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("SCHEMALESS"), "{detail}");
}

/// Same class of blindness, field-level: re-defining a column without
/// `READONLY` keeps its name while disarming the write-once guard.
#[tokio::test]
async fn a_disarmed_readonly_clause_is_detected_as_drift() {
    let (store, cospans) = bootstrapped("cospan_disarmed_readonly").await;
    run(
        &store,
        "DEFINE FIELD OVERWRITE dom_leg ON cospan TYPE array<int>",
    )
    .await;

    let err = cospans
        .assert_schema()
        .await
        .expect_err("a field without READONLY is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("dom_leg"), "{detail}");
}

/// A table that vanishes after open answers "absent", identically from all
/// three read methods — the same condition must not read as `false` from one
/// and as an opaque query error from another.
#[tokio::test]
async fn a_vanished_table_reads_as_absent_from_every_read_method() {
    let (store, cospans) = bootstrapped("cospan_vanished_table").await;
    let addr = cospans.put(&merge()).await.expect("storing a cospan");
    run(&store, "REMOVE TABLE cospan").await;

    assert!(cospans.get(&addr).await.expect("get absorbs it").is_none());
    assert!(!cospans.contains(&addr).await.expect("contains absorbs it"));
    assert_eq!(
        cospans
            .find_by_canon(&merge())
            .await
            .expect("find absorbs it"),
        None
    );
}

/// A table that vanishes after open must make a WRITE loud, not quiet: a bare
/// write against an undefined table auto-creates it `TYPE ANY SCHEMALESS`,
/// silently disarming the unique canonical key — after which two presentations
/// of one morphism store side by side with no [`StoreError::Duplicate`] ever
/// raised. The write guard turns the condition into [`StoreError::Schema`],
/// and the table stays undefined.
#[tokio::test]
async fn a_write_after_the_table_vanishes_is_refused_loudly() {
    let (store, cospans) = bootstrapped("cospan_write_after_drop").await;
    cospans.put(&merge()).await.expect("the first write lands");
    run(&store, "REMOVE TABLE cospan").await;

    let err = cospans
        .put(&id2())
        .await
        .expect_err("a write must not silently re-create the table");
    assert!(matches!(err, StoreError::Schema { .. }), "{err:?}");

    let mut response = store
        .client()
        .query("RETURN (INFO FOR DB).tables.cospan")
        .await
        .expect("reading the table definition");
    let definition: Option<String> = response.take(0).expect("the definition slot");
    assert_eq!(definition, None, "the table must remain undefined");
}

// ------------------------------------------------------- corrupt documents

/// A cospan row, written by hand.
///
/// Mirrors the store's own row so a test can produce a document the store never
/// would — every case below is one field's difference from a faithful record.
#[derive(Debug, Clone, SurrealValue)]
struct RawRow {
    id: RecordId,
    codec: String,
    dom_leg: Vec<i64>,
    cod_leg: Vec<i64>,
    apex: Vec<String>,
    dom_len: i64,
    cod_len: i64,
    apex_len: i64,
    scalar_count: i64,
    canon_key: String,
}

/// A faithful row for `μ`, ready to be corrupted.
fn good_row() -> RawRow {
    let record = cospan::encode(&merge()).expect("a well-formed cospan encodes");
    RawRow {
        id: RecordId::new(schema::COSPAN_TABLE, record.addr().as_str()),
        codec: record.codec().to_owned(),
        dom_leg: record.dom_leg().to_vec(),
        cod_leg: record.cod_leg().to_vec(),
        apex: record.apex().to_vec(),
        dom_len: record.dom_len(),
        cod_len: record.cod_len(),
        apex_len: record.apex_len(),
        scalar_count: record.scalar_count(),
        canon_key: record.canon_key().to_owned(),
    }
}

/// Re-file a tampered row under the content address of the presentation it now
/// holds.
///
/// Revalidation verifies the address *first* — cheapest check, fully decides
/// tampering of the presentation columns — so a fixture that changes a leg while
/// keeping the old id never reaches the stage it means to exercise. The
/// un-re-filed case has its own test below.
fn refile(mut row: RawRow) -> RawRow {
    let addr = cospan::address_of(&row.dom_leg, &row.cod_leg, &row.apex)
        .expect("a presentation is always addressable");
    row.id = RecordId::new(schema::COSPAN_TABLE, addr.as_str());
    row
}

/// Write a row straight through the connection, bypassing the store.
async fn write_raw(store: &Store, row: &RawRow) -> CospanAddr {
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
    CospanAddr::parse(key).expect("the fixture always uses a well-formed address")
}

/// Load a hand-written row and return whatever the store makes of it.
async fn load_raw(database: &str, row: RawRow) -> Result<Option<Cospan<usize>>, StoreError> {
    let (store, cospans) = bootstrapped(database).await;
    let addr = write_raw(&store, &row).await;
    cospans.get(&addr).await
}

fn expect_corrupt(result: Result<Option<Cospan<usize>>, StoreError>) {
    match result {
        Err(StoreError::Corrupt { .. }) => {}
        other => panic!("expected a corrupt-document failure, got {other:?}"),
    }
}

/// A leg column pointing outside the apex, refused by the store's own bounds
/// check on load.
#[tokio::test]
async fn a_row_whose_leg_points_outside_the_apex_is_rejected() {
    let mut row = good_row();
    row.dom_leg = vec![0, 5];
    row.dom_len = 2;
    expect_corrupt(load_raw("cospan_corrupt_bounds", refile(row)).await);
}

/// A non-canonical label spelling is corrupt, not accepted: a forged row
/// spelled `"007"` — filed under its own (different) address but carrying the
/// *canonical* spelling's `canon_key` — would otherwise squat the honest
/// morphism's unique key while the two-identity discipline reads all green.
///
/// `usize::decode` refuses the spelling outright. A `LabelCodec` that accepted
/// it would be stopped one stage later, by the store's re-encode comparison —
/// which `src/cospan.rs` pins over a deliberately lenient codec.
#[tokio::test]
async fn a_non_canonical_label_spelling_is_rejected_on_load() {
    let mut row = good_row();
    row.apex = vec!["007".to_owned()];
    row.canon_key = cospan::canon_key(&merge()).expect("the canonical key");
    let result = load_raw("cospan_squat_load", refile(row)).await;
    match result {
        Err(StoreError::Corrupt { detail, .. }) => {
            assert!(detail.contains("`007`"), "{detail}");
        }
        other => panic!("expected the forged spelling to be refused, got {other:?}"),
    }
}

/// And the lookup path the squat would actually poison: `find_by_canon` must
/// revalidate the matched row rather than trust the stored `canon_key` — a
/// tampered key must surface as an error, never as a confident wrong address.
#[tokio::test]
async fn find_by_canon_rejects_a_row_squatting_a_key_it_does_not_derive() {
    let (store, cospans) = bootstrapped("cospan_squat_find").await;

    // A forged row: the braid's presentation, filed honestly under its own
    // address, but carrying the IDENTITY's canonical key.
    let braid_record = cospan::encode(&braid()).expect("the braid encodes");
    let mut row = RawRow {
        id: RecordId::new(schema::COSPAN_TABLE, braid_record.addr().as_str()),
        codec: braid_record.codec().to_owned(),
        dom_leg: braid_record.dom_leg().to_vec(),
        cod_leg: braid_record.cod_leg().to_vec(),
        apex: braid_record.apex().to_vec(),
        dom_len: braid_record.dom_len(),
        cod_len: braid_record.cod_len(),
        apex_len: braid_record.apex_len(),
        scalar_count: braid_record.scalar_count(),
        canon_key: cospan::canon_key(&id2()).expect("the identity's key"),
    };
    row = refile(row);
    write_raw(&store, &row).await;

    let err = cospans
        .find_by_canon(&id2())
        .await
        .expect_err("a squatting row must not be served as a confident yes");
    assert!(matches!(err, StoreError::Corrupt { .. }), "{err:?}");
}

/// A negative leg entry cannot come from an in-memory cospan, but the column is
/// a signed integer and a hand-written row can hold one.
#[tokio::test]
async fn a_row_with_a_negative_leg_entry_is_rejected() {
    let mut row = good_row();
    row.dom_leg = vec![0, -1];
    row.dom_len = 2;
    expect_corrupt(load_raw("cospan_corrupt_negative", refile(row)).await);
}

/// The canonical key is `UNIQUE` and complete, so a corrupted one would make a
/// key lookup claim two unequal morphisms are equal.
#[tokio::test]
async fn a_row_whose_canonical_key_lies_is_rejected() {
    let mut row = good_row();
    row.canon_key = cospan::canon_key(&braid()).expect("the braid has a key");
    expect_corrupt(load_raw("cospan_corrupt_canon", row).await);
}

#[tokio::test]
async fn a_row_whose_size_columns_lie_is_rejected() {
    let mut row = good_row();
    row.apex_len = 9;
    expect_corrupt(load_raw("cospan_corrupt_apex_len", row).await);

    let mut row = good_row();
    row.scalar_count = 4;
    expect_corrupt(load_raw("cospan_corrupt_scalars", row).await);
}

/// A row parked under an address that is not its own presentation's digest. The
/// id `ASSERT` cannot catch this — it only checks the *format* of the key, and
/// it is skipped on update and under `OPTION IMPORT` besides.
#[tokio::test]
async fn a_row_filed_under_the_wrong_address_is_rejected() {
    let mut row = good_row();
    row.id = RecordId::new(schema::COSPAN_TABLE, format!("b3_{}", "c".repeat(64)));
    expect_corrupt(load_raw("cospan_corrupt_address", row).await);
}

/// An apex label that is not an encoding of this store's label type. Reported as
/// a corrupt document rather than silently dropped or defaulted.
#[tokio::test]
async fn a_row_with_an_undecodable_label_is_rejected() {
    let mut row = good_row();
    row.apex = vec!["not-a-number".to_owned()];
    expect_corrupt(load_raw("cospan_corrupt_label", refile(row)).await);
}

/// A codec version this build does not know is refused rather than read under
/// the current rules.
#[tokio::test]
async fn a_row_written_under_an_unknown_codec_is_rejected() {
    let mut row = good_row();
    row.codec = "cgc99".to_owned();
    match load_raw("cospan_corrupt_codec", row).await {
        Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "codec"),
        other => panic!("expected a type mismatch on the codec column, got {other:?}"),
    }
}

// ------------------------------------------------- database-side defences

/// Defence in depth, pinned so it is noticed if it stops working: the id
/// `ASSERT` rejects a key that is not shaped like a content address.
#[tokio::test]
async fn the_database_refuses_an_id_that_is_not_a_content_address() {
    let (store, _cospans) = bootstrapped("cospan_bad_id").await;
    let mut row = good_row();
    row.id = RecordId::new(schema::COSPAN_TABLE, "not-an-address");

    let outcome = store
        .client()
        .query("CREATE $row.id CONTENT $row RETURN NONE")
        .bind(("row", row))
        .await
        .expect("the statement runs")
        .check();
    assert!(outcome.is_err(), "the id ASSERT must reject this key");
}

/// The element definitions an `array<T>` column creates carry no `READONLY`
/// clause of their own, which looks like a hole and is not: writing an element
/// changes the parent array's value, and the parent is `READONLY`. Pinned here
/// because it is not visible in the definitions.
#[tokio::test]
async fn an_element_level_write_is_refused_by_the_parent_column() {
    let (store, cospans) = bootstrapped("cospan_element_write").await;
    let addr = cospans.put(&id2()).await.expect("storing a cospan");

    let outcome = store
        .client()
        .query("UPDATE $rid SET dom_leg[0] = 9 RETURN NONE")
        .bind(("rid", RecordId::new(schema::COSPAN_TABLE, addr.as_str())))
        .await
        .expect("the statement runs")
        .check();
    assert!(
        outcome.is_err(),
        "the parent array column must refuse an element write"
    );

    // And the stored presentation is untouched.
    let loaded = cospans
        .get(&addr)
        .await
        .expect("reading back")
        .expect("still present");
    assert_eq!(loaded, id2());
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

/// How many rows the cospan table holds.
async fn row_count(store: &Store) -> usize {
    let mut response = store
        .client()
        .query("SELECT VALUE id FROM cospan")
        .await
        .expect("counting the stored rows");
    let ids: Vec<RecordId> = response.take(0).expect("reading the ids");
    ids.len()
}
