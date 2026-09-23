//! End-to-end term store behaviour against the in-memory engine.
//!
//! These need a real engine compiled in, so the whole file is gated on `mem`.
//!
//! The corrupt-document suite below writes rows the store itself would never
//! produce, straight through the connection. That is the point: the store's
//! guarantee is not "documents this store wrote are safe", it is "no document
//! reaches a caller unchecked". Every case here must come back as a typed error
//! and never as a panic.

#![cfg(feature = "mem")]

use std::borrow::Cow;

use catgraph_applied::prop::colored::ColoredExpr;
use catgraph_applied::prop::{Free, PropExpr, PropSignature};
use catgraph_surreal::error::RevalidationStage;
use catgraph_surreal::{Store, StoreBuilder, StoreError, TermAddr, TermStore, schema, term};
use serde::{Deserialize, Serialize};
use surrealdb::types::{RecordId, SurrealValue};

/// A minimal well-behaved signature.
///
/// "Well-behaved" is the load-bearing word: the derived `Serialize` visits four
/// unit variants in a fixed order, with no map or set anywhere in it, so the
/// same term encodes to the same bytes on every run. A generator whose serde
/// went through a `HashMap` would produce a fresh content address each time.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
enum Gen {
    /// `Δ : 1 → 2`
    Copy,
    /// `! : 1 → 0`
    Discard,
    /// `μ : 2 → 1`
    Add,
    /// `η : 0 → 1`
    Zero,
}

/// One variant byte: `Copy` 0, `Discard` 1, `Add` 2, `Zero` 3.
impl catgraph::CanonicalEncode for Gen {
    fn encode_canonical(&self, out: &mut Vec<u8>) {
        out.push(match self {
            Self::Copy => 0,
            Self::Discard => 1,
            Self::Add => 2,
            Self::Zero => 3,
        });
    }
}

impl PropSignature for Gen {
    type Color = ();

    fn source_word(&self) -> Cow<'_, [()]> {
        Cow::Owned(match self {
            Self::Copy | Self::Discard => vec![()],
            Self::Add => vec![(), ()],
            Self::Zero => vec![],
        })
    }

    fn target_word(&self) -> Cow<'_, [()]> {
        Cow::Owned(match self {
            Self::Copy => vec![(), ()],
            Self::Discard => vec![],
            Self::Add | Self::Zero => vec![()],
        })
    }
}

/// `Δ ; μ : 1 → 1`.
fn copy_then_add() -> ColoredExpr<Gen> {
    let expr = Free::compose(Free::generator(Gen::Copy), Free::<Gen>::generator(Gen::Add))
        .expect("Δ ; μ composes: 1 → 2 → 1");
    ColoredExpr::new(vec![()], expr).expect("Δ ; μ type-checks at one wire")
}

/// `η ; ! : 0 → 0`.
fn zero_then_discard() -> ColoredExpr<Gen> {
    let expr = Free::compose(
        Free::generator(Gen::Zero),
        Free::<Gen>::generator(Gen::Discard),
    )
    .expect("η ; ! composes: 0 → 1 → 0");
    ColoredExpr::new(Vec::new(), expr).expect("η ; ! type-checks at no wires")
}

/// `id₁ : 1 → 1`.
fn identity() -> ColoredExpr<Gen> {
    ColoredExpr::new(vec![()], Free::<Gen>::identity(1)).expect("id₁ type-checks at one wire")
}

async fn connect(database: &str) -> Store {
    StoreBuilder::new("memory")
        .namespace("catgraph_test")
        .database(database)
        .connect()
        .await
        .expect("connecting to the in-memory engine")
}

async fn bootstrapped(database: &str) -> (Store, TermStore<Gen>) {
    let store = connect(database).await;
    let terms = TermStore::open(store.clone())
        .await
        .expect("opening the term store bootstraps and verifies");
    (store, terms)
}

// --------------------------------------------------------------- happy paths

/// The contract in one test: what goes in comes back out, revalidated.
#[tokio::test]
async fn a_stored_term_comes_back_equal() {
    let (_store, terms) = bootstrapped("round_trip").await;
    let term = copy_then_add();

    let addr = terms.put(&term).await.expect("storing a well-formed term");
    let loaded = terms
        .get(&addr)
        .await
        .expect("loading a term the store just wrote")
        .expect("the term is present");

    assert_eq!(loaded, term);
}

/// Content addressing makes writing idempotent. Re-storing a term must be a
/// success and must not create a second row — the whole point of an address that
/// is derived from the content rather than allocated.
#[tokio::test]
async fn re_storing_a_term_is_a_no_op() {
    let (store, terms) = bootstrapped("idempotent").await;
    let term = copy_then_add();

    let first = terms.put(&term).await.expect("first write");
    let second = terms
        .put(&term)
        .await
        .expect("second write of the same term");
    assert_eq!(first, second);

    let mut response = store
        .client()
        .query("SELECT VALUE id FROM term")
        .await
        .expect("counting the stored rows");
    let ids: Vec<RecordId> = response.take(0).expect("reading the ids");
    assert_eq!(ids.len(), 1, "the same term must occupy one row, not two");
}

/// The determinism the store depends on, observed end to end: two independently
/// constructed values of the same term resolve to one address and one row.
#[tokio::test]
async fn independently_built_equal_terms_share_one_record() {
    let (store, terms) = bootstrapped("determinism").await;

    let first = terms.put(&copy_then_add()).await.expect("first write");
    let second = terms
        .put(&copy_then_add())
        .await
        .expect("write of a separately built but equal term");
    assert_eq!(first, second);

    let mut response = store
        .client()
        .query("SELECT VALUE id FROM term")
        .await
        .expect("counting the stored rows");
    let ids: Vec<RecordId> = response.take(0).expect("reading the ids");
    assert_eq!(ids.len(), 1);
}

#[tokio::test]
async fn an_absent_term_reads_as_none_rather_than_an_error() {
    let (_store, terms) = bootstrapped("absent").await;
    let addr = TermAddr::from_digest(&"a".repeat(64)).expect("64 lowercase hex chars");
    assert_eq!(
        terms.get(&addr).await.expect("querying for an absent term"),
        None
    );
    assert!(!terms.contains(&addr).await.expect("existence check"));
}

#[tokio::test]
async fn existence_tracks_what_was_written() {
    let (_store, terms) = bootstrapped("contains").await;
    let addr = terms.put(&copy_then_add()).await.expect("storing a term");
    assert!(terms.contains(&addr).await.expect("existence check"));
}

#[tokio::test]
async fn a_batch_stores_every_term_and_returns_addresses_in_order() {
    let (_store, terms) = bootstrapped("batch").await;
    let batch = [copy_then_add(), zero_then_discard(), identity()];

    let addrs = terms.put_many(&batch).await.expect("storing a batch");
    assert_eq!(addrs.len(), batch.len());

    for (addr, term) in addrs.iter().zip(batch.iter()) {
        let loaded = terms
            .get(addr)
            .await
            .expect("loading a batched term")
            .expect("the term is present");
        assert_eq!(&loaded, term);
    }
}

#[tokio::test]
async fn an_empty_batch_touches_nothing() {
    let (_store, terms) = bootstrapped("empty_batch").await;
    let addrs = terms.put_many(&[]).await.expect("an empty batch succeeds");
    assert!(addrs.is_empty());
}

/// A batch containing the same term twice is still idempotent — the second write
/// changes no value, so the read-only columns raise nothing.
#[tokio::test]
async fn a_batch_may_repeat_a_term() {
    let (_store, terms) = bootstrapped("batch_repeat").await;
    let batch = [copy_then_add(), copy_then_add()];
    let addrs = terms.put_many(&batch).await.expect("storing a batch");
    assert_eq!(addrs[0], addrs[1]);
}

// ------------------------------------------------------------ schema drift

/// Bootstrapping is idempotent, which is what lets it run on every open. The
/// interesting half is the second call: `DEFINE TABLE` drops a table's fields, so
/// a bootstrap that re-defined the table would leave the schema empty.
#[tokio::test]
async fn bootstrapping_twice_leaves_the_schema_intact() {
    let (_store, terms) = bootstrapped("bootstrap_twice").await;
    terms.bootstrap().await.expect("second bootstrap");
    terms
        .assert_schema()
        .await
        .expect("the schema still matches");

    // And the table is still writable, which is the property a dropped field set
    // would silently destroy.
    let addr = terms
        .put(&copy_then_add())
        .await
        .expect("storing after re-bootstrap");
    assert!(terms.contains(&addr).await.expect("existence check"));
}

#[tokio::test]
async fn the_bootstrapped_schema_declares_exactly_the_expected_columns() {
    let (store, terms) = bootstrapped("schema_columns").await;
    terms.assert_schema().await.expect("the schema matches");

    let mut response = store
        .client()
        .query("RETURN object::keys((INFO FOR TABLE term).fields)")
        .await
        .expect("reading the live column names");
    let mut live: Vec<String> = response.take(0).expect("reading the key set");
    live.sort();

    let mut expected: Vec<String> = schema::TERM_FIELDS
        .iter()
        .map(|f| (*f).to_owned())
        .collect();
    expected.sort();
    assert_eq!(live, expected);
}

/// The drift guard's reason for existing. Bootstrapping would quietly repair a
/// dropped column — every statement is `IF NOT EXISTS` — so the guard has to be
/// asked, and it has to notice.
#[tokio::test]
async fn a_dropped_column_is_detected() {
    let (store, terms) = bootstrapped("dropped_column").await;
    store
        .client()
        .query("REMOVE FIELD nf_class ON term")
        .await
        .expect("removing a column")
        .check()
        .expect("the removal succeeds");

    let err = terms
        .assert_schema()
        .await
        .expect_err("a dropped column is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("nf_class"), "{detail}");
    assert!(!err.is_conflict());
    assert!(!err.is_shutdown());
}

#[tokio::test]
async fn a_dropped_index_is_detected() {
    let (store, terms) = bootstrapped("dropped_index").await;
    store
        .client()
        .query("REMOVE INDEX term_nf_class ON term")
        .await
        .expect("removing an index")
        .check()
        .expect("the removal succeeds");

    let err = terms
        .assert_schema()
        .await
        .expect_err("a dropped index is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("term_nf_class"), "{detail}");
}

/// A column this build does not know means the database was written by a
/// different version of the store. Reading it under the older shape is how a
/// newer document silently loses a field.
#[tokio::test]
async fn an_unexpected_column_is_detected() {
    let (store, terms) = bootstrapped("extra_column").await;
    store
        .client()
        .query("DEFINE FIELD provenance ON term TYPE string")
        .await
        .expect("adding a column")
        .check()
        .expect("the definition succeeds");

    let err = terms
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
/// on purpose: a `TermStore` value cannot exist without a verified schema.
#[tokio::test]
async fn an_undefined_table_is_detected_as_drift() {
    let store = connect("no_bootstrap").await;
    let err = schema::assert_term_schema(store.client())
        .await
        .expect_err("nothing has been defined");
    assert!(matches!(err, StoreError::Schema { .. }), "{err:?}");
}

/// The auto-create hole, closed and pinned. A raw write against an undefined
/// table makes the engine create it `TYPE ANY SCHEMALESS`; a later bootstrap's
/// `DEFINE TABLE IF NOT EXISTS` then blesses the impostor and every field gets
/// defined on top of it. The definition-level guard is what refuses to serve
/// it — and therefore [`TermStore::open`] must fail on such a database.
#[tokio::test]
async fn an_implicitly_created_table_is_refused_at_open() {
    let store = connect("implicit_table").await;
    store
        .client()
        .query("CREATE term:sneak SET smuggled = true")
        .await
        .expect("the raw write runs")
        .check()
        .expect("the engine auto-creates the table");

    let err = TermStore::<Gen>::open(store)
        .await
        .expect_err("an implicitly created table must not verify");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("SCHEMALESS"), "{detail}");
}

/// The guard compares definitions, not names: `ALTER TABLE … SCHEMALESS` keeps
/// all nine fields defined and every name in place while disarming the schema
/// entirely. A name-level check passed this; the definition check must not.
#[tokio::test]
async fn a_schemaless_alteration_is_detected_as_drift() {
    let (store, terms) = bootstrapped("altered_schemaless").await;
    store
        .client()
        .query("ALTER TABLE term SCHEMALESS")
        .await
        .expect("the alteration runs")
        .check()
        .expect("the alteration succeeds");

    let err = terms
        .assert_schema()
        .await
        .expect_err("a schemaless term table is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("SCHEMALESS"), "{detail}");
}

/// Same class of blindness, field-level: re-defining a column without
/// `READONLY` keeps its name while disarming the collision alarm.
#[tokio::test]
async fn a_disarmed_readonly_clause_is_detected_as_drift() {
    let (store, terms) = bootstrapped("disarmed_readonly").await;
    store
        .client()
        .query("DEFINE FIELD OVERWRITE term_json ON term TYPE string")
        .await
        .expect("the redefinition runs")
        .check()
        .expect("the redefinition succeeds");

    let err = terms
        .assert_schema()
        .await
        .expect_err("a field without READONLY is drift");
    let StoreError::Schema { detail, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert!(detail.contains("term_json"), "{detail}");
}

/// A table that vanishes after open must make a WRITE loud, not quiet: a bare
/// write against an undefined table does not fail — the engine auto-creates it
/// `TYPE ANY SCHEMALESS`, permanently disarming every database-side guard while
/// every documented signal stays green. The write guard turns that into
/// [`StoreError::Schema`], and the table stays undefined.
#[tokio::test]
async fn a_write_after_the_table_vanishes_is_refused_loudly() {
    let (store, terms) = bootstrapped("write_after_drop").await;
    store
        .client()
        .query("REMOVE TABLE term")
        .await
        .expect("the removal runs")
        .check()
        .expect("the removal succeeds");

    let err = terms
        .put(&copy_then_add())
        .await
        .expect_err("a write must not silently re-create the table");
    assert!(matches!(err, StoreError::Schema { .. }), "{err:?}");

    // And the guard aborted before the write could auto-create anything.
    let mut response = store
        .client()
        .query("RETURN (INFO FOR DB).tables.term")
        .await
        .expect("reading the table definition");
    let definition: Option<String> = response.take(0).expect("the definition slot");
    assert_eq!(definition, None, "the table must remain undefined");
}

/// A table that vanishes after open answers "absent", identically from both
/// read methods — the same condition must not read as `false` from one and as
/// an opaque query error from the other.
#[tokio::test]
async fn a_vanished_table_reads_as_absent_from_both_read_methods() {
    let (store, terms) = bootstrapped("vanished_table").await;
    let addr = terms.put(&copy_then_add()).await.expect("storing a term");

    store
        .client()
        .query("REMOVE TABLE term")
        .await
        .expect("the removal runs")
        .check()
        .expect("the removal succeeds");

    assert_eq!(
        terms
            .get(&addr)
            .await
            .expect("get absorbs the vanished table"),
        None
    );
    assert!(
        !terms
            .contains(&addr)
            .await
            .expect("contains absorbs the vanished table")
    );
}

// ------------------------------------------------------- corrupt documents

/// A term row, written by hand.
///
/// Mirrors the store's own row so a test can produce a document the store never
/// would — the whole corrupt-document suite is one field's difference from a
/// good record.
#[derive(Debug, Clone, SurrealValue)]
struct RawRow {
    id: RecordId,
    codec: String,
    term_json: String,
    signature: String,
    source_arity: i64,
    target_arity: i64,
    depth: i64,
    generator_count: i64,
    nf_class: String,
}

/// A faithful row for `Δ ; μ`, ready to be corrupted.
fn good_row() -> RawRow {
    let record = term::encode(&copy_then_add()).expect("a well-formed term encodes");
    RawRow {
        id: RecordId::new(schema::TERM_TABLE, record.addr().as_str()),
        codec: record.codec().to_owned(),
        term_json: record.term_json().to_owned(),
        signature: record.signature().to_owned(),
        source_arity: record.source_arity(),
        target_arity: record.target_arity(),
        depth: record.depth(),
        generator_count: record.generator_count(),
        nf_class: record.nf_class().to_owned(),
    }
}

/// Re-file a (possibly tampered) row under its own encoding's digest.
///
/// Revalidation verifies the content address *first* — cheapest check, fully
/// decides byte-tampering — so a fixture that swaps `term_json` while keeping
/// the old id never reaches the stage it means to exercise. Re-filing makes the
/// address check pass so the later stage fires; the un-re-filed case has its
/// own test (`a_row_filed_under_the_wrong_address_is_rejected`).
fn refile(mut row: RawRow) -> RawRow {
    let digest = blake3::hash(row.term_json.as_bytes()).to_hex().to_string();
    row.id = RecordId::new(schema::TERM_TABLE, format!("b3_{digest}"));
    row
}

/// Write a row straight through the connection, bypassing the store.
async fn write_raw(store: &Store, row: &RawRow) -> TermAddr {
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
    TermAddr::parse(key).expect("the fixture always uses a well-formed address")
}

/// Load a hand-written row and return whatever the store makes of it.
async fn load_raw(database: &str, row: RawRow) -> Result<Option<ColoredExpr<Gen>>, StoreError> {
    let (store, terms) = bootstrapped(database).await;
    let addr = write_raw(&store, &row).await;
    terms.get(&addr).await
}

fn expect_revalidation(
    result: Result<Option<ColoredExpr<Gen>>, StoreError>,
    stage: RevalidationStage,
) {
    match result {
        Err(StoreError::Revalidation { stage: actual, .. }) => assert_eq!(actual, stage),
        other => panic!("expected a {stage}-stage revalidation failure, got {other:?}"),
    }
}

#[tokio::test]
async fn a_row_whose_depth_column_lies_is_rejected() {
    let mut row = good_row();
    row.depth = 99;
    expect_revalidation(
        load_raw("corrupt_depth", row).await,
        RevalidationStage::Depth,
    );
}

#[tokio::test]
async fn a_row_whose_arity_columns_lie_is_rejected() {
    let mut row = good_row();
    row.source_arity = 7;
    expect_revalidation(
        load_raw("corrupt_arity", row).await,
        RevalidationStage::Arity,
    );

    let mut row = good_row();
    row.target_arity = 0;
    expect_revalidation(
        load_raw("corrupt_arity_target", row).await,
        RevalidationStage::Arity,
    );
}

#[tokio::test]
async fn a_row_whose_signature_does_not_match_the_encoding_is_rejected() {
    let mut row = good_row();
    row.signature = "[[null],[null,null]]".to_owned();
    expect_revalidation(
        load_raw("corrupt_signature", row).await,
        RevalidationStage::Check,
    );
}

#[tokio::test]
async fn a_row_whose_generator_count_lies_is_rejected() {
    let mut row = good_row();
    row.generator_count = 5;
    expect_revalidation(
        load_raw("corrupt_generator_count", row).await,
        RevalidationStage::Check,
    );
}

#[tokio::test]
async fn a_row_whose_bucket_does_not_match_the_encoding_is_rejected() {
    let mut row = good_row();
    row.nf_class = "0".repeat(64);
    expect_revalidation(
        load_raw("corrupt_nf_class", row).await,
        RevalidationStage::Check,
    );
}

#[tokio::test]
async fn a_row_whose_encoding_is_not_json_is_rejected() {
    let mut row = good_row();
    row.term_json = "{not json".to_owned();
    let row = refile(row);
    match load_raw("corrupt_malformed", row).await {
        Err(StoreError::Codec(_)) => {}
        other => panic!("expected a codec failure, got {other:?}"),
    }
}

/// The parser's container limit, met head on. It fires before any of the checks
/// below it, which is exactly why it is step one.
#[tokio::test]
async fn a_row_whose_encoding_nests_past_the_parser_limit_is_rejected() {
    let mut row = good_row();
    row.term_json = format!("{}{}", "[".repeat(400), "]".repeat(400));
    let row = refile(row);
    match load_raw("corrupt_over_deep", row).await {
        Err(StoreError::Codec(_)) => {}
        other => panic!("expected a codec failure, got {other:?}"),
    }
}

/// A row parked under an address that is not its own encoding's digest. The id
/// `ASSERT` cannot catch this — it only checks the *format* of the key, and it is
/// skipped on update and under `OPTION IMPORT` besides.
#[tokio::test]
async fn a_row_filed_under_the_wrong_address_is_rejected() {
    let mut row = good_row();
    row.id = RecordId::new(schema::TERM_TABLE, format!("b3_{}", "c".repeat(64)));
    match load_raw("corrupt_address", row).await {
        Err(StoreError::Corrupt { .. }) => {}
        other => panic!("expected a corrupt-document failure, got {other:?}"),
    }
}

/// An encoding that parses but describes an ill-composed expression. `Δ ; !` is
/// `1 → 2` followed by `1 → 0`: the composition does not join.
///
/// This is the screen the fourth step depends on. Without it the content pass
/// below would be handed arities it cannot answer for and would abort rather
/// than return.
#[tokio::test]
async fn a_row_encoding_an_ill_composed_term_is_rejected() {
    let mut row = good_row();
    row.term_json = serde_json::json!({
        "source_word": [null],
        "target_word": [],
        "expr": { "Compose": [ { "Generator": "Copy" }, { "Generator": "Discard" } ] }
    })
    .to_string();
    let row = refile(row);
    expect_revalidation(
        load_raw("corrupt_ill_composed", row).await,
        RevalidationStage::Arity,
    );
}

/// An encoding that parses, composes, and is the declared depth — but whose
/// declared source word does not fit the expression it is attached to. Nothing
/// before the fourth step can see this: only re-running the type check does.
#[tokio::test]
async fn a_row_whose_source_word_does_not_fit_its_expression_is_rejected() {
    let mut row = good_row();
    row.term_json = serde_json::json!({
        "source_word": [],
        "target_word": [],
        "expr": { "Identity": 1 }
    })
    .to_string();
    row.depth = 1;
    let row = refile(row);
    expect_revalidation(
        load_raw("corrupt_word_fit", row).await,
        RevalidationStage::Check,
    );
}

/// A codec version this build does not know is refused rather than read under
/// the current rules.
#[tokio::test]
async fn a_row_written_under_an_unknown_codec_is_rejected() {
    let mut row = good_row();
    row.codec = "cgj99".to_owned();
    match load_raw("corrupt_codec", row).await {
        Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "codec"),
        other => panic!("expected a type mismatch on the codec column, got {other:?}"),
    }
}

// ------------------------------------------------- database-side defences

/// Defence in depth, pinned so it is noticed if it stops working: the id
/// `ASSERT` rejects a key that is not shaped like a content address. This is not
/// the primary guard — it is skipped on update and under `OPTION IMPORT` — but
/// it is the one that stops a foreign row from entering through a raw write.
#[tokio::test]
async fn the_database_refuses_an_id_that_is_not_a_content_address() {
    let (store, _terms) = bootstrapped("bad_id").await;
    let mut row = good_row();
    row.id = RecordId::new(schema::TERM_TABLE, "not-an-address");

    let outcome = store
        .client()
        .query("CREATE $row.id CONTENT $row RETURN NONE")
        .bind(("row", row))
        .await
        .expect("the statement runs")
        .check();
    assert!(outcome.is_err(), "the id ASSERT must reject this key");
}

/// Two distinct encodings resolving to one address would be a hash collision,
/// and the read-only columns are what turn that from a silent overwrite into an
/// error. Simulated by writing a *different* payload under an address that is
/// already taken.
#[tokio::test]
async fn a_changed_payload_under_an_existing_address_is_refused() {
    let (store, terms) = bootstrapped("collision").await;
    let addr = terms.put(&copy_then_add()).await.expect("storing a term");

    let mut row = good_row();
    row.id = RecordId::new(schema::TERM_TABLE, addr.as_str());
    row.term_json = "{}".to_owned();

    let outcome = store
        .client()
        .query("UPSERT $row.id CONTENT $row RETURN NONE")
        .bind(("row", row))
        .await
        .expect("the statement runs")
        .check();
    assert!(
        outcome.is_err(),
        "a read-only column must refuse a changed value"
    );
}

// ------------------------------------------------------------ bucket queries

/// The reason `nf_class` is a column at all: it is queryable, and it is a
/// *bucket*, so two distinct writings of one morphism are two rows sharing one
/// class rather than one row.
#[tokio::test]
async fn distinct_writings_of_one_morphism_share_a_bucket() {
    let (store, terms) = bootstrapped("buckets").await;

    let bare: PropExpr<Gen> = Free::tensor(
        Free::generator(Gen::Discard),
        Free::<Gen>::generator(Gen::Discard),
    );
    let staged = Free::compose(
        Free::tensor(Free::generator(Gen::Discard), Free::<Gen>::identity(1)),
        Free::generator(Gen::Discard),
    )
    .expect("(! ⊗ id) ; ! composes");

    let bare = ColoredExpr::new(vec![(), ()], bare).expect("! ⊗ ! type-checks at two wires");
    let staged =
        ColoredExpr::new(vec![(), ()], staged).expect("(! ⊗ id) ; ! type-checks at two wires");

    let first = terms.put(&bare).await.expect("storing the tensor writing");
    let second = terms
        .put(&staged)
        .await
        .expect("storing the staged writing");
    assert_ne!(first, second, "distinct syntax is distinct identity");

    let class = term::nf_class(&bare).expect("the term has a class");
    let mut response = store
        .client()
        .query("SELECT VALUE id FROM term WHERE nf_class = $class")
        .bind(("class", class))
        .await
        .expect("querying the bucket");
    let ids: Vec<RecordId> = response.take(0).expect("reading the ids");
    assert_eq!(ids.len(), 2, "both writings belong to one bucket");
}
