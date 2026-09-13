//! The export/import round-trip gate.
//!
//! These need a real engine compiled in, so the whole file is gated on `mem`.
//!
//! # What a checkpoint has to survive
//!
//! Every tier is written to a fresh store, exported, and restored into an empty
//! one — and then read back through the same revalidating APIs a consumer uses.
//! That last part is the gate: an export that produced a file and an import that
//! consumed it prove nothing if what comes out no longer validates.
//!
//! Three things get particular attention:
//!
//! - **Weights, bit for bit.** The byte lane exists because the native float
//!   column loses `NaN` payloads and signed zeros *on the export path* — every
//!   `NaN` renders as the bare literal `NaN`, sign and payload gone. The
//!   assertion here is on bit patterns, over a vector chosen to expose exactly
//!   that.
//! - **Content addresses, re-verified.** Every database-side guard is disabled
//!   under `OPTION IMPORT` — the id-format `ASSERT` included — so a restored
//!   dump can carry ids nothing checked. Verification after the restore is what
//!   makes a checkpoint trustworthy, and it is done through the tiers' own
//!   address helpers rather than by trusting the replay.
//! - **The include list.** Tables are selected explicitly rather than by
//!   exclusion, which doubles as a schema-drift guard: a tier added without
//!   being added here fails this test rather than quietly falling out of every
//!   checkpoint.
//!
//! # What a restore does *not* do
//!
//! An import emits **no change-feed entries and no live notifications** — the
//! whole notification path is suppressed for it. A bus consumer's cursor
//! therefore describes a feed that no longer corresponds to the table, and the
//! only correct thing to do afterwards is re-baseline from the durable rows.
//! That is asserted here too, because "the events will show up" is the
//! assumption a restore quietly invalidates.

#![cfg(feature = "mem")]

use std::borrow::Cow;

use catgraph::cospan::Cospan;
use catgraph_applied::prop::colored::ColoredExpr;
use catgraph_applied::prop::presentation::rewrite::{RewriteRule, optimize};
use catgraph_applied::prop::{Free, PropSignature};
use catgraph_dl::para::RModule;
use catgraph_surreal::{
    BusReader, BusWriter, CospanStore, DocStore, LineageStore, ManifestStore, Store, StoreBuilder,
    TermStore, WeightStore, cospan, schema, term, weight,
};
use serde::{Deserialize, Serialize};
use surrealdb::types::RecordId;

/// Every table a checkpoint carries, named explicitly.
///
/// An **include** list rather than an exclusion list, deliberately. Unknown
/// table names in an exclusion list are only warned about, so a typo silently
/// exports everything; and a rollup view in a full dump fails on replay, because
/// a replayed `DEFINE TABLE … AS SELECT` materialises immediately. Naming the
/// source tables sidesteps both.
const CHECKPOINT_TABLES: [&str; 10] = [
    schema::TERM_TABLE,
    schema::COSPAN_TABLE,
    schema::WEIGHT_TABLE,
    schema::RULE_SET_TABLE,
    schema::REWRITE_RUN_TABLE,
    schema::DERIVES_TABLE,
    schema::DOCUMENT_TABLE,
    schema::MANIFEST_TABLE,
    schema::BUS_TABLE,
    schema::BUS_SEQ_TABLE,
];

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

fn copy_then_add() -> ColoredExpr<Gen> {
    let expr = Free::compose(Free::generator(Gen::Copy), Free::<Gen>::generator(Gen::Add))
        .expect("Δ ; μ composes");
    ColoredExpr::new(vec![()], expr).expect("Δ ; μ type-checks")
}

fn identity() -> ColoredExpr<Gen> {
    ColoredExpr::new(vec![()], Free::<Gen>::identity(1)).expect("id₁ type-checks")
}

/// Coordinates chosen so a lossy lane is caught: a `NaN` carrying a payload,
/// both zeros, both infinities, and one ordinary value.
fn awkward_weights() -> RModule<f64> {
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Snapshot {
    beliefs: Vec<f64>,
    boundary: usize,
}

async fn connect(database: &str) -> Store {
    StoreBuilder::new("memory")
        .namespace("catgraph_test")
        .database(database)
        .require_backup()
        .connect()
        .await
        .expect("connecting to the in-memory engine")
}

/// Everything one checkpoint holds, so the assertions after the restore have
/// something to compare against.
struct Fixture {
    cospan: Cospan<usize>,
    weights: RModule<f64>,
    snapshot: Snapshot,
    manifest: serde_json::Value,
}

fn fixture() -> Fixture {
    Fixture {
        // A μ-shape with a scalar: two domain wires onto one apex vertex, one
        // codomain wire, one untouched vertex.
        cospan: Cospan::new(vec![0, 0], vec![0], vec![3usize, 9]).expect("μ's legs are in bounds"),
        weights: awkward_weights(),
        snapshot: Snapshot {
            beliefs: vec![0.25, 0.5, 0.25],
            boundary: 7,
        },
        manifest: serde_json::json!({ "bar": 0.05, "declared": "before the run" }),
    }
}

/// Write one of everything.
async fn populate(store: &Store, fixture: &Fixture) {
    let terms = TermStore::<Gen>::open(store.clone())
        .await
        .expect("opening the term store");
    terms.put(&copy_then_add()).await.expect("storing a term");

    let cospans = CospanStore::<usize>::open(store.clone())
        .await
        .expect("opening the cospan store");
    cospans
        .put(&fixture.cospan)
        .await
        .expect("storing a cospan");

    let weights = WeightStore::open(store.clone())
        .await
        .expect("opening the weight store");
    weights
        .put("genome-a", "layer-0", &fixture.weights)
        .await
        .expect("storing weights");

    let lineage = LineageStore::<Gen>::open(store.clone())
        .await
        .expect("opening the lineage store");
    let rule_set = lineage
        .put_rule_set(&[(copy_then_add(), identity())])
        .await
        .expect("storing a rule set");
    let compiled: Vec<RewriteRule<Gen>> = lineage
        .get_rule_set(&rule_set)
        .await
        .expect("loading it back")
        .expect("present");
    let start = copy_then_add();
    let outcome = optimize(&start, &compiled, 16, |_| 1).expect("the search runs");
    lineage
        .record_run(&rule_set, &start, &outcome, "unit")
        .await
        .expect("recording a run");

    let docs = DocStore::<Snapshot>::open(store.clone())
        .await
        .expect("opening the document store");
    docs.put("snap-1", "solver", &fixture.snapshot)
        .await
        .expect("storing a document");

    let manifests = ManifestStore::<serde_json::Value>::open(store.clone())
        .await
        .expect("opening the manifest store");
    manifests
        .register("run-1", "prereg", &fixture.manifest)
        .await
        .expect("registering a manifest");

    let bus = BusWriter::open(store.clone())
        .await
        .expect("opening the bus");
    for generation in 0..3u32 {
        bus.publish("goals", &serde_json::json!({ "generation": generation }))
            .await
            .expect("publishing");
    }
}

/// Define every table on a fresh store, so the restore replays into a schema
/// rather than creating one.
///
/// Not strictly required — a dump carries its own `DEFINE`s — but it is what a
/// real restore does, and it is what makes the post-restore schema assertions
/// mean something.
async fn bootstrap_all(store: &Store) {
    TermStore::<Gen>::open(store.clone()).await.expect("terms");
    CospanStore::<usize>::open(store.clone())
        .await
        .expect("cospans");
    WeightStore::open(store.clone()).await.expect("weights");
    LineageStore::<Gen>::open(store.clone())
        .await
        .expect("lineage");
    DocStore::<Snapshot>::open(store.clone())
        .await
        .expect("documents");
    ManifestStore::<serde_json::Value>::open(store.clone())
        .await
        .expect("manifests");
    BusWriter::open(store.clone()).await.expect("bus");
}

/// **The gate.** Every tier survives a checkpoint and a restore, and everything
/// that comes back still validates.
#[tokio::test]
async fn every_tier_survives_a_checkpoint_and_a_restore() {
    let fixture = fixture();
    let source = connect("roundtrip_source").await;
    populate(&source, &fixture).await;

    let file = tempfile::NamedTempFile::new().expect("a temporary file");
    source
        .client()
        .export(file.path())
        .with_config()
        // The explicit include list — see `CHECKPOINT_TABLES`.
        .tables(CHECKPOINT_TABLES.to_vec())
        .await
        .expect("exporting the checkpoint");

    let restored = connect("roundtrip_restored").await;
    bootstrap_all(&restored).await;
    restored
        .client()
        .import(file.path())
        .await
        .expect("importing the checkpoint");

    // --- terms: the content address is the identity, and it revalidates.
    let terms = TermStore::<Gen>::open(restored.clone())
        .await
        .expect("opening the term store");
    let addr = term::encode(&copy_then_add())
        .expect("encoding")
        .addr()
        .clone();
    let loaded = terms
        .get(&addr)
        .await
        .expect("loading a restored term")
        .expect("it is present");
    assert_eq!(loaded, copy_then_add());

    // --- cospans: the presentation and the canonical key both survive.
    let cospans = CospanStore::<usize>::open(restored.clone())
        .await
        .expect("opening the cospan store");
    let cospan_addr = cospan::encode(&fixture.cospan)
        .expect("encoding")
        .addr()
        .clone();
    let loaded = cospans
        .get(&cospan_addr)
        .await
        .expect("loading a restored cospan")
        .expect("it is present");
    assert_eq!(loaded.left_to_middle(), fixture.cospan.left_to_middle());
    assert_eq!(loaded.right_to_middle(), fixture.cospan.right_to_middle());
    assert_eq!(loaded.middle(), fixture.cospan.middle());
    assert_eq!(
        cospans
            .find_by_canon(&fixture.cospan)
            .await
            .expect("looking up by canonical key"),
        Some(cospan_addr)
    );

    // --- weights: bit for bit, which is the whole reason for the byte lane.
    let weights = WeightStore::open(restored.clone())
        .await
        .expect("opening the weight store");
    let loaded = weights
        .get("genome-a", "layer-0")
        .await
        .expect("loading restored weights")
        .expect("they are present");
    assert_eq!(bits(&loaded), bits(&fixture.weights));
    // Spelled out, because `==` cannot see either of these differences and the
    // export path is exactly where a native float column loses them.
    assert_eq!(loaded.as_slice()[0].to_bits(), 0x7FF8_0000_DEAD_BEEF);
    assert_eq!(loaded.as_slice()[1].to_bits(), (-0.0f64).to_bits());
    assert_ne!(
        loaded.as_slice()[1].to_bits(),
        loaded.as_slice()[2].to_bits(),
        "the two zeros must not have merged"
    );
    assert_eq!(loaded.as_slice()[3], f64::INFINITY);
    assert_eq!(loaded.as_slice()[4], f64::NEG_INFINITY);

    // --- lineage: the rule set recompiles, the trace reads, the edge is there.
    let lineage = LineageStore::<Gen>::open(restored.clone())
        .await
        .expect("opening the lineage store");
    let rule_set = catgraph_surreal::lineage::encode_rule_set(&[(copy_then_add(), identity())])
        .expect("encoding")
        .addr()
        .clone();
    let compiled: Vec<RewriteRule<Gen>> = lineage
        .get_rule_set(&rule_set)
        .await
        .expect("loading a restored rule set")
        .expect("it is present");
    assert_eq!(compiled.len(), 1);

    let outcome = optimize(&copy_then_add(), &compiled, 16, |_| 1).expect("the search runs");
    let run = catgraph_surreal::lineage::encode_run(&rule_set, &copy_then_add(), &outcome, "unit")
        .expect("encoding")
        .addr()
        .clone();
    let stored = lineage
        .get_run(&run)
        .await
        .expect("loading a restored run")
        .expect("it is present");
    assert_eq!(stored.cost_model(), "unit");
    assert!(!stored.steps().is_empty());
    let edges = lineage
        .edges_of_run(&run)
        .await
        .expect("reading the restored edges");
    assert_eq!(edges.len(), 1, "the relation edge survived the round trip");
    assert_eq!(edges[0].parent(), stored.start());
    assert_eq!(edges[0].child(), stored.best());

    // --- documents and manifests: payload keys and digests intact.
    let docs = DocStore::<Snapshot>::open(restored.clone())
        .await
        .expect("opening the document store");
    assert_eq!(
        docs.get("snap-1")
            .await
            .expect("loading a restored document")
            .expect("it is present"),
        fixture.snapshot
    );

    let manifests = ManifestStore::<serde_json::Value>::open(restored.clone())
        .await
        .expect("opening the manifest store");
    assert_eq!(
        manifests
            .get("run-1")
            .await
            .expect("loading a restored manifest")
            .expect("it is present"),
        fixture.manifest
    );

    // --- bus: the durable rows are all there, in order.
    let reader = BusReader::open(restored.clone(), "auditor")
        .await
        .expect("opening a reader");
    let replayed = reader.replay("goals", 0).await.expect("replaying");
    assert_eq!(
        replayed
            .iter()
            .map(|event| event.seq())
            .collect::<Vec<i64>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        replayed[1].decode::<serde_json::Value>().expect("decoding"),
        serde_json::json!({ "generation": 1 })
    );
}

/// **Post-restore re-verification.** The id-format `ASSERT` is skipped on update
/// *and* under `OPTION IMPORT`, so a restored dump can carry record ids nothing
/// ever checked. This walks the restored content-addressed tiers and re-derives
/// every id from the row it is filed under, which is what the guard would have
/// done had it run.
#[tokio::test]
async fn restored_content_addresses_are_re_verified_rather_than_trusted() {
    let fixture = fixture();
    let source = connect("roundtrip_verify_source").await;
    populate(&source, &fixture).await;

    let file = tempfile::NamedTempFile::new().expect("a temporary file");
    source
        .client()
        .export(file.path())
        .with_config()
        .tables(CHECKPOINT_TABLES.to_vec())
        .await
        .expect("exporting");

    let restored = connect("roundtrip_verify_restored").await;
    bootstrap_all(&restored).await;
    restored
        .client()
        .import(file.path())
        .await
        .expect("importing");

    // Terms: every stored row's id must be the digest of its own encoding, which
    // is exactly what `get` re-derives — so asking for each row by the address
    // the *table* claims and getting it back is the verification.
    let terms = TermStore::<Gen>::open(restored.clone())
        .await
        .expect("opening the term store");
    for addr in ids(&restored, schema::TERM_TABLE).await {
        let addr = catgraph_surreal::TermAddr::parse(&addr)
            .expect("a restored term id has the address shape");
        terms
            .get(&addr)
            .await
            .unwrap_or_else(|e| panic!("restored term `{addr}` does not verify: {e}"))
            .unwrap_or_else(|| panic!("restored term `{addr}` vanished"));
    }

    // Cospans: same, and the canonical key is re-derived alongside the address.
    let cospans = CospanStore::<usize>::open(restored.clone())
        .await
        .expect("opening the cospan store");
    for addr in ids(&restored, schema::COSPAN_TABLE).await {
        let addr = catgraph_surreal::CospanAddr::parse(&addr)
            .expect("a restored cospan id has the address shape");
        cospans
            .get(&addr)
            .await
            .unwrap_or_else(|e| panic!("restored cospan `{addr}` does not verify: {e}"))
            .unwrap_or_else(|| panic!("restored cospan `{addr}` vanished"));
    }

    // Weights: the key is re-derived from the pair the row itself carries.
    let weights = WeightStore::open(restored.clone())
        .await
        .expect("opening the weight store");
    weights
        .get("genome-a", "layer-0")
        .await
        .expect("restored weights verify")
        .expect("they are present");
    assert!(
        ids(&restored, schema::WEIGHT_TABLE)
            .await
            .contains(&weight::record_key("genome-a", "layer-0").expect("a pair has a key")),
        "the restored weight row is filed under the key its own pair derives"
    );

    // Manifests: the digest is the only thing an import cannot forge unnoticed.
    let manifests = ManifestStore::<serde_json::Value>::open(restored.clone())
        .await
        .expect("opening the manifest store");
    assert!(
        manifests.verify_all().await.expect("verifying").is_empty(),
        "every restored manifest agrees with its own digest"
    );
}

/// The check that keeps the include list honest: a tier added without being
/// added to `CHECKPOINT_TABLES` would fall out of every checkpoint silently.
#[tokio::test]
async fn the_include_list_covers_every_populated_table() {
    let fixture = fixture();
    let source = connect("roundtrip_coverage").await;
    populate(&source, &fixture).await;

    let mut response = source
        .client()
        .query("RETURN (INFO FOR DB).tables")
        .await
        .expect("listing the tables");
    let tables: Option<std::collections::BTreeMap<String, String>> =
        response.take(0).expect("the listing reads back");
    let mut uncovered: Vec<String> = tables
        .unwrap_or_default()
        .into_keys()
        .filter(|table| !CHECKPOINT_TABLES.contains(&table.as_str()))
        .collect();
    uncovered.sort();

    // The consumer-cursor table is the one deliberate omission: a checkpoint
    // carries the events, not any particular consumer's position in them, and a
    // restored cursor would point into a change feed the restore did not
    // recreate.
    assert_eq!(
        uncovered,
        vec![schema::BUS_MARK_TABLE.to_owned()],
        "a table exists that no checkpoint would carry"
    );
}

/// **An import is invisible to the notification path.** Change-feed capture,
/// live notifications, and events are all suppressed for it — so a consumer
/// whose cursor predates a restore is describing a feed that no longer
/// corresponds to the table, and re-baselining is the only correct response.
#[tokio::test]
async fn a_restore_emits_no_change_feed_entries() {
    let fixture = fixture();
    let source = connect("roundtrip_feed_source").await;
    populate(&source, &fixture).await;

    let file = tempfile::NamedTempFile::new().expect("a temporary file");
    source
        .client()
        .export(file.path())
        .with_config()
        .tables(CHECKPOINT_TABLES.to_vec())
        .await
        .expect("exporting");

    let restored = connect("roundtrip_feed_restored").await;
    bootstrap_all(&restored).await;
    let mut reader = BusReader::open(restored.clone(), "worker-1")
        .await
        .expect("opening a reader before the restore");
    assert!(
        reader.next_batch().await.expect("catching up").is_empty(),
        "nothing has been published on this store yet"
    );

    restored
        .client()
        .import(file.path())
        .await
        .expect("importing");

    // Three events are now in the table, and none of them is in the feed.
    assert!(
        reader.next_batch().await.expect("catching up").is_empty(),
        "an import must emit no change-feed entries"
    );
    assert_eq!(
        reader.replay("goals", 0).await.expect("replaying").len(),
        3,
        "the durable rows are there regardless"
    );

    // Which is what re-baselining is for: it adopts the rows and reads on.
    reader.rebaseline().await.expect("re-baselining");
    let bus = BusWriter::open(restored.clone())
        .await
        .expect("opening the bus");
    bus.publish("goals", &serde_json::json!({ "generation": 3 }))
        .await
        .expect("publishing after the restore");
    let batch = reader.next_batch().await.expect("catching up");
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].seq(), 3, "the allocator survived the restore too");
}

/// Every record id in a table, as the strings they are stored under.
async fn ids(store: &Store, table: &str) -> Vec<String> {
    let mut response = store
        .client()
        .query(format!("SELECT VALUE id FROM {table}"))
        .await
        .expect("listing the ids");
    let ids: Vec<RecordId> = response.take(0).expect("the ids read back");
    ids.into_iter()
        .map(|id| match id.key {
            surrealdb::types::RecordIdKey::String(key) => key,
            other => panic!("a stored id is not a string key: {other:?}"),
        })
        .collect()
}
