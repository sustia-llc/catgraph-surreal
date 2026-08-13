//! The lineage repository: rule sets, optimizer traces, and the derivation
//! graph.
//!
//! # What is one transaction and what is not
//!
//! Recording a run writes two things that have to agree — the trace row and the
//! edge that puts its endpoints in the graph — so [`LineageStore::record_run`]
//! writes both inside **one** transaction. A trace with no edge is invisible to
//! a traversal; an edge with no trace points at a run nobody can read.
//!
//! The endpoint *terms* are written first, outside that transaction, and
//! deliberately: content-addressed inserts are idempotent single statements that
//! two writers can race harmlessly, and folding them into the contended
//! transaction would widen its write set for nothing. This is the shape the
//! store uses generally — batch the idempotent work outside, keep the
//! transaction to what actually has to be atomic.
//!
//! # Conflicts are a property of the transaction, not of a statement
//!
//! [`LineageStore::record_run`] can fail with a transaction conflict, and the
//! caller retries the **whole** call rather than anything inside it: RocksDB
//! detects conflicts at commit time, so a transaction whose statements all
//! succeeded can still fail at the end. [`retry`](mod@crate::retry) is the reference
//! implementation; the important part is that the unit is this method, and that
//! re-running it re-derives everything rather than reusing what a failed attempt
//! read.
//!
//! # Why the edge write is a content-addressed `RELATE OR UPDATE`
//!
//! The edge's record id is the digest of the tuple it represents — parent,
//! child, and the run that derived one from the other — so re-recording a run
//! targets the same row with the same values: idempotent by construction, with
//! no second uniqueness mechanism to keep in step.
//!
//! `OR UPDATE` is the explicit form of what the engine does anyway. A plain
//! `RELATE` onto an existing edge id does not fail: it logs a warning and
//! overwrites, with nothing surfacing to the client. `OR UPDATE` takes the same
//! path with the intent stated and the warning suppressed. Where create-only
//! semantics are wanted, a unique index is the enforcement — and here they are
//! not wanted, because writing the same derivation twice is a success.
//!
//! What still refuses a *changed* edge is the `READONLY` columns: a different
//! payload under the same tuple is a changed value, and the engine says so.

use std::marker::PhantomData;
use std::sync::LazyLock;

use catgraph_applied::prop::PropSignature;
use catgraph_applied::prop::colored::ColoredExpr;
use catgraph_applied::prop::presentation::rewrite::{RewriteOutcome, RewriteRule};
use serde::Serialize;
use serde::de::DeserializeOwned;
use surrealdb::Surreal;
use surrealdb::engine::any::Any;
use surrealdb::types::{RecordId, RecordIdKey, SurrealValue};

use crate::addr::{DerivationAddr, RuleSetAddr, RunAddr, TermAddr};
use crate::error::{self, Refusals, Result, StoreError};
use crate::lineage::{self, DerivationRecord, RuleSetRecord, RunRecord, TraceStep};
use crate::schema::{
    self, DERIVES_FIELDS, DERIVES_POINTERS, DERIVES_TABLE, REWRITE_RUN_FIELDS, REWRITE_RUN_TABLE,
    RULE_SET_FIELDS, RULE_SET_TABLE, TERM_TABLE,
};
use crate::store::Store;
use crate::term::{self, TermRecord};
use crate::term_store::TermRow;

/// Write one rule set, creating it or leaving an identical row untouched.
const PUT_RULE_SET: &str = "UPSERT $row.id CONTENT $row RETURN NONE";

/// Read one rule set's columns.
static GET_RULE_SET: LazyLock<String> =
    LazyLock::new(|| format!("SELECT {} FROM $rid", RULE_SET_FIELDS.join(", ")));

/// Write a run's trace and its derivation edge as one unit.
///
/// Both writes are idempotent and keyed on a content address, and they run
/// inside the guarded transaction the shared executor composes. The endpoint
/// terms are already stored by the time this runs — see the [module
/// documentation](self).
///
/// The three `LET`s are not decoration. `RELATE`'s grammar accepts a **bare
/// parameter** in each of its three positions and nothing else — not an idiom
/// path, so `$row.parent` is a parse error where `$parent` is fine. Unpacking
/// the one bound object into three parameters is what lets the write still
/// arrive as a single document.
const RECORD_RUN: &str = "\
UPSERT $row.run.id CONTENT $row.run RETURN NONE; \
LET $parent = $row.parent; \
LET $edge = $row.edge; \
LET $child = $row.child; \
RELATE OR UPDATE $parent -> $edge -> $child CONTENT $row.edge_data RETURN NONE";

/// Read one run's columns.
static GET_RUN: LazyLock<String> =
    LazyLock::new(|| format!("SELECT {} FROM $rid", REWRITE_RUN_FIELDS.join(", ")));

/// Read the edges a run recorded.
///
/// The projection names the edge pointers explicitly alongside the declared
/// columns. Bare `SELECT *` does return them at SurrealDB 3.2.4 — this is
/// belt-and-suspenders, and it buys the same thing every explicit projection in
/// this crate buys: a column the database stops returning becomes a failure
/// rather than a silently absent field.
static EDGES_OF_RUN: LazyLock<String> = LazyLock::new(|| {
    format!(
        "SELECT {} FROM {DERIVES_TABLE} WHERE run = $run",
        edge_projection()
    )
});

/// Read the edges leading out of a term.
static EDGES_FROM_TERM: LazyLock<String> = LazyLock::new(|| {
    format!(
        "SELECT {} FROM {DERIVES_TABLE} WHERE in = $rid",
        edge_projection()
    )
});

/// The columns an edge read projects: the declared ones, then the pointers.
fn edge_projection() -> String {
    DERIVES_FIELDS
        .iter()
        .chain(DERIVES_POINTERS.iter())
        .copied()
        .collect::<Vec<&str>>()
        .join(", ")
}

/// One row of the `rule_set` table.
#[derive(Debug, Clone, SurrealValue)]
struct RuleSetRow {
    id: RecordId,
    codec: String,
    rules_json: String,
    rule_count: i64,
}

impl RuleSetRow {
    fn from_record(record: RuleSetRecord) -> Self {
        Self {
            id: RecordId::new(RULE_SET_TABLE, record.addr.as_str()),
            codec: record.codec,
            rules_json: record.rules_json,
            rule_count: record.rule_count,
        }
    }

    fn into_record(self) -> Result<RuleSetRecord> {
        Ok(RuleSetRecord::from_columns(
            key_address(&self.id, RULE_SET_TABLE)?,
            self.codec,
            self.rules_json,
            self.rule_count,
        ))
    }
}

/// One step object inside a `rewrite_run` row.
#[derive(Debug, Clone, SurrealValue)]
struct StepRow {
    rule: i64,
    matched_edges: Vec<i64>,
}

/// One row of the `rewrite_run` table.
#[derive(Debug, Clone, SurrealValue)]
struct RunRow {
    id: RecordId,
    codec: String,
    rule_set: String,
    start: String,
    best: String,
    cost_model: String,
    initial_cost: i64,
    best_cost: i64,
    fuel_exhausted: bool,
    states_explored: i64,
    step_count: i64,
    steps: Vec<StepRow>,
    replayable: bool,
}

impl RunRow {
    fn from_record(record: RunRecord) -> Self {
        Self {
            id: RecordId::new(REWRITE_RUN_TABLE, record.addr.as_str()),
            codec: record.codec,
            rule_set: record.rule_set.as_str().to_owned(),
            start: record.start.as_str().to_owned(),
            best: record.best.as_str().to_owned(),
            cost_model: record.cost_model,
            initial_cost: record.initial_cost,
            best_cost: record.best_cost,
            fuel_exhausted: record.fuel_exhausted,
            states_explored: record.states_explored,
            step_count: record.step_count,
            steps: record
                .steps
                .into_iter()
                .map(|step| StepRow {
                    rule: step.rule(),
                    matched_edges: step.matched_edges().to_vec(),
                })
                .collect(),
            replayable: record.replayable,
        }
    }

    fn into_record(self) -> Result<RunRecord> {
        Ok(RunRecord::from_columns(
            key_address(&self.id, REWRITE_RUN_TABLE)?,
            self.codec,
            parse_addr(&self.rule_set, REWRITE_RUN_TABLE, "rule_set")?,
            parse_addr(&self.start, REWRITE_RUN_TABLE, "start")?,
            parse_addr(&self.best, REWRITE_RUN_TABLE, "best")?,
            self.cost_model,
            self.initial_cost,
            self.best_cost,
            self.fuel_exhausted,
            self.states_explored,
            self.step_count,
            self.steps
                .into_iter()
                .map(|step| TraceStep::from_columns(step.rule, step.matched_edges))
                .collect(),
            self.replayable,
        ))
    }
}

/// One row of the `derives` relation.
///
/// `r#in` binds to the `in` column with no rename attribute: the `SurrealValue`
/// derive strips the raw-identifier prefix from a field name before using it as
/// a key. (`#[serde(rename)]` would be ignored here — the derive does not read
/// serde's attributes — but no rename is needed.)
#[derive(Debug, Clone, SurrealValue)]
struct EdgeRow {
    id: RecordId,
    r#in: RecordId,
    out: RecordId,
    codec: String,
    run: String,
    rule_set: String,
    cost_model: String,
}

impl EdgeRow {
    fn into_record(self) -> Result<DerivationRecord> {
        Ok(DerivationRecord::from_columns(
            key_address(&self.id, DERIVES_TABLE)?,
            self.codec,
            key_address(&self.r#in, TERM_TABLE)?,
            key_address(&self.out, TERM_TABLE)?,
            parse_addr(&self.run, DERIVES_TABLE, "run")?,
            parse_addr(&self.rule_set, DERIVES_TABLE, "rule_set")?,
            self.cost_model,
        ))
    }
}

/// The edge payload a `RELATE` writes, without the pointers the statement sets
/// itself.
#[derive(Debug, Clone, SurrealValue)]
struct EdgeData {
    codec: String,
    run: String,
    rule_set: String,
    cost_model: String,
}

/// Everything one `record_run` write binds, as a single parameter.
///
/// One bound object rather than several parameters, because the guarded executor
/// takes one binding — and because a write whose values arrive as one document
/// cannot accidentally mix values from two runs.
#[derive(Debug, Clone, SurrealValue)]
struct RunWrite {
    run: RunRow,
    edge: RecordId,
    edge_data: EdgeData,
    parent: RecordId,
    child: RecordId,
}

/// Recover an address from a record id read back out of the database.
fn key_address<A: FromKey>(id: &RecordId, table: &'static str) -> Result<A> {
    let RecordIdKey::String(key) = &id.key else {
        return Err(StoreError::Corrupt {
            context: table.to_owned(),
            detail: "record id key is not a string".to_owned(),
        });
    };
    A::parse_key(key).ok_or_else(|| StoreError::Corrupt {
        context: table.to_owned(),
        detail: format!("record id key `{key}` is not an address"),
    })
}

/// Recover an address from a string *column*.
fn parse_addr<A: FromKey>(raw: &str, table: &'static str, field: &str) -> Result<A> {
    A::parse_key(raw).ok_or_else(|| StoreError::Corrupt {
        context: table.to_owned(),
        detail: format!("`{field}` holds `{raw}`, which is not an address"),
    })
}

/// The address types a lineage row can hold, so one recovery helper serves them
/// all.
trait FromKey: Sized {
    fn parse_key(raw: &str) -> Option<Self>;
}

macro_rules! from_key {
    ($($ty:ty),+ $(,)?) => {
        $(impl FromKey for $ty {
            fn parse_key(raw: &str) -> Option<Self> {
                Self::parse(raw)
            }
        })+
    };
}

from_key!(TermAddr, RuleSetAddr, RunAddr, DerivationAddr);

/// Stores and loads rule sets, optimizer traces, and the derivation graph.
///
/// The generator type is a phantom parameter behind a function pointer, so the
/// store's own auto traits do not depend on `G`.
#[derive(Debug, Clone)]
pub struct LineageStore<G> {
    store: Store,
    generator: PhantomData<fn() -> G>,
}

impl<G> LineageStore<G> {
    /// Open the lineage repository: bootstrap the schemas, verify them against
    /// what this build declares, and only then hand back a value that can read
    /// or write.
    ///
    /// This is the only constructor, deliberately — a write against an undefined
    /// table would make SurrealDB auto-create it `SCHEMALESS`, and a later
    /// bootstrap's `DEFINE TABLE IF NOT EXISTS` would bless the impostor rather
    /// than replace it, permanently disarming every database-side guard while
    /// every documented signal stayed green.
    ///
    /// The term table is bootstrapped too, because the derivation edge is
    /// declared `TYPE RELATION IN term OUT term` and both of its endpoints are
    /// term records.
    ///
    /// # Errors
    ///
    /// Fails if a schema cannot be defined or does not verify.
    pub async fn open(store: Store) -> Result<Self> {
        let repository = Self {
            store,
            generator: PhantomData,
        };
        repository.bootstrap().await?;
        Ok(repository)
    }

    /// The connection this store reads and writes through.
    #[must_use]
    pub fn store(&self) -> &Store {
        &self.store
    }

    fn client(&self) -> &Surreal<Any> {
        self.store.client()
    }

    /// Define the lineage tables, their columns, and their indexes, then verify
    /// them.
    ///
    /// Idempotent — safe on every open.
    ///
    /// # Errors
    ///
    /// Fails if a schema cannot be defined, or if the result is not the one this
    /// build declares.
    pub async fn bootstrap(&self) -> Result<()> {
        schema::bootstrap_lineage(self.client()).await
    }

    /// Check the live schemas against what this build declares, without changing
    /// anything.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Schema`] if any of them has drifted.
    pub async fn assert_schema(&self) -> Result<()> {
        schema::assert_lineage_schema(self.client()).await
    }

    /// The edges a run recorded.
    ///
    /// Every edge is revalidated: an edge whose columns are not the tuple its id
    /// addresses is reported rather than returned.
    ///
    /// # Errors
    ///
    /// Fails if the read is rejected, if a row is not shaped like an edge, or if
    /// an edge fails revalidation.
    pub async fn edges_of_run(&self, run: &RunAddr) -> Result<Vec<DerivationRecord>> {
        self.edges(EDGES_OF_RUN.as_str(), ("run", run.as_str().to_owned()))
            .await
    }

    /// The edges leading out of a term — the morphisms some run derived from it.
    ///
    /// # Errors
    ///
    /// As [`Self::edges_of_run`].
    pub async fn edges_from(&self, term: &TermAddr) -> Result<Vec<DerivationRecord>> {
        self.edges(
            EDGES_FROM_TERM.as_str(),
            ("rid", RecordId::new(TERM_TABLE, term.as_str())),
        )
        .await
    }

    async fn edges<V: SurrealValue + 'static>(
        &self,
        statement: &str,
        binding: (&'static str, V),
    ) -> Result<Vec<DerivationRecord>> {
        let mut response = self.client().query(statement).bind(binding).await?;
        // A vanished table answers with no edges, consistently with every other
        // read in this crate; `assert_schema` is the loud detector.
        let Some(rows) =
            error::take_absorbing_missing_table::<Vec<EdgeRow>>(response.take(0), DERIVES_TABLE)?
        else {
            return Ok(Vec::new());
        };
        rows.into_iter()
            .map(|row| {
                let record = row.into_record()?;
                record.revalidate()?;
                Ok(record)
            })
            .collect()
    }

    /// Load a run's trace, revalidating it.
    ///
    /// Returns `None` when nothing is stored under `addr`.
    ///
    /// # Errors
    ///
    /// Fails if the read is rejected, if the row is not shaped like a run, or if
    /// the record fails revalidation. See
    /// [`RunRecord::revalidate`](crate::lineage::RunRecord::revalidate).
    pub async fn get_run(&self, addr: &RunAddr) -> Result<Option<RunRecord>> {
        let mut response = self
            .client()
            .query(GET_RUN.as_str())
            .bind(("rid", RecordId::new(REWRITE_RUN_TABLE, addr.as_str())))
            .await?;
        let Some(row) = error::take_absorbing_missing_table::<Option<RunRow>>(
            response.take(0),
            REWRITE_RUN_TABLE,
        )?
        .flatten() else {
            return Ok(None);
        };
        let record = row.into_record()?;
        record.revalidate()?;
        Ok(Some(record))
    }
}

impl<G> LineageStore<G>
where
    G: PropSignature + Serialize + DeserializeOwned,
    G::Color: Serialize + DeserializeOwned,
{
    /// Store a rule set and return its content address.
    ///
    /// A single statement — no transaction. Content addressing is what makes
    /// that safe: the write is idempotent, so a retry cannot double-apply and two
    /// writers racing on the same rule set write the same bytes.
    ///
    /// Every rule is compiled through `RewriteRule::new` before anything is
    /// written, so a set containing a pair that is not a rule touches the
    /// database not at all.
    ///
    /// # Errors
    ///
    /// [`StoreError::Catgraph`] if a pair is not a well-formed rule — naming the
    /// condition it violated — [`StoreError::Revalidation`] if a side fails a
    /// screen or the set is empty, and a database error if the write is
    /// rejected.
    pub async fn put_rule_set(
        &self,
        rules: &[(ColoredExpr<G>, ColoredExpr<G>)],
    ) -> Result<RuleSetAddr> {
        let record = lineage::encode_rule_set(rules)?;
        let addr = record.addr().clone();
        self.store
            .run_write(
                RULE_SET_TABLE,
                PUT_RULE_SET,
                ("row", RuleSetRow::from_record(record)),
                // A readonly refusal here would mean two distinct encodings
                // produced one content address, which deserves to surface as the
                // raw database error it is.
                Refusals::none(),
            )
            .await?;
        Ok(addr)
    }

    /// Load a rule set, rebuilding every rule through its constructor.
    ///
    /// Returns `None` when nothing is stored under `addr`.
    ///
    /// # Errors
    ///
    /// Fails if the read is rejected, if the row is not shaped like a rule set,
    /// or if the record fails revalidation. See
    /// [`RuleSetRecord::revalidate`](crate::lineage::RuleSetRecord::revalidate).
    pub async fn get_rule_set(&self, addr: &RuleSetAddr) -> Result<Option<Vec<RewriteRule<G>>>> {
        let mut response = self
            .client()
            .query(GET_RULE_SET.as_str())
            .bind(("rid", RecordId::new(RULE_SET_TABLE, addr.as_str())))
            .await?;
        let Some(row) = error::take_absorbing_missing_table::<Option<RuleSetRow>>(
            response.take(0),
            RULE_SET_TABLE,
        )?
        .flatten() else {
            return Ok(None);
        };
        row.into_record()?.revalidate().map(Some)
    }

    /// Record an optimizer run: its endpoints, its trace, and the edge that puts
    /// them in the derivation graph.
    ///
    /// The two endpoint terms are stored first, as idempotent content-addressed
    /// writes outside the transaction; the trace row and the edge then land
    /// together inside one. See the [module documentation](self) for why the
    /// split falls there.
    ///
    /// `cost_model` names the per-generator weighting the run's costs were
    /// measured under and may not be empty — costs under different weightings
    /// are not comparable, and a run that does not say which one it used has
    /// recorded numbers nobody can read.
    ///
    /// Recording the same run twice is a success and changes nothing.
    ///
    /// # Errors
    ///
    /// - [`StoreError::Revalidation`] if `cost_model` is empty or an endpoint
    ///   fails a screen.
    /// - [`StoreError::ReadOnly`] if a *changed* trace or edge is written under
    ///   an existing address, which would mean two distinct runs had produced
    ///   one content address.
    /// - a database error otherwise. A transaction conflict is possible and is
    ///   retried by re-calling **this method**, not anything inside it — see
    ///   [`retry`](mod@crate::retry).
    pub async fn record_run(
        &self,
        rule_set: &RuleSetAddr,
        start: &ColoredExpr<G>,
        outcome: &RewriteOutcome<G>,
        cost_model: &str,
    ) -> Result<RunAddr> {
        let record = lineage::encode_run(rule_set, start, outcome, cost_model)?;
        let edge = lineage::encode_derivation(&record)?;
        let addr = record.addr().clone();

        // The endpoints, outside the transaction: idempotent, content-addressed,
        // and harmless to race on.
        self.put_terms(&[term::encode(start)?, term::encode(outcome.best())?])
            .await?;

        let write = RunWrite {
            parent: RecordId::new(TERM_TABLE, edge.parent().as_str()),
            child: RecordId::new(TERM_TABLE, edge.child().as_str()),
            edge: RecordId::new(DERIVES_TABLE, edge.addr().as_str()),
            edge_data: EdgeData {
                codec: edge.codec().to_owned(),
                run: edge.run().as_str().to_owned(),
                rule_set: edge.rule_set().as_str().to_owned(),
                cost_model: edge.cost_model().to_owned(),
            },
            run: RunRow::from_record(record),
        };
        self.store
            .run_write(
                REWRITE_RUN_TABLE,
                RECORD_RUN,
                ("row", write),
                // Both tables here are write-once by construction, so a readonly
                // refusal is the alarm that two distinct runs produced one
                // address. It is classified rather than left raw because the
                // *edge* shares that fate and naming the column is what tells
                // the two apart.
                Refusals::none().with_readonly_fields(&REWRITE_RUN_FIELDS),
            )
            .await?;
        Ok(addr)
    }

    /// Store the endpoint terms, in one statement.
    async fn put_terms(&self, records: &[TermRecord]) -> Result<()> {
        let rows: Vec<TermRow> = records.iter().cloned().map(TermRow::from_record).collect();
        self.store
            .run_write(
                TERM_TABLE,
                "FOR $row IN $rows { UPSERT $row.id CONTENT $row RETURN NONE; }",
                ("rows", rows),
                Refusals::none(),
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The read projections are derived from the declared schema, so this pins
    /// the *whole* rendered statement rather than per-field substring checks,
    /// which can pass vacuously.
    #[test]
    fn the_read_projections_are_exactly_the_declared_column_lists() {
        assert_eq!(
            GET_RULE_SET.as_str(),
            "SELECT id, codec, rules_json, rule_count FROM $rid"
        );
        assert_eq!(
            GET_RUN.as_str(),
            "SELECT id, codec, rule_set, start, best, cost_model, initial_cost, best_cost, \
             fuel_exhausted, states_explored, step_count, steps, replayable FROM $rid"
        );
    }

    /// An edge read names the pointers explicitly, beside the declared columns.
    #[test]
    fn the_edge_projection_names_the_pointers() {
        assert_eq!(
            EDGES_OF_RUN.as_str(),
            "SELECT id, codec, run, rule_set, cost_model, in, out FROM derives WHERE run = $run"
        );
        assert_eq!(
            EDGES_FROM_TERM.as_str(),
            "SELECT id, codec, run, rule_set, cost_model, in, out FROM derives WHERE in = $rid"
        );
    }

    /// The run write is one transaction over two idempotent writes, and the edge
    /// write is the explicit upsert form rather than the one that overwrites
    /// with a log line.
    #[test]
    fn the_run_write_is_two_idempotent_statements() {
        assert_eq!(RECORD_RUN.matches("RETURN NONE").count(), 2);
        assert!(RECORD_RUN.contains("RELATE OR UPDATE"));
        // Values ride as bound parameters, never as generated text.
        assert!(!RECORD_RUN.contains('\''));
    }

    /// `RELATE`'s three positions take a **bare** parameter — an idiom path is a
    /// parse error there — so the bound document is unpacked first. Pinned
    /// because the failure is at run time, in a statement that looks correct.
    #[test]
    fn the_relate_positions_hold_bare_parameters() {
        assert!(RECORD_RUN.contains("RELATE OR UPDATE $parent -> $edge -> $child"));
        for name in ["parent", "edge", "child"] {
            assert!(
                RECORD_RUN.contains(&format!("LET ${name} = $row.{name};")),
                "`{name}` is used bare but never unpacked"
            );
        }
    }

    /// `ONLY` comes before `OR UPDATE` in the statement's grammar; this store
    /// uses neither position wrongly because it uses no `ONLY` at all, and a
    /// future edit that adds one has to put it first.
    #[test]
    fn the_relate_statement_does_not_use_only() {
        assert!(!RECORD_RUN.contains("ONLY"));
    }
}
