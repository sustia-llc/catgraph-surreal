//! The content-addressed term repository.
//!
//! # Identity is the record id
//!
//! A term's record id *is* its content address (`term:b3_<64 hex>`), which makes
//! writing idempotent for free: storing the same term twice writes the same row
//! twice with the same values, and there is no separate unique key column to
//! keep in step. The `nf_class` column beside it is a semantic *bucket*, indexed
//! but deliberately not unique — see [`crate::term::nf_class`].
//!
//! # There is no trusting read path
//!
//! [`TermStore::get`] revalidates every document it loads, always. That is not
//! defensiveness for its own sake: `ColoredExpr`'s `Deserialize` does not re-run
//! the type check, database-side guards are all void under `OPTION IMPORT`
//! (and the `PERMISSIONS` layer specifically is never evaluated on a default
//! embedded connection — though `ASSERT` and `READONLY` *do* run there, and the
//! collision alarm below relies on that), and the arity screen is what stands
//! between a hand-edited document and an abort deep inside the content pass. A
//! "fast path" that skipped it would be a path that trusts whatever is on disk.
//!
//! # A store value implies a verified schema
//!
//! There is no constructor that skips the schema: [`TermStore::open`]
//! bootstraps and verifies before handing back a value. That is not ceremony —
//! a write against an undefined table would make SurrealDB auto-create it
//! `SCHEMALESS`, and a *later* bootstrap's `DEFINE TABLE IF NOT EXISTS` would
//! then bless the impostor rather than replace it, permanently disarming every
//! database-side guard while every documented signal stays green. Making the
//! unbootstrapped store unrepresentable closes that hole at the type level;
//! the definition-level drift guard closes it against tables created by
//! someone else.

use std::marker::PhantomData;
use std::sync::LazyLock;

use catgraph_applied::prop::PropSignature;
use catgraph_applied::prop::colored::ColoredExpr;
use serde::Serialize;
use serde::de::DeserializeOwned;
use surrealdb::Surreal;
use surrealdb::engine::any::Any;
use surrealdb::types::{RecordId, RecordIdKey, SurrealValue};

use crate::addr::TermAddr;
use crate::error::{self, Refusals, Result, StoreError};
use crate::store::Store;
use crate::term::{self, TermRecord};
use crate::{schema, schema::TERM_TABLE};

/// Write one term, creating it or leaving an identical row untouched.
///
/// `CONTENT` rather than `MERGE` because a term document is written whole and
/// has no partial-update semantics to preserve. Re-writing an identical row
/// changes no value, so the `READONLY` columns raise nothing — while a *changed*
/// value would, which is exactly the alarm wanted if two distinct terms ever
/// resolved to one address.
const PUT: &str = "UPSERT $row.id CONTENT $row RETURN NONE";

/// Write a batch as a single statement over a bound array.
///
/// The array is a **parameter**, never interpolated into the query text. Two
/// reasons, and the first is not the obvious one: generated expression text runs
/// into the parser's own nesting and computation limits (128 levels), so a batch
/// large enough to be worth batching is a batch large enough to fail to parse.
/// The second is the usual one — text assembly around values is an injection
/// surface.
///
/// One statement means **one unit**: a batch containing a row the database
/// rejects lands nothing at all, not a prefix.
const PUT_MANY: &str = "FOR $row IN $rows { UPSERT $row.id CONTENT $row RETURN NONE; }";

/// Read one term's columns.
///
/// The projection is explicit rather than `SELECT *` so that a column this build
/// expects and the database does not is a failure rather than a silently absent
/// field — and it is **derived** from [`schema::TERM_FIELDS`] rather than
/// written out, so the projection and the schema cannot drift apart. (A
/// hand-written copy once had a pin test that passed vacuously: it looked for
/// the substring `"id"`, which `"$rid"` also contains.)
static GET: LazyLock<String> =
    LazyLock::new(|| format!("SELECT {} FROM $rid", schema::TERM_FIELDS.join(", ")));

/// Existence, without materialising the row.
const EXISTS: &str = "RETURN record::exists($rid)";

/// One row of the `term` table.
///
/// Mirrors the schema exactly, integer columns included: the conversion between
/// this and the in-memory counts happens once, in [`crate::term::encode`], where
/// it is checked rather than cast.
///
/// Crate-visible because the lineage tier stores a run's endpoint terms itself,
/// and it must write the same row shape this tier reads — two copies of the
/// mapping would be two things to keep in step.
#[derive(Debug, Clone, SurrealValue)]
pub(crate) struct TermRow {
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

impl TermRow {
    /// Consuming on purpose: the encoding string moves into the row rather than
    /// being cloned per write.
    pub(crate) fn from_record(record: TermRecord) -> Self {
        Self {
            id: record_id(&record.addr),
            codec: record.codec,
            term_json: record.term_json,
            signature: record.signature,
            source_arity: record.source_arity,
            target_arity: record.target_arity,
            depth: record.depth,
            generator_count: record.generator_count,
            nf_class: record.nf_class,
        }
    }

    /// Turn a row into a record, recovering the address from the record id.
    ///
    /// A key that is not a well-formed address means the row was written by
    /// something other than this store — the id `ASSERT` is skipped on update
    /// and under `OPTION IMPORT`, so it cannot be relied on to have run.
    fn into_record(self) -> Result<TermRecord> {
        let RecordIdKey::String(key) = &self.id.key else {
            return Err(StoreError::Corrupt {
                context: TERM_TABLE.to_owned(),
                detail: "record id key is not a string".to_owned(),
            });
        };
        let addr = TermAddr::parse(key).ok_or_else(|| StoreError::Corrupt {
            context: TERM_TABLE.to_owned(),
            detail: format!("record id key `{key}` is not a term address"),
        })?;
        Ok(TermRecord::from_columns(
            addr,
            self.codec,
            self.term_json,
            self.signature,
            self.source_arity,
            self.target_arity,
            self.depth,
            self.generator_count,
            self.nf_class,
        ))
    }
}

/// The record id a term address addresses.
fn record_id(addr: &TermAddr) -> RecordId {
    RecordId::new(TERM_TABLE, addr.as_str())
}

/// Stores and loads terms, addressed by the digest of their canonical encoding.
///
/// The generator type is a phantom parameter behind a function pointer, so the
/// store's own auto traits do not depend on `G`.
#[derive(Debug, Clone)]
pub struct TermStore<G> {
    store: Store,
    generator: PhantomData<fn() -> G>,
}

impl<G> TermStore<G> {
    /// Open the term repository: bootstrap the schema, verify it against what
    /// this build declares, and only then hand back a value that can read or
    /// write.
    ///
    /// This is the only constructor, deliberately — see the [module
    /// documentation](self): a write before the schema exists would auto-create
    /// a `SCHEMALESS` table that every later signal blesses. If the table
    /// already exists, its definitions are verified exactly, so a table created
    /// by an implicit write (or an older or newer version of this store) fails
    /// here with [`StoreError::Schema`] rather than silently serving
    /// unguarded rows.
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined or does not verify.
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

    /// Define the term table, its columns, and its indexes, then verify them.
    ///
    /// Idempotent — safe on every open.
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined, or if the resulting schema is not
    /// the one this build declares.
    pub async fn bootstrap(&self) -> Result<()> {
        schema::bootstrap_terms(self.client()).await
    }

    /// Check the live schema against what this build declares, without changing
    /// anything.
    ///
    /// Worth having separately from [`Self::bootstrap`]: every bootstrap
    /// statement is `IF NOT EXISTS`, so bootstrapping would quietly *repair* a
    /// dropped column instead of reporting it. This asks.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Schema`] if the schema has drifted.
    pub async fn assert_schema(&self) -> Result<()> {
        schema::assert_term_schema(self.client()).await
    }

    /// Whether a term is stored under this address.
    ///
    /// # Which existence primitive, and why
    ///
    /// `record::exists()` rather than a `SELECT … LIMIT 0` probe. The two differ
    /// on exactly one case: a table that does not exist. `SELECT` raises
    /// `TbNotFound`; `record::exists()` absorbs it into `false`.
    ///
    /// `false` is the right answer here. This store defines its schema at
    /// [`Self::open`], and [`Self::assert_schema`] is the dedicated, *loud*
    /// detector for a table that has gone missing since. Routing that same
    /// condition through every existence check as an opaque table-not-found
    /// error would make a bootstrap bug look like a query bug, while telling
    /// `contains` callers nothing they can act on: an absent table really does
    /// contain no terms. [`Self::get`] absorbs the same condition into `None`
    /// for the same reason — the two answers agree.
    ///
    /// # Errors
    ///
    /// Fails if the query cannot be run.
    pub async fn contains(&self, addr: &TermAddr) -> Result<bool> {
        let mut response = self
            .client()
            .query(EXISTS)
            .bind(("rid", record_id(addr)))
            .await?;
        // `record::exists` always answers with a boolean; the fallback only
        // covers the shape of the result, not a real case.
        let exists: Option<bool> = response.take(0)?;
        Ok(exists.unwrap_or(false))
    }
}

impl<G> TermStore<G>
where
    G: PropSignature + Serialize + DeserializeOwned,
    G::Color: Serialize + DeserializeOwned,
{
    /// Store a term and return its content address.
    ///
    /// A single statement — no transaction. Content addressing is what makes
    /// that safe: the write is idempotent, so a retry cannot double-apply and
    /// two writers racing on the same term write the same bytes. Storing a term
    /// that is already present is a success, not a conflict.
    ///
    /// # Errors
    ///
    /// Fails if the term cannot be encoded (see [`crate::term::encode`]) or if
    /// the write is rejected. A rejection naming a read-only column would mean
    /// two distinct encodings had produced one address.
    pub async fn put(&self, term: &ColoredExpr<G>) -> Result<TermAddr> {
        let record = term::encode(term)?;
        let addr = record.addr().clone();
        // The table-existence guard lives on the shared executor — see
        // `Store::run_write`. No unique index and no classified READONLY list
        // on this tier: a readonly refusal here would mean two distinct
        // encodings produced one content address, which deserves to surface as
        // the raw database error it is.
        self.store
            .run_write(
                TERM_TABLE,
                PUT,
                ("row", TermRow::from_record(record)),
                Refusals::none(),
            )
            .await?;
        Ok(addr)
    }

    /// Store many terms in one statement, returning their addresses in order.
    ///
    /// Every term is encoded — and therefore fully screened — before anything is
    /// written, so a batch containing an unencodable term touches the database
    /// not at all. The write itself is one statement over a bound array, which
    /// makes it a single unit: if the database rejects any row, none of them
    /// land.
    ///
    /// # Errors
    ///
    /// Fails if any term cannot be encoded, or if the write is rejected.
    pub async fn put_many(&self, terms: &[ColoredExpr<G>]) -> Result<Vec<TermAddr>> {
        let records = terms
            .iter()
            .map(term::encode)
            .collect::<Result<Vec<TermRecord>>>()?;
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let addrs: Vec<TermAddr> = records.iter().map(|r| r.addr().clone()).collect();
        let rows: Vec<TermRow> = records.into_iter().map(TermRow::from_record).collect();
        self.store
            .run_write(TERM_TABLE, PUT_MANY, ("rows", rows), Refusals::none())
            .await?;
        Ok(addrs)
    }

    /// Load a term, revalidating it.
    ///
    /// Returns `None` when nothing is stored under `addr` — an absent record is
    /// not an error. Everything else is: a document that fails any revalidation
    /// step is reported rather than returned.
    ///
    /// # Errors
    ///
    /// Fails if the read is rejected, if the row is not shaped like a term row,
    /// or if the document fails revalidation. See
    /// [`TermRecord::revalidate`](crate::term::TermRecord::revalidate) for the
    /// stages.
    pub async fn get(&self, addr: &TermAddr) -> Result<Option<ColoredExpr<G>>> {
        let mut response = self
            .client()
            .query(GET.as_str())
            .bind(("rid", record_id(addr)))
            .await?;
        // A vanished table answers `None`, matching `contains` — the same
        // condition must not read as `false` from one method and as an opaque
        // query error from the other. `SELECT` raises where `record::exists`
        // absorbs, and the raise has no structured discriminator, so this is a
        // message match; the integration suite pins it against the engine.
        let Some(row) =
            error::take_absorbing_missing_table::<Option<TermRow>>(response.take(0), TERM_TABLE)?
                .flatten()
        else {
            return Ok(None);
        };
        let record = row.into_record()?;
        record.revalidate().map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The read projection is derived from the declared schema, so this pins
    /// the *whole* rendered statement — not per-field substring checks, whose
    /// previous incarnation passed vacuously (`"$rid"` contains `"id"`).
    #[test]
    fn the_read_projection_is_exactly_the_declared_column_list() {
        assert_eq!(
            GET.as_str(),
            "SELECT id, term_json, codec, signature, source_arity, target_arity, \
             depth, generator_count, nf_class FROM $rid"
        );
    }

    /// Batch values ride as a bound parameter. If this ever regresses into
    /// generated text it will be for a batch small enough to work in testing and
    /// large enough to fail in production.
    #[test]
    fn the_batch_statement_binds_its_rows() {
        assert!(PUT_MANY.contains("$rows"));
        assert!(!PUT_MANY.contains("INSERT INTO"));
    }

    #[test]
    fn an_address_maps_onto_the_term_table() {
        let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let addr = TermAddr::from_digest(digest).expect("64 lowercase hex chars");
        let rid = record_id(&addr);
        assert_eq!(rid.table.as_str(), TERM_TABLE);
        assert_eq!(rid.key, RecordIdKey::String(addr.as_str().to_owned()));
    }
}
