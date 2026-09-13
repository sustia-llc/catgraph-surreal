//! The cospan repository.
//!
//! # One presentation per morphism
//!
//! Two identities live side by side here, and the difference between them is the
//! contract:
//!
//! - A cospan's **record id** is the content address of its *presentation*, so
//!   writing the same presentation twice is idempotent for free.
//! - The **canonical key** column is the *morphism's* identity, and it is
//!   `UNIQUE`. A cospan's canonical form is a complete invariant — two parallel
//!   cospans share it exactly when they are equal as morphisms — so two rows
//!   sharing a key would be two names for one thing.
//!
//! The consequence is worth stating plainly: **writing a second, differently
//! presented spelling of an already-stored morphism is refused**, with
//! [`StoreError::Duplicate`]. It is not a conflict and retrying will not help;
//! the morphism is already stored, under the address
//! [`CospanStore::find_by_canon`] returns. That is a deliberate trade — the
//! alternative, silently returning the stored presentation's address, would mean
//! `get(put(c))` handing back a cospan structurally unlike `c` with nothing
//! said. Refusing keeps the surprise where a caller can see it.
//!
//! This is the opposite choice from the term tier, and for a reason that is
//! about the mathematics rather than the storage: `nf_class` is only a *sound*
//! bucket, so equal terms may land in different buckets and duplicates are
//! expected. A cospan's canonical form decides equality outright.
//!
//! # There is no trusting read path
//!
//! [`CospanStore::get`] revalidates every document it loads: it bounds-checks
//! both legs against the apex, decodes every label, and re-derives every
//! derived column — including the canonical key, which is the one column whose
//! corruption would make a key lookup claim two unequal morphisms are equal.
//!
//! # A store value implies a verified schema
//!
//! There is no constructor that skips the schema: [`CospanStore::open`]
//! bootstraps and verifies before handing back a value. A write against an
//! undefined table would make SurrealDB auto-create it `SCHEMALESS` — and a
//! *later* bootstrap's `DEFINE TABLE IF NOT EXISTS` would bless the impostor
//! rather than replace it, permanently disarming every database-side guard,
//! the unique index included, while every documented signal stayed green.

use std::marker::PhantomData;
use std::sync::LazyLock;

use catgraph::cospan::Cospan;
use surrealdb::Surreal;
use surrealdb::engine::any::Any;
use surrealdb::types::{RecordId, RecordIdKey, SurrealValue};

use crate::addr::CospanAddr;
use crate::codec::LabelCodec;
use crate::cospan::{self, CospanRecord};
use crate::error::{self, Refusals, Result, StoreError};
use crate::schema::{self, COSPAN_CANON_INDEX, COSPAN_TABLE};
use crate::store::Store;

/// Write one cospan, creating it or leaving an identical row untouched.
///
/// `CONTENT` rather than `MERGE` because a cospan document is written whole and
/// has no partial-update semantics to preserve. Re-writing an identical row
/// changes no value, so the `READONLY` columns raise nothing — while a *changed*
/// value would, which is the alarm wanted if two distinct presentations ever
/// resolved to one address.
const PUT: &str = "UPSERT $row.id CONTENT $row RETURN NONE";

/// Write a batch as a single statement over a bound array.
///
/// The array is a **parameter**, never interpolated into the query text:
/// generated expression text runs into the parser's own nesting and computation
/// limits, so a batch large enough to be worth batching is a batch large enough
/// to fail to parse — and text assembly around values is an injection surface
/// besides.
///
/// One statement means **one unit**: a batch containing a row the database
/// rejects lands nothing at all, not a prefix. With a unique key column that is
/// the behaviour to want — a batch holding two presentations of one morphism is
/// a mistake, and half-applying it would be worse than refusing it.
const PUT_MANY: &str = "FOR $row IN $rows { UPSERT $row.id CONTENT $row RETURN NONE; }";

/// Read one cospan's columns.
///
/// The projection is explicit rather than `SELECT *` so that a column this build
/// expects and the database does not is a failure rather than a silently absent
/// field — and it is **derived** from [`schema::COSPAN_FIELDS`] rather than
/// written out, so the projection and the schema cannot drift apart.
static GET: LazyLock<String> =
    LazyLock::new(|| format!("SELECT {} FROM $rid", schema::COSPAN_FIELDS.join(", ")));

/// Find the presentation stored for a morphism, by its canonical key.
///
/// Served by the unique index, so this is a key lookup rather than a scan.
static FIND_BY_CANON: LazyLock<String> = LazyLock::new(|| {
    format!("SELECT VALUE id FROM {COSPAN_TABLE} WHERE canon_key = $canon_key LIMIT 1")
});

/// Existence, without materialising the row.
const EXISTS: &str = "RETURN record::exists($rid)";

/// One row of the `cospan` table.
///
/// Mirrors the schema exactly, integer columns included: the conversion between
/// these and the in-memory indices happens once, in [`crate::cospan::encode`],
/// where it is checked rather than cast.
#[derive(Debug, Clone, SurrealValue)]
struct CospanRow {
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

impl CospanRow {
    /// Consuming on purpose: the payload vectors move into the row rather than
    /// being deep-cloned per write.
    fn from_record(record: CospanRecord) -> Self {
        Self {
            id: record_id(&record.addr),
            codec: record.codec,
            dom_leg: record.dom_leg,
            cod_leg: record.cod_leg,
            apex: record.apex,
            dom_len: record.dom_len,
            cod_len: record.cod_len,
            apex_len: record.apex_len,
            scalar_count: record.scalar_count,
            canon_key: record.canon_key,
        }
    }

    /// Turn a row into a record, recovering the address from the record id.
    ///
    /// A key that is not a well-formed address means the row was written by
    /// something other than this store — the id `ASSERT` is skipped on update
    /// and under `OPTION IMPORT`, so it cannot be relied on to have run.
    fn into_record(self) -> Result<CospanRecord> {
        Ok(CospanRecord::from_columns(
            address_of(&self.id)?,
            self.codec,
            self.dom_leg,
            self.cod_leg,
            self.apex,
            self.dom_len,
            self.cod_len,
            self.apex_len,
            self.scalar_count,
            self.canon_key,
        ))
    }
}

/// The record id a cospan address addresses.
fn record_id(addr: &CospanAddr) -> RecordId {
    RecordId::new(COSPAN_TABLE, addr.as_str())
}

/// Recover an address from a record id read back out of the database.
fn address_of(id: &RecordId) -> Result<CospanAddr> {
    let RecordIdKey::String(key) = &id.key else {
        return Err(StoreError::Corrupt {
            context: COSPAN_TABLE.to_owned(),
            detail: "record id key is not a string".to_owned(),
        });
    };
    CospanAddr::parse(key).ok_or_else(|| StoreError::Corrupt {
        context: COSPAN_TABLE.to_owned(),
        detail: format!("record id key `{key}` is not a cospan address"),
    })
}

/// Stores and loads cospans, addressed by the digest of their presentation.
///
/// The label type is a phantom parameter behind a function pointer, so the
/// store's own auto traits do not depend on `L`.
#[derive(Debug, Clone)]
pub struct CospanStore<L> {
    store: Store,
    label: PhantomData<fn() -> L>,
}

impl<L> CospanStore<L> {
    /// Open the cospan repository: bootstrap the schema, verify it against what
    /// this build declares, and only then hand back a value that can read or
    /// write.
    ///
    /// This is the only constructor, deliberately — see the [module
    /// documentation](self).
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined or does not verify.
    pub async fn open(store: Store) -> Result<Self> {
        let repository = Self {
            store,
            label: PhantomData,
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

    /// Define the cospan table, its columns, and its unique key index, then
    /// verify them.
    ///
    /// Idempotent — safe on every open.
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined, or if the resulting schema is not
    /// the one this build declares.
    pub async fn bootstrap(&self) -> Result<()> {
        schema::bootstrap_cospans(self.client()).await
    }

    /// Check the live schema against what this build declares, without changing
    /// anything.
    ///
    /// Worth having separately from [`Self::bootstrap`]: every bootstrap
    /// statement is `IF NOT EXISTS`, so bootstrapping would quietly *repair* a
    /// dropped column or index instead of reporting it. This asks.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Schema`] if the schema has drifted.
    pub async fn assert_schema(&self) -> Result<()> {
        schema::assert_cospan_schema(self.client()).await
    }

    /// Whether a presentation is stored under this address.
    ///
    /// `record::exists()` rather than a `SELECT … LIMIT 0` probe: the two differ
    /// on exactly one case, a table that does not exist, which `SELECT` raises
    /// and `record::exists()` absorbs into `false`. `false` is the right answer
    /// — an absent table really does contain no cospans, and
    /// [`Self::assert_schema`] is the dedicated, *loud* detector for a table
    /// that has gone missing. [`Self::get`] absorbs the same condition into
    /// `None`, so the two answers agree.
    ///
    /// # Errors
    ///
    /// Fails if the query cannot be run.
    pub async fn contains(&self, addr: &CospanAddr) -> Result<bool> {
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

impl<L: LabelCodec> CospanStore<L> {
    /// Store a cospan and return its content address.
    ///
    /// A single statement — no transaction. Content addressing is what makes
    /// that safe: the write is idempotent, so a retry cannot double-apply and
    /// two writers racing on the same presentation write the same bytes.
    ///
    /// # Errors
    ///
    /// - [`StoreError::Corrupt`] if a leg points outside the apex. `Cospan::new`
    ///   accepts such a value in a release build, so this is where it is caught
    ///   — before it reaches disk, where no loader could safely interpret it.
    /// - [`StoreError::Duplicate`] if this morphism is already stored under a
    ///   *different* presentation. Retrying will not help; the morphism is
    ///   already there, at the address [`Self::find_by_canon`] returns.
    /// - a database error otherwise. A rejection naming a read-only column would
    ///   mean two distinct presentations had produced one address.
    pub async fn put(&self, cospan: &Cospan<L>) -> Result<CospanAddr> {
        let record = cospan::encode(cospan)?;
        let addr = record.addr().clone();
        self.write(PUT, ("row", CospanRow::from_record(record)))
            .await?;
        Ok(addr)
    }

    /// Store many cospans in one statement, returning their addresses in order.
    ///
    /// Every cospan is encoded — and therefore fully screened — before anything
    /// is written, so a batch containing an ill-formed one touches the database
    /// not at all. The write itself is one statement over a bound array, which
    /// makes it a single unit: if the database rejects any row, none of them
    /// land.
    ///
    /// # Errors
    ///
    /// As [`Self::put`]. A batch that contains two presentations of one morphism
    /// fails as a whole with [`StoreError::Duplicate`], writing nothing.
    pub async fn put_many(&self, cospans: &[Cospan<L>]) -> Result<Vec<CospanAddr>> {
        let records = cospans
            .iter()
            .map(cospan::encode)
            .collect::<Result<Vec<CospanRecord>>>()?;
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let addrs: Vec<CospanAddr> = records.iter().map(|r| r.addr().clone()).collect();
        let rows: Vec<CospanRow> = records.into_iter().map(CospanRow::from_record).collect();
        self.write(PUT_MANY, ("rows", rows)).await?;
        Ok(addrs)
    }

    /// Run a write through the shared guarded executor.
    ///
    /// The guard and the refusal classification live on [`Store::run_write`],
    /// so this tier holds the write invariants by construction. No `READONLY`
    /// column list is passed: on this tier a readonly refusal would mean two
    /// distinct presentations produced one content address, which deserves to
    /// surface as the raw database error it is.
    async fn write<V: SurrealValue + 'static>(
        &self,
        statement: &str,
        binding: (&'static str, V),
    ) -> Result<()> {
        self.store
            .run_write(
                &[COSPAN_TABLE],
                statement,
                binding,
                Refusals::none().with_unique_index(COSPAN_CANON_INDEX),
            )
            .await
    }

    /// Load a cospan, revalidating it.
    ///
    /// Returns `None` when nothing is stored under `addr` — an absent record is
    /// not an error. Everything else is: a document that fails any check is
    /// reported rather than returned.
    ///
    /// # Errors
    ///
    /// Fails if the read is rejected, if the row is not shaped like a cospan
    /// row, or if the document fails revalidation. See
    /// [`CospanRecord::revalidate`](crate::cospan::CospanRecord::revalidate).
    pub async fn get(&self, addr: &CospanAddr) -> Result<Option<Cospan<L>>> {
        let mut response = self
            .client()
            .query(GET.as_str())
            .bind(("rid", record_id(addr)))
            .await?;
        // A vanished table answers `None`, matching `contains` — the same
        // condition must not read as `false` from one method and as an opaque
        // query error from the other.
        let Some(row) = error::take_absorbing_missing_table::<Option<CospanRow>>(
            response.take(0),
            COSPAN_TABLE,
        )?
        .flatten() else {
            return Ok(None);
        };
        row.into_record()?.revalidate().map(Some)
    }

    /// The address of the presentation stored for this morphism, if any.
    ///
    /// This is how [`StoreError::Duplicate`] is recovered from, and how a caller
    /// asks "is this morphism stored, however it was spelled?" — the canonical
    /// key decides equality of morphisms outright, so the answer is exact rather
    /// than a candidate set.
    ///
    /// **The matched row is loaded and revalidated before its address is
    /// returned.** The index lookup alone would trust the stored `canon_key`
    /// column — the one read that skips revalidation — and a tampered key
    /// (written raw, or replayed under `OPTION IMPORT`) would hand back the
    /// address of an unrelated cospan as a confident wrong yes. Revalidation
    /// re-derives the key from the row's own presentation, so a squatting row
    /// surfaces as [`StoreError::Corrupt`] here rather than as a silent
    /// misdirection. Costs one row read; correctness of "is this morphism
    /// stored?" is what this method exists for.
    ///
    /// A vanished table answers `None`, consistently with the other reads.
    ///
    /// # Errors
    ///
    /// [`StoreError::Corrupt`] if the matched row fails revalidation (including
    /// a tampered canonical key), or if a stored record id is not an address; a
    /// database error if the read is rejected.
    pub async fn find_by_canon(&self, cospan: &Cospan<L>) -> Result<Option<CospanAddr>> {
        let key = cospan::canon_key(cospan)?;
        let mut response = self
            .client()
            .query(FIND_BY_CANON.as_str())
            .bind(("canon_key", key))
            .await?;
        let Some(ids) =
            error::take_absorbing_missing_table::<Vec<RecordId>>(response.take(0), COSPAN_TABLE)?
        else {
            return Ok(None);
        };
        let Some(id) = ids.first() else {
            return Ok(None);
        };
        let addr = address_of(id)?;
        // Revalidate the matched row; a row that vanished between the two reads
        // is answered as absent, like every other read.
        match self.get(&addr).await? {
            Some(_) => Ok(Some(addr)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The read projection is derived from the declared schema, so this pins the
    /// *whole* rendered statement rather than per-field substring checks, which
    /// can pass vacuously.
    #[test]
    fn the_read_projection_is_exactly_the_declared_column_list() {
        assert_eq!(
            GET.as_str(),
            "SELECT id, codec, dom_leg, cod_leg, apex, dom_len, cod_len, apex_len, \
             scalar_count, canon_key FROM $rid"
        );
    }

    /// The key lookup has to go through the indexed column, and has to bind its
    /// value rather than interpolate it.
    #[test]
    fn the_canonical_lookup_binds_its_key() {
        assert_eq!(
            FIND_BY_CANON.as_str(),
            "SELECT VALUE id FROM cospan WHERE canon_key = $canon_key LIMIT 1"
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
    fn an_address_maps_onto_the_cospan_table() {
        let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let addr = CospanAddr::from_digest(digest).expect("64 lowercase hex chars");
        let rid = record_id(&addr);
        assert_eq!(rid.table.as_str(), COSPAN_TABLE);
        assert_eq!(rid.key, RecordIdKey::String(addr.as_str().to_owned()));
        assert_eq!(address_of(&rid).expect("a well-formed id"), addr);
    }

    /// A record id this store did not write must be refused at the boundary
    /// rather than carried inward as a plausible-looking handle.
    #[test]
    fn a_foreign_record_id_is_corrupt() {
        let numeric = RecordId::new(COSPAN_TABLE, 7i64);
        assert!(matches!(
            address_of(&numeric),
            Err(StoreError::Corrupt { .. })
        ));
        let foreign = RecordId::new(COSPAN_TABLE, "not-an-address");
        assert!(matches!(
            address_of(&foreign),
            Err(StoreError::Corrupt { .. })
        ));
    }
}
