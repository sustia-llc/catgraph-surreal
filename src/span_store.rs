//! The span repository.
//!
//! # One presentation per morphism
//!
//! - A span's **record id** is the content address of its *presentation*, so
//!   writing the same presentation twice is idempotent.
//! - The **canonical key** column is the *morphism's* identity, and it is
//!   `UNIQUE`: two parallel spans share it exactly when they are equal as
//!   morphisms (see [`crate::span`]).
//!
//! Writing a second, differently presented spelling of an already-stored
//! morphism is refused with [`StoreError::Duplicate`]; the stored
//! presentation's address is what [`SpanStore::find_by_canon`] returns.
//!
//! # Every read revalidates
//!
//! [`SpanStore::get`] revalidates every document it loads: it checks every
//! middle pair against both boundaries, decodes every label, and re-derives
//! every derived column, the canonical key included.
//!
//! # A store value implies a verified schema
//!
//! [`SpanStore::open`] is the only constructor, and it bootstraps and verifies
//! the schema before returning.

use std::marker::PhantomData;
use std::sync::LazyLock;

use catgraph::span::Span;
use surrealdb::Surreal;
use surrealdb::engine::any::Any;
use surrealdb::types::{RecordId, RecordIdKey, SurrealValue};

use crate::addr::SpanAddr;
use crate::codec::LabelCodec;
use crate::error::{self, Refusals, Result, StoreError};
use crate::schema::{self, SPAN_CANON_INDEX, SPAN_TABLE};
use crate::span::{self, SpanRecord};
use crate::store::Store;

/// Write one span, creating it or leaving an identical row untouched.
const PUT: &str = "UPSERT $row.id CONTENT $row RETURN NONE";

/// Write a batch as a single statement over a bound array; a batch the
/// database rejects any row of lands nothing.
const PUT_MANY: &str = "FOR $row IN $rows { UPSERT $row.id CONTENT $row RETURN NONE; }";

/// Read one span's columns, projecting exactly [`schema::SPAN_FIELDS`].
static GET: LazyLock<String> =
    LazyLock::new(|| format!("SELECT {} FROM $rid", schema::SPAN_FIELDS.join(", ")));

/// Find the presentation stored for a morphism, by its canonical key.
static FIND_BY_CANON: LazyLock<String> = LazyLock::new(|| {
    format!("SELECT VALUE id FROM {SPAN_TABLE} WHERE canon_key = $canon_key LIMIT 1")
});

/// Existence, without materialising the row.
const EXISTS: &str = "RETURN record::exists($rid)";

/// One row of the `span` table, mirroring the schema.
#[derive(Debug, Clone, SurrealValue)]
struct SpanRow {
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

impl SpanRow {
    /// Move a record's columns into a row.
    fn from_record(record: SpanRecord) -> Self {
        Self {
            id: record_id(&record.addr),
            codec: record.codec,
            dom: record.dom,
            cod: record.cod,
            mid_dom: record.mid_dom,
            mid_cod: record.mid_cod,
            dom_len: record.dom_len,
            cod_len: record.cod_len,
            apex_len: record.apex_len,
            canon_key: record.canon_key,
        }
    }

    /// Turn a row into a record, recovering the address from the record id.
    fn into_record(self) -> Result<SpanRecord> {
        Ok(SpanRecord::from_columns(
            address_of(&self.id)?,
            self.codec,
            self.dom,
            self.cod,
            self.mid_dom,
            self.mid_cod,
            self.dom_len,
            self.cod_len,
            self.apex_len,
            self.canon_key,
        ))
    }
}

/// The record id a span address addresses.
fn record_id(addr: &SpanAddr) -> RecordId {
    RecordId::new(SPAN_TABLE, addr.as_str())
}

/// Recover an address from a record id read back out of the database.
fn address_of(id: &RecordId) -> Result<SpanAddr> {
    let RecordIdKey::String(key) = &id.key else {
        return Err(StoreError::Corrupt {
            context: SPAN_TABLE.to_owned(),
            detail: "record id key is not a string".to_owned(),
        });
    };
    SpanAddr::parse(key).ok_or_else(|| StoreError::Corrupt {
        context: SPAN_TABLE.to_owned(),
        detail: format!("record id key `{key}` is not a span address"),
    })
}

/// Stores and loads spans, addressed by the digest of their presentation.
///
/// The label type is a phantom parameter behind a function pointer, so the
/// store's own auto traits do not depend on `L`.
#[derive(Debug, Clone)]
pub struct SpanStore<L> {
    store: Store,
    label: PhantomData<fn() -> L>,
}

impl<L> SpanStore<L> {
    /// Open the span repository: bootstrap the schema, verify it, and return a
    /// value that can read and write.
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

    /// Define the span table, its columns, and its unique key index, then
    /// verify them. Idempotent.
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined, or if the resulting schema is not
    /// the one this build declares.
    pub async fn bootstrap(&self) -> Result<()> {
        schema::bootstrap_spans(self.client()).await
    }

    /// Check the live schema against what this build declares, without changing
    /// anything.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Schema`] if the schema has drifted.
    pub async fn assert_schema(&self) -> Result<()> {
        schema::assert_span_schema(self.client()).await
    }

    /// Whether a presentation is stored under this address; `false` when the
    /// table does not exist.
    ///
    /// # Errors
    ///
    /// Fails if the query cannot be run.
    pub async fn contains(&self, addr: &SpanAddr) -> Result<bool> {
        let mut response = self
            .client()
            .query(EXISTS)
            .bind(("rid", record_id(addr)))
            .await?;
        let exists: Option<bool> = response.take(0)?;
        Ok(exists.unwrap_or(false))
    }
}

impl<L: LabelCodec> SpanStore<L> {
    /// Store a span and return its content address.
    ///
    /// # Errors
    ///
    /// - [`StoreError::Corrupt`] if a middle pair is out of bounds or names two
    ///   different labels.
    /// - [`StoreError::Duplicate`] if this morphism is already stored under a
    ///   different presentation; [`Self::find_by_canon`] returns its address.
    /// - a database error otherwise.
    pub async fn put(&self, span: &Span<L>) -> Result<SpanAddr> {
        let record = span::encode(span)?;
        let addr = record.addr().clone();
        self.write(PUT, ("row", SpanRow::from_record(record)))
            .await?;
        Ok(addr)
    }

    /// Store many spans in one statement, returning their addresses in order.
    ///
    /// Every span is encoded before anything is written, and the write is one
    /// statement: if any span fails to encode or the database rejects any row,
    /// nothing lands.
    ///
    /// # Errors
    ///
    /// As [`Self::put`]. A batch holding two presentations of one morphism
    /// fails as a whole with [`StoreError::Duplicate`].
    pub async fn put_many(&self, spans: &[Span<L>]) -> Result<Vec<SpanAddr>> {
        let records = spans
            .iter()
            .map(span::encode)
            .collect::<Result<Vec<SpanRecord>>>()?;
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let addrs: Vec<SpanAddr> = records.iter().map(|r| r.addr().clone()).collect();
        let rows: Vec<SpanRow> = records.into_iter().map(SpanRow::from_record).collect();
        self.write(PUT_MANY, ("rows", rows)).await?;
        Ok(addrs)
    }

    /// Run a write through the shared guarded executor, classifying a
    /// canonical-key collision as [`StoreError::Duplicate`].
    async fn write<V: SurrealValue + 'static>(
        &self,
        statement: &str,
        binding: (&'static str, V),
    ) -> Result<()> {
        self.store
            .run_write(
                &[SPAN_TABLE],
                statement,
                binding,
                Refusals::none().with_unique_index(SPAN_CANON_INDEX),
            )
            .await
    }

    /// Load a span, revalidating it. `None` when nothing is stored under `addr`
    /// or the table does not exist.
    ///
    /// # Errors
    ///
    /// Fails if the read is rejected, if the row is not shaped like a span row,
    /// or if the document fails
    /// [`SpanRecord::revalidate`](crate::span::SpanRecord::revalidate).
    pub async fn get(&self, addr: &SpanAddr) -> Result<Option<Span<L>>> {
        let mut response = self
            .client()
            .query(GET.as_str())
            .bind(("rid", record_id(addr)))
            .await?;
        let Some(row) =
            error::take_absorbing_missing_table::<Option<SpanRow>>(response.take(0), SPAN_TABLE)?
                .flatten()
        else {
            return Ok(None);
        };
        row.into_record()?.revalidate().map(Some)
    }

    /// The address of the presentation stored for this morphism, if any.
    ///
    /// The matched row is loaded and revalidated before its address is
    /// returned. `None` when no row matches or the table does not exist.
    ///
    /// # Errors
    ///
    /// [`StoreError::Corrupt`] if the matched row fails revalidation (including
    /// a canonical key its presentation does not derive), or if a stored record
    /// id is not an address; a database error if the read is rejected.
    pub async fn find_by_canon(&self, span: &Span<L>) -> Result<Option<SpanAddr>> {
        let key = span::canon_key(span)?;
        let mut response = self
            .client()
            .query(FIND_BY_CANON.as_str())
            .bind(("canon_key", key))
            .await?;
        let Some(ids) =
            error::take_absorbing_missing_table::<Vec<RecordId>>(response.take(0), SPAN_TABLE)?
        else {
            return Ok(None);
        };
        let Some(id) = ids.first() else {
            return Ok(None);
        };
        let addr = address_of(id)?;
        match self.get(&addr).await? {
            Some(_) => Ok(Some(addr)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_read_projection_is_exactly_the_declared_column_list() {
        assert_eq!(
            GET.as_str(),
            "SELECT id, codec, dom, cod, mid_dom, mid_cod, dom_len, cod_len, apex_len, \
             canon_key FROM $rid"
        );
    }

    #[test]
    fn the_canonical_lookup_binds_its_key() {
        assert_eq!(
            FIND_BY_CANON.as_str(),
            "SELECT VALUE id FROM span WHERE canon_key = $canon_key LIMIT 1"
        );
    }

    #[test]
    fn the_batch_statement_binds_its_rows() {
        assert!(PUT_MANY.contains("$rows"));
        assert!(!PUT_MANY.contains("INSERT INTO"));
    }

    #[test]
    fn an_address_maps_onto_the_span_table() {
        let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let addr = SpanAddr::from_digest(digest).expect("64 lowercase hex chars");
        let rid = record_id(&addr);
        assert_eq!(rid.table.as_str(), SPAN_TABLE);
        assert_eq!(rid.key, RecordIdKey::String(addr.as_str().to_owned()));
        assert_eq!(address_of(&rid).expect("a well-formed id"), addr);
    }

    #[test]
    fn a_foreign_record_id_is_corrupt() {
        let numeric = RecordId::new(SPAN_TABLE, 7i64);
        assert!(matches!(
            address_of(&numeric),
            Err(StoreError::Corrupt { .. })
        ));
        let foreign = RecordId::new(SPAN_TABLE, "not-an-address");
        assert!(matches!(
            address_of(&foreign),
            Err(StoreError::Corrupt { .. })
        ));
    }
}
