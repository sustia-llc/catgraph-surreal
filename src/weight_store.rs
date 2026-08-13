//! The weight repository.
//!
//! # Identity is the caller's key pair
//!
//! A weight row is identified by `(genome, gen_key)`, two **opaque
//! caller-supplied strings**. The store indexes that pair uniquely and derives
//! the record id from it, but it derives no meaning from either half: key
//! derivation, and the guarantee that a key is stable across processes and
//! builds, stay with the caller. (Standard-library hash output in particular is
//! not stable across processes and must never become a key.)
//!
//! # Weights are write-once under their key
//!
//! Every column is `READONLY`, so a stored weight vector cannot be changed.
//! Re-storing an *identical* vector is a no-op; storing a *different* one under
//! the same key is refused with [`StoreError::ReadOnly`].
//!
//! That is a modelling decision, and it has a direct consequence: **a training
//! loop that produces a new vector supplies a new `gen_key`** — the generation,
//! the step, whatever names that version. The alternative, letting a key's value
//! change underneath it, would make every row that references a `(genome,
//! gen_key)` pair ambiguous about which vector it meant, which is exactly the
//! property lineage and checkpoint data exist to have.
//!
//! # Bit-exactness is the point of the byte lane
//!
//! Coordinates are stored as raw little-endian IEEE-754 bytes rather than as
//! native floats. Native floats round-trip non-finites through a store and load
//! — but *not* through an export, which renders every `NaN` as the bare literal
//! `NaN`, sign and payload gone. See [`crate::weight`] for the full argument.
//!
//! # A store value implies a verified schema
//!
//! There is no constructor that skips the schema: [`WeightStore::open`]
//! bootstraps and verifies before handing back a value — otherwise a write
//! against an undefined table would auto-create it `SCHEMALESS`, and a later
//! bootstrap would bless the impostor rather than replace it, silently disarming
//! the unique index and every `READONLY` column at once.

use std::sync::LazyLock;

use catgraph_dl::para::RModule;
use surrealdb::Surreal;
use surrealdb::engine::any::Any;
use surrealdb::types::{Bytes, RecordId, RecordIdKey, SurrealValue};

use crate::error::{self, Refusals, Result, StoreError};
use crate::schema::{self, WEIGHT_FIELDS, WEIGHT_KEY_INDEX, WEIGHT_TABLE};
use crate::store::Store;
use crate::weight::{self, WeightRecord};

/// Write one weight row, creating it or leaving an identical row untouched.
///
/// `CONTENT` because a weight document is written whole. Re-writing an identical
/// row changes no value, so the `READONLY` columns raise nothing — while a
/// changed vector does, which is the write-once contract being enforced.
const PUT: &str = "UPSERT $row.id CONTENT $row RETURN NONE";

/// Read one weight row, by the key pair rather than by the derived record id.
///
/// Going through the indexed pair rather than the id is deliberate: the pair is
/// the row's *declared* identity, so a row written by something else under a
/// different id is found and then rejected by revalidation, instead of being
/// invisible here while colliding with every write.
///
/// The projection is explicit rather than `SELECT *`, and **derived** from
/// [`schema::WEIGHT_FIELDS`] rather than written out, so it cannot drift from
/// the schema.
static GET: LazyLock<String> = LazyLock::new(|| {
    format!(
        "SELECT {} FROM {WEIGHT_TABLE} WHERE genome = $genome AND gen_key = $gen_key LIMIT 1",
        WEIGHT_FIELDS.join(", ")
    )
});

/// Existence, through the same key pair as [`GET`] so the two always agree.
static EXISTS: LazyLock<String> = LazyLock::new(|| {
    format!(
        "SELECT VALUE id FROM {WEIGHT_TABLE} WHERE genome = $genome AND gen_key = $gen_key LIMIT 1"
    )
});

/// One row of the `weight` table.
///
/// `coordinates` is [`Bytes`], not `Vec<u8>`: the SDK deliberately leaves
/// `Vec<u8>` without a `SurrealValue` implementation so that "binary data"
/// versus "an array of small integers" is a choice rather than an inference.
#[derive(Debug, Clone, SurrealValue)]
struct WeightRow {
    id: RecordId,
    codec: String,
    genome: String,
    gen_key: String,
    dim: i64,
    coordinates: Bytes,
    finite: bool,
}

impl WeightRow {
    /// Consuming on purpose: the coordinate buffer moves into the row rather
    /// than being memcpy'd a second time per write.
    fn from_record(record: WeightRecord) -> Self {
        Self {
            id: RecordId::new(WEIGHT_TABLE, record.key.as_str()),
            codec: record.codec,
            genome: record.genome,
            gen_key: record.gen_key,
            dim: record.dim,
            coordinates: Bytes::from(record.coordinates),
            finite: record.finite,
        }
    }

    /// Turn a row into a record, recovering the stored key from the record id.
    ///
    /// A key that is not a string means the row was written by something other
    /// than this store; revalidation then re-derives the key from the pair and
    /// compares, so a misfiled row is caught either way.
    fn into_record(self) -> Result<WeightRecord> {
        let RecordIdKey::String(key) = self.id.key else {
            return Err(StoreError::Corrupt {
                context: WEIGHT_TABLE.to_owned(),
                detail: "record id key is not a string".to_owned(),
            });
        };
        Ok(WeightRecord::from_columns(
            key,
            self.codec,
            self.genome,
            self.gen_key,
            self.dim,
            self.coordinates.to_vec(),
            self.finite,
        ))
    }
}

/// Stores and loads parameter weights, keyed by a caller-supplied pair.
#[derive(Debug, Clone)]
pub struct WeightStore {
    store: Store,
}

impl WeightStore {
    /// Open the weight repository: bootstrap the schema, verify it against what
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
        let repository = Self { store };
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

    /// Define the weight table, its columns, and its unique key index, then
    /// verify them.
    ///
    /// Idempotent — safe on every open.
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined, or if the resulting schema is not
    /// the one this build declares.
    pub async fn bootstrap(&self) -> Result<()> {
        schema::bootstrap_weights(self.client()).await
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
        schema::assert_weight_schema(self.client()).await
    }

    /// Store a weight vector under a key pair.
    ///
    /// A single statement — no transaction. Storing the same vector under the
    /// same key twice is a success and changes nothing.
    ///
    /// # Errors
    ///
    /// - [`StoreError::ReadOnly`] if a *different* vector is already stored
    ///   under this key. Weights are write-once; a new vector needs a new key.
    ///   Retrying will not help.
    /// - [`StoreError::Duplicate`] if some other row already claims this key
    ///   pair under a different record id — a row this store did not write.
    /// - a database error otherwise.
    pub async fn put(&self, genome: &str, gen_key: &str, weights: &RModule<f64>) -> Result<()> {
        let record = weight::encode(genome, gen_key, weights)?;
        // The guard, and the duplicate-before-readonly classification order,
        // live on the shared executor — see `Store::run_write`.
        self.store
            .run_write(
                &[WEIGHT_TABLE],
                PUT,
                ("row", WeightRow::from_record(record)),
                Refusals::none()
                    .with_unique_index(WEIGHT_KEY_INDEX)
                    .with_readonly_fields(&WEIGHT_FIELDS),
            )
            .await
    }

    /// Load a weight vector, revalidating it.
    ///
    /// Returns `None` when nothing is stored under the key pair — an absent
    /// record is not an error, and neither is a table that has been removed
    /// since this store opened (which [`Self::contains`] also reports as
    /// absent, and [`Self::assert_schema`] reports loudly).
    ///
    /// The module handed back has the dimension its row claims: the coordinate
    /// width is checked against the stored `dim` before the bytes are decoded.
    /// Whether that dimension is the one the *architecture* expects is a
    /// question only the caller can ask — `RModule` itself checks nothing on
    /// load, and only `add` rejects a mismatch — so ask it once, here, at your
    /// own entry point.
    ///
    /// # Errors
    ///
    /// Fails if the read is rejected, if the row is not shaped like a weight
    /// row, or if the document fails revalidation. See
    /// [`WeightRecord::revalidate`](crate::weight::WeightRecord::revalidate).
    pub async fn get(&self, genome: &str, gen_key: &str) -> Result<Option<RModule<f64>>> {
        let mut response = self
            .client()
            .query(GET.as_str())
            .bind(("genome", genome.to_owned()))
            .bind(("gen_key", gen_key.to_owned()))
            .await?;
        let Some(rows) =
            error::take_absorbing_missing_table::<Vec<WeightRow>>(response.take(0), WEIGHT_TABLE)?
        else {
            return Ok(None);
        };
        let Some(row) = rows.into_iter().next() else {
            return Ok(None);
        };
        row.into_record()?.revalidate().map(Some)
    }

    /// Whether a weight vector is stored under this key pair.
    ///
    /// Goes through the same indexed pair as [`Self::get`], so the two can never
    /// disagree — including on a table that has gone missing, which both report
    /// as absent.
    ///
    /// # Errors
    ///
    /// Fails if the query cannot be run.
    pub async fn contains(&self, genome: &str, gen_key: &str) -> Result<bool> {
        let mut response = self
            .client()
            .query(EXISTS.as_str())
            .bind(("genome", genome.to_owned()))
            .bind(("gen_key", gen_key.to_owned()))
            .await?;
        let Some(ids) =
            error::take_absorbing_missing_table::<Vec<RecordId>>(response.take(0), WEIGHT_TABLE)?
        else {
            return Ok(false);
        };
        Ok(!ids.is_empty())
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
            "SELECT id, codec, genome, gen_key, dim, coordinates, finite \
             FROM weight WHERE genome = $genome AND gen_key = $gen_key LIMIT 1"
        );
    }

    /// Both reads go through the key pair, so they cannot disagree about what is
    /// present.
    #[test]
    fn both_reads_key_on_the_same_pair() {
        assert_eq!(
            EXISTS.as_str(),
            "SELECT VALUE id FROM weight WHERE genome = $genome AND gen_key = $gen_key LIMIT 1"
        );
    }

    /// Key values are bound, never interpolated — they are caller-supplied
    /// strings the store places no constraints on.
    #[test]
    fn key_values_are_bound_parameters() {
        for statement in [GET.as_str(), EXISTS.as_str()] {
            assert!(statement.contains("$genome"), "{statement}");
            assert!(statement.contains("$gen_key"), "{statement}");
        }
    }

    #[test]
    fn the_write_statement_binds_its_row() {
        assert!(PUT.contains("$row"));
    }
}
