//! The generic document repositories: one mutable, one write-once.
//!
//! # Two handles, because immutability is a property of the table
//!
//! [`DocStore`] reads and writes the `document` table, whose rows can be
//! replaced. [`ManifestStore`] reads and writes the `manifest` table, whose rows
//! cannot. They share an encoding and nothing else, and that is deliberate: a
//! single handle with a mutable flag would put "can this be updated?" in a value
//! where a caller has to check it, instead of in a type where the compiler does.
//!
//! # The write-once stack, and which layer actually fires
//!
//! A manifest is protected four ways, and the layers are not interchangeable —
//! on an embedded connection with no root user configured, permissions are never
//! evaluated and `OPTION IMPORT;` can be prefixed to any query, disabling
//! `READONLY`, `ASSERT`, and events for that statement. **Store-side
//! immutability is the trust boundary; the database backs it up.**
//!
//! 1. **The API.** [`ManifestStore`] exposes no update and no delete. Nothing
//!    below matters if a caller can reach a statement that changes a row.
//! 2. **`READONLY` columns.** Enforced at every authentication level, and the
//!    layer that actually fires on an update: field processing runs *before*
//!    events, so a manifest update carrying a changed value is refused as
//!    [`StoreError::ReadOnly`], naming the column.
//! 3. **Write-once `ASSERT`s.** `$before = NONE OR $value = $before` on every
//!    column whose type evaluates it. Redundant with `READONLY` while `READONLY`
//!    is there, which is the point of defence in depth.
//! 4. **Refusal events.** A synchronous `THROW` on update and on delete, which
//!    aborts the statement *and* rolls the transaction back. Two things reach
//!    this layer and nothing else: a **delete**, which processes no fields at
//!    all so `READONLY` never sees it, and a **re-registration of an identical
//!    manifest**, where no column's value changed so no column objects. Both
//!    arrive as [`StoreError::Immutable`].
//!
//! # Registration is therefore create-only
//!
//! Registering an id that is already registered is refused, whatever the
//! contents — and *which* refusal arrives says which of them it was:
//!
//! | Second registration | Refusal |
//! |---|---|
//! | different contents | [`StoreError::ReadOnly`], naming the column that changed |
//! | identical contents | [`StoreError::Immutable`], naming the update event |
//!
//! A caller that wants to branch rather than be refused asks
//! [`ManifestStore::contains`] first.
//!
//! The engine property behind the second row is worth stating exactly, because
//! the obvious guess is wrong: **whether a no-op write counts as a modification
//! depends on the data clause.** `UPDATE … SET x = <the value it already holds>`
//! is not a modification and fires nothing. `UPSERT … CONTENT <the document it
//! already holds>` **is** one — the document is replaced wholesale, and the
//! replacement counts even when the bytes match. This tier writes `CONTENT`, so
//! its events fire on identical re-writes; `READONLY` still does not, because
//! that clause compares *values* and no value changed. Both halves are pinned by
//! integration tests, and a write-once test that re-wrote an identical document
//! expecting silence would be testing the opposite of what happens.
//!
//! # Restores are re-verified, not trusted
//!
//! Every database-side guard above is void under `OPTION IMPORT`, so a replayed
//! dump can carry rows none of them would have accepted. What survives an import
//! is the digest column, which is why [`ManifestStore::verify_all`] exists:
//! after a restore, re-derive every manifest's digest from its own payload and
//! compare. That is the check a registration record actually rests on.

use std::marker::PhantomData;
use std::sync::LazyLock;

use serde::Serialize;
use serde::de::DeserializeOwned;
use surrealdb::Surreal;
use surrealdb::engine::any::Any;
use surrealdb::types::{RecordId, RecordIdKey, SurrealValue};

use crate::doc::{self, DocumentRecord};
use crate::error::{self, Refusals, Result, StoreError};
use crate::schema::{self, DOCUMENT_FIELDS, DOCUMENT_TABLE, MANIFEST_EVENTS, MANIFEST_TABLE};
use crate::store::Store;

/// Write one document, whole.
///
/// `CONTENT` rather than `MERGE` because a document is written whole and has no
/// partial-update semantics to preserve: a payload the caller did not mention is
/// a payload the caller removed.
const PUT: &str = "UPSERT $row.id CONTENT $row RETURN NONE";

/// Read one document's columns.
///
/// The projection is explicit rather than `SELECT *`, and derived from
/// [`schema::DOCUMENT_FIELDS`] rather than written out, so it cannot drift from
/// the schema. It serves both tables, because both declare the same columns —
/// the table comes from the bound record id rather than from the statement.
static GET: LazyLock<String> =
    LazyLock::new(|| format!("SELECT {} FROM $rid", DOCUMENT_FIELDS.join(", ")));

/// Read every document of one class.
fn list_statement(table: &str) -> String {
    format!(
        "SELECT {} FROM {table} WHERE kind = $kind",
        DOCUMENT_FIELDS.join(", ")
    )
}

/// Read every row of a table.
fn scan_statement(table: &str) -> String {
    format!("SELECT {} FROM {table}", DOCUMENT_FIELDS.join(", "))
}

/// Existence, without materialising the row.
const EXISTS: &str = "RETURN record::exists($rid)";

/// Delete one document.
const DELETE: &str = "DELETE $row.id RETURN NONE";

/// One row of the `document` or `manifest` table.
///
/// `payload` is a `serde_json::Value`, which the SDK maps onto its own value
/// layer — objects to objects, arrays to arrays, integers to integers — so the
/// column stores a real nested object rather than a string a query cannot look
/// inside.
#[derive(Debug, Clone, SurrealValue)]
struct DocumentRow {
    id: RecordId,
    codec: String,
    kind: String,
    digest: String,
    payload: serde_json::Value,
}

impl DocumentRow {
    /// Consuming on purpose: the payload moves into the row rather than being
    /// cloned per write.
    fn from_record(table: &str, record: DocumentRecord) -> Self {
        Self {
            id: RecordId::new(table, record.id.as_str()),
            codec: record.codec,
            kind: record.kind,
            digest: record.digest,
            payload: record.payload,
        }
    }

    /// Turn a row into a record, recovering the caller's id from the record id.
    fn into_record(self, table: &str) -> Result<DocumentRecord> {
        let RecordIdKey::String(id) = self.id.key else {
            return Err(StoreError::Corrupt {
                context: table.to_owned(),
                detail: "record id key is not a string".to_owned(),
            });
        };
        Ok(DocumentRecord::from_columns(
            id,
            self.codec,
            self.kind,
            self.digest,
            self.payload,
        ))
    }
}

/// The row-level operations both tiers share.
///
/// Private, and generic over the table rather than over a trait: the two public
/// handles differ in which of these they expose, which is the whole design.
#[derive(Debug, Clone)]
struct Documents<T> {
    store: Store,
    table: &'static str,
    payload: PhantomData<fn() -> T>,
}

impl<T> Documents<T> {
    fn client(&self) -> &Surreal<Any> {
        self.store.client()
    }

    fn record_id(&self, id: &str) -> RecordId {
        RecordId::new(self.table, id)
    }

    async fn contains(&self, id: &str) -> Result<bool> {
        let mut response = self
            .client()
            .query(EXISTS)
            .bind(("rid", self.record_id(id)))
            .await?;
        let exists: Option<bool> = response.take(0)?;
        Ok(exists.unwrap_or(false))
    }

    async fn read(&self, id: &str) -> Result<Option<DocumentRecord>> {
        let mut response = self
            .client()
            .query(GET.as_str())
            .bind(("rid", self.record_id(id)))
            .await?;
        let Some(row) = error::take_absorbing_missing_table::<Option<DocumentRow>>(
            response.take(0),
            self.table,
        )?
        .flatten() else {
            return Ok(None);
        };
        row.into_record(self.table).map(Some)
    }

    async fn read_many(
        &self,
        statement: String,
        kind: Option<String>,
    ) -> Result<Vec<DocumentRecord>> {
        let query = self.client().query(statement);
        let query = match kind {
            Some(kind) => query.bind(("kind", kind)),
            None => query,
        };
        let mut response = query.await?;
        let Some(rows) =
            error::take_absorbing_missing_table::<Vec<DocumentRow>>(response.take(0), self.table)?
        else {
            return Ok(Vec::new());
        };
        rows.into_iter()
            .map(|row| row.into_record(self.table))
            .collect()
    }
}

/// Stores and loads consumer-shaped documents that can be replaced.
///
/// The payload type is a phantom parameter behind a function pointer, so the
/// store's own auto traits do not depend on `T`.
#[derive(Debug, Clone)]
pub struct DocStore<T> {
    inner: Documents<T>,
}

impl<T> DocStore<T> {
    /// Open the document repository: bootstrap the schema, verify it against
    /// what this build declares, and only then hand back a value that can read
    /// or write.
    ///
    /// This is the only constructor, deliberately — a write against an undefined
    /// table would make SurrealDB auto-create it `SCHEMALESS`, which for this
    /// tier would also drop the `FLEXIBLE` payload column and start discarding
    /// nested keys.
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined or does not verify.
    pub async fn open(store: Store) -> Result<Self> {
        let repository = Self {
            inner: Documents {
                store,
                table: DOCUMENT_TABLE,
                payload: PhantomData,
            },
        };
        repository.bootstrap().await?;
        Ok(repository)
    }

    /// The connection this store reads and writes through.
    #[must_use]
    pub fn store(&self) -> &Store {
        &self.inner.store
    }

    /// Define the document table and its index, then verify them.
    ///
    /// Idempotent — safe on every open.
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined, or if the resulting schema is not
    /// the one this build declares.
    pub async fn bootstrap(&self) -> Result<()> {
        schema::bootstrap_documents(self.inner.client()).await
    }

    /// Check the live schema against what this build declares, without changing
    /// anything.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Schema`] if the schema has drifted.
    pub async fn assert_schema(&self) -> Result<()> {
        schema::assert_document_schema(self.inner.client()).await
    }

    /// Whether a document is stored under this id.
    ///
    /// `record::exists()` rather than a `SELECT … LIMIT 0` probe: the two differ
    /// on exactly one case, a table that does not exist, which `SELECT` raises
    /// and `record::exists()` absorbs into `false` — the right answer, since an
    /// absent table really does contain no documents and
    /// [`Self::assert_schema`] is the dedicated, loud detector.
    ///
    /// # Errors
    ///
    /// Fails if the query cannot be run.
    pub async fn contains(&self, id: &str) -> Result<bool> {
        self.inner.contains(id).await
    }
}

impl<T: Serialize + DeserializeOwned> DocStore<T> {
    /// Store a document under a caller-supplied id, replacing whatever was
    /// there.
    ///
    /// # Errors
    ///
    /// [`StoreError::Revalidation`] if `id` is empty, [`StoreError::TypeMismatch`]
    /// if the payload does not encode to a JSON object, [`StoreError::Codec`] if
    /// it does not serialize, and a database error if the write is rejected.
    pub async fn put(&self, id: &str, kind: &str, payload: &T) -> Result<()> {
        let record = doc::encode(id, kind, payload)?;
        self.inner
            .store
            .run_write(
                &[DOCUMENT_TABLE],
                PUT,
                ("row", DocumentRow::from_record(DOCUMENT_TABLE, record)),
                Refusals::none(),
            )
            .await
    }

    /// Load a document, verifying its digest before deserializing it.
    ///
    /// Returns `None` when nothing is stored under `id` — an absent record is
    /// not an error, and neither is a table that has been removed since this
    /// store opened.
    ///
    /// # Errors
    ///
    /// Fails if the read is rejected, if the row is not shaped like a document,
    /// or if it fails revalidation. See
    /// [`DocumentRecord::revalidate`](crate::doc::DocumentRecord::revalidate).
    pub async fn get(&self, id: &str) -> Result<Option<T>> {
        match self.inner.read(id).await? {
            Some(record) => record.revalidate().map(Some),
            None => Ok(None),
        }
    }

    /// Every document of one class, revalidated.
    ///
    /// # Errors
    ///
    /// As [`Self::get`].
    pub async fn list(&self, kind: &str) -> Result<Vec<T>> {
        self.inner
            .read_many(list_statement(DOCUMENT_TABLE), Some(kind.to_owned()))
            .await?
            .iter()
            .map(DocumentRecord::revalidate)
            .collect()
    }

    /// Remove a document.
    ///
    /// Removing something that is not there is a success: the postcondition —
    /// nothing is stored under `id` — holds either way, and reporting it as a
    /// failure would make every caller write the same existence check.
    ///
    /// # Errors
    ///
    /// Fails if the write is rejected.
    pub async fn delete(&self, id: &str) -> Result<()> {
        #[derive(SurrealValue)]
        struct Target {
            id: RecordId,
        }
        self.inner
            .store
            .run_write(
                &[DOCUMENT_TABLE],
                DELETE,
                (
                    "row",
                    Target {
                        id: self.inner.record_id(id),
                    },
                ),
                Refusals::none(),
            )
            .await
    }
}

/// Stores and loads write-once, manifest-class documents.
///
/// There is no update and no delete on this handle, and that absence is the
/// primary guard rather than an omission — see the [module documentation](self).
#[derive(Debug, Clone)]
pub struct ManifestStore<T> {
    inner: Documents<T>,
}

impl<T> ManifestStore<T> {
    /// Open the manifest repository: bootstrap the schema and its refusal
    /// events, verify them against what this build declares, and only then hand
    /// back a value that can read or write.
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined or does not verify.
    pub async fn open(store: Store) -> Result<Self> {
        let repository = Self {
            inner: Documents {
                store,
                table: MANIFEST_TABLE,
                payload: PhantomData,
            },
        };
        repository.bootstrap().await?;
        Ok(repository)
    }

    /// The connection this store reads and writes through.
    #[must_use]
    pub fn store(&self) -> &Store {
        &self.inner.store
    }

    /// Define the manifest table, its guards, and its refusal events, then
    /// verify them.
    ///
    /// Idempotent — safe on every open.
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined, or if the resulting schema is not
    /// the one this build declares.
    pub async fn bootstrap(&self) -> Result<()> {
        schema::bootstrap_manifests(self.inner.client()).await
    }

    /// Check the live schema against what this build declares, without changing
    /// anything.
    ///
    /// Worth running deliberately on this table: the drift guard compares the
    /// *event* definitions too, and an event that has been removed is a
    /// write-once table that is no longer write-once while every name still
    /// looks right.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Schema`] if the schema has drifted.
    pub async fn assert_schema(&self) -> Result<()> {
        schema::assert_manifest_schema(self.inner.client()).await
    }

    /// Whether a manifest is registered under this id.
    ///
    /// # Errors
    ///
    /// Fails if the query cannot be run.
    pub async fn contains(&self, id: &str) -> Result<bool> {
        self.inner.contains(id).await
    }

    /// Re-derive every stored manifest's digest and compare it against the
    /// stored one.
    ///
    /// This is the post-restore check. Every database-side guard — `READONLY`,
    /// the write-once `ASSERT`s, the refusal events, the id format — is disabled
    /// under `OPTION IMPORT`, so a replayed dump proves nothing about the rows it
    /// carries. What it cannot forge without being noticed is the agreement
    /// between a payload and its digest, and this is where that is checked.
    ///
    /// Returns the ids of the manifests that failed, in table order, rather than
    /// stopping at the first: after a restore the useful answer is *which* rows
    /// are wrong, not that one is.
    ///
    /// # Errors
    ///
    /// Fails if the read is rejected or a row is not shaped like a manifest — a
    /// failure to *read* is not the same as a manifest that does not verify, and
    /// only the second is reported in the returned list.
    pub async fn verify_all(&self) -> Result<Vec<String>> {
        let records = self
            .inner
            .read_many(scan_statement(MANIFEST_TABLE), None)
            .await?;
        Ok(records
            .into_iter()
            .filter(|record| {
                doc::digest_of(record.payload()).is_ok_and(|digest| digest != record.digest())
            })
            .map(|record| record.id().to_owned())
            .collect())
    }
}

impl<T: Serialize + DeserializeOwned> ManifestStore<T> {
    /// Register a manifest under a caller-supplied id.
    ///
    /// **Create-only.** An id that is already registered is refused whatever the
    /// contents; the manifest that is there stays there, because the refusing
    /// event's `THROW` rolls the write back. Ask [`Self::contains`] first if the
    /// intent is to branch rather than to be refused.
    ///
    /// # Errors
    ///
    /// - [`StoreError::ReadOnly`] if a manifest is already registered under this
    ///   id with **different** contents, naming the column that changed. Field
    ///   processing runs before events, so this is the refusal a changed
    ///   registration meets.
    /// - [`StoreError::Immutable`] if the contents are **identical** — no column
    ///   changed, so no column objects, and the update event is what fires. It
    ///   is also the refusal that survives a dropped `READONLY` clause, which is
    ///   the point of having it.
    /// - [`StoreError::Revalidation`] if `id` is empty,
    ///   [`StoreError::TypeMismatch`] if the payload is not a JSON object, and a
    ///   database error otherwise.
    ///
    /// None of these is retryable. See the [module documentation](self) for the
    /// data-clause rule behind the identical-contents case.
    pub async fn register(&self, id: &str, kind: &str, payload: &T) -> Result<()> {
        let record = doc::encode(id, kind, payload)?;
        self.inner
            .store
            .run_write(
                &[MANIFEST_TABLE],
                PUT,
                ("row", DocumentRow::from_record(MANIFEST_TABLE, record)),
                Refusals::none()
                    .with_readonly_fields(&DOCUMENT_FIELDS)
                    .with_events(&MANIFEST_EVENTS),
            )
            .await
    }

    /// Load a manifest, verifying its digest before deserializing it.
    ///
    /// Returns `None` when nothing is registered under `id`.
    ///
    /// # Errors
    ///
    /// As [`DocStore::get`].
    pub async fn get(&self, id: &str) -> Result<Option<T>> {
        match self.inner.read(id).await? {
            Some(record) => record.revalidate().map(Some),
            None => Ok(None),
        }
    }

    /// Every manifest of one class, revalidated.
    ///
    /// # Errors
    ///
    /// As [`Self::get`].
    pub async fn list(&self, kind: &str) -> Result<Vec<T>> {
        self.inner
            .read_many(list_statement(MANIFEST_TABLE), Some(kind.to_owned()))
            .await?
            .iter()
            .map(DocumentRecord::revalidate)
            .collect()
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
            "SELECT id, codec, kind, digest, payload FROM $rid"
        );
        assert_eq!(
            list_statement("manifest"),
            "SELECT id, codec, kind, digest, payload FROM manifest WHERE kind = $kind"
        );
        assert_eq!(
            scan_statement("manifest"),
            "SELECT id, codec, kind, digest, payload FROM manifest"
        );
    }

    /// Caller-supplied ids ride as bound parameters, never interpolated: the
    /// store places no constraints on them beyond being non-empty strings, so
    /// they are exactly the values that must not reach query text.
    #[test]
    fn ids_are_bound_rather_than_interpolated() {
        for statement in [GET.as_str(), EXISTS, DELETE, PUT] {
            assert!(statement.contains('$'), "{statement}");
            assert!(!statement.contains('\''), "{statement}");
        }
    }
}
