//! The store's error type and its retry classifiers.
//!
//! # The three-tier retry contract
//!
//! Not every failure deserves the same response, and treating them uniformly is
//! how a store either loses writes or spins forever. [`StoreError`] exposes two
//! inherent classifiers that sort a failure into exactly one of three tiers:
//!
//! | Tier | Predicate | Correct response |
//! |---|---|---|
//! | Conflict | [`StoreError::is_conflict`] | Retry **in-process** with bounded exponential backoff + jitter. The work was never applied. |
//! | Shutdown | [`StoreError::is_shutdown`] | Reconnect, *then* retry. **Never persist this as a permanent failure.** |
//! | Everything else | neither | **No blind retry.** Surface it. |
//!
//! ## Conflict: classify the whole transaction, not just the statement
//!
//! `is_conflict()` must be applied to the **entire `begin()` … `commit()` unit**,
//! including the value returned by `commit()` itself. Conflicts are detected at
//! different points by different engines — RocksDB detects at commit time, so a
//! transaction whose every statement returned `Ok` can still fail at the commit
//! slot. Classifying only the statement results silently drops that whole class.
//!
//! Engine conflict surfaces also differ, which matters when tuning backoff:
//! the in-memory engine aborts on *read* conflicts too (so it aborts more often
//! than plain snapshot isolation), RocksDB detects at commit, and SurrealKV
//! detects write conflicts only. **Retry policy measured on the memory engine
//! does not transfer to the persistent engines** — tune against RocksDB.
//!
//! ## Shutdown: no structured discriminator exists
//!
//! A datastore shutdown has **no structured discriminator of its own** and
//! surfaces in *different error classes* depending on the path that hits it:
//!
//! - the embedded executor's commit slot reports it as a **query-class** error
//!   (`"Cannot COMMIT: …"` with `QueryError::NotExecuted`) — the shape a
//!   store transaction actually sees;
//! - RPC-handler commit paths report it as a **connection-class** error
//!   wrapped by the key-value layer.
//!
//! `is_shutdown()` therefore keys on the message text alone, across classes.
//! The upstream type documents why the distinction matters: a commit refused
//! during shutdown is usually refused *before* it applied, and callers that
//! persist failure state "must therefore not record this as a permanent
//! error".
//!
//! A plain connection failure is a *different* condition with the *same* remedy.
//! Code asking "should I reconnect and retry?" should treat both alike rather
//! than relying on `is_shutdown()` alone.

use catgraph::errors::CatgraphError;
use surrealdb::types::QueryError;
use thiserror::Error;

/// The exact message an engine reports when it refuses work during shutdown.
///
/// Reaches [`StoreError::is_shutdown`] as a substring under two wrappings: the
/// embedded commit slot's `"Cannot COMMIT: {this}"` and the key-value layer's
/// `"There was a problem with the key-value store: {this}"`.
///
/// ⚠ Mirrored from upstream wording with nothing upstream pinning it: every
/// test here fabricates the string it asserts, so an SDK rewording would keep
/// CI green while silently killing the classifier. **Re-verify this constant
/// against `surrealdb-core` on every SDK upgrade** (`rg "shutting down"`).
const SHUTDOWN_MESSAGE: &str = "The datastore is shutting down";

/// Which step of the revalidation-on-load discipline rejected a document.
///
/// Deserializing a catgraph term does **not** revalidate it — the trust boundary
/// is the store's, not serde's. Loading runs four checks in a fixed order, each
/// of which assumes its predecessors passed:
///
/// 1. JSON parse — reported as [`StoreError::Codec`], not here.
/// 2. [`Self::Depth`] — structural nesting depth, guarding against stack
///    overflow on programmatically-built terms.
/// 3. [`Self::Arity`] — arity well-formedness. **Skipping this makes the next
///    step panic rather than error**, which is why the order is fixed.
/// 4. [`Self::Check`] — re-derive the term's target word and compare it against
///    the stored signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RevalidationStage {
    /// Structural nesting depth exceeded the interpreter's recursion limit.
    Depth,
    /// The term's arities are not well-formed.
    Arity,
    /// Re-running the type check disagreed with the stored signature.
    Check,
}

impl RevalidationStage {
    /// A short, stable label for this stage, suitable for logs and messages.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Depth => "depth",
            Self::Arity => "arity",
            Self::Check => "check",
        }
    }
}

impl std::fmt::Display for RevalidationStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Everything this store can fail with.
///
/// See the [module documentation](self) for the retry contract that
/// [`Self::is_conflict`] and [`Self::is_shutdown`] implement.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StoreError {
    /// The database rejected or could not complete an operation.
    #[error("database error: {0}")]
    Db(#[from] surrealdb::Error),

    /// A catgraph structure rejected an operation — most often a term that
    /// failed to re-check on load.
    #[error("catgraph error: {0}")]
    Catgraph(#[from] CatgraphError),

    /// A stored payload could not be encoded or decoded.
    ///
    /// Also the first line of defence on load: the JSON parser's own container
    /// recursion limit fires before the depth guard gets a chance to.
    #[error("codec error: {0}")]
    Codec(#[from] serde_json::Error),

    /// A document deserialized cleanly but failed the revalidation discipline.
    ///
    /// Deserialization is not validation. A term that round-trips through serde
    /// has *not* been re-checked, and trusting it is how a corrupt or
    /// hand-edited document becomes a panic deep inside an interpreter.
    #[error("revalidation failed at the {stage} stage: {detail}")]
    Revalidation {
        /// Which check rejected the document.
        stage: RevalidationStage,
        /// What specifically was wrong.
        detail: String,
    },

    /// A stored document violates an invariant its constructor only checks
    /// under `debug_assert!`.
    ///
    /// Cospan and span constructors validate leg bounds with `debug_assert!`
    /// only, so a release build handed a corrupt document defers the failure to
    /// a panic somewhere later. The store bounds-checks on load and reports
    /// this instead.
    #[error("corrupt document in {context}: {detail}")]
    Corrupt {
        /// The structure or record the corruption was found in.
        context: String,
        /// What the invariant violation was.
        detail: String,
    },

    /// The live schema is not the one this version of the store declares.
    ///
    /// Raised by the bootstrap's own drift guard. It is not a document problem
    /// and not retryable: something outside the store changed the schema, or a
    /// database written by a different version of the store was opened by this
    /// one. Reading on regardless is how a newer document silently loses a
    /// column.
    #[error("schema drift on `{table}`: {detail}")]
    Schema {
        /// The table whose schema disagreed.
        table: String,
        /// How it disagreed.
        detail: String,
    },

    /// A stored value had a different shape than the schema promised.
    #[error("type mismatch for `{field}`: expected {expected}, found {actual}")]
    TypeMismatch {
        /// The field that disagreed.
        field: String,
        /// The type the store required.
        expected: String,
        /// The type actually found.
        actual: String,
    },

    /// The notification bus lost events: the sequence numbers are not
    /// contiguous.
    ///
    /// Change-capture retention is garbage-collected **silently** — a consumer
    /// whose cursor falls outside the retention window receives whatever
    /// survives, with no error and no gap signal from the database. A monotonic
    /// per-stream sequence number plus a contiguity assertion across catch-up
    /// is what turns that silent truncation into this error. On seeing it, a
    /// consumer must re-baseline from the durable rows rather than trust
    /// catch-up.
    #[error("notification bus gap: expected sequence {expected}, saw {saw}")]
    BusGap {
        /// The sequence number contiguity required.
        expected: u64,
        /// The sequence number actually delivered.
        saw: u64,
    },

    /// A unique index refused a write: an equivalent record is already stored
    /// under a different id.
    ///
    /// This is not a conflict and not corruption — it is the database enforcing
    /// an identity the store declared. The cospan tier raises it when a second
    /// *presentation* of an already-stored morphism is written: the two rows
    /// have different content addresses but one canonical key, and the store
    /// keeps one presentation per morphism. Recovering the address of the
    /// presentation that is already there is a read, not a retry —
    /// [`CospanStore::find_by_canon`](crate::CospanStore::find_by_canon).
    #[error(
        "`{table}` already holds an equivalent record; the unique index `{index}` refused the write"
    )]
    Duplicate {
        /// The table whose index refused the write.
        table: String,
        /// The index that refused it.
        index: String,
    },

    /// A write-once column was written with a *changed* value.
    ///
    /// Every stored column in this crate is `READONLY`, so a record's contents
    /// are fixed once written. Re-writing an identical value is a no-op and
    /// raises nothing — only a change does, which is what makes this the alarm
    /// rather than the noise.
    ///
    /// What it means depends on the tier, and the difference is worth keeping in
    /// view. For the weight tier it is *reachable by design*: a weight vector is
    /// immutable under its key, so training that wants a new vector supplies a
    /// new key. For the content-addressed tiers it is unreachable short of a
    /// digest collision, which is why those leave it as a raw database error
    /// rather than dressing it up as an expected outcome.
    #[error("`{table}.{field}` is write-once; the write changed its value")]
    ReadOnly {
        /// The table the column belongs to.
        table: String,
        /// The column that refused the change.
        field: String,
    },

    /// A write-once record was targeted by an update or delete.
    ///
    /// Store-side immutability is the primary guard; the database-side event is
    /// defence in depth. How a rejecting event's error is detected and mapped
    /// onto this variant is **decided when the write-once tables land**, pinned
    /// there by tests against the real engine — nothing here prescribes
    /// upstream message wording, deliberately (an untested transcription of
    /// upstream text is exactly the fragility [`is_shutdown`](Self::is_shutdown)
    /// has to manage, and this variant has no mitigation yet). One engine
    /// property those tests must respect: a no-op write — one that sets a field
    /// to the value it already holds — fires no event and raises no read-only
    /// error at all, so immutability tests must write a *changed* value or they
    /// pass vacuously.
    #[error("`{table}` is write-once; `{event}` rejected the write")]
    Immutable {
        /// The write-once table.
        table: String,
        /// The database event that rejected the write.
        event: String,
    },
}

impl StoreError {
    /// Whether this failure is a transaction conflict that should be retried
    /// in-process with bounded backoff.
    ///
    /// Apply this to the **whole** `begin()` … `commit()` unit — including the
    /// result of `commit()`. Some engines only detect conflicts at commit time,
    /// so a transaction whose statements all succeeded can still fail here.
    #[must_use]
    pub fn is_conflict(&self) -> bool {
        match self {
            Self::Db(e) => matches!(e.query_details(), Some(QueryError::TransactionConflict)),
            _ => false,
        }
    }

    /// Whether this failure was the datastore refusing work because it is
    /// shutting down.
    ///
    /// Such a failure is **not permanent**: reconnect and retry. Callers that
    /// persist failure state must not record it as terminal.
    ///
    /// Keyed on message text alone, deliberately across error classes: the
    /// embedded commit slot reports shutdown as a *query-class* error while
    /// RPC-handler paths report it as *connection-class*, and no structured
    /// discriminator separates either from its neighbours. Restricting to one
    /// class silently misses the other producer. See the [module
    /// documentation](self).
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        match self {
            Self::Db(e) => e.message().contains(SHUTDOWN_MESSAGE),
            _ => false,
        }
    }
}

/// Whether an error is the engine reporting that `table` is undefined.
///
/// Message-keyed by necessity — the raise carries no structured discriminator —
/// and scoped to the exact rendering for one table so it cannot swallow an
/// unrelated failure. Every repository that uses it pins it with an integration
/// test that removes a real table, which is the discipline that keeps a
/// message-keyed classifier honest.
pub(crate) fn is_missing_table(e: &surrealdb::Error, table: &str) -> bool {
    e.message()
        .contains(&format!("The table '{table}' does not exist"))
}

/// The sentinel the write guard throws when its table has vanished.
///
/// Every write runs inside a transaction that checks the table is still
/// *defined* before touching it — because a write against an undefined table
/// does not fail: the engine silently auto-creates it `TYPE ANY SCHEMALESS`,
/// permanently disarming every database-side guard while every documented
/// signal stays green. Reads absorb a vanished table as "absent"; a write must
/// be **loud** instead, because it is about to re-create the table wrong.
pub(crate) fn undefined_table_sentinel(table: &str) -> String {
    format!("catgraph-surreal: the `{table}` table is not defined; re-open the store")
}

/// Whether an error is the write guard's own sentinel for `table`.
///
/// Matches text this crate itself threw, so unlike the other classifiers it is
/// not at the mercy of upstream rewording.
pub(crate) fn is_undefined_table_guard(e: &surrealdb::Error, table: &str) -> bool {
    e.message().contains(&undefined_table_sentinel(table))
}

/// Whether an error is a unique index on `table` refusing a write.
///
/// Also message-keyed, and also pinned by an integration test that provokes a
/// real collision. It is scoped to the index by name, so an unrelated index on
/// the same table cannot be mistaken for this one.
pub(crate) fn is_duplicate(e: &surrealdb::Error, index: &str) -> bool {
    e.message()
        .contains(&format!("Database index `{index}` already contains"))
}

/// Whether an error is a `READONLY` column refusing a changed value, and if so
/// which column.
///
/// Message-keyed for the same reason as its neighbours — and **anchored to the
/// engine's whole rendering** (`` Found changed value for field `f` ``), not to
/// the bare `` field `f` `` fragment. The distinction is load-bearing: other
/// refusals echo caller-supplied values verbatim (a unique-index message quotes
/// the colliding key), so an unanchored fragment can be *steered into matching*
/// by hostile or unlucky key strings. Pinned by an integration test that
/// provokes a real refusal.
pub(crate) fn read_only_field(e: &surrealdb::Error, fields: &[&str]) -> Option<String> {
    let message = e.message();
    if !message.contains("but field is readonly") {
        return None;
    }
    fields
        .iter()
        .find(|field| message.contains(&format!("Found changed value for field `{field}`")))
        .map(|field| (*field).to_owned())
}

/// Translate a refused write into a typed error, in the one safe order.
///
/// **Duplicate is tested first, deliberately.** The unique-index refusal echoes
/// the colliding *value* — caller-supplied strings included — so a genome or
/// key containing readonly-shaped text could otherwise steer a genuine
/// collision into `ReadOnly`, whose documented remedy ("supply a new key")
/// would file the data under a fabricated identity while a foreign row keeps
/// the real one. The readonly rendering echoes no values, so this order cannot
/// misroute in the other direction.
pub(crate) fn classify_write(
    e: &surrealdb::Error,
    table: &str,
    unique_index: Option<&str>,
    readonly_fields: &[&str],
) -> Option<StoreError> {
    if let Some(index) = unique_index
        && is_duplicate(e, index)
    {
        return Some(StoreError::Duplicate {
            table: table.to_owned(),
            index: index.to_owned(),
        });
    }
    if let Some(field) = read_only_field(e, readonly_fields) {
        return Some(StoreError::ReadOnly {
            table: table.to_owned(),
            field,
        });
    }
    None
}

/// Absorb "the table does not exist" into `None` on a **read** path.
///
/// Reads treat a vanished table as absence — an absent table really does
/// contain no rows, and the schema guard is the loud detector for the
/// condition. Write paths must never use this: they go through the write guard
/// instead, which makes the same condition loud.
pub(crate) fn take_absorbing_missing_table<T>(
    result: std::result::Result<T, surrealdb::Error>,
    table: &str,
) -> Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(e) if is_missing_table(&e, table) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// The store's result alias.
pub type Result<T> = std::result::Result<T, StoreError>;

#[cfg(test)]
mod tests {
    use surrealdb::types::ConnectionError;

    use super::*;

    /// The conflict classifier keys on the structured detail, which is the only
    /// discriminator that survives the wire.
    #[test]
    fn conflict_is_classified_from_structured_details() {
        let err = StoreError::Db(surrealdb::Error::query(
            "read-write conflict".to_owned(),
            QueryError::TransactionConflict,
        ));
        assert!(err.is_conflict());
        assert!(!err.is_shutdown());
    }

    /// A query error that is *not* a conflict must not be retried as one.
    #[test]
    fn non_conflict_query_error_is_not_a_conflict() {
        let err = StoreError::Db(surrealdb::Error::query(
            "some other problem".to_owned(),
            None,
        ));
        assert!(!err.is_conflict());
    }

    /// Shutdown's RPC-handler shape: a connection-class error whose message
    /// carries the engine's shutdown text, wrapped by the key-value layer.
    #[test]
    fn shutdown_is_classified_from_connection_class_and_message() {
        let err = StoreError::Db(surrealdb::Error::connection(
            format!("There was a problem with the key-value store: {SHUTDOWN_MESSAGE}"),
            ConnectionError::ConnectionFailed,
        ));
        assert!(err.is_shutdown());
        assert!(!err.is_conflict());
    }

    /// Shutdown's embedded-commit shape: the executor reports it on the COMMIT
    /// slot as a *query-class* error (`QueryError::NotExecuted`), not a
    /// connection error. This is the shape a store transaction actually sees,
    /// and a classifier restricted to the connection class returns false for
    /// exactly the case it exists to catch.
    #[test]
    fn shutdown_is_classified_from_the_embedded_commit_slot() {
        let err = StoreError::Db(surrealdb::Error::query(
            format!("Cannot COMMIT: {SHUTDOWN_MESSAGE}"),
            QueryError::NotExecuted,
        ));
        assert!(err.is_shutdown());
        assert!(!err.is_conflict());
    }

    /// The load-bearing negative case: an ordinary connection failure shares
    /// shutdown's structured discriminator, so only the message separates them.
    /// If this test ever fails, `is_shutdown` has become indiscriminate.
    #[test]
    fn ordinary_connection_failure_is_not_shutdown() {
        let err = StoreError::Db(surrealdb::Error::connection(
            "dns lookup failed".to_owned(),
            ConnectionError::ConnectionFailed,
        ));
        assert!(!err.is_shutdown());
        assert!(!err.is_conflict());
    }

    /// Neither classifier may fire on the store's own typed variants — they
    /// describe data problems, and retrying them just repeats the failure.
    #[test]
    fn typed_variants_are_neither_conflict_nor_shutdown() {
        let cases = [
            StoreError::Revalidation {
                stage: RevalidationStage::Arity,
                detail: "generator arity 3 exceeds declared 2".to_owned(),
            },
            StoreError::Corrupt {
                context: "cospan:abc".to_owned(),
                detail: "leg index 7 out of bounds for apex length 4".to_owned(),
            },
            StoreError::TypeMismatch {
                field: "coordinates".to_owned(),
                expected: "bytes".to_owned(),
                actual: "array".to_owned(),
            },
            StoreError::Schema {
                table: "term".to_owned(),
                detail: "column set has drifted: missing [depth], unexpected []".to_owned(),
            },
            StoreError::BusGap {
                expected: 42,
                saw: 47,
            },
            StoreError::Immutable {
                table: "manifest".to_owned(),
                event: "manifest_no_update".to_owned(),
            },
            StoreError::Duplicate {
                table: "cospan".to_owned(),
                index: "cospan_canon".to_owned(),
            },
            StoreError::ReadOnly {
                table: "weight".to_owned(),
                field: "coordinates".to_owned(),
            },
        ];
        for err in cases {
            assert!(!err.is_conflict(), "{err} must not classify as a conflict");
            assert!(!err.is_shutdown(), "{err} must not classify as a shutdown");
        }
    }

    /// A gap message has to name both sequence numbers, or an operator cannot
    /// tell how much was lost.
    #[test]
    fn bus_gap_message_reports_both_sequence_numbers() {
        let err = StoreError::BusGap {
            expected: 42,
            saw: 47,
        };
        let msg = err.to_string();
        assert!(msg.contains("42"), "{msg}");
        assert!(msg.contains("47"), "{msg}");
    }

    /// The revalidation message names the stage, which is what makes a failure
    /// actionable — the four checks fail for very different reasons.
    #[test]
    fn revalidation_message_names_the_stage() {
        let err = StoreError::Revalidation {
            stage: RevalidationStage::Depth,
            detail: "nesting depth 300 exceeds limit 256".to_owned(),
        };
        assert!(err.to_string().contains("depth"), "{err}");
    }

    /// Both message-keyed classifiers must be scoped tightly enough that an
    /// unrelated failure cannot satisfy them. The integration suites pin the
    /// positive cases against a real engine; this pins the negatives.
    #[test]
    fn message_keyed_classifiers_are_scoped_to_their_subject() {
        let missing = surrealdb::Error::query("The table 'term' does not exist".to_owned(), None);
        assert!(is_missing_table(&missing, "term"));
        assert!(!is_missing_table(&missing, "cospan"));

        let duplicate = surrealdb::Error::query(
            "Database index `cospan_canon` already contains 'b3_0', with record `cospan:b3_1`"
                .to_owned(),
            None,
        );
        assert!(is_duplicate(&duplicate, "cospan_canon"));
        assert!(!is_duplicate(&duplicate, "weight_key"));
        assert!(!is_duplicate(&missing, "cospan_canon"));

        let read_only = surrealdb::Error::query(
            "Found changed value for field `coordinates`, with record `weight:b3_0`, \
             but field is readonly"
                .to_owned(),
            None,
        );
        assert_eq!(
            read_only_field(&read_only, &["dim", "coordinates"]).as_deref(),
            Some("coordinates")
        );
        // A column this caller does not own must not be claimed as one of its
        // own — the classifier reports which field, and a wrong answer is worse
        // than none.
        assert_eq!(read_only_field(&read_only, &["dim", "finite"]), None);
        assert_eq!(read_only_field(&missing, &["coordinates"]), None);
    }

    /// The catgraph bridge has to be a real `#[from]`, so `?` works across the
    /// boundary without hand-written conversions.
    #[test]
    fn catgraph_errors_convert_through_from() {
        let err: StoreError = CatgraphError::CompositionSizeMismatch {
            expected: 3,
            actual: 5,
        }
        .into();
        assert!(matches!(err, StoreError::Catgraph(_)));
        assert!(!err.is_conflict());
    }
}
