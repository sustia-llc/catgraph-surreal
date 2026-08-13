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
//! A datastore shutdown is deliberately mapped by the engine onto the *same*
//! structured discriminator as an ordinary network failure
//! (`ConnectionError::ConnectionFailed`), so the two cannot be told apart by
//! matching on error details alone. `is_shutdown()` therefore keys on the
//! message text in addition to the connection class. The upstream type documents
//! why the distinction matters: a commit refused during shutdown is usually
//! refused *before* it applied, and callers that persist failure state "must
//! therefore not record this as a permanent error".
//!
//! A plain connection failure is a *different* condition with the *same* remedy.
//! Code asking "should I reconnect and retry?" should treat both alike rather
//! than relying on `is_shutdown()` alone.

use catgraph::errors::CatgraphError;
use surrealdb::types::QueryError;
use thiserror::Error;

/// The exact message an engine reports when it refuses work during shutdown.
///
/// Reaches [`StoreError::is_shutdown`] as a substring: the key-value layer wraps
/// it as `"There was a problem with the key-value store: {this}"`.
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

    /// A write-once record was targeted by an update or delete.
    ///
    /// Store-side immutability is the primary guard; the database-side event is
    /// defence in depth. Two properties of that backing event shape detection:
    ///
    /// - The database wraps a throwing event's error as
    ///   `"Error while processing event {name}: {cause}"`, so detection keys on
    ///   the **event name** inside the wrapped text, not on the inner message.
    /// - A no-op write — one that sets a field to the value it already holds —
    ///   fires no event and raises no read-only error at all. Tests asserting
    ///   immutability must therefore write a *changed* value, or they pass
    ///   vacuously.
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
    /// Keyed on message text by necessity — the engine maps shutdown onto the
    /// same structured discriminator as an ordinary connection failure, so no
    /// structured match can separate them. See the [module
    /// documentation](self).
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        match self {
            Self::Db(e) => e.is_connection() && e.to_string().contains(SHUTDOWN_MESSAGE),
            _ => false,
        }
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

    /// Shutdown arrives as a connection-class error whose message carries the
    /// engine's shutdown text, wrapped by the key-value layer.
    #[test]
    fn shutdown_is_classified_from_connection_class_and_message() {
        let err = StoreError::Db(surrealdb::Error::connection(
            format!("There was a problem with the key-value store: {SHUTDOWN_MESSAGE}"),
            ConnectionError::ConnectionFailed,
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
            StoreError::BusGap {
                expected: 42,
                saw: 47,
            },
            StoreError::Immutable {
                table: "manifest".to_owned(),
                event: "manifest_no_update".to_owned(),
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
