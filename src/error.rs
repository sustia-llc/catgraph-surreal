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
//! It also drops it in a second, less obvious way: the structured
//! `TransactionConflict` discriminator **does not survive a client
//! transaction's `commit()`**, only the query executor's own path. See
//! [`StoreError::is_conflict`] for what that means and how both shapes are
//! recognised.
//!
//! Engine conflict surfaces also differ, which matters when tuning backoff:
//! the in-memory engine additionally aborts on many *read* conflicts (so it
//! aborts more often than plain snapshot isolation — but **its detection is
//! not exhaustive**: a single-transaction read-modify-write has been observed
//! to lose an update, so contended flows put the read and the write in
//! *separate* transactions behind a `WHERE`-guarded advance or a `CREATE`
//! collision, as this store's cursor and sequence flows do, rather than
//! resting on detection), RocksDB detects at commit against a pinned
//! snapshot, and SurrealKV detects write conflicts only. **Retry policy
//! measured on the memory engine does not transfer to the persistent
//! engines** — tune against RocksDB.
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

/// The prefix an engine puts on a transaction conflict.
///
/// Needed because the structured discriminator does **not** survive every path.
/// A conflict raised by the query executor — `db.query("BEGIN; … COMMIT;")` —
/// arrives carrying [`QueryError::TransactionConflict`]. A conflict raised by a
/// *client* transaction's `commit()` does not: that path converts the failure
/// with a plain internal-error constructor, and the kind is dropped on the way
/// out. Since a client transaction is exactly where the interesting conflicts
/// happen — RocksDB detects at commit time — a classifier keyed on the
/// structured detail alone returns `false` for most real conflicts.
///
/// ⚠ Mirrored from upstream wording, like [`SHUTDOWN_MESSAGE`] — but unlike that
/// one, this string is pinned by an integration test that provokes a **real**
/// conflict on both embedded engines. A rewording fails CI rather than silently
/// disabling every retry. **Re-verify on every SDK upgrade** all the same
/// (`rg "Transaction conflict" surrealdb-core`).
const CONFLICT_MESSAGE: &str = "Transaction conflict:";

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

    /// A value violates an invariant no constructor checked: on load, a
    /// column read off disk (leg bounds, derived sizes, content addresses hold
    /// only by re-derivation); on write, a cospan leg pointing outside its
    /// apex, or a span pair pointing outside a boundary or naming two
    /// disagreeing labels. Reported rather than handed to an interpreter that
    /// panics later.
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
    /// an identity the store declared. The cospan and span tiers raise it when a
    /// second *presentation* of an already-stored morphism is written: the two
    /// rows have different content addresses but one canonical key, and the
    /// store keeps one presentation per morphism. Recovering the address of the
    /// presentation that is already there is a read, not a retry —
    /// [`CospanStore::find_by_canon`](crate::CospanStore::find_by_canon) or
    /// [`SpanStore::find_by_canon`](crate::SpanStore::find_by_canon).
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
    /// Every stored column on a write-once table is `READONLY`, so a record's
    /// contents are fixed once written. This clause compares *values*, so
    /// re-writing an identical one raises nothing — only a change does, which is
    /// what makes this the alarm rather than the noise.
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

    /// A write-once record was targeted by an update or delete, and a database
    /// event refused it.
    ///
    /// Store-side immutability is the primary guard; the event is defence in
    /// depth. It is detected from the engine's **wrapped** rendering, keyed on
    /// the event's own name:
    ///
    /// ```text
    /// Error while processing event <name>: An error occurred: <the thrown message>
    /// ```
    ///
    /// Only the wrapper is upstream's; the thrown message is this crate's own
    /// text ([`MANIFEST_IMMUTABLE_MESSAGE`](crate::schema::MANIFEST_IMMUTABLE_MESSAGE)),
    /// and both halves are required to match. That is what keeps the classifier
    /// from being a bare transcription of upstream wording, and an integration
    /// test provokes a real refusal so a rewording fails CI rather than
    /// silently disarming it.
    ///
    /// # Which refusal you actually get
    ///
    /// On the manifest table every column is also `READONLY`, and field
    /// processing runs *before* events on the update path. So a write carrying a
    /// **changed** value is refused as [`Self::ReadOnly`], naming the column;
    /// this variant is what arrives when no column changed and the event is the
    /// only layer left to object — a delete (which processes no fields at all)
    /// or a re-write of an identical document. The sync `THROW` rolls the
    /// transaction back either way, so the record stays as it was.
    ///
    /// One engine property is worth knowing exactly, because the obvious guess
    /// is wrong: whether a no-op write counts as a modification **depends on the
    /// data clause**. `UPDATE … SET x = <the value it already holds>` is not a
    /// modification and fires nothing; `UPSERT … CONTENT <the document it
    /// already holds>` is one, and fires. `READONLY` is unaffected either way —
    /// it compares values, so an unchanged value never trips it.
    #[error("`{table}` is write-once; `{event}` rejected the write")]
    Immutable {
        /// The write-once table.
        table: String,
        /// The database event that rejected the write.
        event: String,
    },

    /// Two readers share one consumer id, and the other one moved the shared
    /// cursor while this batch was being read.
    ///
    /// A cursor is named by its consumer id, so two processes draining under one
    /// id drain *one* cursor — and each sees only what the other has not already
    /// consumed. The guarded advance detects that: the loser's `WHERE` matches
    /// nothing, and it is told so rather than winding the shared mark backwards.
    ///
    /// **The reader stays usable.** Nothing was recorded and the cursor was not
    /// written, so the reader is exactly as it was before the call; the batch it
    /// had decoded is dropped, because the other reader's advance means those
    /// events are that reader's to deliver. A retry is not the remedy — the two
    /// readers would simply race again. Give each one its own consumer id.
    ///
    /// Unlike [`Self::BusGap`] and [`Self::BusStale`] this is a configuration
    /// mistake rather than a data condition, which is why it is neither
    /// [`Self::Corrupt`] nor a re-baseline case.
    #[error(
        "the notification bus cursor for `{consumer}` moved while a batch was being read; \
         another reader shares this consumer id"
    )]
    BusRaced {
        /// The consumer id whose cursor is shared.
        consumer: String,
    },

    /// A bus consumer's cursor is old enough that change capture may already
    /// have discarded events it has not seen.
    ///
    /// Change-capture retention is garbage-collected silently: a cursor that
    /// falls outside the window receives whatever survives, with nothing
    /// distinguishing "no new events" from "the events are gone". A consumer
    /// therefore records *when* it last advanced, and catching up refuses to
    /// trust a cursor whose age has come within reach of the window.
    ///
    /// Unlike [`Self::BusGap`] this is a *prediction*, raised before anything is
    /// read, so no sequence numbers accompany it. The remedy is the same one:
    /// re-baseline from the durable rows, which are the source of truth.
    #[error(
        "notification bus cursor is {age_secs}s old against a {retention_secs}s change-capture \
         window; re-baseline from the durable rows"
    )]
    BusStale {
        /// How long ago the cursor last advanced, in seconds.
        age_secs: u64,
        /// The change-capture retention it is measured against, in seconds.
        retention_secs: u64,
    },
}

impl StoreError {
    /// Whether this failure is a transaction conflict that should be retried
    /// in-process with bounded backoff.
    ///
    /// Apply this to the **whole** `begin()` … `commit()` unit — including the
    /// result of `commit()`. Some engines only detect conflicts at commit time,
    /// so a transaction whose statements all succeeded can still fail here.
    ///
    /// # Two shapes, one condition
    ///
    /// A conflict does **not** always carry a structured discriminator, and the
    /// path where it does not is the common one:
    ///
    /// - The **query executor** propagates conflicts unwrapped, so a
    ///   `db.query("BEGIN; … COMMIT;")` failure arrives as a query-class error
    ///   whose details are [`QueryError::TransactionConflict`].
    /// - A **client transaction**'s `commit()` converts the failure with a plain
    ///   internal-error constructor, which discards the kind. All that survives
    ///   is the message.
    ///
    /// Both are checked. Restricting to the structured detail would classify
    /// almost no real conflicts, because a client transaction is where they
    /// happen — RocksDB detects at commit time, so its conflicts arrive by the
    /// second route exclusively.
    ///
    /// # A third shape, which is this crate's own
    ///
    /// A bus publish allocates its sequence number *before* composing its
    /// transaction — the event's record id is the digest of `(stream, seq)` —
    /// and the transaction re-reads the counter and refuses to proceed if it
    /// moved. Losing that race is optimistic contention in the same sense the
    /// engine's own conflicts are: nothing was applied, another writer got
    /// there first, and re-running is both safe and the correct response. The
    /// publish absorbs a bounded number of those internally; past that budget it
    /// surfaces, and it surfaces as a **conflict** so the documented retry
    /// contract covers exactly the transient contention it exists for.
    ///
    /// The text it is keyed on is a crate-owned sentinel this store throws
    /// itself, not upstream wording it transcribes, so unlike its two
    /// neighbours this arm cannot be disarmed by an SDK rewording.
    #[must_use]
    pub fn is_conflict(&self) -> bool {
        match self {
            Self::Db(e) => {
                matches!(e.query_details(), Some(QueryError::TransactionConflict))
                    || e.message().contains(CONFLICT_MESSAGE)
                    || e.message().contains(STALE_SEQUENCE_SENTINEL)
            }
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

/// The sentinel a bus publish throws when the sequence number it prepared for is
/// no longer the one the stream would hand out.
///
/// A publish has to know its sequence number *before* the transaction runs,
/// because the event's record id is derived from it — so the transaction
/// re-reads the allocator and refuses to proceed if the two disagree. Without
/// that check a publisher whose read was overtaken between the read and the
/// transaction would file an event under one number carrying another, and the
/// mismatch would only surface much later, as a corrupt row.
///
/// Also crate-owned text, for the same reason as the table guard: this is the
/// store detecting its own condition, not transcribing upstream's.
/// It carries no apostrophe and no semicolon, deliberately: it is formatted into
/// query text, where a quote would make the surrounding statement unreadable at
/// best and a semicolon would look like a statement boundary that is not one.
///
/// It is also what [`StoreError::is_conflict`] keys its third arm on: losing
/// this race is optimistic contention, so a publish that exhausts its internal
/// re-preparations must land in the retryable tier rather than outside every
/// classifier.
pub(crate) const STALE_SEQUENCE_SENTINEL: &str = "catgraph-surreal: the next sequence number for this stream moved, so the prepared publish \
     is stale";

/// Whether an error is a publish reporting that its prepared sequence number
/// went stale.
pub(crate) fn is_stale_sequence(error: &StoreError) -> bool {
    match error {
        StoreError::Db(e) => e.message().contains(STALE_SEQUENCE_SENTINEL),
        _ => false,
    }
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

/// Whether an error is a write-once event refusing a write, and if so which
/// event.
///
/// Keyed on **both** halves of the rendering: upstream's wrapper, which names
/// the event, and this crate's own thrown message. Requiring the crate-owned
/// half is what separates this from a bare transcription of upstream text — a
/// foreign event that happened to fire on the same table cannot satisfy it, and
/// neither can a message echoing caller-supplied values. Pinned by an
/// integration test that provokes a real refusal.
pub(crate) fn immutable_event(e: &surrealdb::Error, events: &[&str]) -> Option<String> {
    let message = e.message();
    if !message.contains(crate::schema::MANIFEST_IMMUTABLE_MESSAGE) {
        return None;
    }
    events
        .iter()
        .find(|event| message.contains(&format!("Error while processing event {event}:")))
        .map(|event| (*event).to_owned())
}

/// What a table's write may be refused with, beyond a plain database error.
///
/// Passed to the guarded executor so a tier declares its refusal surfaces once,
/// beside its schema, rather than re-deriving them at each call site.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Refusals {
    /// The unique index whose collisions become [`StoreError::Duplicate`].
    pub unique_index: Option<&'static str>,
    /// The `READONLY` columns whose refusals become [`StoreError::ReadOnly`].
    pub readonly_fields: &'static [&'static str],
    /// The write-once events whose refusals become [`StoreError::Immutable`].
    pub events: &'static [&'static str],
}

impl Refusals {
    /// A tier that classifies nothing: every refusal surfaces as the raw
    /// database error it is.
    ///
    /// The right choice where a classified refusal would be *unreachable* rather
    /// than merely unhandled — on a content-addressed table a read-only refusal
    /// would mean two distinct encodings produced one address, which deserves
    /// the raw error rather than a variant implying a routine outcome.
    pub(crate) const fn none() -> Self {
        Self {
            unique_index: None,
            readonly_fields: &[],
            events: &[],
        }
    }

    /// Also classify collisions on `index`.
    pub(crate) const fn with_unique_index(mut self, index: &'static str) -> Self {
        self.unique_index = Some(index);
        self
    }

    /// Also classify `READONLY` refusals naming one of `fields`.
    pub(crate) const fn with_readonly_fields(mut self, fields: &'static [&'static str]) -> Self {
        self.readonly_fields = fields;
        self
    }

    /// Also classify refusals thrown by one of `events`.
    pub(crate) const fn with_events(mut self, events: &'static [&'static str]) -> Self {
        self.events = events;
        self
    }
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
///
/// The event refusal is tested next. Its rendering is half crate-owned text, so
/// it cannot be reached by an echoed value either, and testing it before
/// `ReadOnly` keeps the two orders from mattering: on a delete only the event
/// can fire, and on an update field processing runs first, so a write never
/// produces both.
pub(crate) fn classify_write(
    e: &surrealdb::Error,
    table: &str,
    refusals: Refusals,
) -> Option<StoreError> {
    if let Some(index) = refusals.unique_index
        && is_duplicate(e, index)
    {
        return Some(StoreError::Duplicate {
            table: table.to_owned(),
            index: index.to_owned(),
        });
    }
    if let Some(event) = immutable_event(e, refusals.events) {
        return Some(StoreError::Immutable {
            table: table.to_owned(),
            event,
        });
    }
    if let Some(field) = read_only_field(e, refusals.readonly_fields) {
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

    /// The shape a *client* transaction's `commit()` actually produces: the
    /// structured kind is discarded on that path, so only the message is left.
    /// A classifier keyed on the discriminator alone returns `false` for exactly
    /// the conflicts a store meets in practice.
    #[test]
    fn conflict_is_classified_from_the_client_transaction_commit_shape() {
        let err = StoreError::Db(surrealdb::Error::internal(
            "Transaction conflict: Write conflict, retry the transaction. This transaction can \
             be retried"
                .to_owned(),
        ));
        assert!(err.query_details_are_absent());
        assert!(err.is_conflict());
        assert!(!err.is_shutdown());
    }

    /// A publish that exhausted its internal re-preparations is reporting
    /// optimistic contention, and the retry contract has to cover it: before
    /// this arm existed the sentinel satisfied neither classifier, so the one
    /// failure the bus's own documentation tells a caller to retry was the one
    /// [`is_retryable`](crate::retry::is_retryable) refused.
    #[test]
    fn an_exhausted_publish_preparation_is_a_conflict() {
        let err = StoreError::Db(surrealdb::Error::query(
            format!("An error occurred: {STALE_SEQUENCE_SENTINEL}"),
            None,
        ));
        assert!(err.query_details_are_absent());
        assert!(err.is_conflict());
        assert!(!err.is_shutdown());
    }

    /// The load-bearing negative: an ordinary internal error must not be swept
    /// into the retry loop by the message check.
    #[test]
    fn an_ordinary_internal_error_is_not_a_conflict() {
        let err = StoreError::Db(surrealdb::Error::internal(
            "something went wrong".to_owned(),
        ));
        assert!(!err.is_conflict());
        assert!(!err.is_shutdown());
    }

    impl StoreError {
        /// Test helper: assert the structured discriminator really is absent, so
        /// the case above is testing the message path rather than accidentally
        /// passing through the structured one.
        fn query_details_are_absent(&self) -> bool {
            match self {
                Self::Db(e) => e.query_details().is_none(),
                _ => true,
            }
        }
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
            StoreError::BusStale {
                age_secs: 90,
                retention_secs: 100,
            },
            StoreError::BusRaced {
                consumer: "worker-1".to_owned(),
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

    /// The write-once classifier needs *both* halves: upstream's wrapper naming
    /// the event, and this crate's own thrown message. Either alone is a
    /// coincidence, and treating a coincidence as immutability would report a
    /// refusal that never happened.
    #[test]
    fn the_write_once_classifier_needs_the_wrapper_and_the_crate_owned_message() {
        let refusal = surrealdb::Error::query(
            format!(
                "Error while processing event manifest_no_delete: An error occurred: {}",
                crate::schema::MANIFEST_IMMUTABLE_MESSAGE
            ),
            None,
        );
        assert_eq!(
            immutable_event(&refusal, &["manifest_no_update", "manifest_no_delete"]).as_deref(),
            Some("manifest_no_delete")
        );
        // An event this caller does not own must not be claimed as one of its
        // own — the classifier reports which event, and a wrong answer is worse
        // than none.
        assert_eq!(immutable_event(&refusal, &["manifest_no_update"]), None);

        // Upstream's wrapper alone: some other event on the same table failed
        // for some other reason, which is not immutability.
        let foreign = surrealdb::Error::query(
            "Error while processing event manifest_no_delete: An error occurred: something else"
                .to_owned(),
            None,
        );
        assert_eq!(immutable_event(&foreign, &["manifest_no_delete"]), None);

        // The crate-owned message alone, echoed by some other failure: no event
        // refused anything.
        let echoed = surrealdb::Error::query(
            format!(
                "Database index `x` already contains '{}'",
                crate::schema::MANIFEST_IMMUTABLE_MESSAGE
            ),
            None,
        );
        assert_eq!(immutable_event(&echoed, &["manifest_no_delete"]), None);
    }

    /// The refusal order is the one safe order: a unique-index message echoes
    /// caller-supplied values, so it is decided first, and neither of the other
    /// two can be reached by an echoed value.
    #[test]
    fn refusal_classification_prefers_the_index_message_it_cannot_trust() {
        let refusals = Refusals::none()
            .with_unique_index("weight_key")
            .with_readonly_fields(&["coordinates"])
            .with_events(&["manifest_no_delete"]);

        // A key that spells out a readonly refusal must still be classified as
        // the collision it is.
        let hostile = surrealdb::Error::query(
            "Database index `weight_key` already contains 'Found changed value for field \
             `coordinates`, with record `weight:x`, but field is readonly', with record \
             `weight:y`"
                .to_owned(),
            None,
        );
        assert!(matches!(
            classify_write(&hostile, "weight", refusals),
            Some(StoreError::Duplicate { .. })
        ));

        let read_only = surrealdb::Error::query(
            "Found changed value for field `coordinates`, with record `weight:x`, but field is \
             readonly"
                .to_owned(),
            None,
        );
        assert!(matches!(
            classify_write(&read_only, "weight", refusals),
            Some(StoreError::ReadOnly { .. })
        ));

        let unrelated = surrealdb::Error::query("something else entirely".to_owned(), None);
        assert!(classify_write(&unrelated, "weight", refusals).is_none());
        assert!(classify_write(&read_only, "weight", Refusals::none()).is_none());
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
