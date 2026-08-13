//! The notification bus: a durable publisher and a catch-up reader.
//!
//! # Two tiers, and how to choose
//!
//! Both tiers write the same durable rows. They differ only in how a consumer
//! learns there is something to read:
//!
//! - **Tier 1 — low volume, latency sensitive.** Goal signals, stopping
//!   decisions, generation boundaries. Subscribe with [`BusReader::subscribe`]
//!   for the wakeup, and read the durable rows with [`BusReader::next_batch`]
//!   when it fires. The subscription buys latency; the rows are still what is
//!   read.
//! - **Tier 2 — high volume.** Per-item progress, per-generation statistics.
//!   Poll [`BusReader::next_batch`] on an interval and do **not** subscribe. A
//!   live subscription on a hot table is a notification per row through a
//!   per-notification task, and every one of those is wasted work for a consumer
//!   that was going to poll anyway.
//!
//! What is never correct is a live-only consumer of either tier. Delivery is
//! best-effort and at-most-once, with no replay and no gap signal, so a listener
//! that never reads rows loses events at every disconnect and under backpressure
//! — quietly.
//!
//! # This module spawns nothing
//!
//! [`BusReader::subscribe`] hands the caller a `Stream`. The caller owns the
//! loop, owns cancellation, and owns ending the subscription — which is done by
//! **dropping the stream**. That is not a stylistic preference: on an embedded
//! engine a raw `KILL` does not end a stream, so dropping it is the mechanism.
//!
//! (The connection itself is not inert — opening a local engine starts a dozen
//! or so background maintenance tasks in-process — but nothing in this module
//! adds to them.)
//!
//! # Sequence allocation, and the fence that makes it safe
//!
//! A publish is one transaction over three statements: read the stream's
//! counter, write it back incremented, create the event at the number that was
//! read. The write-back is the point. Snapshot isolation does not prevent write
//! skew, so two publishers that only *read* the counter could both write event 5;
//! because both also write the counter row, they collide on it and one is
//! refused with a transaction conflict — which is a detectable, retryable
//! outcome instead of two events sharing a number.
//!
//! Behind that sits a second, independent guard: the event's record id is the
//! digest of `(stream, seq)`, and the event is written with `CREATE`, which
//! refuses an existing id. If a duplicate number ever escaped the fence, the
//! write fails rather than overwriting.
//!
//! Deriving the id from the number has one consequence worth stating, because it
//! is the kind of thing that otherwise surfaces months later as an unexplainable
//! corrupt row: the number has to be known *before* the transaction is composed.
//! So a publish reads the counter, prepares an id, and then has the transaction
//! **re-read the counter and refuse to proceed if it moved** — a publisher whose
//! read was overtaken would otherwise file an event under one number while the
//! row carried another. Losing that race is ordinary under contention, so
//! `publish` re-prepares (with a small jittered backoff) and tries again rather
//! than reporting it. Only when that budget runs out does the condition reach
//! the caller — and it reaches them classified as a **conflict**, because that
//! is what it is: nothing was applied and another writer got there first.
//! [`retry`](mod@crate::retry) therefore covers it exactly as it covers the
//! engine's own conflicts.
//!
//! **Publishing is at-least-once under retry.** A commit that succeeded but was
//! reported as failed will be retried, and the retry allocates a *new* number —
//! so the payload appears twice under two sequence numbers. Consumers that
//! cannot tolerate that should carry their own idempotency key in the payload.
//!
//! # Catch-up, and the two ways it can go wrong
//!
//! [`BusReader::next_batch`] reads the change feed from the consumer's persisted
//! high-water mark, pages until the feed is exhausted, and asserts that each
//! stream's sequence numbers are contiguous with what it has already seen.
//!
//! - A **hole** in the sequence means events were lost — retention expired under
//!   the cursor, or a row was removed. That is [`StoreError::BusGap`].
//! - A cursor **older than the retention window** means events may already be
//!   gone before anything is read. Expiry is silent, so this is predicted rather
//!   than detected: that is [`StoreError::BusStale`].
//!
//! Both have the same remedy — [`BusReader::rebaseline`], which reads the
//! durable rows and starts again from what is actually there. A restore is the
//! third case with the same remedy: an import emits no change-feed entries and
//! no notifications at all, so a consumer's cursor means nothing afterwards.
//!
//! A third failure is neither of those and has a different remedy:
//! [`StoreError::BusRaced`] means a *second reader under the same consumer id*
//! moved the shared cursor first. Re-baselining would not help and neither would
//! retrying — the two readers would race again. Consumer ids name cursors, so
//! independent readers need distinct ones.
//!
//! Both the cursor's position and its per-stream expectations live on the cursor
//! row and are written together, which is what lets a restarted reader resume
//! detection rather than start over. A reader that kept its expectations in
//! memory alone would treat every stream as newly sighted after a restart, and a
//! hole punched while it was down would pass unremarked.

use std::sync::LazyLock;
use std::time::Duration;

use serde::Serialize;
use surrealdb::Notification;
use surrealdb::method::QueryStream;
use surrealdb::types::{RecordId, SurrealValue, Value};

use crate::bus::{self, BusEvent};
use crate::error::{self, Refusals, Result, StoreError};
use crate::schema::{
    self, BUS_CHANGEFEED_RETENTION, BUS_FIELDS, BUS_MARK_TABLE, BUS_SEQ_TABLE, BUS_TABLE,
};
use crate::store::Store;

/// Publish one event: re-read, fence, allocate, create.
///
/// Five statements inside the guarded transaction the shared executor composes,
/// and three of them are load-bearing:
///
/// - The **`IF … THROW`** is the fence against a stale preparation. An event's
///   record id is the digest of `(stream, seq)`, so the publisher has to know
///   the number before the transaction runs — and between reading it and the
///   transaction's snapshot, another publisher may have taken it. Re-reading
///   inside the transaction and refusing to proceed on a disagreement is what
///   stops an event being filed under one number while carrying another.
/// - The **`UPSERT` of the allocator** is the write-skew fence against
///   concurrency. Snapshot isolation would happily let two transactions read
///   the same counter; because both also *write* it, they collide on that row
///   and one is refused as a transaction conflict.
/// - **`CREATE`, not `UPSERT`,** for the event. If a number ever escaped both
///   fences, the second event to claim it must fail rather than overwrite the
///   first.
///
/// The trailing `RETURN` hands back the number that was allocated, which is
/// decided here and nowhere else.
static PUBLISH: LazyLock<String> = LazyLock::new(|| {
    format!(
        "LET $seq = (SELECT VALUE next_seq FROM ONLY $row.allocator) ?? 0; \
         IF $seq != $row.expected {{ THROW \"{sentinel}\" }}; \
         UPSERT $row.allocator SET stream = $row.stream, next_seq = $seq + 1; \
         CREATE $row.id CONTENT {{ codec: $row.codec, stream: $row.stream, seq: $seq, \
         payload: $row.payload }} RETURN NONE; \
         RETURN $seq",
        sentinel = error::STALE_SEQUENCE_SENTINEL
    )
});

/// The tables [`PUBLISH`] writes, in the order the guard checks them.
///
/// **Both**, because the statement writes both: the event row and the stream's
/// sequence allocator. Guarding the event table alone would leave `bus_seq`
/// exposed to the condition the guard exists to close — a `REMOVE TABLE bus_seq`
/// followed by a publish would silently re-create it `TYPE ANY SCHEMALESS`, and
/// the allocator is the one row that makes sequence numbers a total order rather
/// than a hope.
///
/// The event table leads, so it is the one a classified refusal names.
const PUBLISH_TABLES: &[&str] = &[BUS_TABLE, BUS_SEQ_TABLE];

/// The slot the publish statement's `RETURN $seq` lands in.
///
/// Four statements precede it, and the guard wrapper occupies the slots before
/// those — one per guarded table, plus `BEGIN`. See
/// [`Store::first_statement_slot`].
const PUBLISH_SEQ_SLOT: usize = Store::first_statement_slot(PUBLISH_TABLES.len()) + 4;

/// How many times a publish re-prepares after losing its sequence number to
/// another publisher.
///
/// This is not the transaction-conflict retry — that one is the caller's, and
/// re-runs the whole call. This is the store re-reading a counter that moved
/// between two of its own statements, which is a normal outcome under
/// contention and not something a caller should have to know about.
///
/// Exhausting the budget is *not* a dead end: the sentinel classifies as a
/// conflict (see [`StoreError::is_conflict`]), so the caller's own
/// [`retry`](mod@crate::retry) loop covers it like any other contention.
const PUBLISH_PREPARATIONS: u32 = 8;

/// The backoff between two of a publish's own re-preparations.
///
/// Small — this is a re-read of one row, not a round of a distributed protocol —
/// but not zero, and jittered. Two publishers that lost the same race would
/// otherwise re-read the counter in lockstep and lose it again on the same
/// schedule, which turns a budget of eight attempts into eight copies of one
/// attempt.
static PUBLISH_BACKOFF: LazyLock<crate::retry::RetryPolicy> = LazyLock::new(|| {
    crate::retry::RetryPolicy::new()
        .base_delay(Duration::from_micros(200))
        .max_delay(Duration::from_millis(5))
});

/// The table the cursor statements write.
const CURSOR_TABLES: &[&str] = &[BUS_MARK_TABLE];

/// Create the consumer's cursor if it is not already there, then read it.
///
/// `UPSERT … SET` rather than `CONTENT` so that an existing cursor is not reset:
/// the write only names the consumer, leaving a mark that is already there
/// alone. A fresh cursor starts at versionstamp zero, which reads the feed from
/// the beginning, and with no streams seen.
const OPEN_CURSOR: &str = "\
UPSERT $row.cursor SET \
    consumer = $row.consumer, \
    versionstamp = versionstamp ?? 0, \
    stamped_at = stamped_at ?? time::now(), \
    seen = seen ?? '[]' \
RETURN NONE; \
SELECT versionstamp, seen, duration::secs(time::now() - stamped_at) AS age \
FROM ONLY $row.cursor";

/// The slot [`OPEN_CURSOR`]'s read lands in.
const OPEN_CURSOR_SLOT: usize = Store::first_statement_slot(CURSOR_TABLES.len()) + 1;

/// Stamp the cursor, but only from the versionstamp it is expected to hold.
///
/// The `WHERE` clause is the second write-skew fence: a reader whose cursor
/// moved underneath it — a second process draining the same consumer id — writes
/// nothing and finds out, rather than winding the shared mark backwards.
///
/// One statement covers both jobs a cursor write has, and deliberately. Moving
/// the mark forward is the obvious one. The other is **renewing freshness on a
/// poll that found nothing**: `stamped_at` is what the staleness prediction is
/// measured against, so a reader that only wrote it when the feed had moved
/// would age past the window on a quiet bus while doing everything right — and
/// then be told to re-baseline, which discards the first event to arrive. A poll
/// that drained the feed to its end has genuinely re-established the cursor's
/// freshness whether or not the end moved, and writing that down is one row
/// update.
const STAMP_CURSOR: &str = "\
UPDATE $row.cursor SET \
    versionstamp = $row.next, stamped_at = time::now(), seen = $row.seen \
WHERE versionstamp = $row.expected RETURN AFTER";

/// The slot [`STAMP_CURSOR`]'s result lands in.
const STAMP_CURSOR_SLOT: usize = Store::first_statement_slot(CURSOR_TABLES.len());

/// Read the highest sequence number on every stream, straight from the durable
/// rows.
///
/// This is what re-baselining reads. It is a `GROUP BY` over the compound index's
/// leading column rather than a scan per stream.
static HIGH_WATER_ROWS: LazyLock<String> = LazyLock::new(|| {
    format!("SELECT stream, math::max(seq) AS seq FROM {BUS_TABLE} GROUP BY stream")
});

/// Read a stream's events from a sequence number onwards, in order.
static ROWS_SINCE: LazyLock<String> = LazyLock::new(|| {
    format!(
        "SELECT {} FROM {BUS_TABLE} WHERE stream = $stream AND seq >= $seq ORDER BY seq",
        BUS_FIELDS.join(", ")
    )
});

/// The live subscription's projection.
///
/// Explicit rather than `SELECT *`, matching every other read in this crate.
static LIVE_SELECT: LazyLock<String> =
    LazyLock::new(|| format!("LIVE SELECT {} FROM {BUS_TABLE}", BUS_FIELDS.join(", ")));

/// The stream a subscription hands back.
///
/// A type alias rather than an opaque wrapper: the caller owns the loop, and
/// wrapping the SDK's stream would hide the `Drop` that ends the subscription.
pub type BusStream = QueryStream<Notification<BusEvent>>;

/// Everything one publish binds, as a single parameter.
#[derive(Debug, Clone, SurrealValue)]
struct PublishWrite {
    id: RecordId,
    allocator: RecordId,
    codec: String,
    stream: String,
    /// The sequence number the record id was derived from. The transaction
    /// re-reads the allocator and refuses to proceed unless it still agrees.
    expected: i64,
    payload: serde_json::Value,
}

/// What [`OPEN_CURSOR`] binds.
#[derive(Debug, Clone, SurrealValue)]
struct CursorOpen {
    cursor: RecordId,
    consumer: String,
}

/// What [`STAMP_CURSOR`] binds.
#[derive(Debug, Clone, SurrealValue)]
struct CursorStamp {
    cursor: RecordId,
    /// The versionstamp the cursor must still hold for the write to land.
    expected: i64,
    /// The versionstamp to leave it at — equal to `expected` on a poll that
    /// found nothing, which renews freshness without moving the mark.
    next: i64,
    /// The per-stream expectations, as the opaque JSON string the column holds.
    seen: String,
}

/// The cursor as it reads back.
#[derive(Debug, Clone, SurrealValue)]
struct CursorRow {
    versionstamp: i64,
    seen: String,
    age: i64,
}

/// Publishes events to the durable bus.
///
/// Cheap to clone, and clones share the connection. Two writers publishing to
/// one stream contend on that stream's allocator row, which is what makes their
/// sequence numbers a total order rather than a hope.
#[derive(Debug, Clone)]
pub struct BusWriter {
    store: Store,
    retention: Duration,
}

impl BusWriter {
    /// Open the bus for publishing, with the default change-capture retention.
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined or does not verify.
    pub async fn open(store: Store) -> Result<Self> {
        Self::with_retention(store, BUS_CHANGEFEED_RETENTION).await
    }

    /// Open the bus for publishing, choosing the change-capture retention.
    ///
    /// The retention is part of the compared schema, so every handle onto one
    /// database must agree on it — opening with a different window is drift, not
    /// a silent re-definition. That is deliberate: a shortened window is exactly
    /// the change that turns a reader's catch-up into silent loss.
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined or does not verify — including
    /// against a retention some other handle already established.
    pub async fn with_retention(store: Store, retention: Duration) -> Result<Self> {
        let writer = Self { store, retention };
        writer.bootstrap().await?;
        Ok(writer)
    }

    /// The connection this writer publishes through.
    #[must_use]
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// The change-capture retention this writer established.
    #[must_use]
    pub fn retention(&self) -> Duration {
        self.retention
    }

    /// Define the bus tables and verify them. Idempotent — safe on every open.
    ///
    /// # Errors
    ///
    /// Fails if a schema cannot be defined, or if the result is not the one this
    /// build declares.
    pub async fn bootstrap(&self) -> Result<()> {
        schema::bootstrap_bus(self.store.client(), self.retention).await
    }

    /// Check the live schemas against what this build declares.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Schema`] if any of them has drifted.
    pub async fn assert_schema(&self) -> Result<()> {
        schema::assert_bus_schema(self.store.client(), self.retention).await
    }

    /// Publish a payload to a stream, returning the sequence number it was
    /// assigned.
    ///
    /// One transaction: allocate from the stream's counter, write the counter
    /// back, create the event. The counter write-back is the fence that makes
    /// two concurrent publishes a conflict rather than a collision.
    ///
    /// # Errors
    ///
    /// - [`StoreError::Revalidation`] if `stream` is empty, and
    ///   [`StoreError::TypeMismatch`] if the payload does not encode to a JSON
    ///   object. Both are checked before the transaction opens.
    /// - A **transaction conflict** if another publisher allocated from the same
    ///   stream concurrently. Retry the whole call — see [`retry`](mod@crate::retry). The
    ///   retry allocates a new number, which is why publishing is at-least-once.
    ///   A publish that exhausted its own re-preparations classifies as a
    ///   conflict too, for the same reason and with the same remedy.
    /// - A database error otherwise.
    pub async fn publish<T: Serialize>(&self, stream: &str, payload: &T) -> Result<i64> {
        let pending = bus::prepare(stream, payload)?;
        let allocator = RecordId::new(BUS_SEQ_TABLE, bus::allocator_key(pending.stream()));

        // The event's record id is the digest of `(stream, seq)`, so the number
        // has to be known before the transaction can be composed. Reading it
        // here and re-checking it inside the transaction is what keeps the id
        // and the number in agreement; losing the race is a normal outcome, so
        // it is re-prepared rather than reported.
        let mut attempt = 0;
        loop {
            let expected = self.next_sequence(&allocator).await?;
            match self.publish_at(&allocator, &pending, expected).await {
                Ok(seq) => return Ok(seq),
                Err(e) if error::is_stale_sequence(&e) && attempt + 1 < PUBLISH_PREPARATIONS => {
                    attempt += 1;
                    // Jittered, so two publishers that lost the same race do not
                    // come back for the counter together.
                    crate::retry::pause(PUBLISH_BACKOFF.delay_before(attempt)).await;
                }
                // Out of re-preparations: this is contention, and saying so is
                // what puts it in the caller's retry tier rather than outside
                // every classifier.
                Err(e) => return Err(e),
            }
        }
    }

    /// One attempt at publishing, prepared for the sequence number `expected`.
    async fn publish_at(
        &self,
        allocator: &RecordId,
        pending: &bus::PendingEvent,
        expected: i64,
    ) -> Result<i64> {
        let write = PublishWrite {
            id: RecordId::new(
                BUS_TABLE,
                bus::event_address(pending.stream(), expected).as_str(),
            ),
            allocator: allocator.clone(),
            codec: bus::BUS_CODEC.to_owned(),
            stream: pending.stream().to_owned(),
            expected,
            payload: pending.payload().clone(),
        };
        let mut response = self
            .store
            .run_write_response(
                PUBLISH_TABLES,
                PUBLISH.as_str(),
                ("row", write),
                Refusals::none().with_readonly_fields(&BUS_FIELDS),
            )
            .await?;
        let assigned: Option<i64> = response.take(PUBLISH_SEQ_SLOT)?;
        assigned.ok_or_else(|| StoreError::Corrupt {
            context: BUS_TABLE.to_owned(),
            detail: "the publish transaction returned no sequence number".to_owned(),
        })
    }

    /// The number the next publish to this stream would be assigned.
    async fn next_sequence(&self, allocator: &RecordId) -> Result<i64> {
        let mut response = self
            .store
            .client()
            .query("SELECT VALUE next_seq FROM ONLY $rid")
            .bind(("rid", allocator.clone()))
            .await?;
        let next =
            error::take_absorbing_missing_table::<Option<i64>>(response.take(0), BUS_SEQ_TABLE)?
                .flatten();
        Ok(next.unwrap_or(0))
    }
}

/// Reads the durable bus: catch-up from a persisted cursor, and a live wakeup.
///
/// A reader keeps the per-stream sequence numbers it has already seen, which is
/// what lets it assert contiguity across batches. Those expectations live **on
/// the cursor row**, beside the versionstamp, and are loaded at construction:
/// held in memory alone they would reset at every reader restart, and a hole
/// punched while a consumer was down would then pass as a first sighting. The
/// two halves of the mark are written together, so a restarted reader resumes
/// with both or neither.
///
/// A stream's *first* sighting still seeds rather than asserts — there is
/// nothing to be contiguous with — so a gap that opened before any reader
/// existed at all is caught by the cursor's age rather than by its sequence
/// numbers.
#[derive(Debug)]
pub struct BusReader {
    store: Store,
    consumer: String,
    cursor: RecordId,
    retention: Duration,
    versionstamp: i64,
    seen: Vec<(String, i64)>,
}

impl BusReader {
    /// How many change-feed entries one page asks for.
    ///
    /// The engine caps this at 1000 whatever is asked, and defaults to 100 when
    /// nothing is — so catch-up pages, always, rather than assuming one call
    /// drains the feed.
    const PAGE: u32 = 1000;

    /// The fraction of the retention window at which a cursor is declared stale.
    ///
    /// Not the whole window: expiry is silent and collection is periodic, so a
    /// cursor at 100% of the window has already lost the race. Three quarters
    /// leaves room to notice.
    const STALE_AT: u32 = 3;
    /// The denominator of [`Self::STALE_AT`].
    const STALE_OF: u32 = 4;

    /// Open the bus for reading under a consumer id, with the default retention.
    ///
    /// The consumer id names the cursor, so two processes sharing one id share a
    /// cursor — and each will see only what the other has not already consumed.
    /// Independent consumers need distinct ids.
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined or does not verify, or if the
    /// cursor cannot be read.
    pub async fn open(store: Store, consumer: &str) -> Result<Self> {
        Self::with_retention(store, consumer, BUS_CHANGEFEED_RETENTION).await
    }

    /// Open the bus for reading, choosing the change-capture retention.
    ///
    /// Must agree with the retention the publisher established — see
    /// [`BusWriter::with_retention`].
    ///
    /// # Errors
    ///
    /// Fails if the schema cannot be defined or does not verify, if `consumer`
    /// is empty, or if the cursor cannot be read.
    pub async fn with_retention(store: Store, consumer: &str, retention: Duration) -> Result<Self> {
        if consumer.is_empty() {
            return Err(StoreError::Revalidation {
                stage: error::RevalidationStage::Check,
                detail: "a consumer id may not be empty".to_owned(),
            });
        }
        schema::bootstrap_bus(store.client(), retention).await?;
        let cursor = RecordId::new(BUS_MARK_TABLE, bus::cursor_key(consumer));
        let mut reader = Self {
            store,
            consumer: consumer.to_owned(),
            cursor,
            retention,
            versionstamp: 0,
            seen: Vec::new(),
        };
        let row = reader.open_cursor().await?;
        reader.versionstamp = row.versionstamp;
        // The expectations a previous reader under this id left behind. Without
        // them a restart would have nothing to be contiguous with, and a hole
        // punched while it was down would read as a first sighting.
        reader.seen = decode_seen(&row.seen)?;
        Ok(reader)
    }

    /// The connection this reader reads through.
    #[must_use]
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// The consumer id this reader's cursor is filed under.
    #[must_use]
    pub fn consumer(&self) -> &str {
        &self.consumer
    }

    /// The change-capture versionstamp this reader has consumed up to.
    #[must_use]
    pub fn versionstamp(&self) -> i64 {
        self.versionstamp
    }

    /// Check the live schemas against what this build declares.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Schema`] if any of them has drifted.
    pub async fn assert_schema(&self) -> Result<()> {
        schema::assert_bus_schema(self.store.client(), self.retention).await
    }

    /// Read everything published since this reader last advanced.
    ///
    /// Pages the change feed until it is exhausted, decodes the created bus rows
    /// out of it, checks each stream's sequence numbers for contiguity, and
    /// writes the persisted cursor — all before returning, so a caller that
    /// drops the batch has still consumed it. A caller that must not lose a batch
    /// should process it before the next call rather than relying on the cursor.
    ///
    /// The cursor is written even when the batch is empty. That is not a wasted
    /// round trip: the write is what renews the freshness the staleness
    /// prediction is measured against, and without it a reader polling a quiet
    /// bus ages out of its own window — see [`Self::rebaseline`] for the remedy
    /// it would then be pointlessly sent to.
    ///
    /// # Change-feed mechanics worth knowing
    ///
    /// The catch-up convention is `SINCE versionstamp + 1`. The feed's `SINCE` is
    /// **inclusive**, so reading from the stored mark would re-deliver the last
    /// batch every time; the `+ 1` is what makes the cursor a high-water mark
    /// rather than a low one.
    ///
    /// The feed also carries entries that are not row writes — a table
    /// definition, for one — and those are skipped.
    ///
    /// ⚠ The page limit counts raw feed entries across the **whole database**,
    /// and the table filter is applied after the scan. This store defines a
    /// change feed on the bus table alone, so every entry in range is a bus
    /// entry; if another table in the same database is given one, a page can
    /// come back empty while bus entries remain, and catch-up will make progress
    /// over several calls rather than one. Nothing is lost either way — the
    /// cursor only advances over what was read.
    ///
    /// # A batch is checked, then committed — in that order
    ///
    /// Contiguity is asserted against a **scratch** copy of the expectations
    /// that is advanced event by event, and the scratch replaces the reader's
    /// own only once the cursor write has landed. Two things fall out of that,
    /// and both were wrong before it:
    ///
    /// - Two events on one stream *inside a single batch* are contiguous with
    ///   **each other**. Checking a whole batch against a view frozen at its
    ///   start reports a gap on the second one — and the documented remedy for a
    ///   gap, re-baselining, then discards the very events that were there all
    ///   along.
    /// - A raced cursor advance leaves the reader **unchanged**. Recording
    ///   sequence numbers and then failing to move the mark would mean the
    ///   reader had noted events it returned to nobody, and every later batch
    ///   would be measured against expectations built from them.
    ///
    /// # Errors
    ///
    /// - [`StoreError::BusStale`] if the cursor is old enough that retention may
    ///   already have discarded events. Nothing is read; call
    ///   [`Self::rebaseline`].
    /// - [`StoreError::BusGap`] if a stream's sequence numbers are not
    ///   contiguous. Events were lost; call [`Self::rebaseline`].
    /// - [`StoreError::BusRaced`] if another reader sharing this consumer id
    ///   moved the cursor first. The reader is untouched and still usable; the
    ///   remedy is a consumer id of its own, not a retry.
    /// - [`StoreError::Corrupt`] if an event is not filed under its own
    ///   `(stream, seq)` address, or if a change-feed entry carries no
    ///   versionstamp this cursor can address.
    /// - A database error otherwise.
    pub async fn next_batch(&mut self) -> Result<Vec<BusEvent>> {
        let cursor = self.open_cursor().await?;
        self.versionstamp = cursor.versionstamp;
        self.check_freshness(cursor.age)?;

        let mut events = Vec::new();
        let mut mark = self.versionstamp;
        loop {
            let (page, last) = self.page(mark).await?;
            let Some(last) = last else { break };
            events.extend(page);
            mark = last;
        }

        // One live view, advanced as it is checked — see the note above on why
        // a view frozen at the batch's start reports gaps that are not there.
        let mut seen = self.seen.clone();
        for event in &events {
            check_contiguity(&seen, event)?;
            record_seen(&mut seen, event);
        }

        // The cursor moves first and the expectations are adopted only if it
        // did, so a raced advance costs this reader nothing but the batch.
        self.stamp(mark, &seen).await?;
        self.seen = seen;
        Ok(events)
    }

    /// Subscribe to the live wakeup for the bus table.
    ///
    /// The stream is the caller's: the caller owns the loop, the cancellation,
    /// and ending the subscription — which is done by **dropping the stream**,
    /// because a raw `KILL` does not end one on an embedded engine.
    ///
    /// # What arrives, and what it means
    ///
    /// Each item is a `Notification` carrying a [`NotificationAction`] and a
    /// decoded [`BusEvent`]. There are five actions:
    ///
    /// - `Create` — the only one this table produces in normal operation, since
    ///   bus rows are write-once.
    /// - `Update` and `Delete` — not produced by this store's own writes; seeing
    ///   one means something else touched the table.
    /// - `Killed` — the subscription ended. The stream yields nothing after it.
    /// - `Error` — an evaluation failure. Match it defensively, but do **not**
    ///   rely on it: the per-subscription evaluation-failure path does not emit
    ///   it, it downgrades to a debug log instead. A subscription that delivers
    ///   nothing is diagnosed by checking the query, not by waiting for this.
    ///
    /// And the standing rule: an arriving notification means *look again*, not
    /// *this happened*. Ordering between notifications is not structurally
    /// guaranteed, delivery is at-most-once, and nothing is replayed after a
    /// disconnect. Read [`Self::next_batch`] for what actually happened.
    ///
    /// # Generic listen loops
    ///
    /// A loop generic over the notification type needs `R: SurrealValue + Unpin`
    /// on the payload — the stream's `Stream` implementation requires it. The
    /// concrete stream this returns already satisfies it.
    ///
    /// # Errors
    ///
    /// Fails if the subscription cannot be registered — including because the
    /// transport does not support live queries at all, which
    /// [`StoreBuilder::require_live_queries`](crate::StoreBuilder::require_live_queries)
    /// turns into a failure at connection time instead.
    pub async fn subscribe(&self) -> Result<BusStream> {
        let mut response = self.store.client().query(LIVE_SELECT.as_str()).await?;
        Ok(response.stream::<Notification<BusEvent>>(0)?)
    }

    /// Forget the change feed and start again from the durable rows.
    ///
    /// This is the answer to every way catch-up can stop being trustworthy: a
    /// sequence gap, a stale cursor, or a restore — an import emits no
    /// change-feed entries and no notifications, so after one the cursor
    /// describes a feed that no longer corresponds to the table.
    ///
    /// It reads the highest sequence number on each stream, adopts those as the
    /// numbers already seen, and stamps the cursor at the feed's current end so
    /// the next batch carries only what arrives afterwards. **Events published
    /// before this call are not returned by any later batch** — recovering them
    /// is [`Self::replay`], which reads rows directly.
    ///
    /// # Errors
    ///
    /// Fails if the durable rows or the cursor cannot be read or written, or
    /// with [`StoreError::BusRaced`] if another reader sharing this consumer id
    /// moved the cursor first.
    pub async fn rebaseline(&mut self) -> Result<()> {
        let mut response = self.store.client().query(HIGH_WATER_ROWS.as_str()).await?;
        #[derive(SurrealValue)]
        struct HighWater {
            stream: String,
            seq: i64,
        }
        let rows =
            error::take_absorbing_missing_table::<Vec<HighWater>>(response.take(0), BUS_TABLE)?
                .unwrap_or_default();
        let seen: Vec<(String, i64)> = rows.into_iter().map(|row| (row.stream, row.seq)).collect();

        // Drain the feed without decoding it, so the cursor lands at its current
        // end rather than at a point the next batch would re-read.
        let mut mark = self.versionstamp;
        loop {
            let (_, last) = self.page(mark).await?;
            let Some(last) = last else { break };
            mark = last;
        }
        // Unconditionally, even when the feed has not moved: this is the
        // documented remedy for a *stale* cursor, and a remedy that skipped the
        // write whenever there was nothing new would leave the cursor exactly as
        // old as it was — reporting the same staleness on the next call, for
        // ever.
        self.stamp(mark, &seen).await?;
        self.seen = seen;
        Ok(())
    }

    /// Read a stream's durable rows from a sequence number onwards.
    ///
    /// The recovery path, and the one read in this module that does not involve
    /// the change feed at all: the rows are the source of truth, so this answers
    /// "what did I miss?" even when the feed cannot.
    ///
    /// # Errors
    ///
    /// Fails if the read is rejected or a row is not filed under its own
    /// `(stream, seq)` address.
    pub async fn replay(&self, stream: &str, from_seq: i64) -> Result<Vec<BusEvent>> {
        let mut response = self
            .store
            .client()
            .query(ROWS_SINCE.as_str())
            .bind(("stream", stream.to_owned()))
            .bind(("seq", from_seq))
            .await?;
        let Some(events) =
            error::take_absorbing_missing_table::<Vec<BusEvent>>(response.take(0), BUS_TABLE)?
        else {
            return Ok(Vec::new());
        };
        for event in &events {
            event.revalidate()?;
        }
        Ok(events)
    }

    /// Create the cursor if absent, and read it back with its age.
    ///
    /// The age is computed in the database rather than against a clock here: the
    /// question is how long the row has been sitting there, and the database is
    /// where both the row and the reference clock are.
    async fn open_cursor(&self) -> Result<CursorRow> {
        let write = CursorOpen {
            cursor: self.cursor.clone(),
            consumer: self.consumer.clone(),
        };
        let mut response = self
            .store
            .run_write_response(CURSOR_TABLES, OPEN_CURSOR, ("row", write), Refusals::none())
            .await?;
        let row: Option<CursorRow> = response.take(OPEN_CURSOR_SLOT)?;
        row.ok_or_else(|| StoreError::Corrupt {
            context: BUS_MARK_TABLE.to_owned(),
            detail: "the cursor was neither found nor created".to_owned(),
        })
    }

    /// Refuse to read a feed the cursor may already have fallen out of.
    fn check_freshness(&self, age_secs: i64) -> Result<()> {
        let age = u64::try_from(age_secs).unwrap_or(0);
        if age >= stale_after(self.retention) {
            return Err(StoreError::BusStale {
                age_secs: age,
                retention_secs: self.retention.as_secs(),
            });
        }
        Ok(())
    }

    /// Read one page of the change feed, returning its events and the
    /// versionstamp to continue from.
    ///
    /// `None` for the continuation means the feed is exhausted.
    async fn page(&self, from: i64) -> Result<(Vec<BusEvent>, Option<i64>)> {
        // `SINCE` and `LIMIT` take literals, not parameters — the statement's
        // grammar has no place to bind one. Both values are integers this crate
        // computed, never anything a caller supplies, and the table name is a
        // crate constant.
        let since = from.saturating_add(1);
        let statement = format!(
            "SHOW CHANGES FOR TABLE {BUS_TABLE} SINCE {since} LIMIT {}",
            Self::PAGE
        );
        let mut response = self.store.client().query(statement).await?;
        let changesets: Vec<Value> = response.take(0)?;
        if changesets.is_empty() {
            return Ok((Vec::new(), None));
        }

        let mut events = Vec::new();
        let mut last = None;
        for changeset in changesets {
            let Value::Object(object) = changeset else {
                continue;
            };
            // A changeset this reader cannot address is loud, not skipped. The
            // cursor advances to the *last* versionstamp a page yielded, so an
            // entry whose own stamp went unread would have its rows delivered
            // and then delivered again on the next call — a duplicate the
            // contiguity check reports as a gap, from a cause nothing names.
            // Unreachable until versionstamps outgrow a signed 64-bit integer;
            // reachable in principle, and silent is the wrong failure mode.
            let Some(Value::Number(number)) = object.get("versionstamp") else {
                return Err(StoreError::Corrupt {
                    context: BUS_TABLE.to_owned(),
                    detail: "a change-feed entry carries no numeric versionstamp".to_owned(),
                });
            };
            let Some(versionstamp) = number.to_int() else {
                return Err(StoreError::Corrupt {
                    context: BUS_TABLE.to_owned(),
                    detail: format!(
                        "change-feed versionstamp `{number}` does not fit a signed 64-bit \
                         integer, so a cursor cannot address it"
                    ),
                });
            };
            last = Some(versionstamp);
            let Some(Value::Array(changes)) = object.get("changes") else {
                continue;
            };
            for change in changes.iter() {
                let Value::Object(change) = change else {
                    continue;
                };
                // Entries that are not row writes — a replayed table definition,
                // a delete — carry no `update` key and are skipped.
                let Some(row) = change.get("update") else {
                    continue;
                };
                let event = BusEvent::from_value(row.clone())?;
                event.revalidate()?;
                events.push(event);
            }
        }
        Ok((events, last))
    }

    /// Write the cursor's mark and its expectations, but only from the
    /// versionstamp it is expected to hold.
    ///
    /// The guarded update is the write-skew fence for the cursor: two processes
    /// draining the same consumer id cannot wind the shared mark backwards,
    /// because the loser's `WHERE` matches nothing and it is told so — as
    /// [`StoreError::BusRaced`], which names the condition rather than reporting
    /// a corrupt document. Nothing is mutated on that path, here or in the
    /// caller, so the reader is exactly as it was.
    ///
    /// `to` may equal the versionstamp already held: that is a poll that found
    /// nothing, and the write is what renews the freshness the staleness
    /// prediction is measured against.
    async fn stamp(&mut self, to: i64, seen: &[(String, i64)]) -> Result<()> {
        let write = CursorStamp {
            cursor: self.cursor.clone(),
            expected: self.versionstamp,
            next: to,
            seen: encode_seen(seen)?,
        };
        let mut response = self
            .store
            .run_write_response(
                CURSOR_TABLES,
                STAMP_CURSOR,
                ("row", write),
                Refusals::none(),
            )
            .await?;
        let updated: Vec<CursorMark> = response.take(STAMP_CURSOR_SLOT)?;
        if updated.is_empty() {
            return Err(StoreError::BusRaced {
                consumer: self.consumer.clone(),
            });
        }
        self.versionstamp = to;
        Ok(())
    }
}

/// Assert that an event follows the last one seen on its stream, in `seen`.
///
/// A free function over the view rather than a method on the reader, because the
/// view it is asked about is the *scratch* one a batch is checked against — not
/// the reader's own, which is only replaced once the batch is committed.
fn check_contiguity(seen: &[(String, i64)], event: &BusEvent) -> Result<()> {
    let Some((_, last)) = seen.iter().find(|(name, _)| name == event.stream()) else {
        // First sighting of this stream: nothing to be contiguous with. A gap
        // that opened before any reader existed is the cursor's age to catch,
        // not this.
        return Ok(());
    };
    let expected = last.saturating_add(1);
    if event.seq() != expected {
        return Err(StoreError::BusGap {
            expected: u64::try_from(expected).unwrap_or(0),
            saw: u64::try_from(event.seq()).unwrap_or(0),
        });
    }
    Ok(())
}

/// Remember an event as the last seen on its stream.
fn record_seen(seen: &mut Vec<(String, i64)>, event: &BusEvent) {
    match seen.iter_mut().find(|(name, _)| name == event.stream()) {
        Some((_, last)) => *last = event.seq(),
        None => seen.push((event.stream().to_owned(), event.seq())),
    }
}

/// The per-stream expectations, as the opaque JSON string the cursor column
/// holds.
///
/// Sorted by stream name, so the same set of expectations is the same string
/// whatever order the streams were first seen in — a stored value that varies
/// with history is a value nobody can compare two of.
fn encode_seen(seen: &[(String, i64)]) -> Result<String> {
    let mut sorted: Vec<(&str, i64)> = seen
        .iter()
        .map(|(stream, last)| (stream.as_str(), *last))
        .collect();
    sorted.sort_unstable();
    Ok(serde_json::to_string(&sorted)?)
}

/// The inverse of [`encode_seen`].
///
/// # Errors
///
/// [`StoreError::Codec`] if the column does not hold this encoding — which for a
/// cursor written by any build of this store it does, so the failure means
/// something else wrote it.
fn decode_seen(raw: &str) -> Result<Vec<(String, i64)>> {
    Ok(serde_json::from_str(raw)?)
}

/// The age at which a cursor is declared untrustworthy, for a given retention.
///
/// Deliberately short of the window itself: expiry is silent and collection is
/// periodic, so a cursor that has only just reached the edge has already lost
/// the race.
///
/// The arithmetic order is load-bearing. Multiplying before dividing keeps a
/// short retention's limit from rounding to zero — which would declare every
/// cursor stale from the moment it was created, and an alarm that is always on
/// is the same as no alarm at all. The floor of one second covers the remaining
/// degenerate cases.
fn stale_after(retention: Duration) -> u64 {
    retention
        .as_secs()
        .saturating_mul(u64::from(BusReader::STALE_AT))
        .checked_div(u64::from(BusReader::STALE_OF))
        .unwrap_or(0)
        .max(1)
}

/// The shape [`STAMP_CURSOR`] returns.
#[derive(Debug, Clone, SurrealValue)]
struct CursorMark {
    versionstamp: i64,
}

/// Re-export so a caller matching on notification actions does not have to reach
/// into the SDK for the enum.
pub use surrealdb::types::Action as NotificationAction;

#[cfg(test)]
mod tests {
    use super::*;

    /// The publish statement's shape is load-bearing four times over: the
    /// staleness check keeps the record id and the sequence number in
    /// agreement, the counter write-back is the write-skew fence, `CREATE` is
    /// what refuses a duplicated number, and the trailing `RETURN` is the only
    /// place the allocated number is visible.
    #[test]
    fn the_publish_statement_fences_creates_and_returns() {
        let publish = PUBLISH.as_str();
        assert!(publish.contains("IF $seq != $row.expected"));
        assert!(publish.contains(error::STALE_SEQUENCE_SENTINEL));
        assert!(publish.contains("UPSERT $row.allocator SET"));
        assert!(publish.contains("next_seq = $seq + 1"));
        assert!(publish.contains("CREATE $row.id"));
        assert!(!publish.contains("UPSERT $row.id"));
        assert!(publish.trim_end().ends_with("RETURN $seq"));
        // Values ride as bound parameters; the only text formatted into the
        // statement is this crate's own sentinel.
        assert!(!publish.contains('\''));
    }

    /// The slot arithmetic is the one thing here that cannot be checked by
    /// reading the statement: `BEGIN` and `COMMIT` occupy slots too, and so does
    /// **each** table's existence guard — so a statement that guards two tables
    /// starts one slot later than one that guards a single table.
    #[test]
    fn the_returned_slot_follows_the_guards_and_the_preceding_statements() {
        assert_eq!(Store::first_statement_slot(1), 2);
        assert_eq!(Store::first_statement_slot(2), 3);
        assert_eq!(PUBLISH.matches(';').count(), 4);
        assert_eq!(PUBLISH_SEQ_SLOT, 7);
        assert_eq!(OPEN_CURSOR_SLOT, 3);
        assert_eq!(STAMP_CURSOR_SLOT, 2);
    }

    /// A publish writes the event *and* the allocator, so both are guarded. A
    /// guard on the event table alone would let `REMOVE TABLE bus_seq` be
    /// silently repaired as a schemaless impostor by the next publish.
    #[test]
    fn the_guarded_tables_are_every_table_the_statement_writes() {
        assert_eq!(PUBLISH_TABLES, &[BUS_TABLE, BUS_SEQ_TABLE]);
        assert_eq!(CURSOR_TABLES, &[BUS_MARK_TABLE]);
        // The primary table leads: it is the one a classified refusal names.
        assert_eq!(PUBLISH_TABLES.first(), Some(&BUS_TABLE));
    }

    /// The cursor's creation must not reset a cursor that is already there —
    /// that would wind every consumer back to the start of the feed on open, and
    /// throw away the expectations that make gap detection survive a restart.
    #[test]
    fn opening_a_cursor_preserves_an_existing_mark() {
        assert!(OPEN_CURSOR.contains("versionstamp = versionstamp ?? 0"));
        assert!(OPEN_CURSOR.contains("seen = seen ?? '[]'"));
        assert!(!OPEN_CURSOR.contains("CONTENT"));
        // Both halves of the mark are read back: a reader that loaded the
        // versionstamp without the expectations would resume in the feed while
        // starting over on detection.
        assert!(OPEN_CURSOR.contains("SELECT versionstamp, seen, "));
    }

    /// The cursor write is conditional, which is what makes a second reader on
    /// the same consumer id a detectable condition rather than a lost mark — and
    /// it writes the freshness stamp and the expectations alongside the mark, in
    /// one statement, so the three cannot disagree.
    #[test]
    fn stamping_a_cursor_is_guarded_by_the_versionstamp_it_expects() {
        assert!(STAMP_CURSOR.contains("WHERE versionstamp = $row.expected"));
        assert!(STAMP_CURSOR.contains("RETURN AFTER"));
        assert!(STAMP_CURSOR.contains("stamped_at = time::now()"));
        assert!(STAMP_CURSOR.contains("seen = $row.seen"));
    }

    /// The stored expectations are order-independent: the same set is the same
    /// string however the streams were first sighted, or two cursors that agree
    /// would not compare equal.
    #[test]
    fn the_stored_expectations_are_canonical_and_round_trip() {
        let forwards = vec![("goals".to_owned(), 4i64), ("ticks".to_owned(), 0)];
        let backwards = vec![("ticks".to_owned(), 0i64), ("goals".to_owned(), 4)];
        let encoded = encode_seen(&forwards).expect("expectations encode");
        assert_eq!(encoded, r#"[["goals",4],["ticks",0]]"#);
        assert_eq!(encode_seen(&backwards).expect("encodes"), encoded);
        assert_eq!(decode_seen(&encoded).expect("decodes"), forwards);
        assert!(
            decode_seen("[]")
                .expect("an empty cursor decodes")
                .is_empty()
        );
    }

    /// Contiguity is asked of a *view*, not of the reader, which is what lets a
    /// batch be checked against a scratch copy before anything is committed —
    /// and what makes two events on one stream contiguous with each other rather
    /// than both with the batch's starting point.
    #[test]
    fn contiguity_is_checked_against_a_view_that_advances() {
        let event = |stream: &str, seq: i64| {
            BusEvent::from_columns(
                bus::event_address(stream, seq),
                bus::BUS_CODEC.to_owned(),
                stream.to_owned(),
                seq,
                serde_json::json!({}),
            )
        };

        let mut seen = Vec::new();
        // A first sighting seeds rather than asserts, whatever number it holds.
        check_contiguity(&seen, &event("goals", 7)).expect("a first sighting seeds");
        record_seen(&mut seen, &event("goals", 7));

        check_contiguity(&seen, &event("goals", 8)).expect("8 follows 7");
        record_seen(&mut seen, &event("goals", 8));
        check_contiguity(&seen, &event("goals", 9)).expect("9 follows 8 in the same pass");
        check_contiguity(&seen, &event("ticks", 3)).expect("another stream is its own sighting");

        let err = check_contiguity(&seen, &event("goals", 10)).expect_err("10 does not follow 8");
        assert!(
            matches!(
                err,
                StoreError::BusGap {
                    expected: 9,
                    saw: 10
                }
            ),
            "{err}"
        );
    }

    /// The read projections are derived from the declared schema, so this pins
    /// the whole rendered statements.
    #[test]
    fn the_read_projections_are_exactly_the_declared_column_lists() {
        assert_eq!(
            LIVE_SELECT.as_str(),
            "LIVE SELECT id, codec, stream, seq, payload FROM bus"
        );
        assert_eq!(
            ROWS_SINCE.as_str(),
            "SELECT id, codec, stream, seq, payload FROM bus \
             WHERE stream = $stream AND seq >= $seq ORDER BY seq"
        );
        assert_eq!(
            HIGH_WATER_ROWS.as_str(),
            "SELECT stream, math::max(seq) AS seq FROM bus GROUP BY stream"
        );
    }

    /// The page size has to sit at the engine's hard cap, not above it: asking
    /// for more silently gets 1000, and a catch-up loop that believed otherwise
    /// would think it had drained the feed.
    #[test]
    fn the_page_size_is_the_engines_cap() {
        assert_eq!(BusReader::PAGE, 1000);
    }

    /// Staleness is declared before the window closes, because expiry is silent
    /// and collection is periodic — a cursor at the edge has already lost.
    ///
    /// The arithmetic order matters and is easy to get wrong: dividing first
    /// rounds a short retention's limit to zero, which would declare every
    /// cursor stale from the moment it was created — an alarm that is always on
    /// is the same as no alarm at all.
    #[test]
    fn a_cursor_is_stale_before_the_window_closes() {
        let limit = stale_after;
        assert_eq!(limit(Duration::from_secs(100)), 75);
        assert_eq!(limit(Duration::from_secs(4)), 3);
        // Short windows must not round to "always stale".
        assert_eq!(limit(Duration::from_secs(2)), 1);
        assert_eq!(limit(Duration::from_secs(1)), 1);
        assert_eq!(limit(Duration::ZERO), 1);
        assert!(limit(BUS_CHANGEFEED_RETENTION) < BUS_CHANGEFEED_RETENTION.as_secs());
    }
}
