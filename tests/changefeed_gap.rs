//! The change-capture gap gate.
//!
//! These need a real engine compiled in, so the whole file is gated on `mem`.
//!
//! # The condition being tested
//!
//! Change-capture retention is garbage-collected **silently**. A consumer whose
//! cursor falls outside the window gets whatever survives, with nothing
//! distinguishing "no new events" from "the events are gone" — no error, no
//! signal, no shorter batch than expected. Left alone that is the worst kind of
//! failure: a bus that appears to be working and is not.
//!
//! The store closes it two ways, and both are gates here:
//!
//! - **Prediction.** The cursor records *when* it last advanced, and catching up
//!   refuses to read a cursor whose age has come within reach of the retention
//!   window. That is [`StoreError::BusStale`], raised before anything is read.
//! - **Detection.** Every event carries a per-stream sequence number, and
//!   catch-up asserts contiguity across batches. A hole is
//!   [`StoreError::BusGap`].
//!
//! Both have the same remedy — re-baseline from the durable rows, which are the
//! source of truth and are not garbage-collected.
//!
//! # Why the collection interval is configurable
//!
//! Expired entries are removed by a background task, so a test that only
//! shortened the retention would be waiting on a timer whose period it did not
//! control. Setting both makes expiry observable inside the test's own lifetime.

#![cfg(feature = "mem")]

use std::time::Duration;

use catgraph_surreal::{BusReader, BusWriter, Store, StoreBuilder, StoreError, schema};
use serde::{Deserialize, Serialize};
use surrealdb::types::RecordId;

/// Short enough that a test can outlive it, long enough that a publish and a
/// read inside one are not racing it.
const RETENTION: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Tick {
    generation: u32,
}

async fn connect(database: &str) -> Store {
    StoreBuilder::new("memory")
        .namespace("catgraph_test")
        .database(database)
        // Without this the collection sweep runs on the engine's own schedule,
        // which is far longer than any test should wait.
        .changefeed_gc_interval(Duration::from_millis(100))
        .connect()
        .await
        .expect("connecting to the in-memory engine")
}

async fn bus(database: &str) -> (Store, BusWriter) {
    let store = connect(database).await;
    let writer = BusWriter::with_retention(store.clone(), RETENTION)
        .await
        .expect("opening the bus with a short retention");
    (store, writer)
}

/// Move a consumer's cursor timestamp into the past, which is what the passage
/// of time would do — without the test having to spend it.
///
/// The cursor row is not a write-once table, so this is an ordinary update
/// rather than a privileged one.
async fn age_cursor(store: &Store, consumer: &str, by: Duration) {
    let seconds = i64::try_from(by.as_secs()).expect("a test-sized duration fits");
    store
        .client()
        .query("UPDATE $rid SET stamped_at = stamped_at - $age RETURN NONE")
        .bind((
            "rid",
            RecordId::new(
                schema::BUS_MARK_TABLE,
                catgraph_surreal::bus::cursor_key(consumer),
            ),
        ))
        .bind(("age", surrealdb::types::Duration::from_secs(seconds as u64)))
        .await
        .expect("ageing the cursor")
        .check()
        .expect("and it ages");
}

/// **The prediction gate.** A cursor old enough that retention may already have
/// discarded events must be refused *before* anything is read — silently
/// returning whatever survived is the failure this exists to prevent.
#[tokio::test]
async fn a_cursor_older_than_the_window_is_refused_rather_than_trusted() {
    let (store, writer) = bus("changefeed_stale").await;
    let mut reader = BusReader::with_retention(store.clone(), "worker-1", RETENTION)
        .await
        .expect("opening a reader");

    writer
        .publish("ticks", &Tick { generation: 0 })
        .await
        .expect("publishing");
    assert_eq!(reader.next_batch().await.expect("catching up").len(), 1);

    // Age the cursor past three quarters of the window — the point at which a
    // cursor is declared unreliable, deliberately short of the window itself
    // because expiry is silent and collection is periodic.
    age_cursor(&store, "worker-1", RETENTION).await;
    writer
        .publish("ticks", &Tick { generation: 1 })
        .await
        .expect("publishing");

    let err = reader
        .next_batch()
        .await
        .expect_err("a stale cursor must not be trusted");
    let StoreError::BusStale {
        age_secs,
        retention_secs,
    } = &err
    else {
        panic!("expected a stale cursor, got {err:?}");
    };
    assert!(*age_secs >= RETENTION.as_secs(), "{age_secs}");
    assert_eq!(*retention_secs, RETENTION.as_secs());
    // Not retryable: waiting makes a stale cursor staler.
    assert!(!err.is_conflict());
    assert!(!err.is_shutdown());

    // And the remedy works: re-baselining adopts the durable rows and reads on.
    reader.rebaseline().await.expect("re-baselining");
    writer
        .publish("ticks", &Tick { generation: 2 })
        .await
        .expect("publishing");
    let batch = reader.next_batch().await.expect("catching up");
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].seq(), 2);
}

/// A fresh cursor is *not* stale, which is what keeps the check above from
/// being a permanently-armed alarm.
#[tokio::test]
async fn a_fresh_cursor_reads_normally() {
    let (_store, writer) = bus("changefeed_fresh").await;
    let mut reader = BusReader::with_retention(writer.store().clone(), "worker-1", RETENTION)
        .await
        .expect("opening a reader");
    for generation in 0..3 {
        writer
            .publish("ticks", &Tick { generation })
            .await
            .expect("publishing");
    }
    assert_eq!(reader.next_batch().await.expect("catching up").len(), 3);
}

/// **The detection gate**, over a window that really has expired.
///
/// The change feed is left to run past its retention with nobody reading, the
/// collection sweep removes the expired entries, and a consumer then catches up
/// over a feed that no longer contains everything the table does. What must not
/// happen is a batch that silently skips.
#[tokio::test]
async fn events_lost_to_expiry_do_not_pass_as_a_complete_batch() {
    let (store, writer) = bus("changefeed_expiry").await;
    let mut reader = BusReader::with_retention(store.clone(), "worker-1", RETENTION)
        .await
        .expect("opening a reader");

    // Seen: 0. The reader now has an expectation to be contiguous with.
    writer
        .publish("ticks", &Tick { generation: 0 })
        .await
        .expect("publishing");
    assert_eq!(reader.next_batch().await.expect("catching up").len(), 1);

    // Event 1 is published and then left to expire out of the feed. The durable
    // row survives — retention governs the feed, not the table.
    writer
        .publish("ticks", &Tick { generation: 1 })
        .await
        .expect("publishing");
    tokio::time::sleep(RETENTION + Duration::from_millis(1500)).await;

    // Event 2 arrives after the sweep, so the feed holds 2 but not 1.
    writer
        .publish("ticks", &Tick { generation: 2 })
        .await
        .expect("publishing");

    // The cursor is now older than the window too, so the prediction fires
    // first — which is the designed order: a consumer should be told its cursor
    // cannot be trusted before it is told what it found with it.
    let err = reader
        .next_batch()
        .await
        .expect_err("a consumer that slept through the window must be told");
    assert!(
        matches!(err, StoreError::BusStale { .. } | StoreError::BusGap { .. }),
        "expected a stale cursor or a gap, got {err:?}"
    );

    // Whichever fired, the remedy is the same and it works. The durable rows
    // still hold every event, including the one the feed dropped.
    reader.rebaseline().await.expect("re-baselining");
    let replayed = reader.replay("ticks", 0).await.expect("replaying");
    assert_eq!(
        replayed
            .iter()
            .map(|event| event.seq())
            .collect::<Vec<i64>>(),
        vec![0, 1, 2],
        "expiry governs the change feed, not the table"
    );
}

/// The gate the whole file rests on: retention really does expire entries out of
/// the feed. If this ever stops being true the tests above would pass by
/// accident.
#[tokio::test]
async fn the_change_feed_really_is_collected() {
    let (store, writer) = bus("changefeed_collection").await;
    writer
        .publish("ticks", &Tick { generation: 0 })
        .await
        .expect("publishing");

    let entries = |store: Store| async move {
        let mut response = store
            .client()
            .query(format!(
                "SHOW CHANGES FOR TABLE {} SINCE 0 LIMIT 1000",
                schema::BUS_TABLE
            ))
            .await
            .expect("reading the feed");
        let changesets: Vec<surrealdb::types::Value> = response.take(0).expect("it reads back");
        changesets.len()
    };

    assert!(
        entries(store.clone()).await > 0,
        "the feed must hold the publish that just happened"
    );

    tokio::time::sleep(RETENTION + Duration::from_millis(1500)).await;
    // A write is what gives the collector something to advance past.
    writer
        .publish("ticks", &Tick { generation: 1 })
        .await
        .expect("publishing");
    tokio::time::sleep(Duration::from_millis(500)).await;

    let remaining = entries(store).await;
    assert!(
        remaining <= 1,
        "the expired entries were never collected: {remaining} changesets remain"
    );
}
