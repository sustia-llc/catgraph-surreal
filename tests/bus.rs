//! End-to-end notification-bus behaviour against the in-memory engine.
//!
//! These need a real engine compiled in, so the whole file is gated on `mem`.
//!
//! Three properties are pinned here, and none of them is visible from the
//! encoding alone. **Sequence allocation** is one: numbers have to be contiguous
//! per stream and independent between streams, and the counter has to survive
//! concurrent publishers. **Catch-up** is the second: the change feed's `SINCE`
//! is inclusive, so the `+ 1` convention is what makes a cursor a high-water
//! mark rather than a permanent re-delivery of the last batch. And the **live
//! wakeup** is the third — it is a `Stream` the caller owns, which means the
//! test owns it too, drop and all.

#![cfg(feature = "mem")]

use std::time::Duration;

use catgraph_surreal::{BusReader, BusWriter, Store, StoreBuilder, StoreError, bus, schema};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use surrealdb::types::{Action, RecordId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Goal {
    reached: bool,
    generation: u32,
}

async fn connect(database: &str) -> Store {
    StoreBuilder::new("memory")
        .namespace("catgraph_test")
        .database(database)
        .require_live_queries()
        .connect()
        .await
        .expect("connecting to the in-memory engine")
}

async fn writer(database: &str) -> (Store, BusWriter) {
    let store = connect(database).await;
    let writer = BusWriter::open(store.clone())
        .await
        .expect("opening the bus bootstraps and verifies");
    (store, writer)
}

// ------------------------------------------------------------- publishing

#[tokio::test]
async fn sequence_numbers_start_at_zero_and_are_contiguous() {
    let (_store, writer) = writer("bus_sequence").await;
    for expected in 0..5 {
        let seq = writer
            .publish(
                "goals",
                &Goal {
                    reached: false,
                    generation: expected,
                },
            )
            .await
            .expect("publishing");
        assert_eq!(seq, i64::from(expected));
    }
}

#[tokio::test]
async fn streams_number_independently() {
    let (_store, writer) = writer("bus_streams").await;
    assert_eq!(
        writer
            .publish(
                "goals",
                &Goal {
                    reached: true,
                    generation: 0
                }
            )
            .await
            .expect("publishing"),
        0
    );
    assert_eq!(
        writer
            .publish(
                "ticks",
                &Goal {
                    reached: false,
                    generation: 0
                }
            )
            .await
            .expect("publishing"),
        0
    );
    assert_eq!(
        writer
            .publish(
                "goals",
                &Goal {
                    reached: true,
                    generation: 1
                }
            )
            .await
            .expect("publishing"),
        1
    );
}

#[tokio::test]
async fn a_published_event_reads_back_with_its_payload() {
    let (store, writer) = writer("bus_payload").await;
    let goal = Goal {
        reached: true,
        generation: 3,
    };
    let seq = writer.publish("goals", &goal).await.expect("publishing");

    let reader = BusReader::open(store, "worker-1")
        .await
        .expect("opening a reader");
    let events = reader.replay("goals", 0).await.expect("replaying");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].stream(), "goals");
    assert_eq!(events[0].seq(), seq);
    assert_eq!(events[0].decode::<Goal>().expect("decoding"), goal);
}

/// An event's record id is the digest of `(stream, seq)`, so a sequence number
/// handed out twice would collide on the id rather than overwrite. Publishing
/// the same payload twice must therefore produce two rows, not one.
#[tokio::test]
async fn republishing_an_identical_payload_produces_a_second_event() {
    let (store, writer) = writer("bus_duplicate_payload").await;
    let goal = Goal {
        reached: true,
        generation: 0,
    };
    assert_eq!(writer.publish("goals", &goal).await.expect("first"), 0);
    assert_eq!(writer.publish("goals", &goal).await.expect("second"), 1);
    assert_eq!(row_count(&store, schema::BUS_TABLE).await, 2);
}

#[tokio::test]
async fn a_payload_that_is_not_an_object_never_reaches_the_database() {
    let (store, writer) = writer("bus_bad_payload").await;
    let err = writer
        .publish("goals", &42)
        .await
        .expect_err("a scalar is not a bus payload");
    assert!(matches!(err, StoreError::TypeMismatch { .. }), "{err}");
    assert_eq!(row_count(&store, schema::BUS_TABLE).await, 0);
}

/// **The retry contract, end to end.** Four publishers on one stream under a
/// modest budget: every publish must eventually land.
///
/// The budget is what this pins. A publish absorbs a bounded number of lost
/// sequence-number races itself, and what happens *past* that bound is the
/// point — the condition surfaces as a conflict, so the caller's own loop
/// carries it. While it classified as neither a conflict nor a shutdown,
/// `retry` returned it on the first attempt however generous the policy, and
/// the one failure the bus's documentation predicts was the one no budget could
/// cover.
#[tokio::test]
async fn contending_publishers_succeed_within_a_modest_retry_budget() {
    use catgraph_surreal::{RetryPolicy, retry};

    let (store, writer) = writer("bus_contention_budget").await;
    let mut handles = Vec::new();
    for worker in 0..4u32 {
        let writer = writer.clone();
        handles.push(tokio::spawn(async move {
            for round in 0..4u32 {
                let goal = Goal {
                    reached: false,
                    generation: worker * 100 + round,
                };
                retry(RetryPolicy::new().attempts(16), |_| {
                    let writer = writer.clone();
                    let goal = goal.clone();
                    async move { writer.publish("goals", &goal).await }
                })
                .await
                .expect("a contended publish must land within the budget");
            }
        }));
    }
    for handle in handles {
        handle.await.expect("a publisher task must not panic");
    }

    let reader = BusReader::open(store.clone(), "auditor")
        .await
        .expect("opening a reader");
    let seqs: Vec<i64> = reader
        .replay("goals", 0)
        .await
        .expect("replaying")
        .iter()
        .map(|event| event.seq())
        .collect();
    assert_eq!(seqs, (0..16).collect::<Vec<i64>>(), "{seqs:?}");
}

/// Concurrent publishers contend on one allocator row. Whatever the interleaving
/// — conflicts retried, or serialised by the engine — the outcome must be a
/// contiguous run of distinct numbers, never two events sharing one.
#[tokio::test]
async fn concurrent_publishers_never_share_a_sequence_number() {
    use catgraph_surreal::{RetryPolicy, retry};

    let (store, writer) = writer("bus_concurrent").await;
    let mut handles = Vec::new();
    for worker in 0..4u32 {
        // A session per worker, minted from the same connection. Cloning is what
        // does that — a second `connect("memory")` would be a second datastore
        // rather than a second session, and the workers would not contend at
        // all.
        let writer = writer.clone();
        handles.push(tokio::spawn(async move {
            for round in 0..5u32 {
                let goal = Goal {
                    reached: false,
                    generation: worker * 100 + round,
                };
                retry(RetryPolicy::new().attempts(40), |_| {
                    let writer = writer.clone();
                    let goal = goal.clone();
                    async move { writer.publish("goals", &goal).await }
                })
                .await
                .expect("publishing under contention");
            }
        }));
    }
    for handle in handles {
        handle.await.expect("a publisher task must not panic");
    }

    let reader = BusReader::open(store.clone(), "auditor")
        .await
        .expect("opening a reader");
    let events = reader.replay("goals", 0).await.expect("replaying");
    let seqs: Vec<i64> = events.iter().map(|event| event.seq()).collect();
    assert_eq!(seqs, (0..20).collect::<Vec<i64>>(), "{seqs:?}");
    assert_eq!(row_count(&store, schema::BUS_TABLE).await, 20);
}

// ---------------------------------------------------------------- catch-up

#[tokio::test]
async fn catch_up_delivers_what_was_published_and_then_nothing() {
    let (store, writer) = writer("bus_catch_up").await;
    let mut reader = BusReader::open(store, "worker-1")
        .await
        .expect("opening a reader");

    for generation in 0..3u32 {
        writer
            .publish(
                "goals",
                &Goal {
                    reached: false,
                    generation,
                },
            )
            .await
            .expect("publishing");
    }

    let batch = reader.next_batch().await.expect("catching up");
    assert_eq!(batch.len(), 3);
    assert_eq!(
        batch.iter().map(|e| e.seq()).collect::<Vec<i64>>(),
        vec![0, 1, 2]
    );

    // The cursor advanced, so a second call delivers nothing rather than the
    // same batch again — which is what the `SINCE versionstamp + 1` convention
    // buys, the feed's own `SINCE` being inclusive.
    assert!(reader.next_batch().await.expect("catching up").is_empty());

    // And new events are picked up from where it left off.
    writer
        .publish(
            "goals",
            &Goal {
                reached: true,
                generation: 3,
            },
        )
        .await
        .expect("publishing");
    let batch = reader.next_batch().await.expect("catching up");
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].seq(), 3);
}

#[tokio::test]
async fn catch_up_interleaves_streams_and_keeps_each_contiguous() {
    let (store, writer) = writer("bus_catch_up_streams").await;
    let mut reader = BusReader::open(store, "worker-1")
        .await
        .expect("opening a reader");

    for round in 0..3u32 {
        writer
            .publish(
                "goals",
                &Goal {
                    reached: false,
                    generation: round,
                },
            )
            .await
            .expect("publishing");
        writer
            .publish(
                "ticks",
                &Goal {
                    reached: false,
                    generation: round,
                },
            )
            .await
            .expect("publishing");
    }

    let batch = reader.next_batch().await.expect("catching up");
    assert_eq!(batch.len(), 6);
    let goals: Vec<i64> = batch
        .iter()
        .filter(|e| e.stream() == "goals")
        .map(|e| e.seq())
        .collect();
    let ticks: Vec<i64> = batch
        .iter()
        .filter(|e| e.stream() == "ticks")
        .map(|e| e.seq())
        .collect();
    assert_eq!(goals, vec![0, 1, 2]);
    assert_eq!(ticks, vec![0, 1, 2]);
}

/// A cursor is per consumer id, so two consumers each see everything and
/// neither consumes the other's batch.
#[tokio::test]
async fn consumers_have_independent_cursors() {
    let (store, writer) = writer("bus_cursors").await;
    let mut first = BusReader::open(store.clone(), "worker-1")
        .await
        .expect("opening");
    let mut second = BusReader::open(store, "worker-2").await.expect("opening");

    writer
        .publish(
            "goals",
            &Goal {
                reached: true,
                generation: 0,
            },
        )
        .await
        .expect("publishing");

    assert_eq!(first.next_batch().await.expect("catching up").len(), 1);
    assert_eq!(second.next_batch().await.expect("catching up").len(), 1);
    assert!(first.next_batch().await.expect("catching up").is_empty());
}

/// **Two events on one stream, in one batch, after the reader already knows the
/// stream.** Both are contiguous — with each other, not with where the batch
/// started.
///
/// This is the regression the whole check-then-commit shape exists for. While a
/// batch was measured against expectations frozen at its start, the second event
/// of every multi-event batch was reported as a gap; and the documented remedy
/// for a gap re-baselines, which throws away exactly the events that were there
/// all along. Only the *first* batch escaped it, because a first sighting seeds
/// rather than asserts — which is why this test's first batch is deliberately
/// not the one under test.
#[tokio::test]
async fn a_later_batch_may_carry_several_events_from_one_stream() {
    let (store, writer) = writer("bus_multi_event_batch").await;
    let mut reader = BusReader::open(store, "worker-1")
        .await
        .expect("opening a reader");

    // The seeding batch: after this the reader has an expectation to be
    // contiguous with, which is what makes the next one a real test.
    writer
        .publish(
            "goals",
            &Goal {
                reached: false,
                generation: 0,
            },
        )
        .await
        .expect("publishing");
    assert_eq!(reader.next_batch().await.expect("catching up").len(), 1);

    for generation in 1..4u32 {
        writer
            .publish(
                "goals",
                &Goal {
                    reached: false,
                    generation,
                },
            )
            .await
            .expect("publishing");
    }

    let batch = reader
        .next_batch()
        .await
        .expect("three contiguous events are not a gap");
    assert_eq!(
        batch.iter().map(|e| e.seq()).collect::<Vec<i64>>(),
        vec![1, 2, 3]
    );
    // And the expectations moved with them: the next batch continues from 3.
    writer
        .publish(
            "goals",
            &Goal {
                reached: true,
                generation: 4,
            },
        )
        .await
        .expect("publishing");
    let batch = reader.next_batch().await.expect("catching up");
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].seq(), 4);
}

/// A cursor survives the reader that made it: a second reader under the same
/// consumer id resumes rather than replaying.
#[tokio::test]
async fn a_cursor_persists_across_readers() {
    let (store, writer) = writer("bus_cursor_persists").await;
    writer
        .publish(
            "goals",
            &Goal {
                reached: false,
                generation: 0,
            },
        )
        .await
        .expect("publishing");

    {
        let mut reader = BusReader::open(store.clone(), "worker-1")
            .await
            .expect("opening");
        assert_eq!(reader.next_batch().await.expect("catching up").len(), 1);
    }

    let mut resumed = BusReader::open(store, "worker-1")
        .await
        .expect("re-opening under the same consumer id");
    assert!(
        resumed.next_batch().await.expect("catching up").is_empty(),
        "a resumed reader must not replay what its cursor already covered"
    );
}

/// Skip a stream's sequence allocator forward, the way a restored or
/// hand-edited counter would, so the next publish leaves a hole.
///
/// Deleting a *row* would not do it: the change feed is a log of what happened,
/// so removing the row afterwards leaves the entry that recorded its creation.
/// The hole a reader can actually meet is one in the numbering itself.
async fn skip_sequence(store: &Store, stream: &str, to: i64) {
    store
        .client()
        .query("UPDATE $rid SET next_seq = $to RETURN NONE")
        .bind((
            "rid",
            RecordId::new(schema::BUS_SEQ_TABLE, bus::allocator_key(stream)),
        ))
        .bind(("to", to))
        .await
        .expect("moving the allocator")
        .check()
        .expect("and it moves");
}

/// A hole in a stream's sequence numbers means events were lost, and the reader
/// has to say so rather than deliver a batch that skips.
#[tokio::test]
async fn a_missing_event_is_reported_as_a_gap() {
    let (store, writer) = writer("bus_gap").await;
    let mut reader = BusReader::open(store.clone(), "worker-1")
        .await
        .expect("opening");

    writer
        .publish(
            "goals",
            &Goal {
                reached: false,
                generation: 0,
            },
        )
        .await
        .expect("publishing");
    let batch = reader.next_batch().await.expect("catching up");
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].seq(), 0);

    // Event 1 never happens: the allocator is moved past it, so what arrives
    // next is 2.
    skip_sequence(&store, "goals", 2).await;
    writer
        .publish(
            "goals",
            &Goal {
                reached: false,
                generation: 2,
            },
        )
        .await
        .expect("publishing");

    let err = reader
        .next_batch()
        .await
        .expect_err("a hole in the sequence is a gap");
    let StoreError::BusGap { expected, saw } = &err else {
        panic!("expected a bus gap, got {err:?}");
    };
    assert_eq!((*expected, *saw), (1, 2));
    assert!(!err.is_conflict());
    assert!(!err.is_shutdown());
}

/// **A hole punched while the consumer was down is still a hole.** Gap detection
/// has to survive a restart, which means the per-stream expectations have to
/// live on the cursor row beside the versionstamp rather than in the reader.
///
/// Held in memory alone they reset with the process, and a resumed reader would
/// treat the first event it saw on each stream as a first sighting — seeding
/// silently past exactly the loss the sequence numbers exist to catch. The
/// cursor's *position* survived a restart already; its expectations did not, so
/// a restarted reader resumed in the right place with the wrong question.
#[tokio::test]
async fn a_restarted_reader_detects_a_hole_punched_while_it_was_down() {
    let (store, writer) = writer("bus_restart_gap").await;

    {
        let mut reader = BusReader::open(store.clone(), "worker-1")
            .await
            .expect("opening");
        writer
            .publish(
                "goals",
                &Goal {
                    reached: false,
                    generation: 0,
                },
            )
            .await
            .expect("publishing");
        assert_eq!(reader.next_batch().await.expect("catching up").len(), 1);
    }
    // The reader is gone. Event 1 never happens: the allocator is moved past it,
    // so what arrives next is 2.
    skip_sequence(&store, "goals", 2).await;
    writer
        .publish(
            "goals",
            &Goal {
                reached: false,
                generation: 2,
            },
        )
        .await
        .expect("publishing");

    let mut resumed = BusReader::open(store.clone(), "worker-1")
        .await
        .expect("re-opening under the same consumer id");
    let err = resumed
        .next_batch()
        .await
        .expect_err("the hole opened while this consumer was down, and is still a hole");
    let StoreError::BusGap { expected, saw } = &err else {
        panic!("expected a bus gap, got {err:?}");
    };
    assert_eq!((*expected, *saw), (1, 2));

    // And the remedy still works across the restart.
    resumed.rebaseline().await.expect("re-baselining");
    assert!(resumed.next_batch().await.expect("catching up").is_empty());
}

/// **Two readers, one consumer id.** A cursor is named by its consumer id, so
/// sharing one means sharing a cursor — and the loser of the guarded write is
/// told, by name.
///
/// It is reported as [`StoreError::BusRaced`] rather than as a corrupt document:
/// nothing is damaged, it is a configuration mistake, and the remedy is a
/// consumer id of one's own rather than a retry or a re-baseline. The reader
/// that lost is left exactly as it was, which this checks by using it
/// afterwards.
#[tokio::test]
async fn a_cursor_moved_by_a_second_reader_is_reported_as_a_race() {
    let (store, writer) = writer("bus_shared_consumer").await;
    let mut first = BusReader::open(store.clone(), "worker-1")
        .await
        .expect("opening");
    let mut second = BusReader::open(store.clone(), "worker-1")
        .await
        .expect("opening a second reader under the same id");

    writer
        .publish(
            "goals",
            &Goal {
                reached: true,
                generation: 0,
            },
        )
        .await
        .expect("publishing");
    assert_eq!(first.next_batch().await.expect("catching up").len(), 1);

    // `rebaseline` writes from the mark the reader itself holds, which the other
    // reader has since moved — the shared-cursor collision, deterministically.
    let err = second
        .rebaseline()
        .await
        .expect_err("the cursor moved underneath this reader");
    let StoreError::BusRaced { consumer } = &err else {
        panic!("expected a shared-cursor race, got {err:?}");
    };
    assert_eq!(consumer, "worker-1");
    // Not a conflict and not a shutdown: retrying re-runs the same collision.
    assert!(!err.is_conflict());
    assert!(!err.is_shutdown());

    // The reader is still usable — it re-reads the shared cursor and carries on.
    assert!(
        second
            .next_batch()
            .await
            .expect("a raced reader is still usable")
            .is_empty()
    );
}

/// Re-baselining is the remedy for every way catch-up stops being trustworthy:
/// it adopts the durable rows as the truth and starts again from the feed's
/// current end.
#[tokio::test]
async fn rebaselining_recovers_from_a_gap() {
    let (store, writer) = writer("bus_rebaseline").await;
    let mut reader = BusReader::open(store.clone(), "worker-1")
        .await
        .expect("opening");
    writer
        .publish(
            "goals",
            &Goal {
                reached: false,
                generation: 0,
            },
        )
        .await
        .expect("publishing");
    assert_eq!(reader.next_batch().await.expect("catching up").len(), 1);

    skip_sequence(&store, "goals", 2).await;
    writer
        .publish(
            "goals",
            &Goal {
                reached: false,
                generation: 2,
            },
        )
        .await
        .expect("publishing");
    assert!(matches!(
        reader.next_batch().await,
        Err(StoreError::BusGap { .. })
    ));

    reader.rebaseline().await.expect("re-baselining");
    // Nothing arrives from before the re-baseline...
    assert!(reader.next_batch().await.expect("catching up").is_empty());
    // ...and what follows does, contiguously with what the durable rows hold.
    writer
        .publish(
            "goals",
            &Goal {
                reached: true,
                generation: 3,
            },
        )
        .await
        .expect("publishing");
    let batch = reader.next_batch().await.expect("catching up");
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].seq(), 3);

    // The durable rows are the recovery path for what the feed skipped.
    let replayed = reader.replay("goals", 0).await.expect("replaying");
    assert_eq!(
        replayed.iter().map(|e| e.seq()).collect::<Vec<i64>>(),
        vec![0, 2, 3]
    );
}

/// An event whose sequence number was rewritten in place is not filed under its
/// own address, and a read says so rather than letting a fabricated number
/// through the contiguity check.
#[tokio::test]
async fn an_event_filed_under_the_wrong_address_is_corrupt() {
    let (store, writer) = writer("bus_corrupt_event").await;
    writer
        .publish(
            "goals",
            &Goal {
                reached: false,
                generation: 0,
            },
        )
        .await
        .expect("publishing");

    let addr = bus::event_address("goals", 0);
    store
        .client()
        .query("OPTION IMPORT; UPDATE $rid SET seq = 9 RETURN NONE")
        .bind(("rid", RecordId::new(schema::BUS_TABLE, addr.as_str())))
        .await
        .expect("the edit runs")
        .check()
        .expect("and is accepted under OPTION IMPORT");

    let reader = BusReader::open(store, "worker-1").await.expect("opening");
    let err = reader
        .replay("goals", 0)
        .await
        .expect_err("a rewritten sequence number is corrupt");
    assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
}

// ------------------------------------------------------------ the live tier

/// The wakeup. The stream is the caller's — this test owns the loop, the
/// timeout, and the drop that ends the subscription.
#[tokio::test]
async fn a_subscription_wakes_on_a_publish() {
    let (store, writer) = writer("bus_live").await;
    let reader = BusReader::open(store, "worker-1").await.expect("opening");
    let mut stream = reader.subscribe().await.expect("subscribing");

    let goal = Goal {
        reached: true,
        generation: 7,
    };
    writer.publish("goals", &goal).await.expect("publishing");

    let notification = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("a notification arrives within the timeout")
        .expect("the stream is not exhausted")
        .expect("the notification is not an error");

    assert_eq!(notification.action, Action::Create);
    let event = notification.data;
    assert_eq!(event.stream(), "goals");
    assert_eq!(event.seq(), 0);
    assert_eq!(event.decode::<Goal>().expect("decoding"), goal);
    // The payload is a wakeup, but it is still a well-formed event.
    event.revalidate().expect("the delivered event revalidates");

    // Dropping the stream is what ends the subscription — a raw KILL does not,
    // on an embedded engine.
    drop(stream);
}

/// The subscription is a wakeup, and the durable rows are the truth. Both
/// describe the same events, which is what makes "subscribe for latency, read
/// for correctness" a coherent instruction rather than two answers.
#[tokio::test]
async fn the_wakeup_and_the_durable_rows_agree() {
    let (store, writer) = writer("bus_live_agrees").await;
    let mut reader = BusReader::open(store, "worker-1").await.expect("opening");
    let mut stream = reader.subscribe().await.expect("subscribing");

    for generation in 0..3u32 {
        writer
            .publish(
                "goals",
                &Goal {
                    reached: false,
                    generation,
                },
            )
            .await
            .expect("publishing");
    }

    let mut live = Vec::new();
    for _ in 0..3 {
        let notification = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("a notification arrives")
            .expect("the stream is not exhausted")
            .expect("the notification is not an error");
        live.push(notification.data.seq());
    }
    live.sort_unstable();
    drop(stream);

    let durable: Vec<i64> = reader
        .next_batch()
        .await
        .expect("catching up")
        .iter()
        .map(|event| event.seq())
        .collect();
    assert_eq!(live, durable);
}

// ------------------------------------------------------------------- schema

#[tokio::test]
async fn opening_twice_is_idempotent_and_verifies() {
    let store = connect("bus_bootstrap").await;
    let first = BusWriter::open(store.clone())
        .await
        .expect("the first open bootstraps");
    let second = BusWriter::open(store.clone())
        .await
        .expect("the second open is a no-op");
    first.assert_schema().await.expect("still verified");
    second.assert_schema().await.expect("still verified");
}

/// A shortened change-capture window is exactly the drift that turns catch-up
/// into silent loss, so the retention is part of the compared schema rather
/// than a setting each handle chooses for itself.
#[tokio::test]
async fn disagreeing_about_the_retention_is_drift() {
    let store = connect("bus_retention_drift").await;
    BusWriter::open(store.clone())
        .await
        .expect("the default retention establishes the schema");

    let err = BusWriter::with_retention(store, Duration::from_secs(1))
        .await
        .expect_err("a different retention is drift, not a re-definition");
    let StoreError::Schema { table, detail } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert_eq!(table, schema::BUS_TABLE);
    assert!(detail.contains("CHANGEFEED"), "{detail}");
}

/// The retention reaches the engine's own rendering of the table definition,
/// which is what the guard compares against.
#[tokio::test]
async fn the_retention_is_visible_in_the_table_definition() {
    let store = connect("bus_retention_rendering").await;
    BusWriter::with_retention(store.clone(), Duration::from_secs(90))
        .await
        .expect("opening with a chosen retention");

    let mut response = store
        .client()
        .query(format!("RETURN (INFO FOR DB).tables.{}", schema::BUS_TABLE))
        .await
        .expect("reading the table definition");
    let definition: Option<String> = response.take(0).expect("it reads back");
    assert_eq!(
        definition.as_deref(),
        Some(schema::bus_table_definition(Duration::from_secs(90)).as_str())
    );
}

/// Reads absorb a vanished table as absence; a write must be loud.
#[tokio::test]
async fn a_removed_table_is_absent_to_reads_and_loud_to_writes() {
    let (store, writer) = writer("bus_removed_table").await;
    let reader = BusReader::open(store.clone(), "worker-1")
        .await
        .expect("opening");
    writer
        .publish(
            "goals",
            &Goal {
                reached: false,
                generation: 0,
            },
        )
        .await
        .expect("publishing");

    store
        .client()
        .query(format!("REMOVE TABLE {}", schema::BUS_TABLE))
        .await
        .expect("removing the table")
        .check()
        .expect("and it is removed");

    assert!(
        reader
            .replay("goals", 0)
            .await
            .expect("reads absorb")
            .is_empty()
    );

    let err = writer
        .publish(
            "goals",
            &Goal {
                reached: false,
                generation: 1,
            },
        )
        .await
        .expect_err("a write against an undefined table must be loud");
    let StoreError::Schema { table, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert_eq!(table, schema::BUS_TABLE);
}

/// **The other table a publish writes.** A publish writes the event *and* the
/// stream's sequence allocator, so removing the allocator table has to be as
/// loud as removing the event table.
///
/// This is the hole a single-table guard leaves. `REMOVE TABLE bus_seq` followed
/// by a publish would auto-create it `TYPE ANY SCHEMALESS` — the counter that
/// makes sequence numbers a total order, silently reset to zero and stripped of
/// every guard, while `assert_schema` on the event table stayed green.
#[tokio::test]
async fn a_removed_allocator_table_is_loud_to_writes() {
    let (store, writer) = writer("bus_removed_allocator").await;
    writer
        .publish(
            "goals",
            &Goal {
                reached: false,
                generation: 0,
            },
        )
        .await
        .expect("publishing creates the allocator");

    store
        .client()
        .query(format!("REMOVE TABLE {}", schema::BUS_SEQ_TABLE))
        .await
        .expect("removing the allocator table")
        .check()
        .expect("and it is removed");

    let err = writer
        .publish(
            "goals",
            &Goal {
                reached: false,
                generation: 1,
            },
        )
        .await
        .expect_err("a write against an undefined table must be loud");
    let StoreError::Schema { table, .. } = &err else {
        panic!("expected schema drift, got {err:?}");
    };
    assert_eq!(table, schema::BUS_SEQ_TABLE);

    // And the guard refused rather than repaired: the table stays undefined.
    let mut response = store
        .client()
        .query(format!(
            "RETURN (INFO FOR DB).tables.{}",
            schema::BUS_SEQ_TABLE
        ))
        .await
        .expect("reading the table definition");
    let definition: Option<String> = response.take(0).expect("the definition slot");
    assert_eq!(definition, None, "the table must remain undefined");
}

/// The cursor row carries both halves of a consumer's mark, and the engine's
/// rendering of the column is what the drift guard compares against.
#[tokio::test]
async fn the_cursor_columns_are_the_engines_renderings() {
    let (store, _writer) = writer("bus_cursor_columns").await;
    BusReader::open(store.clone(), "worker-1")
        .await
        .expect("opening a reader creates the cursor");

    let mut response = store
        .client()
        .query(format!(
            "RETURN (INFO FOR TABLE {}).fields",
            schema::BUS_MARK_TABLE
        ))
        .await
        .expect("reading the field definitions");
    let live: Option<std::collections::BTreeMap<String, String>> =
        response.take(0).expect("the definitions read back");
    let live = live.unwrap_or_default();
    for (name, definition) in schema::BUS_MARK_FIELD_DEFINITIONS {
        assert_eq!(
            live.get(name).map(String::as_str),
            Some(definition),
            "{name}"
        );
    }
}

// ------------------------------------------------------------------ helpers

async fn row_count(store: &Store, table: &str) -> usize {
    let mut response = store
        .client()
        .query(format!("SELECT VALUE id FROM {table}"))
        .await
        .expect("counting the stored rows");
    let ids: Vec<RecordId> = response.take(0).expect("reading the ids");
    ids.len()
}

/// The encoding constant is part of the on-disk format, so a change to it is a
/// migration.
#[test]
fn the_bus_codec_is_pinned() {
    assert_eq!(bus::BUS_CODEC, "cgb1");
}
