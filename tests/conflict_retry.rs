//! The conflict-retry gate, on **both** embedded engines.
//!
//! # Why both, and why that is not duplication
//!
//! The two engines detect conflicts at different points and on different
//! conditions, so a retry policy validated against one is not validated against
//! the other:
//!
//! - The **in-memory** engine additionally detects many *read* conflicts —
//!   but **not reliably enough to carry a correctness argument**: an
//!   *unguarded* read-modify-write racing four workers has been observed to
//!   lose an update on the order of once in a dozen runs at SurrealDB 3.2.4.
//!   Engine detection is the fast path, never the guarantee.
//! - **RocksDB** pins a snapshot to the transaction and detects at **commit
//!   time** — a transaction whose every statement returned `Ok` can still fail
//!   at the commit call.
//!
//! The second point shapes the retry unit: a retry wrapped around a single
//! statement would never see a RocksDB conflict, because there is nothing wrong
//! with any of the statements. The unit that has to be classified and re-run is
//! the whole `begin()` … `commit()` block, and that is what these tests exercise.
//!
//! The first point is why the exactly-once assertion runs on **RocksDB only**.
//! A `WHERE`-guarded write cannot rescue a single-transaction read-modify-write
//! — inside one snapshot the guard is a tautology (the rival's commit is
//! invisible to the WHERE), so exactly-once increments rest entirely on the
//! engine's commit-time write-write detection. RocksDB's, pinned to the
//! transaction snapshot, has never been observed to miss here; the in-memory
//! engine's demonstrably does (~1 lost update per dozen 4×8 races at 3.2.4).
//! The store's own contended flows do not rely on that detection either way —
//! they put the read and the write in *separate* transactions with a
//! `WHERE`-guarded advance, or collide on a `CREATE` — and their end-to-end
//! gate is `tests/bus.rs::contending_publishers_succeed_within_a_modest_retry_budget`.
//! The memory engine still runs the classification and permanence gates below.
//!
//! Everything here provokes **real** conflicts by racing sessions on one row —
//! no fabricated errors. A fabricated conflict tests the classifier; a real one
//! tests the contract.
//!
//! # Harness notes
//!
//! The persistent engines lock their data directory, so every RocksDB test gets
//! a `TempDir` of its own. Nothing asserts on disk space or on a directory's
//! contents after a `REMOVE DATABASE`: reclamation is deferred by design, and a
//! test that waited for it would be testing a timer.

#![cfg(any(feature = "mem", feature = "rocksdb"))]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use catgraph_surreal::retry::{RetryPolicy, is_retryable, retry};
use catgraph_surreal::{Result, Store, StoreBuilder, StoreError};

/// A counter table, defined before anything transacts against it.
///
/// The order is not incidental. A transaction that is cancelled leaves no table
/// metadata behind, so a workload that relied on the first write to create the
/// table would find the table gone after its first rollback — and, worse, would
/// have it re-created `SCHEMALESS` by the next write.
const DDL: &str = "\
DEFINE TABLE IF NOT EXISTS counter SCHEMAFULL TYPE NORMAL;
DEFINE FIELD IF NOT EXISTS n ON counter TYPE int;
";

async fn prepared(store: &Store) {
    store
        .client()
        .query(DDL)
        .await
        .expect("defining the counter table")
        .check()
        .expect("and it is defined");
    store
        .client()
        .query("UPSERT counter:shared SET n = 0 RETURN NONE")
        .await
        .expect("seeding the counter")
        .check()
        .expect("and it is seeded");
}

/// One read-modify-write transaction over the shared counter.
///
/// The **whole** block is the unit: the read, the write, and the commit. The
/// commit is included deliberately — RocksDB detects conflicts there, so a
/// helper that returned before committing would report success for a
/// transaction that never landed.
///
/// The write is deliberately **unguarded**: inside a single snapshot
/// transaction a `WHERE n = $current` guard is a tautology (the rival's commit
/// is invisible to it), so exactly-once behaviour here rests on the engine's
/// commit-time write-write detection and nothing else. That is the point of
/// the RocksDB gate — and the reason the in-memory engine, whose detection
/// demonstrably misses, does not run it (see the module documentation).
///
/// A failure anywhere cancels the transaction before returning. Dropping one
/// without committing or cancelling leaves it open on the datastore, which is a
/// slower and much more confusing failure than the one being reported.
async fn increment(store: &Store) -> Result<i64> {
    let tx = store.session().begin().await?;

    let attempt = async {
        let mut response = tx.query("SELECT VALUE n FROM ONLY counter:shared").await?;
        let current: Option<i64> = response.take(0)?;
        let next = current.unwrap_or(0) + 1;
        tx.query("UPSERT counter:shared SET n = $n RETURN NONE")
            .bind(("n", next))
            .await?
            .check()?;
        Ok::<i64, StoreError>(next)
    }
    .await;

    match attempt {
        Ok(next) => {
            tx.commit().await?;
            Ok(next)
        }
        Err(e) => {
            // Cancel before propagating: an abandoned transaction is a resource
            // leak wearing the costume of the error that caused it.
            tx.cancel().await?;
            Err(e)
        }
    }
}

/// Race `workers` sessions, each incrementing `rounds` times under the policy.
///
/// Returns how many attempts were retried, so a test can assert that conflicts
/// actually happened rather than that the workload was accidentally serial.
///
/// Only the RocksDB gate drives this — see the module documentation for why
/// the exactly-once assertion does not run on the memory engine.
#[cfg(feature = "rocksdb")]
async fn race(store: &Store, workers: u32, rounds: u32, policy: RetryPolicy) -> u32 {
    let retries = Arc::new(AtomicU32::new(0));
    let mut handles = Vec::new();
    for _ in 0..workers {
        // A session per worker, minted from the same connection: sessions are
        // the isolation boundary that makes a conflict meaningful, and starting
        // a transaction needs an owned handle anyway.
        let store = store.clone();
        let retries = Arc::clone(&retries);
        handles.push(tokio::spawn(async move {
            for _ in 0..rounds {
                retry(policy, |attempt| {
                    if attempt > 0 {
                        retries.fetch_add(1, Ordering::Relaxed);
                    }
                    let store = store.clone();
                    async move { increment(&store).await }
                })
                .await
                .expect("a conflicting transaction must eventually land");
            }
        }));
    }
    for handle in handles {
        handle.await.expect("a worker must not panic");
    }
    retries.load(Ordering::Relaxed)
}

#[cfg(feature = "rocksdb")]
async fn counter(store: &Store) -> i64 {
    let mut response = store
        .client()
        .query("SELECT VALUE n FROM ONLY counter:shared")
        .await
        .expect("reading the counter");
    let value: Option<i64> = response.take(0).expect("it reads back");
    value.expect("the counter was seeded")
}

/// The gate itself: every increment lands exactly once, so the counter equals
/// the number of increments. A lost update would show up as a number that is too
/// small — which is precisely what an unretried conflict, or a retry classified
/// on the wrong unit, produces.
///
/// RocksDB-only — the memory engine's commit-time detection demonstrably
/// misses (see the module documentation), so this exactly-once claim is not
/// one it can carry.
#[cfg(feature = "rocksdb")]
async fn conflicting_increments_all_land(store: Store) {
    prepared(&store).await;
    let workers = 4;
    let rounds = 8;
    let retries = race(
        &store,
        workers,
        rounds,
        RetryPolicy::new()
            .attempts(64)
            .base_delay(std::time::Duration::from_micros(200)),
    )
    .await;

    assert_eq!(
        counter(&store).await,
        i64::from(workers * rounds),
        "an increment was lost: the retry did not cover every conflicting attempt"
    );
    assert!(
        retries > 0,
        "no attempt was ever retried, so this ran serially and tested nothing"
    );
}

/// The other direction: a failure retrying cannot fix must come back from the
/// attempt that produced it, not after the budget is burnt. A schema error is
/// the honest example — it is a real database failure, and it will still be one
/// on the sixty-fourth attempt.
async fn a_permanent_failure_is_not_retried(store: Store) {
    prepared(&store).await;
    let attempts = Arc::new(AtomicU32::new(0));
    let counted = Arc::clone(&attempts);
    let result: Result<()> = retry(RetryPolicy::new().attempts(16), move |_| {
        counted.fetch_add(1, Ordering::Relaxed);
        let store = store.clone();
        async move {
            let tx = store.session().begin().await?;
            let outcome = tx
                .query("UPSERT counter:shared SET n = 'not an integer' RETURN NONE")
                .await?
                .check()
                .map(|_| ());
            match outcome {
                Ok(()) => {
                    tx.commit().await?;
                    Ok(())
                }
                Err(e) => {
                    tx.cancel().await?;
                    Err(StoreError::from(e))
                }
            }
        }
    })
    .await;

    let err = result.expect_err("a type error is not a conflict");
    assert!(!is_retryable(&err), "{err}");
    assert_eq!(
        attempts.load(Ordering::Relaxed),
        1,
        "a permanent failure must not burn the retry budget"
    );
}

/// A conflict is classified from the structured detail the engine attaches, and
/// it must be visible on the **whole** unit. This drives two sessions into a
/// deliberate collision with no retry at all, so the error itself is the result.
async fn a_bare_conflict_is_classified_as_one(store: Store) {
    prepared(&store).await;
    // Enough concurrency, and no retry, that at least one attempt is refused.
    let outcomes: Vec<Result<i64>> = {
        let mut handles = Vec::new();
        for _ in 0..8 {
            let store = store.clone();
            handles.push(tokio::spawn(async move { increment(&store).await }));
        }
        let mut outcomes = Vec::new();
        for handle in handles {
            outcomes.push(handle.await.expect("a worker must not panic"));
        }
        outcomes
    };

    let failures: Vec<&StoreError> = outcomes
        .iter()
        .filter_map(|outcome| outcome.as_ref().err())
        .collect();
    assert!(
        !failures.is_empty(),
        "eight racing sessions produced no conflict at all"
    );
    for err in failures {
        assert!(
            err.is_conflict(),
            "a refused transaction must classify as a conflict, not as {err}"
        );
        assert!(!err.is_shutdown());
        assert!(is_retryable(err));
    }
}

// ------------------------------------------------------------------- memory

#[cfg(feature = "mem")]
mod memory {
    use super::*;

    async fn store(database: &str) -> Store {
        StoreBuilder::new("memory")
            .namespace("catgraph_test")
            .database(database)
            .connect()
            .await
            .expect("connecting to the in-memory engine")
    }

    // No `conflicting_increments_all_land` here, deliberately: the in-memory
    // engine's commit-time write-write detection intermittently misses (an
    // observed lost update roughly once per dozen 4×8 races at 3.2.4), and an
    // unguarded read-modify-write's exactly-once behaviour has nothing else to
    // rest on inside a single transaction. The assertion runs on RocksDB,
    // whose snapshot-pinned commit-time detection has never been observed to
    // miss; the store's own contended flows never rely on the leaky path (see
    // the module documentation).

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_permanent_failure_is_not_retried() {
        super::a_permanent_failure_is_not_retried(store("conflict_permanent").await).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_bare_conflict_is_classified_as_one() {
        super::a_bare_conflict_is_classified_as_one(store("conflict_classified").await).await;
    }
}

// ------------------------------------------------------------------ rocksdb

/// The engine the store is actually meant to run on, and the one whose conflict
/// surface the retry policy should be tuned against — measurements taken on the
/// memory engine do not transfer.
#[cfg(feature = "rocksdb")]
mod rocksdb {
    use super::*;

    /// A store on a directory of its own.
    ///
    /// The `TempDir` is returned alongside, and has to outlive the store: the
    /// engine holds a lock on the directory, and dropping the guard early takes
    /// the data out from under it.
    async fn store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let store = StoreBuilder::new(format!("rocksdb://{}", dir.path().display()))
            .namespace("catgraph_test")
            .database("conflict")
            .connect()
            .await
            .expect("connecting to the RocksDB engine");
        (dir, store)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn conflicting_increments_all_land() {
        let (dir, store) = store().await;
        super::conflicting_increments_all_land(store).await;
        drop(dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_permanent_failure_is_not_retried() {
        let (dir, store) = store().await;
        super::a_permanent_failure_is_not_retried(store).await;
        drop(dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_bare_conflict_is_classified_as_one() {
        let (dir, store) = store().await;
        super::a_bare_conflict_is_classified_as_one(store).await;
        drop(dir);
    }
}
