//! Bounded retry with exponential backoff and jitter.
//!
//! This is the reference implementation of the retry contract [`crate::error`]
//! describes, and it exists because that contract has three parts that are easy
//! to get individually right and collectively wrong.
//!
//! # What is retried
//!
//! Exactly two conditions, and nothing else:
//!
//! - a **transaction conflict**, which means the work was never applied and
//!   re-running it is safe;
//! - a **shutdown refusal**, which means the datastore declined the work while
//!   stopping — also unapplied, but not fixed by waiting on the same connection.
//!
//! Everything else is surfaced immediately. A schema error, a corrupt document,
//! a write-once refusal: none of those become true on the ninth attempt, and
//! retrying them turns a clear failure into a slow one.
//!
//! # The unit is the whole transaction
//!
//! The operation handed to [`retry`] must be the **entire** `begin()` …
//! `commit()` unit, not one statement inside it. Conflicts are detected at
//! different points by different engines — RocksDB detects at commit time, so a
//! transaction whose every statement returned `Ok` can still fail at the commit
//! call. A retry wrapped around a single statement classifies the wrong thing
//! and drops that whole class of conflict.
//!
//! It follows that the operation must also be **restartable from the top**: it
//! is called again from the beginning, with everything the previous attempt read
//! discarded. Reading a value in one attempt and using it in the next is exactly
//! the write-skew the retry is supposed to make impossible.
//!
//! # The engine decides the policy, not the test suite
//!
//! Conflict surfaces differ per engine, and the differences are not small. The
//! in-memory engine aborts on *read* conflicts too, so it aborts more often than
//! plain snapshot isolation; RocksDB detects at commit; SurrealKV detects write
//! conflicts only. **A backoff tuned against the memory engine does not transfer
//! to the persistent ones.** Measure against the engine actually deployed.
//!
//! # Jitter, and where the randomness comes from
//!
//! Without jitter, two writers that collide once tend to collide again on the
//! same schedule. The jitter here is drawn from a process-local counter seeded
//! by the standard library's own randomly-keyed hasher, which is enough to
//! decorrelate two writers and costs no dependency. It is not, and must not be
//! used as, a source of cryptographic randomness.

use std::hash::{BuildHasher, Hasher, RandomState};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::error::{Result, StoreError};

/// How long to wait, and how many times.
///
/// The default is deliberately modest — five attempts over roughly a tenth of a
/// second — because the failure this exists for is contention between writers in
/// one process, not an outage. A policy generous enough to ride out an outage is
/// a policy that hides one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    attempts: u32,
    base_delay: Duration,
    max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            attempts: 5,
            base_delay: Duration::from_millis(5),
            max_delay: Duration::from_millis(200),
        }
    }
}

impl RetryPolicy {
    /// The default policy: five attempts, 5 ms doubling to at most 200 ms.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many times the operation may run in total, including the first try.
    ///
    /// Clamped to at least one: an operation that is never attempted cannot
    /// report anything useful.
    #[must_use]
    pub fn attempts(mut self, attempts: u32) -> Self {
        self.attempts = attempts.max(1);
        self
    }

    /// The delay before the second attempt, doubled before each attempt after.
    #[must_use]
    pub fn base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// The ceiling the doubling stops at.
    #[must_use]
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// How long to wait before attempt `attempt`, counting the first attempt as
    /// zero.
    ///
    /// Exponential up to the ceiling, then jittered down by up to half. Jittering
    /// *down* rather than around the target keeps the ceiling a real ceiling.
    #[must_use]
    pub fn delay_before(self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }
        let doubled = self
            .base_delay
            .checked_mul(1u32.checked_shl(attempt - 1).unwrap_or(u32::MAX))
            .unwrap_or(self.max_delay)
            .min(self.max_delay);
        let nanos = u64::try_from(doubled.as_nanos()).unwrap_or(u64::MAX);
        // Half the delay is fixed, half is drawn — so a backoff never collapses
        // to zero and two writers do not stay in lockstep.
        let jitter = (nanos / 2).saturating_add(1);
        Duration::from_nanos(nanos - (next_random() % jitter))
    }
}

/// Run `operation`, retrying while it fails with a condition that retrying can
/// fix.
///
/// `operation` is called with the attempt number, counting from zero, and must
/// be the whole `begin()` … `commit()` unit — see the [module
/// documentation](self) for why, and for what "restartable from the top" means
/// in practice.
///
/// The last attempt's error is returned unchanged: a caller that exhausted its
/// budget on conflicts should see a conflict, not a wrapper claiming the budget
/// was the problem.
///
/// # Errors
///
/// Whatever the final attempt failed with. A failure that is neither a conflict
/// nor a shutdown refusal is returned from the attempt that produced it, without
/// waiting.
///
/// # Examples
///
/// ```no_run
/// # use catgraph_surreal::{Result, Store, retry::{retry, RetryPolicy}};
/// # async fn example(store: &Store) -> Result<()> {
/// retry(RetryPolicy::new(), |_attempt| async {
///     let tx = store.session().begin().await?;
///     tx.query("UPDATE counter:a SET n += 1").await?;
///     // Classified as part of the same unit: some engines only detect a
///     // conflict here.
///     tx.commit().await?;
///     Ok(())
/// })
/// .await
/// # }
/// ```
pub async fn retry<F, Fut, T>(policy: RetryPolicy, mut operation: F) -> Result<T>
where
    F: FnMut(u32) -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut attempt = 0;
    loop {
        let error = match operation(attempt).await {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };
        attempt += 1;
        let retryable = error.is_conflict() || error.is_shutdown();
        if !retryable || attempt >= policy.attempts {
            return Err(error);
        }
        sleep(policy.delay_before(attempt)).await;
    }
}

/// Whether a failure is one [`retry`] would try again.
///
/// Exposed so a caller writing its own loop classifies the same way this one
/// does, rather than reaching for one of the two predicates and forgetting the
/// other.
#[must_use]
pub fn is_retryable(error: &StoreError) -> bool {
    error.is_conflict() || error.is_shutdown()
}

/// Wait, where waiting is possible.
#[cfg(not(target_family = "wasm"))]
async fn sleep(delay: Duration) {
    tokio::time::sleep(delay).await;
}

/// The wasm counterpart: yield rather than wait.
///
/// There is no portable timer on this target, and the browser-side engine is
/// single-threaded anyway — the contention this backs off from is between
/// concurrent *sessions*, which a single-threaded runtime interleaves rather
/// than runs in parallel. Retrying without a delay is therefore the honest
/// behaviour rather than a degraded one, but it is a real difference and callers
/// budgeting attempts on this target should know the budget is spent
/// immediately.
#[cfg(target_family = "wasm")]
async fn sleep(_delay: Duration) {}

/// A process-local pseudo-random number.
///
/// Seeded once from the standard library's randomly-keyed hasher — the same
/// source `HashMap` uses to make its iteration order unpredictable — and
/// advanced by a linear congruential step. Adequate for decorrelating two
/// backoff schedules; adequate for nothing that needs unpredictability.
fn next_random() -> u64 {
    static STATE: LazyLock<AtomicU64> = LazyLock::new(|| {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(0x9E37_79B9_7F4A_7C15);
        AtomicU64::new(hasher.finish() | 1)
    });
    // Numerical Recipes' LCG constants: full period over the 64-bit state.
    let next = STATE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |state| {
            Some(
                state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407),
            )
        })
        .unwrap_or(0);
    // The high bits of an LCG are the well-behaved ones.
    next >> 11
}

#[cfg(test)]
mod tests {
    use surrealdb::types::{ConnectionError, QueryError};

    use super::*;
    use crate::error::RevalidationStage;

    fn conflict() -> StoreError {
        StoreError::Db(surrealdb::Error::query(
            "read-write conflict".to_owned(),
            QueryError::TransactionConflict,
        ))
    }

    fn shutdown() -> StoreError {
        StoreError::Db(surrealdb::Error::connection(
            "There was a problem with the key-value store: The datastore is shutting down"
                .to_owned(),
            ConnectionError::ConnectionFailed,
        ))
    }

    fn permanent() -> StoreError {
        StoreError::Revalidation {
            stage: RevalidationStage::Check,
            detail: "the encoded target word is not the one the expression derives".to_owned(),
        }
    }

    #[tokio::test]
    async fn a_conflict_is_retried_until_it_succeeds() {
        let policy = RetryPolicy::new()
            .attempts(4)
            .base_delay(Duration::from_micros(1));
        let result = retry(policy, |attempt| async move {
            if attempt < 2 {
                Err(conflict())
            } else {
                Ok(attempt)
            }
        })
        .await;
        assert_eq!(result.expect("the third attempt succeeds"), 2);
    }

    #[tokio::test]
    async fn a_shutdown_refusal_is_retried_too() {
        let policy = RetryPolicy::new()
            .attempts(3)
            .base_delay(Duration::from_micros(1));
        let result = retry(policy, |attempt| async move {
            if attempt == 0 {
                Err(shutdown())
            } else {
                Ok(())
            }
        })
        .await;
        assert!(result.is_ok());
    }

    /// The load-bearing negative: a failure retrying cannot fix must be returned
    /// from the attempt that produced it, without burning the budget.
    #[tokio::test]
    async fn a_permanent_failure_is_not_retried() {
        let mut attempts = 0;
        let policy = RetryPolicy::new()
            .attempts(5)
            .base_delay(Duration::from_micros(1));
        let result: Result<()> = retry(policy, |_| {
            attempts += 1;
            async { Err(permanent()) }
        })
        .await;
        assert!(matches!(result, Err(StoreError::Revalidation { .. })));
        assert_eq!(attempts, 1);
    }

    /// Exhausting the budget returns the last failure unchanged, so a caller
    /// sees the condition rather than a wrapper describing the budget.
    #[tokio::test]
    async fn an_exhausted_budget_returns_the_final_error() {
        let mut attempts = 0;
        let policy = RetryPolicy::new()
            .attempts(3)
            .base_delay(Duration::from_micros(1));
        let result: Result<()> = retry(policy, |_| {
            attempts += 1;
            async { Err(conflict()) }
        })
        .await;
        assert!(result.expect_err("every attempt conflicted").is_conflict());
        assert_eq!(attempts, 3);
    }

    /// A zero-attempt policy would never run the operation at all, which is
    /// never what a caller means.
    #[tokio::test]
    async fn a_policy_always_runs_at_least_once() {
        let policy = RetryPolicy::new().attempts(0);
        assert_eq!(policy.attempts, 1);
        let result = retry(policy, |_| async { Ok(7) }).await;
        assert_eq!(result.expect("one attempt is enough"), 7);
    }

    #[test]
    fn backoff_grows_and_stops_at_the_ceiling() {
        let policy = RetryPolicy::new()
            .base_delay(Duration::from_millis(10))
            .max_delay(Duration::from_millis(40));
        assert_eq!(policy.delay_before(0), Duration::ZERO);
        // Each delay is at most its exponential target and more than half of it,
        // which is what jittering down by up to half means.
        for (attempt, target) in [(1u32, 10u64), (2, 20), (3, 40), (4, 40), (9, 40)] {
            let delay = policy.delay_before(attempt).as_nanos();
            let target = u128::from(target) * 1_000_000;
            assert!(delay <= target, "attempt {attempt}: {delay} > {target}");
            assert!(
                delay >= target / 2,
                "attempt {attempt}: {delay} < {target}/2"
            );
        }
    }

    /// Two schedules drawn back to back must not be identical, or the jitter is
    /// decorating rather than decorrelating.
    #[test]
    fn jitter_varies_between_draws() {
        let policy = RetryPolicy::new().base_delay(Duration::from_millis(100));
        let draws: Vec<Duration> = (0..16).map(|_| policy.delay_before(1)).collect();
        assert!(draws.iter().any(|d| *d != draws[0]));
    }

    #[test]
    fn the_shared_classifier_agrees_with_the_loop() {
        assert!(is_retryable(&conflict()));
        assert!(is_retryable(&shutdown()));
        assert!(!is_retryable(&permanent()));
    }
}
