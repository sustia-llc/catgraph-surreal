//! Connect-and-probe smoke tests against the in-memory engine.
//!
//! These need a real engine compiled in, so the whole file is gated on `mem`.

#![cfg(feature = "mem")]

use catgraph_surreal::{Capability, Requirements, StoreBuilder, StoreError, capability};

/// The in-memory engine is a local engine, so it grants everything. If this ever
/// fails, the capability table has drifted from the SDK.
#[tokio::test]
async fn memory_endpoint_satisfies_every_requirement() {
    let caps = capability::capabilities_of("memory").expect("`memory` is a known endpoint");
    assert!(caps.grants(Capability::Backup));
    assert!(caps.grants(Capability::LiveQueries));

    let all = Requirements::none().with_backup().with_live_queries();
    assert!(capability::check("memory", all).is_ok());
}

/// The end-to-end path the builder promises: probe, connect, select namespace
/// and database, and come back with a usable handle.
#[tokio::test]
async fn builder_connects_to_the_memory_engine() {
    let store = StoreBuilder::new("memory")
        .namespace("catgraph_test")
        .database("smoke")
        .require_backup()
        .require_live_queries()
        .connect()
        .await
        .expect("connecting to the in-memory engine");

    // The handle is live: a trivial query has to round-trip.
    let mut response = store
        .client()
        .query("RETURN 1 + 1")
        .await
        .expect("querying a connected store");
    let answer: Option<i64> = response.take(0).expect("reading the query result");
    assert_eq!(answer, Some(2));
}

/// Each clone is a *new session*, which is what a concurrent transaction worker
/// needs — starting a transaction consumes the handle, so a shared reference
/// cannot begin one.
#[tokio::test]
async fn sessions_are_independent_and_usable() {
    let store = StoreBuilder::new("memory")
        .namespace("catgraph_test")
        .database("sessions")
        .connect()
        .await
        .expect("connecting to the in-memory engine");

    let session = store.session();
    let mut response = session
        .query("RETURN 'session is live'")
        .await
        .expect("querying through a minted session");
    let answer: Option<String> = response.take(0).expect("reading the query result");
    assert_eq!(answer.as_deref(), Some("session is live"));
}

/// The fail-fast contract. A WebSocket endpoint cannot export, so requiring
/// backup must fail at construction — before any connection is attempted, which
/// is why this passes without a server to connect to.
#[tokio::test]
async fn requiring_backup_over_websocket_fails_before_connecting() {
    let err = StoreBuilder::new("ws://127.0.0.1:8000")
        .require_backup()
        .connect()
        .await
        .expect_err("a WebSocket endpoint cannot export");

    let StoreError::Db(inner) = &err else {
        panic!("expected a database-class error, got {err:?}");
    };
    assert!(inner.is_configuration(), "{inner}");
    assert!(!err.is_conflict());
    assert!(!err.is_shutdown());
}
