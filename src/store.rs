//! The connection handle and its builder.

use surrealdb::Surreal;
use surrealdb::engine::any::Any;
use surrealdb::types::SurrealValue;

use std::time::Duration;

use crate::capability::{self, Requirements};
use crate::error::{self, Refusals, Result, StoreError};

/// Configure and open a [`Store`].
///
/// The endpoint decides the engine: `memory` for an in-process store, a
/// `rocksdb://` path for the durable one, `ws://` or `http://` to reach a
/// server. Whichever engines are reachable is a build-time question — each is
/// behind its own cargo feature.
#[derive(Debug, Clone)]
pub struct StoreBuilder {
    endpoint: String,
    namespace: String,
    database: String,
    requirements: Requirements,
    changefeed_gc_interval: Option<Duration>,
}

impl StoreBuilder {
    /// Start configuring a store against `endpoint`.
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            namespace: "catgraph".to_owned(),
            database: "main".to_owned(),
            requirements: Requirements::none(),
            changefeed_gc_interval: None,
        }
    }

    /// Set the namespace. Defaults to `catgraph`.
    #[must_use]
    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// Set the database. Defaults to `main`.
    #[must_use]
    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.database = database.into();
        self
    }

    /// Require that the endpoint supports export and import.
    #[must_use]
    pub fn require_backup(mut self) -> Self {
        self.requirements = self.requirements.with_backup();
        self
    }

    /// Require that the endpoint supports live queries.
    #[must_use]
    pub fn require_live_queries(mut self) -> Self {
        self.requirements = self.requirements.with_live_queries();
        self
    }

    /// How often the engine collects expired change-capture entries.
    ///
    /// Only meaningful for the local engines, which run collection as a
    /// background task. Leaving it unset takes the engine's own default, which
    /// is what a deployment normally wants; setting it short is how a test makes
    /// retention expiry observable within the test's own lifetime, since a
    /// consumer cannot tell an expired cursor from an idle one until collection
    /// has actually run.
    #[must_use]
    pub fn changefeed_gc_interval(mut self, interval: Duration) -> Self {
        self.changefeed_gc_interval = Some(interval);
        self
    }

    /// Check capabilities, connect, and select the namespace and database.
    ///
    /// Capabilities are checked **first**, before any connection is opened, so a
    /// transport that cannot do what the store needs fails here rather than at
    /// the first export or subscription — potentially hours later, and in the
    /// backup case as an error carrying no structured discriminator at all.
    ///
    /// # Errors
    ///
    /// Fails if the endpoint lacks a required capability, if the connection
    /// cannot be established, or if the namespace and database cannot be
    /// selected.
    pub async fn connect(self) -> Result<Store> {
        capability::check(&self.endpoint, self.requirements)?;

        let config =
            surrealdb::opt::Config::new().changefeed_gc_interval(self.changefeed_gc_interval);
        let client = surrealdb::engine::any::connect((self.endpoint.clone(), config)).await?;
        client
            .use_ns(&self.namespace)
            .use_db(&self.database)
            .await?;

        Ok(Store { client })
    }
}

/// An open connection to the backing database.
///
/// # Sessions and transactions
///
/// Starting a transaction consumes an **owned** `Surreal` handle. That does
/// not strand a shared `Arc<Store>`: [`Self::session`] mints a fresh owned
/// handle from a shared reference, so `store.session().begin()` works from
/// behind an `Arc`. The distinction that actually matters is *sessions*:
///
/// - **One session per concurrent transaction worker.** Each [`Self::session`]
///   call (equivalently, each [`Clone`] of the handle) is a *new session*,
///   snapshot-inheriting namespace, database, auth, and variables at the
///   moment it is minted; sessions are the isolation boundary that makes
///   `TransactionConflict` meaningful.
/// - **A shared handle is a shared session.** Fine for plain concurrent
///   queries; for transactions, mint a session per worker rather than
///   funnelling workers through one.
///
/// # Background tasks
///
/// The repositories built on this handle spawn nothing themselves, but the
/// handle is not inert: opening a local engine starts roughly a dozen background
/// maintenance tasks in-process, and their intervals are not configurable
/// through the public options. Anything budgeting tasks or reasoning about
/// shutdown should account for them.
#[derive(Debug, Clone)]
pub struct Store {
    client: Surreal<Any>,
}

impl Store {
    /// Borrow the underlying client.
    ///
    /// Repositories are built on top of this. Note that transactions need an
    /// owned handle — see [`Self::session`].
    #[must_use]
    pub fn client(&self) -> &Surreal<Any> {
        &self.client
    }

    /// Mint a new session sharing this connection.
    ///
    /// Returns an owned handle, which is what starting a transaction requires.
    /// The new session snapshot-inherits the current namespace, database, auth,
    /// and variables; later changes on either side do not cross over.
    #[must_use]
    pub fn session(&self) -> Surreal<Any> {
        self.client.clone()
    }

    /// Run a repository write, guarded and classified.
    ///
    /// Every repository write goes through here, and the shared shape is the
    /// point — the invariants below hold for a tier *by construction* rather
    /// than by each repository re-copying them:
    ///
    /// - **The table-existence guard.** The statement runs inside a transaction
    ///   that first checks the table is still *defined* and `THROW`s a
    ///   crate-owned sentinel otherwise. A bare write against an undefined
    ///   table would not fail — the engine auto-creates the table
    ///   `TYPE ANY SCHEMALESS`, silently disarming every schema-level guard —
    ///   so where the read paths absorb a vanished table as absence, a write
    ///   surfaces it as [`StoreError::Schema`]: re-open the store.
    /// - **Refusal classification in the one safe order** — see
    ///   [`error::classify_write`]: unique-index collisions are recognised
    ///   before write-once events and `READONLY` refusals, because the collision
    ///   message echoes caller-supplied values that can imitate either of the
    ///   other two renderings. Which refusals a tier can produce is declared
    ///   once, as [`Refusals`], beside its schema.
    ///
    /// The composed query text embeds only crate-owned constants (the table
    /// name and the statement); values ride as bound parameters.
    pub(crate) async fn run_write<V: SurrealValue + 'static>(
        &self,
        table: &'static str,
        statement: &str,
        binding: (&'static str, V),
        refusals: Refusals,
    ) -> Result<()> {
        self.run_write_response(table, statement, binding, refusals)
            .await
            .map(|_| ())
    }

    /// The slot the first of a write's own statements lands in.
    ///
    /// The guard wraps the statement as `BEGIN; IF …; <statement>; COMMIT;`, and
    /// **`BEGIN` and `COMMIT` each occupy a result slot** — verified against the
    /// engine, and the reason this is a named constant rather than an
    /// open-coded `0`. A multi-statement write's `k`-th statement is therefore at
    /// `FIRST_STATEMENT_SLOT + k`.
    pub(crate) const FIRST_STATEMENT_SLOT: usize = 2;

    /// [`Self::run_write`], keeping the response so a write can return a value
    /// the database computed.
    ///
    /// Sequence allocation needs this: the number a publish is assigned is
    /// decided inside the guarded transaction, and reading it back afterwards
    /// would be both a second round trip and a different number under
    /// concurrency. Callers index the response with
    /// [`Self::FIRST_STATEMENT_SLOT`].
    pub(crate) async fn run_write_response<V: SurrealValue + 'static>(
        &self,
        table: &'static str,
        statement: &str,
        binding: (&'static str, V),
        refusals: Refusals,
    ) -> Result<surrealdb::IndexedResults> {
        let guarded = format!(
            "BEGIN; \
             IF (INFO FOR DB).tables.{table} == NONE {{ THROW \"{sentinel}\" }}; \
             {statement}; \
             COMMIT;",
            sentinel = error::undefined_table_sentinel(table),
        );
        let mut response = self.client.query(guarded).bind(binding).await?;
        // `check()` would surface the FIRST error slot — which, inside the
        // guard transaction, is the guard statement's "not executed due to a
        // failed transaction" boilerplate whenever the *write* is what failed.
        // The classifiable message lives on the failing statement's own slot,
        // so every slot is inspected, in slot order.
        let mut errors: Vec<(usize, surrealdb::Error)> =
            response.take_errors().into_iter().collect();
        if errors.is_empty() {
            return Ok(response);
        }
        errors.sort_by_key(|(slot, _)| *slot);

        for (_, e) in &errors {
            if error::is_undefined_table_guard(e, table) {
                return Err(StoreError::Schema {
                    table: table.to_owned(),
                    detail: "the table is no longer defined — it was removed after this store \
                             opened; writing would silently re-create it schemaless, so re-open \
                             (bootstrap and verify) instead"
                        .to_owned(),
                });
            }
        }
        for (_, e) in &errors {
            if let Some(classified) = error::classify_write(e, table, refusals) {
                return Err(classified);
            }
        }
        // Nothing classified: prefer the slot carrying a real message over the
        // rolled-back boilerplate, falling back to the first slot.
        let (_, first) = errors.remove(0);
        let informative = errors.into_iter().map(|(_, e)| e).find(|e| {
            !matches!(
                e.query_details(),
                Some(surrealdb::types::QueryError::NotExecuted)
            )
        });
        match informative {
            Some(e)
                if matches!(
                    first.query_details(),
                    Some(surrealdb::types::QueryError::NotExecuted)
                ) =>
            {
                Err(e.into())
            }
            _ => Err(first.into()),
        }
    }
}
