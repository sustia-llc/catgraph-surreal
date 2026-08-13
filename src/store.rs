//! The connection handle and its builder.

use surrealdb::Surreal;
use surrealdb::engine::any::Any;

use crate::capability::{self, Requirements};
use crate::error::Result;

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

        let client = surrealdb::engine::any::connect(&self.endpoint).await?;
        client
            .use_ns(&self.namespace)
            .use_db(&self.database)
            .await?;

        Ok(Store { client })
    }
}

/// An open connection to the backing database.
///
/// # Sessions, and why `Arc` is the wrong tool for transactions
///
/// Starting a transaction **consumes** the handle, so a shared
/// `Arc<Store>` cannot begin one — there is nothing to consume. The two
/// sharing modes are therefore not interchangeable:
///
/// - **Per-worker [`Clone`]** — each clone is a *new session*, snapshot-
///   inheriting namespace, database, auth, and variables at the moment of the
///   clone. This is what concurrent transaction workers need: one clone each.
/// - **`Arc<Store>`** — one shared session, for non-transactional use only.
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
}
