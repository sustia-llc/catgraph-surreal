//! Fail-fast capability checking for a connection endpoint.
//!
//! # Why this module exists
//!
//! Not every transport can do everything, and the SDK does not tell you until
//! you try. The split that matters here:
//!
//! | Endpoint | Backup (export/import) | Live queries |
//! |---|---|---|
//! | `memory` / `mem://` | yes | yes |
//! | `rocksdb://` | yes | yes |
//! | `surrealkv://` | yes | yes |
//! | `tikv://` | yes | yes |
//! | `http://`, `https://` | yes | **no** |
//! | `ws://`, `wss://` | **no** | yes |
//! | `indxdb://` (wasm) | **no** | yes |
//!
//! A store configured to both checkpoint and subscribe therefore cannot run on a
//! single remote transport — it needs both, which is why the `server` feature
//! turns on both protocols.
//!
//! Left to the SDK, a missing capability surfaces at the first use, which can be
//! hours after start-up — and in the case of a missing backup capability it
//! surfaces as an *internal* error carrying no structured discriminator at all,
//! which is close to undiagnosable. Checking at construction converts that into
//! an immediate, typed failure.
//!
//! # How the capabilities are determined
//!
//! By endpoint scheme. The SDK does compute exactly this information while
//! connecting, but keeps it private: both the capability enum and the field
//! holding it are crate-internal, so there is no public API — documented or
//! otherwise — to read a live connection's capability set. Probing at runtime
//! instead was rejected as the primary mechanism: a live-query probe registers a
//! real subscription that then has to be torn down, and a backup probe means
//! running an export purely for its error.
//!
//! The table above is therefore mirrored from the scheme dispatch the SDK itself
//! uses to populate that private set. It is keyed on the endpoint string the
//! caller supplies, which is stable public API. The mirroring is the tradeoff:
//! if upstream ever changes which scheme grants what, this table must move with
//! it.

use surrealdb::types::ConfigurationError;

use crate::error::{Result, StoreError};

/// A transport capability the store may depend on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Capability {
    /// Export and import — the checkpoint and restore path.
    Backup,
    /// Live queries — the low-latency wakeup tier of the notification bus.
    ///
    /// Delivery is best-effort and at-most-once with no replay, so this is only
    /// ever a wakeup: durable rows remain the source of truth.
    LiveQueries,
}

impl Capability {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Backup => "backup (export/import)",
            Self::LiveQueries => "live queries",
        }
    }
}

/// What the store requires of its transport.
///
/// Default is "requires nothing", so a caller opts in to exactly the checks it
/// needs rather than inheriting assumptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Requirements {
    backup: bool,
    live_queries: bool,
}

impl Requirements {
    /// Require nothing.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            backup: false,
            live_queries: false,
        }
    }

    /// Also require the ability to export and import.
    #[must_use]
    pub const fn with_backup(mut self) -> Self {
        self.backup = true;
        self
    }

    /// Also require live queries.
    #[must_use]
    pub const fn with_live_queries(mut self) -> Self {
        self.live_queries = true;
        self
    }

    /// The capabilities required, in check order.
    fn required(self) -> impl Iterator<Item = Capability> {
        [
            self.backup.then_some(Capability::Backup),
            self.live_queries.then_some(Capability::LiveQueries),
        ]
        .into_iter()
        .flatten()
    }
}

/// The capabilities an endpoint scheme grants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointCapabilities {
    backup: bool,
    live_queries: bool,
}

impl EndpointCapabilities {
    const fn new(backup: bool, live_queries: bool) -> Self {
        Self {
            backup,
            live_queries,
        }
    }

    /// Whether this endpoint grants `capability`.
    #[must_use]
    pub const fn grants(self, capability: Capability) -> bool {
        match capability {
            Capability::Backup => self.backup,
            Capability::LiveQueries => self.live_queries,
        }
    }
}

/// The scheme portion of an endpoint string, lowercased.
///
/// Handles the bare aliases the SDK accepts for the in-memory engine
/// (`memory`, `memory?...`) alongside ordinary `scheme://rest` forms.
fn scheme_of(endpoint: &str) -> Option<String> {
    let trimmed = endpoint.trim();
    if trimmed.eq_ignore_ascii_case("memory") || trimmed.to_ascii_lowercase().starts_with("memory?")
    {
        return Some("mem".to_owned());
    }
    let (scheme, _) = trimmed.split_once("://")?;
    if scheme.is_empty() {
        None
    } else {
        Some(scheme.to_ascii_lowercase())
    }
}

/// The capabilities granted by an endpoint, or `None` for an unrecognised
/// scheme.
///
/// An unrecognised scheme is deliberately *not* an error here — connecting will
/// produce the SDK's own, more specific message (including the "engine not
/// enabled in this build" case), and pre-empting it with a worse one helps
/// nobody.
#[must_use]
pub fn capabilities_of(endpoint: &str) -> Option<EndpointCapabilities> {
    match scheme_of(endpoint)?.as_str() {
        // Local engines: full capability.
        "mem" | "rocksdb" | "surrealkv" | "tikv" => Some(EndpointCapabilities::new(true, true)),
        // HTTP carries backups but cannot subscribe.
        "http" | "https" => Some(EndpointCapabilities::new(true, false)),
        // WebSocket subscribes but cannot export.
        "ws" | "wss" => Some(EndpointCapabilities::new(false, true)),
        // The browser engine subscribes but has no backup path.
        "indxdb" => Some(EndpointCapabilities::new(false, true)),
        _ => None,
    }
}

/// Check an endpoint against what the store needs, before connecting.
///
/// # Errors
///
/// Returns a configuration-class [`StoreError::Db`] naming the first missing
/// capability. The live-query case carries the same structured detail the SDK
/// itself would raise later, so callers can match it either way.
pub fn check(endpoint: &str, requirements: Requirements) -> Result<()> {
    let Some(capabilities) = capabilities_of(endpoint) else {
        // Unknown scheme: let the connection attempt produce the better message.
        return Ok(());
    };

    for capability in requirements.required() {
        if !capabilities.grants(capability) {
            return Err(missing(endpoint, capability));
        }
    }
    Ok(())
}

/// Build the error for a capability the endpoint cannot provide.
fn missing(endpoint: &str, capability: Capability) -> StoreError {
    let message = format!(
        "the endpoint `{endpoint}` does not support {}, which this store requires",
        capability.as_str()
    );
    // Configuration class for both, so `is_conflict`/`is_shutdown` stay false
    // and nothing retries a misconfiguration. The live-query case additionally
    // carries the SDK's own structured detail; there is no equivalent detail for
    // backup, whose native failure is an undiscriminated internal error.
    let details = match capability {
        Capability::LiveQueries => Some(ConfigurationError::LiveQueryNotSupported),
        Capability::Backup => None,
    };
    StoreError::Db(surrealdb::Error::configuration(message, details))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_aliases_resolve_to_the_local_engine() {
        assert_eq!(scheme_of("memory").as_deref(), Some("mem"));
        assert_eq!(scheme_of("Memory").as_deref(), Some("mem"));
        assert_eq!(scheme_of("memory?versioned=true").as_deref(), Some("mem"));
        assert_eq!(scheme_of("mem://").as_deref(), Some("mem"));
    }

    #[test]
    fn schemes_are_lowercased_and_unparseable_endpoints_yield_none() {
        assert_eq!(scheme_of("WSS://example.test").as_deref(), Some("wss"));
        assert_eq!(scheme_of("rocksdb:///tmp/data").as_deref(), Some("rocksdb"));
        assert_eq!(scheme_of("no-scheme-here"), None);
        assert_eq!(scheme_of("://missing"), None);
    }

    /// The whole point of the module: the two remote transports are
    /// complementary, never interchangeable.
    #[test]
    fn remote_transports_have_complementary_capabilities() {
        let http = capabilities_of("http://example.test").expect("http is a known scheme");
        assert!(http.grants(Capability::Backup));
        assert!(!http.grants(Capability::LiveQueries));

        let ws = capabilities_of("ws://example.test").expect("ws is a known scheme");
        assert!(!ws.grants(Capability::Backup));
        assert!(ws.grants(Capability::LiveQueries));
    }

    #[test]
    fn local_engines_grant_everything() {
        for endpoint in ["memory", "mem://", "rocksdb:///tmp/x", "surrealkv:///tmp/y"] {
            let caps = capabilities_of(endpoint).expect("local engines are known schemes");
            assert!(caps.grants(Capability::Backup), "{endpoint}");
            assert!(caps.grants(Capability::LiveQueries), "{endpoint}");
        }
    }

    #[test]
    fn a_local_engine_satisfies_every_requirement() {
        let all = Requirements::none().with_backup().with_live_queries();
        assert!(check("memory", all).is_ok());
        assert!(check("rocksdb:///tmp/x", all).is_ok());
    }

    #[test]
    fn websocket_is_rejected_when_backup_is_required() {
        let err = check("ws://example.test", Requirements::none().with_backup())
            .expect_err("ws cannot export");
        let StoreError::Db(inner) = &err else {
            panic!("expected a database-class error, got {err:?}");
        };
        assert!(inner.is_configuration(), "{inner}");
        // Never retried: a misconfiguration does not fix itself.
        assert!(!err.is_conflict());
        assert!(!err.is_shutdown());
    }

    /// The live-query rejection reuses the SDK's own structured detail, so a
    /// caller matching on it sees the same thing whether the failure arrives
    /// here or later from the SDK.
    #[test]
    fn http_rejection_carries_the_sdk_live_query_detail() {
        let err = check(
            "http://example.test",
            Requirements::none().with_live_queries(),
        )
        .expect_err("http cannot subscribe");
        let StoreError::Db(inner) = &err else {
            panic!("expected a database-class error, got {err:?}");
        };
        assert_eq!(
            inner.configuration_details(),
            Some(&ConfigurationError::LiveQueryNotSupported)
        );
    }

    #[test]
    fn requiring_nothing_accepts_any_known_endpoint() {
        for endpoint in ["ws://example.test", "http://example.test", "memory"] {
            assert!(check(endpoint, Requirements::none()).is_ok(), "{endpoint}");
        }
    }

    /// An unknown scheme defers to the connection attempt, which has a better
    /// message than anything this module could invent.
    #[test]
    fn unknown_schemes_defer_to_the_connection_attempt() {
        let all = Requirements::none().with_backup().with_live_queries();
        assert!(check("bogus://example.test", all).is_ok());
        assert!(check("not-an-endpoint", all).is_ok());
    }
}
