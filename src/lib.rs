//! SurrealDB persistence for catgraph's category-theoretic structures.
//!
//! This crate stores and reloads the structures catgraph builds — terms, cospans,
//! parameter weights — in SurrealDB, embedded or over a server connection.
//!
//! # Status
//!
//! Early scaffold. What is here today is the substrate the repositories will be
//! built on: the [error type](error) and its retry classifiers, the
//! [label codec](codec) bridging catgraph's generic labels to their stored form,
//! the [term address](addr) newtype, and a [capability-checked](capability)
//! [`Store`] handle. The repositories themselves, the schema, and the
//! notification bus land next.
//!
//! ```no_run
//! use catgraph_surreal::StoreBuilder;
//!
//! # async fn example() -> catgraph_surreal::Result<()> {
//! let store = StoreBuilder::new("memory")
//!     .namespace("catgraph")
//!     .database("main")
//!     .require_backup()
//!     .connect()
//!     .await?;
//! # let _ = store;
//! # Ok(())
//! # }
//! ```
//!
//! # Engines
//!
//! Each engine is behind a cargo feature, and `default` is `rocksdb`:
//!
//! | Feature | Endpoint | Use |
//! |---|---|---|
//! | `mem` | `memory` | Tests and correctness work |
//! | `rocksdb` | `rocksdb://path` | The primary durable engine |
//! | `kv` | `surrealkv://path` | Edge candidate; see the caveat below |
//! | `server` | `ws://…`, `http://…` | Server-tier client |
//! | `wasm` | `indxdb://name` | Browser |
//!
//! Two engine caveats worth knowing before choosing one:
//!
//! - **SurrealKV is the unsoaked engine.** It is documented as beta for
//!   embedded use, detects write conflicts only, and has no durability soak
//!   history behind it here. Data whose loss would be silent should not live
//!   on it until a soak says otherwise.
//! - **Conflict behaviour differs per engine.** The in-memory engine aborts on
//!   read conflicts too, RocksDB detects at commit time, SurrealKV detects write
//!   conflicts only. Retry tuning measured on the memory engine does not
//!   transfer; tune against the engine actually deployed.
//!
//! # Design commitments
//!
//! Two are worth stating up front, because they are load-bearing rather than
//! stylistic:
//!
//! - **Deserialization is not validation.** A term read back out of the database
//!   has not been re-checked, and trusting it is how a corrupt document becomes
//!   a panic deep inside an interpreter. Loading revalidates; see
//!   [`RevalidationStage`].
//! - **Keys are caller-supplied and stable.** The store indexes key strings and
//!   enforces uniqueness where asked, but deriving them — and guaranteeing their
//!   stability — stays with the caller. Standard-library hash output in
//!   particular is not stable across processes and must never be persisted.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

// Compile the README's code fences as doctests, so the Usage example cannot
// drift from the real API without CI noticing.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}

pub mod addr;
pub mod capability;
pub mod codec;
pub mod error;
pub mod store;

pub use addr::TermAddr;
pub use capability::{Capability, EndpointCapabilities, Requirements};
pub use codec::LabelCodec;
pub use error::{Result, RevalidationStage, StoreError};
pub use store::{Store, StoreBuilder};
