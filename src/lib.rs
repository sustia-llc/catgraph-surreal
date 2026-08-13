//! SurrealDB persistence for catgraph's category-theoretic structures.
//!
//! This crate stores and reloads the structures catgraph builds — terms, cospans,
//! parameter weights — in SurrealDB, embedded or over a server connection.
//!
//! # Status
//!
//! Early. The substrate is in place — the [error type](error) and its retry
//! classifiers, the [label codec](codec) bridging catgraph's generic labels to
//! their stored form, the [content-address newtypes](addr), and a
//! [capability-checked](capability) [`Store`] handle — and three repositories
//! with it: the content-addressed [term store](term_store), the
//! [cospan store](cospan_store) with its complete canonical key, and the
//! [weight store](weight_store) on a bit-exact byte lane. The lineage and
//! document tiers and the notification bus land next.
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
//! - **Keys are caller-supplied where the store cannot derive them — and
//!   store-derived where it can.** The term tier derives its own record ids
//!   (a term's id *is* the digest of its canonical encoding; [`TermAddr`] is
//!   never accepted from outside). The tiers where identity genuinely lives
//!   with the consumer — document ids, weight keys, canonical cospan keys —
//!   take caller-supplied stable strings, and there deriving them and
//!   guaranteeing their stability stays with the caller. Standard-library hash
//!   output in particular is not stable across processes and must never be
//!   persisted.

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
pub mod cospan;
pub mod cospan_store;
pub mod error;
pub mod schema;
pub mod store;
pub mod term;
pub mod term_store;
pub mod weight;
pub mod weight_store;

pub use addr::{CospanAddr, TermAddr};
pub use capability::{Capability, EndpointCapabilities, Requirements};
pub use codec::LabelCodec;
pub use cospan::CospanRecord;
pub use cospan_store::CospanStore;
pub use error::{Result, RevalidationStage, StoreError};
pub use store::{Store, StoreBuilder};
pub use term::TermRecord;
pub use term_store::TermStore;
pub use weight::WeightRecord;
pub use weight_store::WeightStore;
