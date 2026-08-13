# catgraph-surreal

SurrealDB persistence for [catgraph](https://github.com/sustia-llc/catgraph)'s
category-theoretic structures — terms, cospans, and parameter weights — embedded
or over a server connection.

> **Status: early.** The substrate is in place — the error type and its retry
> classifiers, the label codec, the term-address newtype, and a
> capability-checked connection handle — and the first repository with it: the
> content-addressed term store, its schema, and the revalidation discipline that
> guards every load. The cospan, weight, lineage, and document tiers and the
> notification bus land next.

## Why this crate exists

catgraph is a pure, in-memory library: it builds and manipulates structures but
takes no opinion on where they live between runs. This crate is the seam. It
keeps the dependency direction one-way — catgraph never learns about a database —
and it keeps the storage concerns that genuinely need care (revalidating what
comes back off disk, classifying which failures are worth retrying, addressing
terms by content) in one place instead of spread across every consumer.

It is a **fresh implementation**, not a port. An earlier crate of the same name
predates SurrealDB 3.x and is unrelated to this lineage.

## Install

```toml
[dependencies]
catgraph-surreal = { git = "https://github.com/sustia-llc/catgraph-surreal" }
```

The crate is public but unpublished. Its catgraph dependencies are git-tag
sources over HTTPS, which crates.io does not permit in a published package —
fetching over HTTPS from CI, without credentials, is the supported path and is
exercised as a requirement rather than assumed.

## Usage

```rust
use catgraph_surreal::{Result, StoreBuilder};

async fn open() -> Result<()> {
    let store = StoreBuilder::new("rocksdb:///var/lib/catgraph")
        .namespace("catgraph")
        .database("main")
        .require_backup()
        .connect()
        .await?;
    let _ = store;
    Ok(())
}
```

## Terms

`TermStore` keeps catgraph terms addressed by the digest of their canonical
encoding: a term's record id *is* its content address, which makes storing one
idempotent and makes two writers racing on the same term write the same bytes.

Three decisions are worth knowing before using it.

**A term is stored as one opaque JSON string, not as a native nested object.**
Expanding it would break twice: the SDK writes `usize` through an unchecked
`as i64` cast, and catgraph deliberately produces `usize::MAX` saturation
sentinels — one of those becomes `-1` with nothing raised anywhere — and content
addressing needs the stored bytes to be the bytes that were hashed. The columns
beside it (`signature`, arities, `depth`, generator count, `nf_class`) are all
derived from that string, and every one is re-derived and compared on load.

**Loading always revalidates. There is no trusting path.** `ColoredExpr`'s
`Deserialize` does not re-run the type check; database-side guards are void
under `OPTION IMPORT` and are never evaluated at all on a default embedded
connection. So every load runs four checks in a fixed order — JSON parse, depth,
arity well-formedness, then a re-run of the type check against the stored
signature — and the order is load-bearing: skipping the arity screen makes the
next step abort rather than return an error.

**The generator type's `Serialize` must be deterministic.** No `HashMap` or
`HashSet` in a generator's serde representation: their iteration order is
unspecified, so two runs would encode one term two ways, producing two addresses
and two rows for one term. Debug builds round-trip every encoding and assert the
bytes reproduce, which catches this at the first write.

Two limits follow from the encoding. `PropExpr` serializes externally tagged, so
each nesting level costs two JSON containers against `serde_json`'s 128-container
parser limit — roughly **64 levels**, well under catgraph's own structural limit
of 256. That is a safety property and a round-trip cap, so encoding parses its
own output back before returning: what this store writes, it can read.

Term identity has two columns with different strengths. The record id is
*representation-level* — distinct syntax is distinct identity, which is what a
population store wants. `nf_class` is a **sound semantic bucket**: equal values
mean the terms are equal in the free symmetric monoidal category, while equal
morphisms may still land in different buckets. It is indexed and deliberately
*not* unique; duplicates within a bucket are expected. Complete semantic
deduplication is an in-process concern over a working set, not something to
persist.

## Engines

Every engine sits behind a cargo feature; `default` is `rocksdb`.

| Feature | Endpoint | Use |
|---|---|---|
| `mem` | `memory` | Tests and correctness work |
| `rocksdb` (default) | `rocksdb://path` | Primary durable engine |
| `kv` | `surrealkv://path` | Edge candidate — see below |
| `server` | `ws://…`, `http://…` | Server-tier client |
| `wasm` | `indxdb://name` | Browser |

The SurrealDB dependency is taken with `default-features = false` on purpose:
the SDK's defaults pull a WebSocket stack and TLS into what would otherwise be a
purely embedded build.

Two engine caveats worth knowing before picking one:

- **SurrealKV is the unsoaked engine.** It is documented as beta for embedded
  use and has no durability soak history behind it here. Data whose loss would
  be silent should not live there until a soak says otherwise.
- **Conflict behaviour is engine-specific.** The in-memory engine aborts on read
  conflicts too, RocksDB detects at commit time, SurrealKV detects write
  conflicts only. Retry tuning measured on the memory engine does not transfer.

### Why `server` enables both protocols

The two remote transports are complementary, not interchangeable: **HTTP can
export and import but cannot subscribe; WebSocket can subscribe but cannot
export.** A server-tier client that both checkpoints and listens needs both, so
the `server` feature turns on both rather than offering a choice that would
quietly be wrong half the time.

`server` does **not** currently enable a TLS backend, so `wss://` and `https://`
are not yet usable — plain `ws://` and `http://` are. Adding a TLS feature is a
follow-up.

The connection handle checks these capabilities against what the store was told
it needs **before opening a connection**, so a mismatch fails at construction.
Left to the SDK it would surface at the first export or subscription — possibly
hours in, and in the export case as an error carrying no structured
discriminator at all.

## Error handling

`StoreError` sorts every failure into one of three tiers, via two inherent
classifiers:

| Tier | Predicate | Correct response |
|---|---|---|
| Conflict | `is_conflict()` | Retry in-process, bounded backoff with jitter |
| Shutdown | `is_shutdown()` | Reconnect, then retry — **never persist as permanent** |
| Everything else | neither | No blind retry |

Two subtleties are documented on the type itself, because getting either wrong
loses data rather than merely erroring:

- `is_conflict()` must be applied to the **whole** `begin()` … `commit()` unit,
  including the result of `commit()`. Some engines only detect conflicts at
  commit time, so a transaction whose statements all succeeded can still fail
  there.
- Shutdown has **no structured discriminator**, and it surfaces in *different
  error classes* depending on the path (the embedded commit slot reports it as a
  query-class error; RPC-handler paths as connection-class) — so `is_shutdown()`
  necessarily keys on message text alone, across classes.

## MSRV

**1.94**, measured rather than declared by habit: the maximum required Rust
version across the resolved dependency graph, checked over every feature combo
(`mem`, `rocksdb`, `kv`, `server`). It is forced by `fastnum`, a hard dependency
of `surrealdb-core`, and is identical across all feature combinations — no
feature raises it. `roaring` is the next highest at 1.90.

Re-measure whenever a dependency moves.

## Development

```sh
cargo test --workspace
cargo test --workspace --no-default-features --features mem
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

CI additionally runs `cargo check` over each engine feature in isolation.

The `wasm` feature is **excluded from the host feature matrix** and checked on a
separate `wasm32-unknown-unknown` lane, because it cannot build for a native
target at all: the `indxdb` engine is built on browser APIs that are neither
`Send` nor `Sync`, which cannot satisfy the transaction bounds SurrealDB's core
requires natively. `cargo check --features wasm` on the host is expected to fail;
`cargo check --features wasm --target wasm32-unknown-unknown` is the supported
invocation.

## License

MIT — see [LICENSE](LICENSE).
