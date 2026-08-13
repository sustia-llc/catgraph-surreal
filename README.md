# catgraph-surreal

SurrealDB persistence for [catgraph](https://github.com/sustia-llc/catgraph)'s
category-theoretic structures — terms, cospans, and parameter weights — embedded
or over a server connection.

> **Status: early scaffold.** What is here is the substrate the repositories will
> be built on: the error type and its retry classifiers, the label codec, the
> term-address newtype, and a capability-checked connection handle. The
> repositories, the schema, and the notification bus land next.

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

```rust,no_run
use catgraph_surreal::StoreBuilder;

# async fn example() -> catgraph_surreal::Result<()> {
let store = StoreBuilder::new("rocksdb:///var/lib/catgraph")
    .namespace("catgraph")
    .database("main")
    .require_backup()
    .connect()
    .await?;
# Ok(())
# }
```

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

- **SurrealKV commits are not fsyncs.** Every transaction commits with eventual
  durability, leaving the sync to the operating system. Data whose loss would be
  silent should not live there without a durability soak first.
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
- Shutdown has **no structured discriminator** — the engine maps it onto the same
  detail as an ordinary connection failure — so `is_shutdown()` necessarily keys
  on message text as well as error class.

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
