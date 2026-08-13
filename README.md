# catgraph-surreal

SurrealDB persistence for [catgraph](https://github.com/sustia-llc/catgraph)'s
category-theoretic structures — terms, cospans, and parameter weights — embedded
or over a server connection.

> **Status: early.** The substrate is in place — the error type and its retry
> classifiers, the label codec, the content-address newtypes, and a
> capability-checked connection handle — and three repositories with it: the
> content-addressed term store, the cospan store with its complete canonical
> key, and the weight store on a bit-exact byte lane. Every one of them
> revalidates what it loads. The lineage and document tiers and the notification
> bus land next.

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
`Deserialize` does not re-run the type check, and every database-side guard is
void under `OPTION IMPORT`. (On a default embedded connection the `PERMISSIONS`
layer specifically is never evaluated — `ASSERT` and `READONLY` *do* run there,
and the store's collision alarm relies on that — but none of it is the trust
boundary.) So every load verifies the content address against the stored bytes
first, then runs four checks in a fixed order — JSON parse, depth, arity
well-formedness, then a re-run of the type check against the stored signature —
and the order is load-bearing: skipping the arity screen makes the next step
abort rather than return an error.

**The generator type's `Serialize` must be deterministic — and
equality-faithful.** No `HashMap` or `HashSet` in a generator's serde
representation: their iteration order is unspecified, so two runs would encode
one term two ways, producing two addresses and two rows for one term. Debug
builds round-trip every encoding and assert the bytes reproduce, which catches
this at the first write. Equality-faithful is the subtler half: values the
consumer considers equal must serialize to equal bytes, and floats are the
standing hazard — `-0.0` and `0.0` compare equal but encode differently (two
addresses for one morphism), and a `NaN` color fails to encode with an error
that never names the float. Prefer integral color representations, or
canonicalize floats before they reach serde.

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

## Cospans

`CospanStore` keeps two identities side by side, and the difference between them
is the contract.

**A cospan's record id is the content address of its *presentation*** — the two
leg maps and the labelled apex, exactly as built. Legs are stored as ordered
arrays rather than as graph edges, because a leg is an ordered function: as
arrays the ordering is structural, while as edges every read would need an index
column and an `ORDER BY`, and an edge lost anywhere in that pipeline would
silently change which morphism the record describes rather than failing.

**The `canon_key` column is the *morphism's* identity, and it is `UNIQUE`.** Two
parallel cospans are equal as morphisms exactly when some bijection of their
apexes commutes with both legs, and catgraph's `CospanCanon` decides that — so
the key is a **complete** invariant, unlike the term tier's `nf_class`, which is
only sound. The consequence is worth stating plainly: **writing a second,
differently presented spelling of an already-stored morphism is refused**, with
`StoreError::Duplicate`. The morphism is already there, at the address
`find_by_canon` returns. Refusing keeps the surprise where a caller can see it;
silently returning the stored presentation's address would mean `get(put(c))`
handing back a cospan structurally unlike `c` with nothing said.

The store computes that key itself. `CospanCanon` carries no serializable form —
its equivalence data is private, and standard-library hash output is not stable
across processes — but the data is fully recomputable from the public accessors,
so the store re-derives it, encodes it deterministically, and hashes that. A
proptest asserts the equivalence in **both** directions against
`Cospan::canonical_form`: equal keys must mean equal morphisms, or the index
would refuse a genuinely new one; equal morphisms must mean equal keys, or the
store would hold duplicates it promised not to.

One more guard exists because catgraph cannot provide it: `Cospan::new`
bounds-checks legs against the apex under `debug_assert!` only, so a release
build accepts an out-of-bounds leg and defers the failure to a panic somewhere
else. This store bounds-checks on both sides — on write, so such a value never
reaches disk, and on load, where it arrives as a tampered column.

## Weights

`WeightStore` keeps `RModule<f64>` coordinates as raw **little-endian IEEE-754
bytes**, eight per coordinate, beside a `dim` column and a `finite` flag.

Bytes rather than native floats, because of the export path. SurrealDB's value
layer does round-trip non-finites bit-exactly through a store and a load — but a
dump renders every `NaN`, whatever its sign or payload, as the bare literal
`NaN`. So a checkpoint of a float column is lossy precisely where a training run
most wants the evidence. Bytes export as `b"<HEX>"` and come back byte-identical,
they sidestep the serde-JSON lane (which turns non-finites into `null` and then
fails to read that back), and they are index-safe: float index keys go through a
decimal encoding that collapses `±NaN` onto one key and `-0.0` onto `0.0`.

A row is identified by `(genome, gen_key)`, two **opaque caller-supplied
strings** carried by a unique index. The store derives no meaning from either
half; key derivation and its stability guarantee stay with the caller.

**Weights are write-once under their key.** Every column is `READONLY`:
re-storing an identical vector is a no-op, and storing a different one is refused
with `StoreError::ReadOnly`. A training loop that produces a new vector therefore
supplies a new `gen_key`. Letting a key's value change underneath it would make
every row referencing a `(genome, gen_key)` pair ambiguous about which vector it
meant, which is the one property lineage and checkpoint data exist to have.

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

Two variants look retryable and are not, so they are worth naming: `Duplicate`
(a unique index refusing a write — the equivalent record is already stored, and
finding it is a read) and `ReadOnly` (a write-once column refusing a *changed*
value). Both classify as neither conflict nor shutdown.

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
