# catgraph-surreal

SurrealDB persistence for [catgraph](https://github.com/sustia-llc/catgraph)'s
category-theoretic structures — terms, cospans, parameter weights, rewrite
lineage — plus a consumer-shaped document tier and a durable notification bus,
embedded or over a server connection.

> **Status: early, but complete in outline.** The substrate is in place — the
> error type with its retry classifiers and the retry loop that implements them,
> the label codec, the content-address newtypes, and a capability-checked
> connection handle — and every tier is built on it. Each one revalidates what it
> loads rather than trusting what came off disk.

| Tier | What it stores |
|---|---|
| [Terms](#terms) | Content-addressed `ColoredExpr`s, revalidated on load |
| [Cospans](#cospans) | Presentations, with a complete canonical key |
| [Weights](#weights) | Coordinate vectors on a bit-exact byte lane |
| [Lineage](#lineage) | Rule sets, optimizer traces, and the derivation graph |
| [Documents](#documents) | Consumer-shaped serde types, mutable or write-once |
| [Bus](#notification-bus) | Durable notifications with a live wakeup |

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

One more guard sits on both sides of the store. `Cospan::new` bounds-checks
legs against the apex in every build profile, but `Cospan::new_unchecked` and
`Cospan::add_boundary_node_unchecked` check only under `debug_assert!`, so a
release build accepts an out-of-bounds leg through either. This store
bounds-checks on write, so such a value never reaches disk, and on load, where
it arrives as a tampered column.

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

## Lineage

`LineageStore` keeps three things: the **rule sets** an optimizer ran under, the
**traces** it produced, and the **derivation graph** those traces imply.

A rule set is stored as the canonical JSON of its equation pairs, and loading
rebuilds every rule through `RewriteRule::new` — which is where the four
conditions a rewrite site relies on are actually checked. Serde checks none of
them, so a rule that skipped the constructor would fail at a match site instead
of at the boundary, or not fail at all and match where it should not have.

A run records its endpoints, its costs, and its steps — for each step, which rule
fired and on which hyperedges. **`cost_model` is a mandatory column**, because
costs are sums of a per-generator weighting that arrives as a closure and cannot
be persisted: two runs measured under different weightings produce numbers that
look comparable and are not. A run that did not say which weighting produced its
numbers has recorded numbers nobody can read.

A stored trace *replays*. `replay_run` loads the run's start term, rebuilds its
rules in stored order, and hands the steps to catgraph's `replay`, which
re-derives each one against the state it has reached — so a trace that is not a
legal derivation of that start under those rules comes back as an error rather
than as an endpoint. Rule *order* is load-bearing: a step binds an index, not a
rule identity. The `replayable` column says which side of the change a row was
written on — `true` for runs this build records, `false` for older rows.

The `derives` edge is a `TYPE RELATION` table whose record id is the digest of
the tuple it represents: parent, child, and the run that derived one from the
other. That makes writing an edge idempotent without a second uniqueness
mechanism, and it is written with `RELATE OR UPDATE` — the explicit form of what
the engine does anyway, since a plain `RELATE` onto an existing edge id logs a
warning and overwrites with nothing surfacing to the client.

A run's trace and its edge are written in **one transaction**; the endpoint terms
are written before it, outside, because content-addressed inserts are idempotent
single statements two writers can race harmlessly.

## Documents

`DocStore<T>` and `ManifestStore<T>` store consumer-defined serde types the
crate knows nothing about — a solver state, an archive entry, a registration
manifest — without depending on the crate that defines them. What it still
guarantees is that what comes back is what went in: a digest column is re-derived
and compared before anything is deserialized.

The payload column is `object FLEXIBLE`, which is what lets it hold a shape the
schema does not describe. This is a **JSON lane**, with the limit that implies:
a non-finite float becomes `null` on the way in, silently and irreversibly.
Consumers needing bit-exact floats want the weight tier's byte lane instead.

`ManifestStore` is the write-once half, and immutability is enforced four ways:
no update or delete on the handle, `READONLY` on every column, a write-once
`ASSERT` on every column whose type evaluates one, and synchronous refusal events
on update and delete. On an embedded connection with no root user, permissions
are never evaluated and `OPTION IMPORT;` disables the rest for a statement —
**store-side immutability is the trust boundary; the database backs it up.** So a
restore re-verifies digests rather than trusting the replay, which is what
`verify_all()` is for.

Registration is therefore create-only: re-registering an id is refused whatever
the contents, with `ReadOnly` naming the column when they differ and `Immutable`
naming the event when they are identical.

## Notification bus

`BusWriter` and `BusReader` are a durable bus with a live wakeup. Two tiers, one
mechanism:

- **Tier 1 — low volume, latency sensitive.** Subscribe for the wakeup, read the
  durable rows when it fires.
- **Tier 2 — high volume.** Poll `next_batch()` on an interval; do not subscribe.

What is never correct is a live-only consumer. Notifications are best-effort and
at-most-once, never replayed, with no gap detection and no structurally
guaranteed ordering — so a listener that never reads rows loses events at every
disconnect and under backpressure, quietly. **The row is the event; the
notification says "look again".**

Each stream numbers its events from zero, contiguously, from a per-stream counter
allocated inside the same transaction that writes the event. That counter is also
the write-skew fence: two publishers that only *read* it could both take the same
number, but because both also write it they collide and one is refused as a
retryable conflict. A publisher also absorbs a bounded number of *lost* number
races internally, with jittered backoff; past that bound the condition surfaces
classified as a conflict, so `retry()` covers it like any other contention.

Catch-up reads the change feed from a persisted high-water mark and asserts
contiguity, which turns the two ways a bus can go wrong into named errors:
`BusGap` when a stream's numbers skip, and `BusStale` when the cursor is old
enough that retention may already have discarded events — silently, since expiry
signals nothing. Both have one remedy: `rebaseline()`, which adopts the durable
rows and starts again. A restore needs it too, because an import emits no
change-feed entries and no notifications at all.

The mark is *both* the versionstamp and the per-stream sequence expectations, and
both live on the cursor row: expectations kept only in memory reset at every
reader restart, so a hole punched while a consumer was down would pass as a first
sighting. A poll that finds nothing still writes the cursor, which is what keeps
a healthy reader on a quiet bus from ageing into `BusStale` while doing
everything right.

A third named error is not a data condition at all. `BusRaced` means a **second
reader under the same consumer id** moved the shared cursor first — consumer ids
name cursors, so sharing one means sharing a cursor. Neither retrying nor
re-baselining helps; give each reader an id of its own. The reader that lost is
left untouched and stays usable.

`subscribe()` hands the caller a `Stream` and this crate spawns nothing: the
caller owns the loop, the cancellation, and ending the subscription — which is
done by **dropping the stream**, since a raw `KILL` does not end one on an
embedded engine.

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
- **Conflict behaviour is engine-specific.** The in-memory engine additionally
  aborts on many read conflicts — but not exhaustively: its commit-time
  detection has been observed to miss a racing write, so contended flows in
  this store split the read and the write across transactions behind a
  `WHERE`-guarded advance or a `CREATE` collision rather than resting on
  detection. RocksDB detects at commit time against a pinned snapshot;
  SurrealKV detects write conflicts only. Retry tuning measured on the memory
  engine does not transfer.

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

`retry()` is the reference implementation of that table, and the shape it
enforces is the point: the operation it takes must be the **whole** `begin()` …
`commit()` unit, restartable from the top.

Several variants look retryable and are not, so they are worth naming:
`Duplicate` (a unique index refusing a write — the equivalent record is already
stored, and finding it is a read), `ReadOnly` (a write-once column refusing a
*changed* value), `Immutable` (a write-once event refusing an update or delete),
`BusGap` / `BusStale` (events lost, or a cursor that can no longer be trusted —
both call for `rebaseline()`, not another attempt), and `BusRaced` (two readers
sharing one consumer id, whose remedy is a second id rather than a second try).

Three subtleties are documented on the type itself, because getting any of them
wrong loses data rather than merely erroring:

- `is_conflict()` must be applied to the **whole** `begin()` … `commit()` unit,
  including the result of `commit()`. Some engines only detect conflicts at
  commit time, so a transaction whose statements all succeeded can still fail
  there.
- The structured conflict discriminator **does not survive a client
  transaction's `commit()`** — that path converts the failure with a plain
  internal-error constructor and the kind is dropped. Since a client transaction
  is exactly where the interesting conflicts happen, a classifier keyed on the
  discriminator alone would classify almost none of them, so both shapes are
  checked.
- Shutdown has **no structured discriminator** at all, and it surfaces in
  *different error classes* depending on the path (the embedded commit slot
  reports it as a query-class error; RPC-handler paths as connection-class) — so
  `is_shutdown()` necessarily keys on message text alone, across classes.

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
