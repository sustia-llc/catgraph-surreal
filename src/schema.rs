//! Idempotent schema bootstrap, and the drift guard that keeps it honest.
//!
//! # Why the DDL is written the way it is
//!
//! Three properties of SurrealDB's DDL shape every statement below, and getting
//! any of them wrong is silent rather than loud:
//!
//! - **Table first, then fields — always.** Re-issuing `DEFINE TABLE` *drops
//!   that table's field definitions*. The order here is not cosmetic: it is what
//!   makes a replay of this DDL (a restore, a second process, a future
//!   migration) end with the fields present rather than a `SCHEMAFULL` table
//!   that accepts nothing.
//! - **`IF NOT EXISTS`, never `OVERWRITE`.** `DEFINE INDEX … OVERWRITE` is a
//!   synchronous delete plus a full rebuild — on *every* store open. On a table
//!   of any size that turns opening a connection into an unbounded stall.
//!   `IF NOT EXISTS` makes the bootstrap genuinely idempotent instead.
//! - **A `SCHEMAFULL` table with no undeclared columns.** Every column is
//!   declared with a non-optional, non-`any` type. That is a requirement, not a
//!   preference: an `ASSERT` is skipped when a field is unset *and* its type
//!   admits `NONE`, so `option<T>` and `any` would quietly disarm the guards
//!   below.
//!
//! # Three kinds of field definition
//!
//! `INFO FOR TABLE` reports more field definitions than the DDL writes, and the
//! drift guard compares the live set **exactly**, so all three kinds have to be
//! accounted for:
//!
//! - **Columns** — what the DDL declares and what a read projects. The read
//!   projections are derived from these lists.
//! - **Nested fields** — declared by the DDL but
//!   not projectable, because they live inside a column. A `rewrite_run`'s
//!   `steps.*.rule` is one: a read selects `steps`, never `steps.*.rule`.
//!   Declaring them is not optional decoration. In a `SCHEMAFULL` table an
//!   `object` column rejects nested keys nobody declared — `Found field
//!   'steps[0].rule', but no such field exists for table 'rewrite_run'` — so an
//!   undeclared step shape does not store badly, it does not store at all.
//!   Either declare the nested shape, as `rewrite_run` does, or declare the
//!   column `FLEXIBLE`, as the document tier does. (Verified at SurrealDB 3.2.4:
//!   the refusal is loud. Nothing here relies on that — both tables would be
//!   wrong either way — but it is worth knowing which failure to expect.)
//! - **Implicit fields** — created by the
//!   engine, never written here. Declaring `TYPE array<int>` creates a `.*`
//!   element definition beside it; `TYPE RELATION IN t OUT t` creates `in` and
//!   `out`. Writing them in the DDL would be **dead text**: the parent statement
//!   already created them, so a following `DEFINE FIELD … IF NOT EXISTS` no-ops
//!   — verified, including that adding `READONLY` to such a statement changes
//!   nothing, because the statement never runs.
//!
//! Implicit fields carry no `READONLY` clause, and that is not a hole. An
//! element write (`SET dom_leg[0] = 9`) changes the parent array's value, so the
//! *parent's* `READONLY` refuses it, and a relation's `in`/`out` are written by
//! `RELATE` itself, which refuses to move an existing edge's endpoints. Both are
//! pinned by integration tests, because neither is obvious from the definitions
//! alone.
//!
//! # No numeric column joins a unique index
//!
//! Index keys normalise numbers: `0`, `0.0`, and `0dec` become one key. A unique
//! index that includes a numeric column therefore refuses rows that are not
//! duplicates. Every unique index here is made of `string` columns only, and
//! that is a standing constraint rather than an accident of the current schema.
//! Non-unique indexes are unaffected, which is why `bus_stream_seq` may pair a
//! string with an integer.
//!
//! # The drift guard compares definitions, not names
//!
//! [`assert_term_schema`] and its siblings compare the **engine-rendered
//! definition strings** (`INFO FOR DB` / `INFO FOR TABLE`) against the ones this
//! build expects — not merely the field, index, and event *names*. A name-only
//! check is blind to exactly the drift that disarms the guards: `ALTER TABLE …
//! SCHEMALESS` keeps every field defined, a dropped `READONLY` clause keeps the
//! field's name, a dropped `UNIQUE` keeps the index's name, a shortened
//! `CHANGEFEED` retention keeps the table's name, and a field re-typed to
//! `option<T>` still answers to its name. The definition strings are the
//! engine's own normalization at SurrealDB 3.2.4, pinned by integration tests
//! against a fresh bootstrap — an SDK upgrade that changes the rendering fails
//! those tests loudly, which is the moment to re-pin the strings deliberately
//! rather than discover the change in production.
//!
//! # The guards are defence in depth, not the trust boundary
//!
//! On an embedded connection with no root user configured, table and field
//! permissions are never evaluated, and prefixing `OPTION IMPORT;` to a query
//! disables `READONLY`, `ASSERT`, type processing, and events for that query.
//! So the `READONLY` columns, the write-once events, and the id-format `ASSERT`
//! do not *guarantee* anything: the store's own validation is the trust boundary
//! and the database backs it up. This is why [`crate::term`]'s revalidation runs
//! on every load, why [`crate::cospan`] bounds-checks every leg it reads, and
//! why a restore re-verifies record ids rather than trusting the replay.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::time::Duration;

use surrealdb::Surreal;
use surrealdb::engine::any::Any;

use crate::error::{Result, StoreError};

// ---------------------------------------------------------------------- terms

/// The table holding content-addressed terms.
pub const TERM_TABLE: &str = "term";

/// Every column the [`TERM_TABLE`] schema declares, in the order the DDL
/// defines them.
///
/// The read projection is derived from this list, and the drift guard's
/// expected definitions are pinned to the same names by a unit test — adding a
/// column means touching this list, the DDL, and [`TERM_FIELD_DEFINITIONS`]
/// together, and any mismatch is caught in-process rather than at run time.
pub const TERM_FIELDS: [&str; 9] = [
    "id",
    "term_json",
    "codec",
    "signature",
    "source_arity",
    "target_arity",
    "depth",
    "generator_count",
    "nf_class",
];

/// Every index the [`TERM_TABLE`] schema declares.
///
/// `nf_class` is indexed but **not** `UNIQUE`, and that is the whole design:
/// it is a *sound semantic bucket*, so equal values mean the terms are equal in
/// the symmetric monoidal category, while distinct-but-equal terms may still
/// land in different buckets. Duplicates within a bucket are expected. Term
/// identity proper is the record id, which *is* the content address — no
/// separate unique key column exists or is wanted.
pub const TERM_INDEXES: [&str; 1] = ["term_nf_class"];

/// The term table's own definition, as the engine renders it.
///
/// `SCHEMAFULL` is the load-bearing token: `ALTER TABLE term SCHEMALESS`
/// renders as `TYPE NORMAL SCHEMALESS` and fails the comparison, which is the
/// entire point — a schemaless term table accepts undeclared columns and
/// silently disarms every field guard.
pub const TERM_TABLE_DEFINITION: &str = "DEFINE TABLE term TYPE NORMAL SCHEMAFULL PERMISSIONS NONE";

/// Every column's definition, as the engine renders it.
///
/// These strings carry the guards the name-only view cannot see: the `READONLY`
/// clause on every derived column, the non-optional types, and the id `ASSERT`.
pub const TERM_FIELD_DEFINITIONS: [(&str, &str); 9] = [
    (
        "id",
        "DEFINE FIELD id ON term TYPE string ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/ PERMISSIONS FULL",
    ),
    (
        "term_json",
        "DEFINE FIELD term_json ON term TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "codec",
        "DEFINE FIELD codec ON term TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "signature",
        "DEFINE FIELD signature ON term TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "source_arity",
        "DEFINE FIELD source_arity ON term TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "target_arity",
        "DEFINE FIELD target_arity ON term TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "depth",
        "DEFINE FIELD depth ON term TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "generator_count",
        "DEFINE FIELD generator_count ON term TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "nf_class",
        "DEFINE FIELD nf_class ON term TYPE string READONLY PERMISSIONS FULL",
    ),
];

/// Every declared index's definition, as the engine renders it.
pub const TERM_INDEX_DEFINITIONS: [(&str, &str); 1] = [(
    "term_nf_class",
    "DEFINE INDEX term_nf_class ON term FIELDS nf_class",
)];

/// The term table's DDL.
///
/// Every statement is `IF NOT EXISTS`, so replaying this is a no-op against an
/// already-bootstrapped database.
///
/// The `id` `ASSERT` is the content-address format guard. It is genuinely
/// defence in depth and nothing more, for two reasons that are easy to forget:
/// an `ASSERT` on `id` is **skipped on update** (only a create evaluates it),
/// and it is skipped entirely under `OPTION IMPORT` — so a restored dump can
/// carry ids this expression would have rejected. The primary guard is
/// [`crate::TermAddr`], which the store derives itself and never accepts from
/// outside; the post-restore path must re-verify term ids rather than assume
/// this clause ran.
///
/// Note the binding: there is no `$key` in an `ASSERT`. On the `id` field
/// `$value` is the **whole record id**, which is why the key is extracted with
/// `record::id($value)` before matching.
const TERM_DDL: &str = "\
DEFINE TABLE IF NOT EXISTS term SCHEMAFULL TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON term TYPE string
    ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/;
DEFINE FIELD IF NOT EXISTS term_json ON term TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS codec ON term TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS signature ON term TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS source_arity ON term TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS target_arity ON term TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS depth ON term TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS generator_count ON term TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS nf_class ON term TYPE string READONLY;

DEFINE INDEX IF NOT EXISTS term_nf_class ON term FIELDS nf_class;
";

/// The term table's schema, as this build declares it.
const TERM_SCHEMA: TableSchema = TableSchema {
    table: TERM_TABLE,
    ddl: Cow::Borrowed(TERM_DDL),
    table_definition: Cow::Borrowed(TERM_TABLE_DEFINITION),
    fields: &TERM_FIELD_DEFINITIONS,
    nested_fields: &[],
    // No array column, so the engine creates nothing beside them.
    implicit_fields: &[],
    indexes: &TERM_INDEX_DEFINITIONS,
    events: &[],
    write_once: true,
};

// -------------------------------------------------------------------- cospans

/// The table holding content-addressed cospan presentations.
pub const COSPAN_TABLE: &str = "cospan";

/// Every column the [`COSPAN_TABLE`] schema declares, in DDL order.
///
/// The legs are `array<int>` and the apex is `array<string>` — flattened into
/// the row rather than reified as edges, so that leg *ordering* is structural
/// (see [`crate::cospan`]).
pub const COSPAN_FIELDS: [&str; 10] = [
    "id",
    "codec",
    "dom_leg",
    "cod_leg",
    "apex",
    "dom_len",
    "cod_len",
    "apex_len",
    "scalar_count",
    "canon_key",
];

/// The unique index on a cospan's canonical key.
///
/// The contrast with the term table's `nf_class` is the point: a cospan's
/// canonical form is a **complete** invariant for equality of morphisms, so two
/// rows sharing a key would be two names for one thing. `nf_class` is only
/// *sound*, so duplicates there are expected and that index is deliberately not
/// unique.
///
/// Named, because the store reads it back: a write this index refuses reports no
/// structured discriminator, only a message naming the index.
pub const COSPAN_CANON_INDEX: &str = "cospan_canon";

/// Every index the [`COSPAN_TABLE`] schema declares.
pub const COSPAN_INDEXES: [&str; 1] = [COSPAN_CANON_INDEX];

/// The cospan table's own definition, as the engine renders it.
pub const COSPAN_TABLE_DEFINITION: &str =
    "DEFINE TABLE cospan TYPE NORMAL SCHEMAFULL PERMISSIONS NONE";

/// Every column's definition, as the engine renders it.
pub const COSPAN_FIELD_DEFINITIONS: [(&str, &str); 10] = [
    (
        "id",
        "DEFINE FIELD id ON cospan TYPE string ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/ PERMISSIONS FULL",
    ),
    (
        "codec",
        "DEFINE FIELD codec ON cospan TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "dom_leg",
        "DEFINE FIELD dom_leg ON cospan TYPE array<int> READONLY PERMISSIONS FULL",
    ),
    (
        "cod_leg",
        "DEFINE FIELD cod_leg ON cospan TYPE array<int> READONLY PERMISSIONS FULL",
    ),
    (
        "apex",
        "DEFINE FIELD apex ON cospan TYPE array<string> READONLY PERMISSIONS FULL",
    ),
    (
        "dom_len",
        "DEFINE FIELD dom_len ON cospan TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "cod_len",
        "DEFINE FIELD cod_len ON cospan TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "apex_len",
        "DEFINE FIELD apex_len ON cospan TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "scalar_count",
        "DEFINE FIELD scalar_count ON cospan TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "canon_key",
        "DEFINE FIELD canon_key ON cospan TYPE string READONLY PERMISSIONS FULL",
    ),
];

/// The element definitions the engine creates for the `array<T>` columns above,
/// as it renders them.
///
/// Nobody writes these — declaring `dom_leg TYPE array<int>` creates
/// `dom_leg.*` as a side effect — but the drift guard compares the live field
/// set exactly, so they belong in the expected set. They are kept apart from
/// [`COSPAN_FIELD_DEFINITIONS`] because they are not columns a read can project.
///
/// The absence of `READONLY` here is the engine's doing and is not a hole: an
/// element write changes the parent array's value, and the parent *is*
/// `READONLY`. See the [module documentation](self).
pub const COSPAN_ELEMENT_DEFINITIONS: [(&str, &str); 3] = [
    (
        "dom_leg.*",
        "DEFINE FIELD dom_leg.* ON cospan TYPE int PERMISSIONS FULL",
    ),
    (
        "cod_leg.*",
        "DEFINE FIELD cod_leg.* ON cospan TYPE int PERMISSIONS FULL",
    ),
    (
        "apex.*",
        "DEFINE FIELD apex.* ON cospan TYPE string PERMISSIONS FULL",
    ),
];

/// Every declared index's definition, as the engine renders it.
pub const COSPAN_INDEX_DEFINITIONS: [(&str, &str); 1] = [(
    COSPAN_CANON_INDEX,
    "DEFINE INDEX cospan_canon ON cospan FIELDS canon_key UNIQUE",
)];

/// The cospan table's DDL.
const COSPAN_DDL: &str = "\
DEFINE TABLE IF NOT EXISTS cospan SCHEMAFULL TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON cospan TYPE string
    ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/;
DEFINE FIELD IF NOT EXISTS codec ON cospan TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS dom_leg ON cospan TYPE array<int> READONLY;
DEFINE FIELD IF NOT EXISTS cod_leg ON cospan TYPE array<int> READONLY;
DEFINE FIELD IF NOT EXISTS apex ON cospan TYPE array<string> READONLY;
DEFINE FIELD IF NOT EXISTS dom_len ON cospan TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS cod_len ON cospan TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS apex_len ON cospan TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS scalar_count ON cospan TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS canon_key ON cospan TYPE string READONLY;

DEFINE INDEX IF NOT EXISTS cospan_canon ON cospan FIELDS canon_key UNIQUE;
";

/// The cospan table's schema, as this build declares it.
const COSPAN_SCHEMA: TableSchema = TableSchema {
    table: COSPAN_TABLE,
    ddl: Cow::Borrowed(COSPAN_DDL),
    table_definition: Cow::Borrowed(COSPAN_TABLE_DEFINITION),
    fields: &COSPAN_FIELD_DEFINITIONS,
    nested_fields: &[],
    implicit_fields: &COSPAN_ELEMENT_DEFINITIONS,
    indexes: &COSPAN_INDEX_DEFINITIONS,
    events: &[],
    write_once: true,
};

// -------------------------------------------------------------------- weights

/// The table holding parameter weights.
pub const WEIGHT_TABLE: &str = "weight";

/// Every column the [`WEIGHT_TABLE`] schema declares, in DDL order.
pub const WEIGHT_FIELDS: [&str; 7] = [
    "id",
    "codec",
    "genome",
    "gen_key",
    "dim",
    "coordinates",
    "finite",
];

/// The unique index on a weight row's `(genome, gen_key)` pair.
///
/// That pair is the row's declared identity — both halves are caller-supplied
/// opaque strings, which is also what keeps the no-numeric-column rule (see the
/// [module documentation](self)) satisfied. Named for the same reason as
/// [`COSPAN_CANON_INDEX`].
pub const WEIGHT_KEY_INDEX: &str = "weight_key";

/// The non-unique index on the `finite` flag.
///
/// The flag exists so "which checkpoints went non-finite?" is a query rather
/// than a scan of the store's largest table — which requires the index, not
/// just the column.
pub const WEIGHT_FINITE_INDEX: &str = "weight_finite";

/// Every index the [`WEIGHT_TABLE`] schema declares.
pub const WEIGHT_INDEXES: [&str; 2] = [WEIGHT_KEY_INDEX, WEIGHT_FINITE_INDEX];

/// The weight table's own definition, as the engine renders it.
pub const WEIGHT_TABLE_DEFINITION: &str =
    "DEFINE TABLE weight TYPE NORMAL SCHEMAFULL PERMISSIONS NONE";

/// Every column's definition, as the engine renders it.
pub const WEIGHT_FIELD_DEFINITIONS: [(&str, &str); 7] = [
    (
        "id",
        "DEFINE FIELD id ON weight TYPE string ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/ PERMISSIONS FULL",
    ),
    (
        "codec",
        "DEFINE FIELD codec ON weight TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "genome",
        "DEFINE FIELD genome ON weight TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "gen_key",
        "DEFINE FIELD gen_key ON weight TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "dim",
        "DEFINE FIELD dim ON weight TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "coordinates",
        "DEFINE FIELD coordinates ON weight TYPE bytes READONLY PERMISSIONS FULL",
    ),
    (
        "finite",
        "DEFINE FIELD finite ON weight TYPE bool READONLY PERMISSIONS FULL",
    ),
];

/// Every declared index's definition, as the engine renders it.
pub const WEIGHT_INDEX_DEFINITIONS: [(&str, &str); 2] = [
    (
        WEIGHT_KEY_INDEX,
        "DEFINE INDEX weight_key ON weight FIELDS genome, gen_key UNIQUE",
    ),
    (
        WEIGHT_FINITE_INDEX,
        "DEFINE INDEX weight_finite ON weight FIELDS finite",
    ),
];

/// The weight table's DDL.
const WEIGHT_DDL: &str = "\
DEFINE TABLE IF NOT EXISTS weight SCHEMAFULL TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON weight TYPE string
    ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/;
DEFINE FIELD IF NOT EXISTS codec ON weight TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS genome ON weight TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS gen_key ON weight TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS dim ON weight TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS coordinates ON weight TYPE bytes READONLY;
DEFINE FIELD IF NOT EXISTS finite ON weight TYPE bool READONLY;

DEFINE INDEX IF NOT EXISTS weight_key ON weight FIELDS genome, gen_key UNIQUE;
DEFINE INDEX IF NOT EXISTS weight_finite ON weight FIELDS finite;
";

/// The weight table's schema, as this build declares it.
const WEIGHT_SCHEMA: TableSchema = TableSchema {
    table: WEIGHT_TABLE,
    ddl: Cow::Borrowed(WEIGHT_DDL),
    table_definition: Cow::Borrowed(WEIGHT_TABLE_DEFINITION),
    fields: &WEIGHT_FIELD_DEFINITIONS,
    nested_fields: &[],
    implicit_fields: &[],
    indexes: &WEIGHT_INDEX_DEFINITIONS,
    events: &[],
    write_once: true,
};

// ------------------------------------------------------------------ rule sets

/// The table holding content-addressed rewrite rule sets.
pub const RULE_SET_TABLE: &str = "rule_set";

/// Every column the [`RULE_SET_TABLE`] schema declares, in DDL order.
pub const RULE_SET_FIELDS: [&str; 4] = ["id", "codec", "rules_json", "rule_count"];

/// The rule-set table's own definition, as the engine renders it.
pub const RULE_SET_TABLE_DEFINITION: &str =
    "DEFINE TABLE rule_set TYPE NORMAL SCHEMAFULL PERMISSIONS NONE";

/// Every column's definition, as the engine renders it.
pub const RULE_SET_FIELD_DEFINITIONS: [(&str, &str); 4] = [
    (
        "id",
        "DEFINE FIELD id ON rule_set TYPE string ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/ PERMISSIONS FULL",
    ),
    (
        "codec",
        "DEFINE FIELD codec ON rule_set TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "rules_json",
        "DEFINE FIELD rules_json ON rule_set TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "rule_count",
        "DEFINE FIELD rule_count ON rule_set TYPE int READONLY PERMISSIONS FULL",
    ),
];

/// The rule-set table's DDL.
///
/// A rule set is stored the way a term is: one opaque canonical JSON string plus
/// derived columns, never a native nested object. The reasons carry over
/// unchanged — `usize` does not survive the value layer intact, and a content
/// address has to be the digest of the bytes that are actually stored.
const RULE_SET_DDL: &str = "\
DEFINE TABLE IF NOT EXISTS rule_set SCHEMAFULL TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON rule_set TYPE string
    ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/;
DEFINE FIELD IF NOT EXISTS codec ON rule_set TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS rules_json ON rule_set TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS rule_count ON rule_set TYPE int READONLY;
";

/// The rule-set table's schema, as this build declares it.
const RULE_SET_SCHEMA: TableSchema = TableSchema {
    table: RULE_SET_TABLE,
    ddl: Cow::Borrowed(RULE_SET_DDL),
    table_definition: Cow::Borrowed(RULE_SET_TABLE_DEFINITION),
    fields: &RULE_SET_FIELD_DEFINITIONS,
    nested_fields: &[],
    implicit_fields: &[],
    indexes: &[],
    events: &[],
    write_once: true,
};

// --------------------------------------------------------------- rewrite runs

/// The table holding optimizer traces.
pub const REWRITE_RUN_TABLE: &str = "rewrite_run";

/// Every column the [`REWRITE_RUN_TABLE`] schema declares, in DDL order.
pub const REWRITE_RUN_FIELDS: [&str; 13] = [
    "id",
    "codec",
    "rule_set",
    "start",
    "best",
    "cost_model",
    "initial_cost",
    "best_cost",
    "fuel_exhausted",
    "states_explored",
    "step_count",
    "steps",
    "replayable",
];

/// The index on a run's starting term.
pub const REWRITE_RUN_START_INDEX: &str = "rewrite_run_start";

/// The index on a run's rule set.
pub const REWRITE_RUN_RULE_SET_INDEX: &str = "rewrite_run_rule_set";

/// Every index the [`REWRITE_RUN_TABLE`] schema declares.
pub const REWRITE_RUN_INDEXES: [&str; 2] = [REWRITE_RUN_START_INDEX, REWRITE_RUN_RULE_SET_INDEX];

/// The rewrite-run table's own definition, as the engine renders it.
pub const REWRITE_RUN_TABLE_DEFINITION: &str =
    "DEFINE TABLE rewrite_run TYPE NORMAL SCHEMAFULL PERMISSIONS NONE";

/// Every column's definition, as the engine renders it.
///
/// `cost_model` is a plain, mandatory `string`, and its mandatoriness is the
/// point rather than an oversight: the per-generator weighting an optimizer ran
/// under is a closure nothing can persist, and two costs measured under
/// different weightings are not comparable. A run that did not say which
/// weighting produced its numbers has recorded numbers nobody can use.
///
/// `replayable` carries `DEFAULT false` so that the day a persisted trace can be
/// replayed on load, new runs can start recording `true` without a schema
/// migration.
pub const REWRITE_RUN_FIELD_DEFINITIONS: [(&str, &str); 13] = [
    (
        "id",
        "DEFINE FIELD id ON rewrite_run TYPE string ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/ PERMISSIONS FULL",
    ),
    (
        "codec",
        "DEFINE FIELD codec ON rewrite_run TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "rule_set",
        "DEFINE FIELD rule_set ON rewrite_run TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "start",
        "DEFINE FIELD start ON rewrite_run TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "best",
        "DEFINE FIELD best ON rewrite_run TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "cost_model",
        "DEFINE FIELD cost_model ON rewrite_run TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "initial_cost",
        "DEFINE FIELD initial_cost ON rewrite_run TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "best_cost",
        "DEFINE FIELD best_cost ON rewrite_run TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "fuel_exhausted",
        "DEFINE FIELD fuel_exhausted ON rewrite_run TYPE bool READONLY PERMISSIONS FULL",
    ),
    (
        "states_explored",
        "DEFINE FIELD states_explored ON rewrite_run TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "step_count",
        "DEFINE FIELD step_count ON rewrite_run TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "steps",
        "DEFINE FIELD steps ON rewrite_run TYPE array<object> READONLY PERMISSIONS FULL",
    ),
    (
        "replayable",
        "DEFINE FIELD replayable ON rewrite_run TYPE bool DEFAULT false READONLY PERMISSIONS FULL",
    ),
];

/// The step object's declared shape, as the engine renders it.
///
/// Declaring these is what makes the trace storable at all. In a `SCHEMAFULL`
/// table an `object` column refuses keys nobody declared, so without these two
/// definitions every write of a non-empty trace would be rejected — and the
/// alternative, declaring the column `FLEXIBLE`, would accept a step of any
/// shape whatsoever, which for a fixed two-field record is worse than useless.
pub const REWRITE_RUN_NESTED_DEFINITIONS: [(&str, &str); 2] = [
    (
        "steps.*.rule",
        "DEFINE FIELD steps.*.rule ON rewrite_run TYPE int PERMISSIONS FULL",
    ),
    (
        "steps.*.matched_edges",
        "DEFINE FIELD steps.*.matched_edges ON rewrite_run TYPE array<int> PERMISSIONS FULL",
    ),
];

/// The element definitions the engine creates beside the array columns above.
pub const REWRITE_RUN_ELEMENT_DEFINITIONS: [(&str, &str); 2] = [
    (
        "steps.*",
        "DEFINE FIELD steps.* ON rewrite_run TYPE object PERMISSIONS FULL",
    ),
    (
        "steps.*.matched_edges.*",
        "DEFINE FIELD steps.*.matched_edges.* ON rewrite_run TYPE int PERMISSIONS FULL",
    ),
];

/// Every declared index's definition, as the engine renders it.
pub const REWRITE_RUN_INDEX_DEFINITIONS: [(&str, &str); 2] = [
    (
        REWRITE_RUN_START_INDEX,
        "DEFINE INDEX rewrite_run_start ON rewrite_run FIELDS start",
    ),
    (
        REWRITE_RUN_RULE_SET_INDEX,
        "DEFINE INDEX rewrite_run_rule_set ON rewrite_run FIELDS rule_set",
    ),
];

/// The rewrite-run table's DDL.
///
/// The `steps` column has to be defined before its nested fields: the parent's
/// `DEFINE FIELD` is what creates the `steps.*` element the nested definitions
/// hang off.
const REWRITE_RUN_DDL: &str = "\
DEFINE TABLE IF NOT EXISTS rewrite_run SCHEMAFULL TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON rewrite_run TYPE string
    ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/;
DEFINE FIELD IF NOT EXISTS codec ON rewrite_run TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS rule_set ON rewrite_run TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS start ON rewrite_run TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS best ON rewrite_run TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS cost_model ON rewrite_run TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS initial_cost ON rewrite_run TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS best_cost ON rewrite_run TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS fuel_exhausted ON rewrite_run TYPE bool READONLY;
DEFINE FIELD IF NOT EXISTS states_explored ON rewrite_run TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS step_count ON rewrite_run TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS steps ON rewrite_run TYPE array<object> READONLY;
DEFINE FIELD IF NOT EXISTS steps.*.rule ON rewrite_run TYPE int;
DEFINE FIELD IF NOT EXISTS steps.*.matched_edges ON rewrite_run TYPE array<int>;
DEFINE FIELD IF NOT EXISTS replayable ON rewrite_run TYPE bool DEFAULT false READONLY;

DEFINE INDEX IF NOT EXISTS rewrite_run_start ON rewrite_run FIELDS start;
DEFINE INDEX IF NOT EXISTS rewrite_run_rule_set ON rewrite_run FIELDS rule_set;
";

/// The rewrite-run table's schema, as this build declares it.
const REWRITE_RUN_SCHEMA: TableSchema = TableSchema {
    table: REWRITE_RUN_TABLE,
    ddl: Cow::Borrowed(REWRITE_RUN_DDL),
    table_definition: Cow::Borrowed(REWRITE_RUN_TABLE_DEFINITION),
    fields: &REWRITE_RUN_FIELD_DEFINITIONS,
    nested_fields: &REWRITE_RUN_NESTED_DEFINITIONS,
    implicit_fields: &REWRITE_RUN_ELEMENT_DEFINITIONS,
    indexes: &REWRITE_RUN_INDEX_DEFINITIONS,
    events: &[],
    write_once: true,
};

// -------------------------------------------------------------- derives edges

/// The relation table linking a term to the term a rewrite run derived from it.
pub const DERIVES_TABLE: &str = "derives";

/// Every column the [`DERIVES_TABLE`] schema declares, in DDL order.
///
/// The edge *pointers* are not here — see [`DERIVES_POINTERS`].
pub const DERIVES_FIELDS: [&str; 5] = ["id", "codec", "run", "rule_set", "cost_model"];

/// The edge pointers, which `TYPE RELATION` defines rather than the DDL.
///
/// They are projected explicitly on every edge read. Bare `SELECT *` and a bare
/// `LIVE SELECT *` both do deliver them at SurrealDB 3.2.4 — this is
/// belt-and-suspenders, not a workaround: an explicit projection is what makes a
/// column the database stops returning a *failure* rather than a silently absent
/// field.
pub const DERIVES_POINTERS: [&str; 2] = ["in", "out"];

/// The index on the run an edge was recorded by.
pub const DERIVES_RUN_INDEX: &str = "derives_run";

/// The index on an edge's incoming pointer — the term it leads out of.
///
/// The forward traversal ("what did some run derive from this morphism?") is the
/// whole reason the tier keeps a graph, and it filters on `in`. The planner
/// resolves candidate rows only from *defined* indexes, so without this the
/// traversal is a full scan of the edge table on every hop — which is fine at
/// ten edges and is the tier's dominant cost at ten million.
pub const DERIVES_IN_INDEX: &str = "derives_in";

/// Every index the [`DERIVES_TABLE`] schema declares.
pub const DERIVES_INDEXES: [&str; 2] = [DERIVES_RUN_INDEX, DERIVES_IN_INDEX];

/// The derives table's own definition, as the engine renders it.
pub const DERIVES_TABLE_DEFINITION: &str =
    "DEFINE TABLE derives TYPE RELATION IN term OUT term SCHEMAFULL PERMISSIONS NONE";

/// Every column's definition, as the engine renders it.
pub const DERIVES_FIELD_DEFINITIONS: [(&str, &str); 5] = [
    (
        "id",
        "DEFINE FIELD id ON derives TYPE string ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/ PERMISSIONS FULL",
    ),
    (
        "codec",
        "DEFINE FIELD codec ON derives TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "run",
        "DEFINE FIELD run ON derives TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "rule_set",
        "DEFINE FIELD rule_set ON derives TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "cost_model",
        "DEFINE FIELD cost_model ON derives TYPE string READONLY PERMISSIONS FULL",
    ),
];

/// The pointer definitions `TYPE RELATION IN term OUT term` creates.
///
/// Writing them in the DDL would be dead text — verified, `READONLY` included:
/// `DEFINE TABLE … TYPE RELATION` has already defined them, so a following
/// `DEFINE FIELD IF NOT EXISTS in …` never runs and the `READONLY` it carries
/// never lands. That is not a hole either: an edge's endpoints are set by
/// `RELATE`, and re-relating the same edge id writes the same endpoints, so a
/// *changed* endpoint would need a statement this store does not issue.
pub const DERIVES_POINTER_DEFINITIONS: [(&str, &str); 2] = [
    (
        "in",
        "DEFINE FIELD in ON derives TYPE record<term> PERMISSIONS FULL",
    ),
    (
        "out",
        "DEFINE FIELD out ON derives TYPE record<term> PERMISSIONS FULL",
    ),
];

/// Every declared index's definition, as the engine renders it.
pub const DERIVES_INDEX_DEFINITIONS: [(&str, &str); 2] = [
    (
        DERIVES_RUN_INDEX,
        "DEFINE INDEX derives_run ON derives FIELDS run",
    ),
    (
        DERIVES_IN_INDEX,
        "DEFINE INDEX derives_in ON derives FIELDS in",
    ),
];

/// The derives table's DDL.
const DERIVES_DDL: &str = "\
DEFINE TABLE IF NOT EXISTS derives SCHEMAFULL TYPE RELATION IN term OUT term;

DEFINE FIELD IF NOT EXISTS id ON derives TYPE string
    ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/;
DEFINE FIELD IF NOT EXISTS codec ON derives TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS run ON derives TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS rule_set ON derives TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS cost_model ON derives TYPE string READONLY;

DEFINE INDEX IF NOT EXISTS derives_run ON derives FIELDS run;
DEFINE INDEX IF NOT EXISTS derives_in ON derives FIELDS in;
";

/// The derives table's schema, as this build declares it.
const DERIVES_SCHEMA: TableSchema = TableSchema {
    table: DERIVES_TABLE,
    ddl: Cow::Borrowed(DERIVES_DDL),
    table_definition: Cow::Borrowed(DERIVES_TABLE_DEFINITION),
    fields: &DERIVES_FIELD_DEFINITIONS,
    nested_fields: &[],
    implicit_fields: &DERIVES_POINTER_DEFINITIONS,
    indexes: &DERIVES_INDEX_DEFINITIONS,
    events: &[],
    write_once: true,
};

// ------------------------------------------------------------------ documents

/// The table holding consumer-shaped documents.
pub const DOCUMENT_TABLE: &str = "document";

/// The table holding write-once, manifest-class documents.
pub const MANIFEST_TABLE: &str = "manifest";

/// Every column the document and manifest schemas declare, in DDL order.
///
/// The two tables carry the same columns and differ only in their guards, which
/// is the whole design: a manifest is a document that cannot change.
pub const DOCUMENT_FIELDS: [&str; 5] = ["id", "codec", "kind", "digest", "payload"];

/// The index on a document's caller-supplied class tag.
pub const DOCUMENT_KIND_INDEX: &str = "document_kind";

/// The index on a manifest's caller-supplied class tag.
pub const MANIFEST_KIND_INDEX: &str = "manifest_kind";

/// The event that refuses an update to a stored manifest.
pub const MANIFEST_NO_UPDATE_EVENT: &str = "manifest_no_update";

/// The event that refuses a delete of a stored manifest.
pub const MANIFEST_NO_DELETE_EVENT: &str = "manifest_no_delete";

/// Every event the [`MANIFEST_TABLE`] schema declares.
pub const MANIFEST_EVENTS: [&str; 2] = [MANIFEST_NO_UPDATE_EVENT, MANIFEST_NO_DELETE_EVENT];

/// The message both manifest events throw.
///
/// It is crate-owned text, which is what makes the classifier that keys on it
/// (see [`crate::error`]) something other than a transcription of upstream
/// wording: the only part of the rendering this store does not author is the
/// `Error while processing event <name>: An error occurred: ` wrapper, and that
/// is pinned by an integration test against a real refusal.
pub const MANIFEST_IMMUTABLE_MESSAGE: &str =
    "catgraph-surreal: a stored manifest cannot be updated or deleted";

/// The document table's own definition, as the engine renders it.
pub const DOCUMENT_TABLE_DEFINITION: &str =
    "DEFINE TABLE document TYPE NORMAL SCHEMAFULL PERMISSIONS NONE";

/// The manifest table's own definition, as the engine renders it.
pub const MANIFEST_TABLE_DEFINITION: &str =
    "DEFINE TABLE manifest TYPE NORMAL SCHEMAFULL PERMISSIONS NONE";

/// Every document column's definition, as the engine renders it.
///
/// `payload` is `object FLEXIBLE`, and the `FLEXIBLE` is not optional. Without
/// it a `SCHEMAFULL` object column accepts only the keys the schema declares and
/// **refuses every write carrying another** — `Found field 'payload.whatever',
/// but no such field exists for table 'document'`. Since the whole point of this
/// tier is to store shapes the store does not know, `FLEXIBLE` is what makes it
/// work at all. The clause order is also fixed: it must follow `TYPE`
/// immediately.
///
/// The id `ASSERT` is deliberately weak, because the ids here are
/// caller-supplied opaque strings rather than digests this store derives: it
/// rejects an empty key and nothing more. `TYPE string` does the load-bearing
/// half by refusing a numeric key.
pub const DOCUMENT_FIELD_DEFINITIONS: [(&str, &str); 5] = [
    (
        "id",
        "DEFINE FIELD id ON document TYPE string ASSERT record::id($value) != '' PERMISSIONS FULL",
    ),
    (
        "codec",
        "DEFINE FIELD codec ON document TYPE string PERMISSIONS FULL",
    ),
    (
        "kind",
        "DEFINE FIELD kind ON document TYPE string PERMISSIONS FULL",
    ),
    (
        "digest",
        "DEFINE FIELD digest ON document TYPE string PERMISSIONS FULL",
    ),
    (
        "payload",
        "DEFINE FIELD payload ON document TYPE object FLEXIBLE PERMISSIONS FULL",
    ),
];

/// Every manifest column's definition, as the engine renders it.
///
/// Three of the four layers of the write-once stack are visible here: `READONLY`
/// on every column, and the `ASSERT $before = NONE OR $value = $before` guard on
/// the columns whose types allow it. The guard is skipped when a field is unset
/// *and* its type admits `NONE`, so it is only meaningful on non-optional,
/// non-`any` columns — which is every column on this table.
pub const MANIFEST_FIELD_DEFINITIONS: [(&str, &str); 5] = [
    (
        "id",
        "DEFINE FIELD id ON manifest TYPE string ASSERT record::id($value) != '' PERMISSIONS FULL",
    ),
    (
        "codec",
        "DEFINE FIELD codec ON manifest TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "kind",
        "DEFINE FIELD kind ON manifest TYPE string READONLY ASSERT $before = NONE OR $value = $before PERMISSIONS FULL",
    ),
    (
        "digest",
        "DEFINE FIELD digest ON manifest TYPE string READONLY ASSERT $before = NONE OR $value = $before PERMISSIONS FULL",
    ),
    (
        "payload",
        "DEFINE FIELD payload ON manifest TYPE object FLEXIBLE READONLY ASSERT $before = NONE OR $value = $before PERMISSIONS FULL",
    ),
];

/// The document table's index definition, as the engine renders it.
pub const DOCUMENT_INDEX_DEFINITIONS: [(&str, &str); 1] = [(
    DOCUMENT_KIND_INDEX,
    "DEFINE INDEX document_kind ON document FIELDS kind",
)];

/// The manifest table's index definition, as the engine renders it.
pub const MANIFEST_INDEX_DEFINITIONS: [(&str, &str); 1] = [(
    MANIFEST_KIND_INDEX,
    "DEFINE INDEX manifest_kind ON manifest FIELDS kind",
)];

/// The manifest table's event definitions, as the engine renders them.
///
/// Both are **synchronous**, which is the property that matters: a sync `THROW`
/// aborts the statement *and* rolls the transaction back. An `ASYNC` event does
/// not abort anything, so it is never an enforcement mechanism.
pub const MANIFEST_EVENT_DEFINITIONS: [(&str, &str); 2] = [
    (
        MANIFEST_NO_UPDATE_EVENT,
        "DEFINE EVENT manifest_no_update ON manifest WHEN $event = 'UPDATE' THEN { THROW 'catgraph-surreal: a stored manifest cannot be updated or deleted' }",
    ),
    (
        MANIFEST_NO_DELETE_EVENT,
        "DEFINE EVENT manifest_no_delete ON manifest WHEN $event = 'DELETE' THEN { THROW 'catgraph-surreal: a stored manifest cannot be updated or deleted' }",
    ),
];

/// The document table's DDL.
const DOCUMENT_DDL: &str = "\
DEFINE TABLE IF NOT EXISTS document SCHEMAFULL TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON document TYPE string
    ASSERT record::id($value) != \"\";
DEFINE FIELD IF NOT EXISTS codec ON document TYPE string;
DEFINE FIELD IF NOT EXISTS kind ON document TYPE string;
DEFINE FIELD IF NOT EXISTS digest ON document TYPE string;
DEFINE FIELD IF NOT EXISTS payload ON document TYPE object FLEXIBLE;

DEFINE INDEX IF NOT EXISTS document_kind ON document FIELDS kind;
";

/// The manifest table's DDL.
const MANIFEST_DDL: &str = "\
DEFINE TABLE IF NOT EXISTS manifest SCHEMAFULL TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON manifest TYPE string
    ASSERT record::id($value) != \"\";
DEFINE FIELD IF NOT EXISTS codec ON manifest TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS kind ON manifest TYPE string READONLY
    ASSERT $before = NONE OR $value = $before;
DEFINE FIELD IF NOT EXISTS digest ON manifest TYPE string READONLY
    ASSERT $before = NONE OR $value = $before;
DEFINE FIELD IF NOT EXISTS payload ON manifest TYPE object FLEXIBLE READONLY
    ASSERT $before = NONE OR $value = $before;

DEFINE INDEX IF NOT EXISTS manifest_kind ON manifest FIELDS kind;

DEFINE EVENT IF NOT EXISTS manifest_no_update ON manifest WHEN $event = \"UPDATE\"
    THEN { THROW \"catgraph-surreal: a stored manifest cannot be updated or deleted\"; };
DEFINE EVENT IF NOT EXISTS manifest_no_delete ON manifest WHEN $event = \"DELETE\"
    THEN { THROW \"catgraph-surreal: a stored manifest cannot be updated or deleted\"; };
";

/// The document table's schema, as this build declares it.
const DOCUMENT_SCHEMA: TableSchema = TableSchema {
    table: DOCUMENT_TABLE,
    ddl: Cow::Borrowed(DOCUMENT_DDL),
    table_definition: Cow::Borrowed(DOCUMENT_TABLE_DEFINITION),
    fields: &DOCUMENT_FIELD_DEFINITIONS,
    nested_fields: &[],
    implicit_fields: &[],
    indexes: &DOCUMENT_INDEX_DEFINITIONS,
    events: &[],
    write_once: false,
};

/// The manifest table's schema, as this build declares it.
const MANIFEST_SCHEMA: TableSchema = TableSchema {
    table: MANIFEST_TABLE,
    ddl: Cow::Borrowed(MANIFEST_DDL),
    table_definition: Cow::Borrowed(MANIFEST_TABLE_DEFINITION),
    fields: &MANIFEST_FIELD_DEFINITIONS,
    nested_fields: &[],
    implicit_fields: &[],
    indexes: &MANIFEST_INDEX_DEFINITIONS,
    events: &MANIFEST_EVENT_DEFINITIONS,
    write_once: true,
};

// ------------------------------------------------------------------------ bus

/// The table holding durable notification-bus events.
pub const BUS_TABLE: &str = "bus";

/// The table holding one sequence allocator per stream.
pub const BUS_SEQ_TABLE: &str = "bus_seq";

/// The table holding one catch-up cursor per consumer.
pub const BUS_MARK_TABLE: &str = "bus_mark";

/// Every column the [`BUS_TABLE`] schema declares, in DDL order.
pub const BUS_FIELDS: [&str; 5] = ["id", "codec", "stream", "seq", "payload"];

/// Every column the [`BUS_SEQ_TABLE`] schema declares, in DDL order.
pub const BUS_SEQ_FIELDS: [&str; 3] = ["id", "stream", "next_seq"];

/// Every column the [`BUS_MARK_TABLE`] schema declares, in DDL order.
pub const BUS_MARK_FIELDS: [&str; 5] = ["id", "consumer", "versionstamp", "stamped_at", "seen"];

/// The compound index on `(stream, seq)`.
///
/// Non-unique, deliberately: uniqueness of `(stream, seq)` is carried by the
/// record id, which is the digest of exactly that pair, and a numeric column may
/// not join a unique index (see the [module documentation](self)). Its job is to
/// serve the two access paths that matter — every event on a stream in order,
/// and the highest sequence number on a stream, which is what re-baselining
/// after a change-capture gap reads.
pub const BUS_STREAM_SEQ_INDEX: &str = "bus_stream_seq";

/// Every index the [`BUS_TABLE`] schema declares.
pub const BUS_INDEXES: [&str; 1] = [BUS_STREAM_SEQ_INDEX];

/// The default change-capture retention on the bus table.
///
/// Long enough that an ordinary consumer restart catches up from the feed rather
/// than re-baselining, short enough that the feed is not an unbounded log. It is
/// configurable per store — see [`bootstrap_bus`] — because "long enough" is a
/// property of the consumer's downtime, which this crate cannot know.
pub const BUS_CHANGEFEED_RETENTION: Duration = Duration::from_secs(24 * 60 * 60);

/// The bus table's own definition, as the engine renders it, for a given
/// change-capture retention.
///
/// The retention is part of the compared definition on purpose: a shortened
/// `CHANGEFEED` is exactly the drift that turns catch-up into silent loss, and a
/// name-only check would not see it.
#[must_use]
pub fn bus_table_definition(retention: Duration) -> String {
    format!(
        "DEFINE TABLE bus TYPE NORMAL SCHEMAFULL CHANGEFEED {} PERMISSIONS NONE",
        render_duration(retention)
    )
}

/// Every bus column's definition, as the engine renders it.
pub const BUS_FIELD_DEFINITIONS: [(&str, &str); 5] = [
    (
        "id",
        "DEFINE FIELD id ON bus TYPE string ASSERT record::id($value) = /^b3_[0-9a-f]{64}$/ PERMISSIONS FULL",
    ),
    (
        "codec",
        "DEFINE FIELD codec ON bus TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "stream",
        "DEFINE FIELD stream ON bus TYPE string READONLY PERMISSIONS FULL",
    ),
    (
        "seq",
        "DEFINE FIELD seq ON bus TYPE int READONLY PERMISSIONS FULL",
    ),
    (
        "payload",
        "DEFINE FIELD payload ON bus TYPE object FLEXIBLE READONLY PERMISSIONS FULL",
    ),
];

/// The bus index definition, as the engine renders it.
pub const BUS_INDEX_DEFINITIONS: [(&str, &str); 1] = [(
    BUS_STREAM_SEQ_INDEX,
    "DEFINE INDEX bus_stream_seq ON bus FIELDS stream, seq",
)];

/// The sequence-allocator table's own definition, as the engine renders it.
pub const BUS_SEQ_TABLE_DEFINITION: &str =
    "DEFINE TABLE bus_seq TYPE NORMAL SCHEMAFULL PERMISSIONS NONE";

/// Every sequence-allocator column's definition, as the engine renders it.
///
/// Nothing here is `READONLY`, and that is the point: this is the one mutable
/// table in the store. `next_seq` is the shared row two racing publishers both
/// write, which is what turns a lost-update race into a detectable transaction
/// conflict instead of two events sharing a sequence number.
pub const BUS_SEQ_FIELD_DEFINITIONS: [(&str, &str); 3] = [
    (
        "id",
        "DEFINE FIELD id ON bus_seq TYPE string ASSERT record::id($value) != '' PERMISSIONS FULL",
    ),
    (
        "stream",
        "DEFINE FIELD stream ON bus_seq TYPE string PERMISSIONS FULL",
    ),
    (
        "next_seq",
        "DEFINE FIELD next_seq ON bus_seq TYPE int PERMISSIONS FULL",
    ),
];

/// The consumer-cursor table's own definition, as the engine renders it.
pub const BUS_MARK_TABLE_DEFINITION: &str =
    "DEFINE TABLE bus_mark TYPE NORMAL SCHEMAFULL PERMISSIONS NONE";

/// Every consumer-cursor column's definition, as the engine renders it.
///
/// `stamped_at` is a `datetime` rather than an integer so the staleness question
/// — "is this cursor older than the change-capture window?" — can be asked in
/// the database, where `time::now()` is, rather than by comparing a stored
/// number against a clock the store would have to trust.
///
/// `seen` holds the consumer's per-stream sequence expectations, and it is an
/// opaque JSON **string** for the same reason a term is: a native nested value
/// would have to be declared, key by key, on a `SCHEMAFULL` table whose keys are
/// stream names nobody can enumerate in a DDL. Held in memory alone the
/// expectations reset at every reader restart, and a hole punched while a
/// consumer was down would pass as a first sighting — which is exactly the
/// silent loss the sequence numbers exist to catch.
pub const BUS_MARK_FIELD_DEFINITIONS: [(&str, &str); 5] = [
    (
        "id",
        "DEFINE FIELD id ON bus_mark TYPE string ASSERT record::id($value) != '' PERMISSIONS FULL",
    ),
    (
        "consumer",
        "DEFINE FIELD consumer ON bus_mark TYPE string PERMISSIONS FULL",
    ),
    (
        "versionstamp",
        "DEFINE FIELD versionstamp ON bus_mark TYPE int PERMISSIONS FULL",
    ),
    (
        "stamped_at",
        "DEFINE FIELD stamped_at ON bus_mark TYPE datetime PERMISSIONS FULL",
    ),
    (
        "seen",
        "DEFINE FIELD seen ON bus_mark TYPE string PERMISSIONS FULL",
    ),
];

/// The bus table's DDL, for a given change-capture retention.
///
/// ⚠ A change feed defined on the **database** overrides every table-level one.
/// This store never defines a database-level feed, but an operator who does
/// silently replaces the retention below for every table at once — including
/// this one, whose consumers then catch up over a window nobody here chose.
fn bus_ddl(retention: Duration) -> String {
    format!(
        "\
DEFINE TABLE IF NOT EXISTS bus SCHEMAFULL TYPE NORMAL CHANGEFEED {retention};

DEFINE FIELD IF NOT EXISTS id ON bus TYPE string
    ASSERT record::id($value) = /^b3_[0-9a-f]{{64}}$/;
DEFINE FIELD IF NOT EXISTS codec ON bus TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS stream ON bus TYPE string READONLY;
DEFINE FIELD IF NOT EXISTS seq ON bus TYPE int READONLY;
DEFINE FIELD IF NOT EXISTS payload ON bus TYPE object FLEXIBLE READONLY;

DEFINE INDEX IF NOT EXISTS bus_stream_seq ON bus FIELDS stream, seq;
",
        retention = render_duration(retention)
    )
}

/// The sequence-allocator table's DDL.
const BUS_SEQ_DDL: &str = "\
DEFINE TABLE IF NOT EXISTS bus_seq SCHEMAFULL TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON bus_seq TYPE string
    ASSERT record::id($value) != \"\";
DEFINE FIELD IF NOT EXISTS stream ON bus_seq TYPE string;
DEFINE FIELD IF NOT EXISTS next_seq ON bus_seq TYPE int;
";

/// The consumer-cursor table's DDL.
const BUS_MARK_DDL: &str = "\
DEFINE TABLE IF NOT EXISTS bus_mark SCHEMAFULL TYPE NORMAL;

DEFINE FIELD IF NOT EXISTS id ON bus_mark TYPE string
    ASSERT record::id($value) != \"\";
DEFINE FIELD IF NOT EXISTS consumer ON bus_mark TYPE string;
DEFINE FIELD IF NOT EXISTS versionstamp ON bus_mark TYPE int;
DEFINE FIELD IF NOT EXISTS stamped_at ON bus_mark TYPE datetime;
DEFINE FIELD IF NOT EXISTS seen ON bus_mark TYPE string;
";

/// The bus table's schema for a given retention.
fn bus_schema(retention: Duration) -> TableSchema {
    TableSchema {
        table: BUS_TABLE,
        ddl: Cow::Owned(bus_ddl(retention)),
        table_definition: Cow::Owned(bus_table_definition(retention)),
        fields: &BUS_FIELD_DEFINITIONS,
        nested_fields: &[],
        implicit_fields: &[],
        indexes: &BUS_INDEX_DEFINITIONS,
        events: &[],
        write_once: true,
    }
}

/// The sequence-allocator table's schema, as this build declares it.
const BUS_SEQ_SCHEMA: TableSchema = TableSchema {
    table: BUS_SEQ_TABLE,
    ddl: Cow::Borrowed(BUS_SEQ_DDL),
    table_definition: Cow::Borrowed(BUS_SEQ_TABLE_DEFINITION),
    fields: &BUS_SEQ_FIELD_DEFINITIONS,
    nested_fields: &[],
    implicit_fields: &[],
    indexes: &[],
    events: &[],
    write_once: false,
};

/// The consumer-cursor table's schema, as this build declares it.
const BUS_MARK_SCHEMA: TableSchema = TableSchema {
    table: BUS_MARK_TABLE,
    ddl: Cow::Borrowed(BUS_MARK_DDL),
    table_definition: Cow::Borrowed(BUS_MARK_TABLE_DEFINITION),
    fields: &BUS_MARK_FIELD_DEFINITIONS,
    nested_fields: &[],
    implicit_fields: &[],
    indexes: &[],
    events: &[],
    write_once: false,
};

/// Render a duration the way SurrealQL does, so an expected table definition can
/// be compared against the engine's own rendering.
///
/// The rendering is the SDK value type's, not this crate's: a duration reaches
/// the engine's `INFO` output through exactly this formatter, which reduces
/// `3600s` to `1h` and emits `µs` for microseconds, so a hand-rolled mirror
/// would be a second implementation of a normalization the drift guard compares
/// *exactly* against. There is no version of that mirror that is safer than
/// calling the thing being mirrored — a divergence in either direction reports
/// drift on a correctly bootstrapped database.
fn render_duration(duration: Duration) -> String {
    surrealdb::types::Duration::from(duration).to_string()
}

// -------------------------------------------------------------- the machinery

/// One table's declared schema: its DDL and everything the drift guard compares
/// against.
#[derive(Debug, Clone)]
struct TableSchema {
    table: &'static str,
    ddl: Cow<'static, str>,
    table_definition: Cow<'static, str>,
    /// The columns the DDL declares — also the read projection.
    fields: &'static [(&'static str, &'static str)],
    /// Fields the DDL declares that are not projectable columns, because they
    /// live inside one.
    nested_fields: &'static [(&'static str, &'static str)],
    /// Field definitions the engine creates on its own, which the DDL must not
    /// repeat.
    implicit_fields: &'static [(&'static str, &'static str)],
    indexes: &'static [(&'static str, &'static str)],
    events: &'static [(&'static str, &'static str)],
    /// Whether every column on this table is meant to be `READONLY`.
    ///
    /// A *declaration*, not a derivation, and read only by the unit tests that
    /// enforce it. Deriving it from the definitions would make the check
    /// circular — "every column is read-only because every column is read-only"
    /// — where stating it separately makes a dropped `READONLY` clause a
    /// disagreement between two independent statements of the same intent.
    #[cfg_attr(not(test), allow(dead_code))]
    write_once: bool,
}

impl TableSchema {
    /// Every field definition the engine should be holding.
    fn all_fields(&self) -> Vec<(&'static str, &'static str)> {
        self.fields
            .iter()
            .chain(self.nested_fields)
            .chain(self.implicit_fields)
            .copied()
            .collect()
    }
}

/// Define the term table, its columns, and its indexes.
///
/// Idempotent: running it against an already-bootstrapped database changes
/// nothing. It finishes by running [`assert_term_schema`], so a database whose
/// schema has drifted out from under the DDL — a dropped column, a table
/// carrying columns this version does not know, a table something else created
/// implicitly before this ran (`IF NOT EXISTS` blesses the impostor rather
/// than replacing it) — fails here rather than at the first read that quietly
/// returns the wrong shape.
///
/// # Errors
///
/// Fails if any statement is rejected, or if the resulting schema does not
/// match what this version declares.
pub async fn bootstrap_terms(client: &Surreal<Any>) -> Result<()> {
    bootstrap(client, &TERM_SCHEMA).await
}

/// Check the live term schema against what this version of the store declares.
///
/// Deliberately callable on its own, not just as [`bootstrap_terms`]'s tail:
/// this is the guard that catches drift introduced *after* a bootstrap — a
/// column removed by hand, a restore that replayed a different schema version.
/// Bootstrapping would silently repair a dropped column (every statement is
/// `IF NOT EXISTS`), so "bootstrap and hope" is not a substitute for asking.
///
/// **Definitions must match exactly, in both directions** — see the [module
/// documentation](self) for why names alone are not enough. A missing column is
/// obviously drift; an *extra* one means the database was written by a version
/// of this store that knows a column this one does not, and reading it as if it
/// were the older shape is how a newer document silently loses a field.
/// Declared indexes and events must match their expected definitions; an *extra*
/// index is an operator's performance decision and costs correctness nothing,
/// while an extra *event* is not tolerated — an event can rewrite or refuse
/// writes, so one this build does not know about is a change in behaviour.
///
/// A table that does not exist at all reports no definition, so it lands here
/// as drift rather than as an opaque "table not found".
///
/// # Errors
///
/// Returns [`StoreError::Schema`] naming the difference, or a database error if
/// the schema could not be read.
pub async fn assert_term_schema(client: &Surreal<Any>) -> Result<()> {
    assert_schema(client, &TERM_SCHEMA).await
}

/// Define the cospan table, its columns, and its unique canonical-key index.
///
/// Idempotent, and verified — see [`bootstrap_terms`].
///
/// # Errors
///
/// Fails if any statement is rejected, or if the resulting schema does not
/// match what this version declares.
pub async fn bootstrap_cospans(client: &Surreal<Any>) -> Result<()> {
    bootstrap(client, &COSPAN_SCHEMA).await
}

/// Check the live cospan schema against what this version of the store
/// declares. See [`assert_term_schema`].
///
/// # Errors
///
/// Returns [`StoreError::Schema`] naming the difference, or a database error if
/// the schema could not be read.
pub async fn assert_cospan_schema(client: &Surreal<Any>) -> Result<()> {
    assert_schema(client, &COSPAN_SCHEMA).await
}

/// Define the weight table, its columns, and its unique key index.
///
/// Idempotent, and verified — see [`bootstrap_terms`].
///
/// # Errors
///
/// Fails if any statement is rejected, or if the resulting schema does not
/// match what this version declares.
pub async fn bootstrap_weights(client: &Surreal<Any>) -> Result<()> {
    bootstrap(client, &WEIGHT_SCHEMA).await
}

/// Check the live weight schema against what this version of the store
/// declares. See [`assert_term_schema`].
///
/// # Errors
///
/// Returns [`StoreError::Schema`] if the schema has drifted.
pub async fn assert_weight_schema(client: &Surreal<Any>) -> Result<()> {
    assert_schema(client, &WEIGHT_SCHEMA).await
}

/// Define the lineage tables: rule sets, rewrite runs, and the derivation edge.
///
/// The term table is defined first, because the edge's `TYPE RELATION IN term
/// OUT term` names it and because both endpoints of an edge are term records.
/// Idempotent, and verified — see [`bootstrap_terms`].
///
/// # Errors
///
/// Fails if any statement is rejected, or if the resulting schema does not
/// match what this version declares.
pub async fn bootstrap_lineage(client: &Surreal<Any>) -> Result<()> {
    bootstrap(client, &TERM_SCHEMA).await?;
    bootstrap(client, &RULE_SET_SCHEMA).await?;
    bootstrap(client, &REWRITE_RUN_SCHEMA).await?;
    bootstrap(client, &DERIVES_SCHEMA).await
}

/// Check the live lineage schemas against what this version of the store
/// declares. See [`assert_term_schema`].
///
/// # Errors
///
/// Returns [`StoreError::Schema`] if any of them has drifted.
pub async fn assert_lineage_schema(client: &Surreal<Any>) -> Result<()> {
    assert_schema(client, &TERM_SCHEMA).await?;
    assert_schema(client, &RULE_SET_SCHEMA).await?;
    assert_schema(client, &REWRITE_RUN_SCHEMA).await?;
    assert_schema(client, &DERIVES_SCHEMA).await
}

/// Define the mutable document table.
///
/// Idempotent, and verified — see [`bootstrap_terms`].
///
/// # Errors
///
/// Fails if any statement is rejected, or if the resulting schema does not
/// match what this version declares.
pub async fn bootstrap_documents(client: &Surreal<Any>) -> Result<()> {
    bootstrap(client, &DOCUMENT_SCHEMA).await
}

/// Check the live document schema against what this version of the store
/// declares. See [`assert_term_schema`].
///
/// # Errors
///
/// Returns [`StoreError::Schema`] if the schema has drifted.
pub async fn assert_document_schema(client: &Surreal<Any>) -> Result<()> {
    assert_schema(client, &DOCUMENT_SCHEMA).await
}

/// Define the write-once manifest table, its guards, and its refusal events.
///
/// Idempotent, and verified — see [`bootstrap_terms`].
///
/// # Errors
///
/// Fails if any statement is rejected, or if the resulting schema does not
/// match what this version declares.
pub async fn bootstrap_manifests(client: &Surreal<Any>) -> Result<()> {
    bootstrap(client, &MANIFEST_SCHEMA).await
}

/// Check the live manifest schema against what this version of the store
/// declares. See [`assert_term_schema`].
///
/// # Errors
///
/// Returns [`StoreError::Schema`] if the schema has drifted.
pub async fn assert_manifest_schema(client: &Surreal<Any>) -> Result<()> {
    assert_schema(client, &MANIFEST_SCHEMA).await
}

/// Define the bus tables: the durable event table with its change feed, the
/// per-stream sequence allocator, and the per-consumer catch-up cursor.
///
/// `retention` is the change-capture window on the event table and is part of
/// the compared schema, so opening the same database with a different retention
/// is drift rather than a silent re-definition.
///
/// Idempotent, and verified — see [`bootstrap_terms`].
///
/// # Errors
///
/// Fails if any statement is rejected, or if the resulting schema does not
/// match what this version declares.
pub async fn bootstrap_bus(client: &Surreal<Any>, retention: Duration) -> Result<()> {
    bootstrap(client, &bus_schema(retention)).await?;
    bootstrap(client, &BUS_SEQ_SCHEMA).await?;
    bootstrap(client, &BUS_MARK_SCHEMA).await
}

/// Check the live bus schemas against what this version of the store declares.
/// See [`assert_term_schema`].
///
/// # Errors
///
/// Returns [`StoreError::Schema`] if any of them has drifted — including a
/// change-capture retention that is not the one asked for.
pub async fn assert_bus_schema(client: &Surreal<Any>, retention: Duration) -> Result<()> {
    assert_schema(client, &bus_schema(retention)).await?;
    assert_schema(client, &BUS_SEQ_SCHEMA).await?;
    assert_schema(client, &BUS_MARK_SCHEMA).await
}

/// Run one table's DDL, then verify the result.
async fn bootstrap(client: &Surreal<Any>, schema: &TableSchema) -> Result<()> {
    client.query(schema.ddl.as_ref()).await?.check()?;
    assert_schema(client, schema).await
}

/// Compare one table's live definitions against what this build declares.
async fn assert_schema(client: &Surreal<Any>, schema: &TableSchema) -> Result<()> {
    let mut response = client.query(info_query(schema.table)).await?;
    let table: Option<String> = response.take(0)?;
    let fields: Option<BTreeMap<String, String>> = response.take(1)?;
    let indexes: Option<BTreeMap<String, String>> = response.take(2)?;
    let events: Option<BTreeMap<String, String>> = response.take(3)?;

    let Some(table) = table else {
        return Err(drift(schema.table, "the table is not defined"));
    };
    if table != schema.table_definition {
        return Err(drift(
            schema.table,
            &format!(
                "table definition is `{table}`, expected `{}`",
                schema.table_definition
            ),
        ));
    }

    let fields = fields.unwrap_or_default();
    compare_definitions(schema.table, "column", &fields, &schema.all_fields(), true)?;

    let indexes = indexes.unwrap_or_default();
    compare_definitions(schema.table, "index", &indexes, schema.indexes, false)?;

    let events = events.unwrap_or_default();
    compare_definitions(schema.table, "event", &events, schema.events, true)?;

    Ok(())
}

/// Read a table's own rendered definition (or `NONE` if undefined), its field
/// definitions, its index definitions, and its event definitions, in that order.
///
/// The table name is formatted into the statement rather than bound, because a
/// table name is not a value and cannot be a parameter. It is always one of this
/// module's own constants — never anything a caller supplies.
fn info_query(table: &str) -> String {
    format!(
        "RETURN (INFO FOR DB).tables.{table};\n\
         RETURN (INFO FOR TABLE {table}).fields;\n\
         RETURN (INFO FOR TABLE {table}).indexes;\n\
         RETURN (INFO FOR TABLE {table}).events;"
    )
}

/// Compare live rendered definitions against the expected set.
///
/// `exact` demands the live set carry nothing beyond the expected one; indexes
/// pass `false` so an operator-added index is tolerated.
fn compare_definitions(
    table: &str,
    kind: &str,
    live: &BTreeMap<String, String>,
    expected: &[(&str, &str)],
    exact: bool,
) -> Result<()> {
    for (name, definition) in expected {
        match live.get(*name) {
            None => return Err(drift(table, &format!("{kind} `{name}` is missing"))),
            Some(found) if found != definition => {
                return Err(drift(
                    table,
                    &format!("{kind} `{name}` is defined as `{found}`, expected `{definition}`"),
                ));
            }
            Some(_) => {}
        }
    }
    if exact {
        for name in live.keys() {
            if !expected
                .iter()
                .any(|(expected_name, _)| expected_name == name)
            {
                return Err(drift(table, &format!("unexpected {kind} `{name}`")));
            }
        }
    }
    Ok(())
}

/// The uniform drift error.
fn drift(table: &str, detail: &str) -> StoreError {
    StoreError::Schema {
        table: table.to_owned(),
        detail: detail.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every table this build knows about. The properties below are properties
    /// of *the discipline*, not of one table, so each test walks all of them —
    /// which is also what stops a new table from being added without them.
    fn all_schemas() -> Vec<TableSchema> {
        vec![
            TERM_SCHEMA,
            COSPAN_SCHEMA,
            WEIGHT_SCHEMA,
            RULE_SET_SCHEMA,
            REWRITE_RUN_SCHEMA,
            DERIVES_SCHEMA,
            DOCUMENT_SCHEMA,
            MANIFEST_SCHEMA,
            bus_schema(BUS_CHANGEFEED_RETENTION),
            BUS_SEQ_SCHEMA,
            BUS_MARK_SCHEMA,
        ]
    }

    /// The DDL and the declared column list are two statements of the same
    /// fact, and the drift guard compares the live schema against the *second*.
    /// If they ever disagree, the guard reports drift on a correctly-bootstrapped
    /// database — so pin them together here rather than discovering it against
    /// an engine.
    #[test]
    fn every_declared_field_appears_in_the_ddl() {
        for schema in all_schemas() {
            for (field, _) in schema.fields.iter().chain(schema.nested_fields) {
                let definition = format!("DEFINE FIELD IF NOT EXISTS {field} ON {} ", schema.table);
                assert!(
                    schema.ddl.contains(&definition),
                    "`{field}` is declared but not defined in the {} DDL",
                    schema.table
                );
            }
        }
    }

    #[test]
    fn every_declared_index_appears_in_the_ddl() {
        for schema in all_schemas() {
            for (index, _) in schema.indexes {
                let definition = format!("DEFINE INDEX IF NOT EXISTS {index} ON {} ", schema.table);
                assert!(
                    schema.ddl.contains(&definition),
                    "`{index}` is declared but not defined in the {} DDL",
                    schema.table
                );
            }
        }
    }

    #[test]
    fn every_declared_event_appears_in_the_ddl() {
        for schema in all_schemas() {
            for (event, _) in schema.events {
                let definition = format!("DEFINE EVENT IF NOT EXISTS {event} ON {} ", schema.table);
                assert!(
                    schema.ddl.contains(&definition),
                    "`{event}` is declared but not defined in the {} DDL",
                    schema.table
                );
            }
        }
    }

    /// The public name lists and the expected rendered definitions must agree
    /// with each other — the read projections are derived from the first and the
    /// drift guard compares against the second.
    #[test]
    fn field_definitions_cover_exactly_the_declared_fields() {
        for (names, definitions) in [
            (&TERM_FIELDS[..], &TERM_FIELD_DEFINITIONS[..]),
            (&COSPAN_FIELDS[..], &COSPAN_FIELD_DEFINITIONS[..]),
            (&WEIGHT_FIELDS[..], &WEIGHT_FIELD_DEFINITIONS[..]),
            (&RULE_SET_FIELDS[..], &RULE_SET_FIELD_DEFINITIONS[..]),
            (&REWRITE_RUN_FIELDS[..], &REWRITE_RUN_FIELD_DEFINITIONS[..]),
            (&DERIVES_FIELDS[..], &DERIVES_FIELD_DEFINITIONS[..]),
            (&DOCUMENT_FIELDS[..], &DOCUMENT_FIELD_DEFINITIONS[..]),
            (&DOCUMENT_FIELDS[..], &MANIFEST_FIELD_DEFINITIONS[..]),
            (&BUS_FIELDS[..], &BUS_FIELD_DEFINITIONS[..]),
            (&BUS_SEQ_FIELDS[..], &BUS_SEQ_FIELD_DEFINITIONS[..]),
            (&BUS_MARK_FIELDS[..], &BUS_MARK_FIELD_DEFINITIONS[..]),
        ] {
            let declared: Vec<&str> = definitions.iter().map(|(n, _)| *n).collect();
            assert_eq!(declared, names);
        }
        for (names, definitions) in [
            (&TERM_INDEXES[..], &TERM_INDEX_DEFINITIONS[..]),
            (&COSPAN_INDEXES[..], &COSPAN_INDEX_DEFINITIONS[..]),
            (&WEIGHT_INDEXES[..], &WEIGHT_INDEX_DEFINITIONS[..]),
            (&REWRITE_RUN_INDEXES[..], &REWRITE_RUN_INDEX_DEFINITIONS[..]),
            (&DERIVES_INDEXES[..], &DERIVES_INDEX_DEFINITIONS[..]),
            (&BUS_INDEXES[..], &BUS_INDEX_DEFINITIONS[..]),
        ] {
            let declared: Vec<&str> = definitions.iter().map(|(n, _)| *n).collect();
            assert_eq!(declared, names);
        }
        let declared: Vec<&str> = MANIFEST_EVENT_DEFINITIONS.iter().map(|(n, _)| *n).collect();
        assert_eq!(declared, &MANIFEST_EVENTS[..]);
    }

    /// Every load-bearing token the guard exists to protect must actually be in
    /// the expected definitions it compares against.
    ///
    /// **Nested fields are walked too, by their root column.** A nested
    /// definition carries no `READONLY` of its own and is not supposed to:
    /// writing `steps[0].rule` changes the value of the `steps` column, so the
    /// *root* column's clause is what refuses it — the same relationship an
    /// array element has to its parent. Chasing the chain rather than skipping
    /// the field is what makes that a checked claim instead of an assumption;
    /// for a top-level column the root is the column itself, so the two cases
    /// are one rule.
    #[test]
    fn expected_definitions_carry_the_guards() {
        for schema in all_schemas() {
            assert!(
                schema.table_definition.contains("SCHEMAFULL"),
                "{}",
                schema.table
            );
            for (name, definition) in schema.fields.iter().chain(schema.nested_fields) {
                if *name == "id" {
                    assert!(definition.contains("ASSERT record::id($value)"), "{name}");
                } else if schema.write_once {
                    let root = name
                        .split('.')
                        .next()
                        .expect("invariant: splitting a name yields at least one part");
                    let (_, root_definition) = schema
                        .fields
                        .iter()
                        .find(|(field, _)| *field == root)
                        .unwrap_or_else(|| {
                            panic!("`{name}` has no declared root column on {}", schema.table)
                        });
                    assert!(
                        root_definition.contains("READONLY"),
                        "`{}.{name}` is on a write-once table but its root column `{root}` is \
                         not READONLY — nothing would refuse a write to it",
                        schema.table
                    );
                }
                assert!(!definition.contains("option<"), "{name}");
                assert!(!definition.contains("TYPE any"), "{name}");
            }
        }
    }

    /// The write-once guard `$before = NONE OR $value = $before` is only
    /// meaningful on a non-optional, non-`any` column: the assertion is skipped
    /// when a field is unset *and* its declared type admits `NONE`. Every column
    /// carrying it must therefore be typed tightly enough for it to run.
    #[test]
    fn the_write_once_assert_sits_only_on_types_that_evaluate_it() {
        for (name, definition) in MANIFEST_FIELD_DEFINITIONS {
            if !definition.contains("$before") {
                continue;
            }
            assert!(!definition.contains("option<"), "{name}");
            assert!(!definition.contains("TYPE any"), "{name}");
            assert!(definition.contains("READONLY"), "{name}");
        }
    }

    /// Implicit fields are the engine's, not ours, so the rules differ: none may
    /// appear in the DDL (the statement would never run), and a `.*` element
    /// must belong to an array-typed parent this build declares — with that
    /// parent carrying the `READONLY` that refuses an element write, on tables
    /// where write-once is the contract.
    #[test]
    fn implicit_definitions_are_never_declared_and_hang_off_declared_parents() {
        for schema in all_schemas() {
            let declared = schema.all_fields();
            for (name, definition) in schema.implicit_fields {
                // Anchored to the whole `DEFINE FIELD` prefix rather than to
                // `<name> ON <table>`: a pointer named `in` is a suffix of the
                // index name `derives_in`, so the loose form reports the
                // *index* statement as a dead field definition.
                assert!(
                    !schema.ddl.contains(&format!(
                        "DEFINE FIELD IF NOT EXISTS {name} ON {}",
                        schema.table
                    )),
                    "the {} DDL declares `{name}`, which the engine already created — \
                     the statement is dead text",
                    schema.table
                );
                assert!(!definition.contains("option<"), "{name}");
                assert!(!definition.contains("TYPE any"), "{name}");

                let Some(parent) = name.strip_suffix(".*") else {
                    // A relation's `in`/`out`: no parent column, and their
                    // values are written by RELATE rather than by a field write.
                    assert!(
                        schema.table_definition.contains("TYPE RELATION"),
                        "`{name}` is not an element definition and `{}` is not a relation",
                        schema.table
                    );
                    continue;
                };
                let (_, parent_definition) = declared
                    .iter()
                    .find(|(field, _)| *field == parent)
                    .unwrap_or_else(|| panic!("`{name}` has no declared parent field"));
                assert!(
                    parent_definition.contains("TYPE array<"),
                    "`{parent}` is not an array field, so `{name}` would not exist"
                );
                if schema.write_once && schema.fields.iter().any(|(f, _)| *f == parent) {
                    assert!(
                        parent_definition.contains("READONLY"),
                        "`{parent}` must be READONLY: that is what refuses an element write"
                    );
                }
            }
        }
    }

    /// And the other direction: an `array<T>` field that has no element
    /// definition listed would make the drift guard reject a correctly
    /// bootstrapped database, because the engine creates one regardless.
    #[test]
    fn every_array_field_lists_its_element_definition() {
        for schema in all_schemas() {
            for (name, definition) in schema.fields.iter().chain(schema.nested_fields) {
                if !definition.contains("TYPE array<") {
                    continue;
                }
                let element = format!("{name}.*");
                assert!(
                    schema
                        .implicit_fields
                        .iter()
                        .any(|(field, _)| *field == element),
                    "`{name}` is an array field, so the engine defines `{element}` — \
                     which the drift guard would then report as unexpected"
                );
            }
        }
    }

    /// In a `SCHEMAFULL` table an `object` column refuses keys nobody declared,
    /// so every object-typed field must either be `FLEXIBLE` or have its shape
    /// declared. A column that is neither accepts only `{}`.
    #[test]
    fn every_object_field_is_flexible_or_has_a_declared_shape() {
        for schema in all_schemas() {
            let declared = schema.all_fields();
            for (name, definition) in schema.fields.iter().chain(schema.implicit_fields) {
                let is_object =
                    definition.contains("TYPE object") || definition.contains("TYPE array<object>");
                if !is_object || definition.contains("FLEXIBLE") {
                    continue;
                }
                let prefix = format!("{name}.");
                assert!(
                    declared
                        .iter()
                        .any(|(field, _)| field.starts_with(&prefix) && field != name),
                    "`{}.{name}` is a non-FLEXIBLE object field with no declared inner \
                     shape — its keys would be silently dropped",
                    schema.table
                );
            }
        }
    }

    /// The DDL defines exactly the declared fields and no others — otherwise
    /// bootstrapping would create a field the drift guard then reports as
    /// unexpected, failing every open.
    #[test]
    fn the_ddl_defines_no_undeclared_field() {
        for schema in all_schemas() {
            let defined = schema
                .ddl
                .lines()
                .filter_map(|line| line.trim().strip_prefix("DEFINE FIELD IF NOT EXISTS "))
                .filter_map(|rest| rest.split_whitespace().next())
                .count();
            assert_eq!(
                defined,
                schema.fields.len() + schema.nested_fields.len(),
                "{}",
                schema.table
            );
        }
    }

    /// The table has to be defined before any of its fields: re-issuing
    /// `DEFINE TABLE` drops that table's field definitions, so a DDL that
    /// interleaved them would leave a replay with a schemafull table and no
    /// columns.
    #[test]
    fn the_table_is_defined_before_its_fields() {
        for schema in all_schemas() {
            let table = schema
                .ddl
                .find("DEFINE TABLE")
                .expect("invariant: every DDL defines its table");
            let first_field = schema
                .ddl
                .find("DEFINE FIELD")
                .expect("invariant: every DDL defines at least one field");
            assert!(table < first_field, "{}", schema.table);
        }
    }

    /// A nested field hangs off an element the parent column's definition
    /// creates, so the parent must come first in the DDL.
    #[test]
    fn nested_fields_follow_the_column_they_live_in() {
        for schema in all_schemas() {
            for (name, _) in schema.nested_fields {
                let parent = name
                    .split_once(".*")
                    .map(|(head, _)| head)
                    .expect("invariant: a nested field names an element of a column");
                let parent_at = schema
                    .ddl
                    .find(&format!("{parent} ON {}", schema.table))
                    .unwrap_or_else(|| panic!("`{parent}` is not defined in the DDL"));
                let nested_at = schema
                    .ddl
                    .find(&format!("{name} ON {}", schema.table))
                    .unwrap_or_else(|| panic!("`{name}` is not defined in the DDL"));
                assert!(parent_at < nested_at, "{name}");
            }
        }
    }

    /// `OVERWRITE` on an index is a synchronous delete plus a full rebuild, and
    /// this DDL runs on every store open.
    #[test]
    fn the_ddl_never_overwrites() {
        for schema in all_schemas() {
            assert!(!schema.ddl.contains("OVERWRITE"), "{}", schema.table);
        }
    }

    /// An optional or `any`-typed column would disarm its own `ASSERT`: the
    /// unset-skip fires when the value is `NONE` *and* the declared type admits
    /// `NONE`.
    #[test]
    fn no_column_is_optional_or_any() {
        for schema in all_schemas() {
            assert!(!schema.ddl.contains("option<"), "{}", schema.table);
            assert!(!schema.ddl.contains("TYPE any"), "{}", schema.table);
        }
    }

    /// There is no `$key` binding in an `ASSERT`. On `id`, `$value` is the whole
    /// record id, so the key has to be extracted before it can be matched.
    #[test]
    fn the_id_assert_extracts_the_key_from_the_record_id() {
        for schema in all_schemas() {
            assert!(
                schema.ddl.contains("record::id($value)"),
                "{}",
                schema.table
            );
            assert!(!schema.ddl.contains("$key"), "{}", schema.table);
        }
    }

    /// A write-once event must be synchronous. An `ASYNC` event runs after the
    /// fact and cannot abort anything, so it would refuse nothing while looking
    /// exactly like a guard.
    #[test]
    fn write_once_events_are_synchronous_and_throw() {
        for schema in all_schemas() {
            for (name, definition) in schema.events {
                assert!(!definition.contains("ASYNC"), "{name}");
                assert!(definition.contains("THROW"), "{name}");
                assert!(
                    definition.contains(MANIFEST_IMMUTABLE_MESSAGE),
                    "`{name}` throws a message the classifier does not know"
                );
            }
        }
    }

    /// Index keys normalise numbers, so a unique index containing a numeric
    /// column refuses rows that are not duplicates. Every unique index here is
    /// built from `string` columns, and this keeps it that way. Non-unique
    /// indexes are unaffected and may pair a string with an integer.
    #[test]
    fn no_numeric_column_joins_a_unique_index() {
        for schema in all_schemas() {
            for (name, definition) in schema.indexes {
                if !definition.contains("UNIQUE") {
                    continue;
                }
                let columns = definition
                    .split_once(" FIELDS ")
                    .map(|(_, rest)| rest.trim_end_matches(" UNIQUE"))
                    .expect("invariant: an index definition names its fields");
                for column in columns.split(", ") {
                    let (_, field) = schema
                        .fields
                        .iter()
                        .find(|(field, _)| *field == column)
                        .unwrap_or_else(|| {
                            panic!("index `{name}` names `{column}`, which is not a column")
                        });
                    assert!(
                        field.contains("TYPE string"),
                        "unique index `{name}` includes the non-string column `{column}`"
                    );
                }
            }
        }
    }

    /// The `INFO` statement is built from a table name, and it has to read all
    /// four things the guard compares, in the order it takes them.
    #[test]
    fn the_info_query_reads_the_table_then_its_fields_indexes_and_events() {
        assert_eq!(
            info_query("cospan"),
            "RETURN (INFO FOR DB).tables.cospan;\n\
             RETURN (INFO FOR TABLE cospan).fields;\n\
             RETURN (INFO FOR TABLE cospan).indexes;\n\
             RETURN (INFO FOR TABLE cospan).events;"
        );
    }

    /// The definition comparison rejects a changed definition, a missing entry,
    /// and (when exact) an extra one — and tolerates the extra otherwise.
    #[test]
    fn definition_comparison_detects_each_drift_class() {
        let expected = [("a", "DEFINE a"), ("b", "DEFINE b")];
        let live = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect()
        };

        assert!(
            compare_definitions(
                "t",
                "column",
                &live(&[("a", "DEFINE a"), ("b", "DEFINE b")]),
                &expected,
                true
            )
            .is_ok()
        );
        assert!(
            compare_definitions(
                "t",
                "column",
                &live(&[("a", "ALTERED"), ("b", "DEFINE b")]),
                &expected,
                true
            )
            .is_err()
        );
        assert!(
            compare_definitions("t", "column", &live(&[("a", "DEFINE a")]), &expected, true)
                .is_err()
        );
        let with_extra = live(&[("a", "DEFINE a"), ("b", "DEFINE b"), ("c", "DEFINE c")]);
        assert!(compare_definitions("t", "column", &with_extra, &expected, true).is_err());
        assert!(compare_definitions("t", "index", &with_extra, &expected, false).is_ok());
    }

    /// The retention reaches both the DDL and the expected definition, so a
    /// store opened with a different window sees drift rather than a silent
    /// re-definition.
    #[test]
    fn the_change_feed_retention_reaches_the_ddl_and_the_expected_definition() {
        let schema = bus_schema(Duration::from_secs(1));
        assert!(schema.ddl.contains("CHANGEFEED 1s"), "{}", schema.ddl);
        assert!(
            schema.table_definition.contains("CHANGEFEED 1s"),
            "{}",
            schema.table_definition
        );
        let other = bus_schema(Duration::from_secs(3600));
        assert_ne!(schema.table_definition, other.table_definition);
    }

    /// A relation's pointers are not declared columns, but every edge read
    /// projects them, so the two lists have to stay disjoint and complete.
    #[test]
    fn edge_pointers_are_implicit_rather_than_declared() {
        for pointer in DERIVES_POINTERS {
            assert!(!DERIVES_FIELDS.contains(&pointer));
            assert!(
                DERIVES_POINTER_DEFINITIONS
                    .iter()
                    .any(|(name, _)| *name == pointer)
            );
        }
    }
}
