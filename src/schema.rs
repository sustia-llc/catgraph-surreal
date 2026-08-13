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
//! # An `array<T>` column defines a second field you did not write
//!
//! Declaring `TYPE array<int>` makes the engine create an *element* definition
//! beside it — `DEFINE FIELD dom_leg.* ON cospan TYPE int` — which then shows up
//! in `INFO FOR TABLE`. Three consequences, all learned the hard way:
//!
//! - The drift guard compares the live field set **exactly**, so those element
//!   definitions have to be in the expected set or every bootstrap fails on a
//!   column nobody wrote. They are listed separately from the declared columns
//!   ([`COSPAN_ELEMENT_DEFINITIONS`]) because they are not projectable: a read
//!   selects `dom_leg`, never `dom_leg.*`.
//! - Declaring them in the DDL would be dead text. The parent's `DEFINE FIELD`
//!   creates them, so a following `DEFINE FIELD … IF NOT EXISTS dom_leg.*` is a
//!   no-op — verified: adding `READONLY` to such a statement changes nothing,
//!   because the statement never runs.
//! - They carry **no `READONLY` clause**, and that is fine rather than a hole:
//!   an element write (`SET dom_leg[0] = 9`) changes the parent array's value,
//!   so the *parent's* `READONLY` refuses it. Pinned by an integration test,
//!   because it is not obvious from the definitions alone.
//!
//! # No numeric column joins a unique index
//!
//! Index keys normalise numbers: `0`, `0.0`, and `0dec` become one key. A unique
//! index that includes a numeric column therefore refuses rows that are not
//! duplicates. Both unique indexes here — a cospan's canonical key, a weight's
//! `(genome, gen_key)` pair — are made of `string` columns only, and that is a
//! standing constraint rather than an accident of the current schema.
//!
//! # The drift guard compares definitions, not names
//!
//! [`assert_term_schema`] and its siblings compare the **engine-rendered
//! definition strings** (`INFO FOR DB` / `INFO FOR TABLE`) against the ones this
//! build expects — not merely the field and index *names*. A name-only check is
//! blind to exactly the drift that disarms the guards: `ALTER TABLE …
//! SCHEMALESS` keeps every field defined, a dropped `READONLY` clause keeps the
//! field's name, a dropped `UNIQUE` keeps the index's name, and a field re-typed
//! to `option<T>` still answers to its name. The definition strings are the
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
//! So the `READONLY` columns and the id-format `ASSERT` do not *guarantee*
//! anything: the store's own validation is the trust boundary and the database
//! backs it up. This is why [`crate::term`]'s revalidation runs on every load,
//! why [`crate::cospan`] bounds-checks every leg it reads, and why a restore
//! re-verifies record ids rather than trusting the replay.

use std::collections::BTreeMap;

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
    ddl: TERM_DDL,
    table_definition: TERM_TABLE_DEFINITION,
    fields: &TERM_FIELD_DEFINITIONS,
    // No array column, so the engine creates nothing beside them.
    element_fields: &[],
    indexes: &TERM_INDEX_DEFINITIONS,
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
    ddl: COSPAN_DDL,
    table_definition: COSPAN_TABLE_DEFINITION,
    fields: &COSPAN_FIELD_DEFINITIONS,
    element_fields: &COSPAN_ELEMENT_DEFINITIONS,
    indexes: &COSPAN_INDEX_DEFINITIONS,
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
    ddl: WEIGHT_DDL,
    table_definition: WEIGHT_TABLE_DEFINITION,
    fields: &WEIGHT_FIELD_DEFINITIONS,
    element_fields: &[],
    indexes: &WEIGHT_INDEX_DEFINITIONS,
};

// -------------------------------------------------------------- the machinery

/// One table's declared schema: its DDL and everything the drift guard compares
/// against.
#[derive(Debug, Clone, Copy)]
struct TableSchema {
    table: &'static str,
    ddl: &'static str,
    table_definition: &'static str,
    /// The columns the DDL declares — also the read projection.
    fields: &'static [(&'static str, &'static str)],
    /// The element definitions an `array<T>` column brings with it. Empty for a
    /// table with no array columns.
    element_fields: &'static [(&'static str, &'static str)],
    indexes: &'static [(&'static str, &'static str)],
}

impl TableSchema {
    /// Every field definition the engine should be holding: the declared
    /// columns and the element definitions their array types created.
    fn all_fields(self) -> Vec<(&'static str, &'static str)> {
        self.fields
            .iter()
            .chain(self.element_fields)
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
    bootstrap(client, TERM_SCHEMA).await
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
/// Declared indexes must match their expected definitions; an *extra* index is
/// an operator's performance decision and costs correctness nothing.
///
/// A table that does not exist at all reports no definition, so it lands here
/// as drift rather than as an opaque "table not found".
///
/// # Errors
///
/// Returns [`StoreError::Schema`] naming the difference, or a database error if
/// the schema could not be read.
pub async fn assert_term_schema(client: &Surreal<Any>) -> Result<()> {
    assert_schema(client, TERM_SCHEMA).await
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
    bootstrap(client, COSPAN_SCHEMA).await
}

/// Check the live cospan schema against what this version of the store
/// declares. See [`assert_term_schema`].
///
/// # Errors
///
/// Returns [`StoreError::Schema`] naming the difference, or a database error if
/// the schema could not be read.
pub async fn assert_cospan_schema(client: &Surreal<Any>) -> Result<()> {
    assert_schema(client, COSPAN_SCHEMA).await
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
    bootstrap(client, WEIGHT_SCHEMA).await
}

/// Check the live weight schema against what this version of the store
/// declares. See [`assert_term_schema`].
///
/// # Errors
///
/// Returns [`StoreError::Schema`] naming the difference, or a database error if
/// the schema could not be read.
pub async fn assert_weight_schema(client: &Surreal<Any>) -> Result<()> {
    assert_schema(client, WEIGHT_SCHEMA).await
}

/// Run one table's DDL, then verify the result.
async fn bootstrap(client: &Surreal<Any>, schema: TableSchema) -> Result<()> {
    client.query(schema.ddl).await?.check()?;
    assert_schema(client, schema).await
}

/// Compare one table's live definitions against what this build declares.
async fn assert_schema(client: &Surreal<Any>, schema: TableSchema) -> Result<()> {
    let mut response = client.query(info_query(schema.table)).await?;
    let table: Option<String> = response.take(0)?;
    let fields: Option<BTreeMap<String, String>> = response.take(1)?;
    let indexes: Option<BTreeMap<String, String>> = response.take(2)?;

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

    Ok(())
}

/// Read a table's own rendered definition (or `NONE` if undefined), its field
/// definitions, and its index definitions, in that order.
///
/// The table name is formatted into the statement rather than bound, because a
/// table name is not a value and cannot be a parameter. It is always one of this
/// module's own constants — never anything a caller supplies.
fn info_query(table: &str) -> String {
    format!(
        "RETURN (INFO FOR DB).tables.{table};\n\
         RETURN (INFO FOR TABLE {table}).fields;\n\
         RETURN (INFO FOR TABLE {table}).indexes;"
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
    /// of *the discipline*, not of one table, so each test walks all three —
    /// which is also what stops a fourth table from being added without them.
    const ALL_SCHEMAS: [TableSchema; 3] = [TERM_SCHEMA, COSPAN_SCHEMA, WEIGHT_SCHEMA];

    /// The DDL and the declared column list are two statements of the same
    /// fact, and the drift guard compares the live schema against the *second*.
    /// If they ever disagree, the guard reports drift on a correctly-bootstrapped
    /// database — so pin them together here rather than discovering it against
    /// an engine.
    #[test]
    fn every_declared_field_appears_in_the_ddl() {
        for schema in ALL_SCHEMAS {
            for (field, _) in schema.fields {
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
        for schema in ALL_SCHEMAS {
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

    /// The public name lists and the expected rendered definitions must agree
    /// with each other — the read projections are derived from the first and the
    /// drift guard compares against the second.
    #[test]
    fn field_definitions_cover_exactly_the_declared_fields() {
        for (names, definitions) in [
            (&TERM_FIELDS[..], &TERM_FIELD_DEFINITIONS[..]),
            (&COSPAN_FIELDS[..], &COSPAN_FIELD_DEFINITIONS[..]),
            (&WEIGHT_FIELDS[..], &WEIGHT_FIELD_DEFINITIONS[..]),
        ] {
            let declared: Vec<&str> = definitions.iter().map(|(n, _)| *n).collect();
            assert_eq!(declared, names);
        }
        for (names, definitions) in [
            (&TERM_INDEXES[..], &TERM_INDEX_DEFINITIONS[..]),
            (&COSPAN_INDEXES[..], &COSPAN_INDEX_DEFINITIONS[..]),
            (&WEIGHT_INDEXES[..], &WEIGHT_INDEX_DEFINITIONS[..]),
        ] {
            let declared: Vec<&str> = definitions.iter().map(|(n, _)| *n).collect();
            assert_eq!(declared, names);
        }
    }

    /// Every load-bearing token the guard exists to protect must actually be in
    /// the expected definitions it compares against.
    #[test]
    fn expected_definitions_carry_the_guards() {
        for schema in ALL_SCHEMAS {
            assert!(
                schema.table_definition.contains("SCHEMAFULL"),
                "{}",
                schema.table
            );
            for (name, definition) in schema.fields {
                if *name == "id" {
                    assert!(definition.contains("ASSERT record::id($value)"), "{name}");
                } else {
                    assert!(definition.contains("READONLY"), "{name}");
                }
                assert!(!definition.contains("option<"), "{name}");
                assert!(!definition.contains("TYPE any"), "{name}");
            }
        }
    }

    /// Element definitions are the engine's, not ours, so the rules differ:
    /// each must belong to a declared `array<T>` column, that parent must be the
    /// one carrying `READONLY` (which is what refuses an element write), and the
    /// DDL must not pretend to declare the element — the statement would never
    /// run.
    #[test]
    fn element_definitions_belong_to_readonly_array_columns() {
        for schema in ALL_SCHEMAS {
            for (name, definition) in schema.element_fields {
                let parent = name
                    .strip_suffix(".*")
                    .expect("an element definition names its parent");
                let (_, parent_definition) = schema
                    .fields
                    .iter()
                    .find(|(field, _)| *field == parent)
                    .unwrap_or_else(|| panic!("`{name}` has no declared parent column"));
                assert!(
                    parent_definition.contains("TYPE array<"),
                    "`{parent}` is not an array column, so `{name}` would not exist"
                );
                assert!(
                    parent_definition.contains("READONLY"),
                    "`{parent}` must be READONLY: that is what refuses an element write"
                );
                assert!(!definition.contains("option<"), "{name}");
                assert!(!definition.contains("TYPE any"), "{name}");
                assert!(
                    !schema.ddl.contains(&format!("{name} ON {}", schema.table)),
                    "the {} DDL declares `{name}`, which the parent column already \
                     created — the statement is dead text",
                    schema.table
                );
            }
        }
    }

    /// And the other direction: an `array<T>` column that has no element
    /// definition listed would make the drift guard reject a correctly
    /// bootstrapped database, because the engine creates one regardless.
    #[test]
    fn every_array_column_lists_its_element_definition() {
        for schema in ALL_SCHEMAS {
            for (name, definition) in schema.fields {
                if !definition.contains("TYPE array<") {
                    continue;
                }
                let element = format!("{name}.*");
                assert!(
                    schema
                        .element_fields
                        .iter()
                        .any(|(field, _)| *field == element),
                    "`{name}` is an array column, so the engine defines `{element}` — \
                     which the drift guard would then report as unexpected"
                );
            }
        }
    }

    /// The DDL defines exactly the declared columns and no others — otherwise
    /// bootstrapping would create a column the drift guard then reports as
    /// unexpected, failing every open.
    #[test]
    fn the_ddl_defines_no_undeclared_field() {
        for schema in ALL_SCHEMAS {
            let defined = schema
                .ddl
                .lines()
                .filter_map(|line| line.trim().strip_prefix("DEFINE FIELD IF NOT EXISTS "))
                .filter_map(|rest| rest.split_whitespace().next())
                .count();
            assert_eq!(defined, schema.fields.len(), "{}", schema.table);
        }
    }

    /// The table has to be defined before any of its fields: re-issuing
    /// `DEFINE TABLE` drops that table's field definitions, so a DDL that
    /// interleaved them would leave a replay with a schemafull table and no
    /// columns.
    #[test]
    fn the_table_is_defined_before_its_fields() {
        for schema in ALL_SCHEMAS {
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

    /// `OVERWRITE` on an index is a synchronous delete plus a full rebuild, and
    /// this DDL runs on every store open.
    #[test]
    fn the_ddl_never_overwrites() {
        for schema in ALL_SCHEMAS {
            assert!(!schema.ddl.contains("OVERWRITE"), "{}", schema.table);
        }
    }

    /// An optional or `any`-typed column would disarm its own `ASSERT`: the
    /// unset-skip fires when the value is `NONE` *and* the declared type admits
    /// `NONE`.
    #[test]
    fn no_column_is_optional_or_any() {
        for schema in ALL_SCHEMAS {
            assert!(!schema.ddl.contains("option<"), "{}", schema.table);
            assert!(!schema.ddl.contains("TYPE any"), "{}", schema.table);
        }
    }

    /// There is no `$key` binding in an `ASSERT`. On `id`, `$value` is the whole
    /// record id, so the key has to be extracted before it can be matched.
    #[test]
    fn the_id_assert_extracts_the_key_from_the_record_id() {
        for schema in ALL_SCHEMAS {
            assert!(
                schema.ddl.contains("record::id($value)"),
                "{}",
                schema.table
            );
            assert!(!schema.ddl.contains("$key"), "{}", schema.table);
        }
    }

    /// Index keys normalise numbers, so a unique index containing a numeric
    /// column refuses rows that are not duplicates. Both unique indexes here are
    /// built from `string` columns, and this keeps it that way.
    #[test]
    fn no_numeric_column_joins_a_unique_index() {
        for schema in ALL_SCHEMAS {
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
    /// three things the guard compares, in the order it takes them.
    #[test]
    fn the_info_query_reads_the_table_then_its_fields_then_its_indexes() {
        assert_eq!(
            info_query("cospan"),
            "RETURN (INFO FOR DB).tables.cospan;\n\
             RETURN (INFO FOR TABLE cospan).fields;\n\
             RETURN (INFO FOR TABLE cospan).indexes;"
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
}
