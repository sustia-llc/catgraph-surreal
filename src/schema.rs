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
//! # The drift guard compares definitions, not names
//!
//! [`assert_term_schema`] compares the **engine-rendered definition strings**
//! (`INFO FOR DB` / `INFO FOR TABLE`) against the ones this build expects — not
//! merely the field and index *names*. A name-only check is blind to exactly
//! the drift that disarms the guards: `ALTER TABLE … SCHEMALESS` keeps every
//! field defined, a dropped `READONLY` clause keeps the field's name, and a
//! field re-typed to `option<T>` still answers to its name. The definition
//! strings are the engine's own normalization at SurrealDB 3.2.4, pinned by an
//! integration test against a fresh bootstrap — an SDK upgrade that changes the
//! rendering fails that test loudly, which is the moment to re-pin the strings
//! deliberately rather than discover the change in production.
//!
//! # The guards are defence in depth, not the trust boundary
//!
//! On an embedded connection with no root user configured, table and field
//! permissions are never evaluated, and prefixing `OPTION IMPORT;` to a query
//! disables `READONLY`, `ASSERT`, type processing, and events for that query.
//! So the `READONLY` columns and the id-format `ASSERT` do not *guarantee*
//! anything: the store's own validation is the trust boundary and the database
//! backs it up. This is why [`crate::term`]'s revalidation runs on every load
//! and why a restore re-verifies term ids rather than trusting the replay.

use std::collections::BTreeMap;

use surrealdb::Surreal;
use surrealdb::engine::any::Any;

use crate::error::{Result, StoreError};

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

/// Reads the term table's own rendered definition (or `NONE` if undefined),
/// its field definitions, and its index definitions, in that order.
const TERM_INFO: &str = "\
RETURN (INFO FOR DB).tables.term;
RETURN (INFO FOR TABLE term).fields;
RETURN (INFO FOR TABLE term).indexes;
";

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
    client.query(TERM_DDL).await?.check()?;
    assert_term_schema(client).await
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
    let mut response = client.query(TERM_INFO).await?;
    let table: Option<String> = response.take(0)?;
    let fields: Option<BTreeMap<String, String>> = response.take(1)?;
    let indexes: Option<BTreeMap<String, String>> = response.take(2)?;

    let Some(table) = table else {
        return Err(drift("the table is not defined"));
    };
    if table != TERM_TABLE_DEFINITION {
        return Err(drift(&format!(
            "table definition is `{table}`, expected `{TERM_TABLE_DEFINITION}`"
        )));
    }

    let fields = fields.unwrap_or_default();
    compare_definitions("column", &fields, &TERM_FIELD_DEFINITIONS, true)?;

    let indexes = indexes.unwrap_or_default();
    compare_definitions("index", &indexes, &TERM_INDEX_DEFINITIONS, false)?;

    Ok(())
}

/// Compare live rendered definitions against the expected set.
///
/// `exact` demands the live set carry nothing beyond the expected one; indexes
/// pass `false` so an operator-added index is tolerated.
fn compare_definitions(
    kind: &str,
    live: &BTreeMap<String, String>,
    expected: &[(&str, &str)],
    exact: bool,
) -> Result<()> {
    for (name, definition) in expected {
        match live.get(*name) {
            None => return Err(drift(&format!("{kind} `{name}` is missing"))),
            Some(found) if found != definition => {
                return Err(drift(&format!(
                    "{kind} `{name}` is defined as `{found}`, expected `{definition}`"
                )));
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
                return Err(drift(&format!("unexpected {kind} `{name}`")));
            }
        }
    }
    Ok(())
}

/// The uniform drift error.
fn drift(detail: &str) -> StoreError {
    StoreError::Schema {
        table: TERM_TABLE.to_owned(),
        detail: detail.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The DDL and [`TERM_FIELDS`] are two statements of the same fact, and the
    /// drift guard compares the live schema against the *second*. If they ever
    /// disagree, the guard reports drift on a correctly-bootstrapped database —
    /// so pin them together here rather than discovering it against an engine.
    #[test]
    fn every_declared_field_appears_in_the_ddl() {
        for field in TERM_FIELDS {
            let definition = format!("DEFINE FIELD IF NOT EXISTS {field} ON {TERM_TABLE} ");
            assert!(
                TERM_DDL.contains(&definition),
                "`{field}` is declared but not defined in the DDL"
            );
        }
    }

    #[test]
    fn every_declared_index_appears_in_the_ddl() {
        for index in TERM_INDEXES {
            let definition = format!("DEFINE INDEX IF NOT EXISTS {index} ON {TERM_TABLE} ");
            assert!(
                TERM_DDL.contains(&definition),
                "`{index}` is declared but not defined in the DDL"
            );
        }
    }

    /// The three statements of the column list — names, DDL, and expected
    /// rendered definitions — must agree with each other.
    #[test]
    fn field_definitions_cover_exactly_the_declared_fields() {
        let names: Vec<&str> = TERM_FIELD_DEFINITIONS.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, TERM_FIELDS);
        let index_names: Vec<&str> = TERM_INDEX_DEFINITIONS.iter().map(|(n, _)| *n).collect();
        assert_eq!(index_names, TERM_INDEXES);
    }

    /// Every load-bearing token the guard exists to protect must actually be in
    /// the expected definitions it compares against.
    #[test]
    fn expected_definitions_carry_the_guards() {
        assert!(TERM_TABLE_DEFINITION.contains("SCHEMAFULL"));
        for (name, definition) in TERM_FIELD_DEFINITIONS {
            if name == "id" {
                assert!(definition.contains("ASSERT record::id($value)"), "{name}");
            } else {
                assert!(definition.contains("READONLY"), "{name}");
            }
            assert!(!definition.contains("option<"), "{name}");
            assert!(!definition.contains("TYPE any"), "{name}");
        }
    }

    /// The DDL defines exactly the declared columns and no others — otherwise
    /// bootstrapping would create a column the drift guard then reports as
    /// unexpected, failing every open.
    #[test]
    fn the_ddl_defines_no_undeclared_field() {
        let defined = TERM_DDL
            .lines()
            .filter_map(|line| line.trim().strip_prefix("DEFINE FIELD IF NOT EXISTS "))
            .filter_map(|rest| rest.split_whitespace().next())
            .count();
        assert_eq!(defined, TERM_FIELDS.len());
    }

    /// The table has to be defined before any of its fields: re-issuing
    /// `DEFINE TABLE` drops that table's field definitions, so a DDL that
    /// interleaved them would leave a replay with a schemafull table and no
    /// columns.
    #[test]
    fn the_table_is_defined_before_its_fields() {
        let table = TERM_DDL
            .find("DEFINE TABLE")
            .expect("invariant: the DDL defines the term table");
        let first_field = TERM_DDL
            .find("DEFINE FIELD")
            .expect("invariant: the DDL defines at least one field");
        assert!(table < first_field);
    }

    /// `OVERWRITE` on an index is a synchronous delete plus a full rebuild, and
    /// this DDL runs on every store open.
    #[test]
    fn the_ddl_never_overwrites() {
        assert!(!TERM_DDL.contains("OVERWRITE"));
    }

    /// An optional or `any`-typed column would disarm its own `ASSERT`: the
    /// unset-skip fires when the value is `NONE` *and* the declared type admits
    /// `NONE`.
    #[test]
    fn no_column_is_optional_or_any() {
        assert!(!TERM_DDL.contains("option<"));
        assert!(!TERM_DDL.contains("TYPE any"));
    }

    /// There is no `$key` binding in an `ASSERT`. On `id`, `$value` is the whole
    /// record id, so the key has to be extracted before it can be matched.
    #[test]
    fn the_id_assert_extracts_the_key_from_the_record_id() {
        assert!(TERM_DDL.contains("record::id($value)"));
        assert!(!TERM_DDL.contains("$key"));
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
                "column",
                &live(&[("a", "DEFINE a"), ("b", "DEFINE b")]),
                &expected,
                true
            )
            .is_ok()
        );
        assert!(
            compare_definitions(
                "column",
                &live(&[("a", "ALTERED"), ("b", "DEFINE b")]),
                &expected,
                true
            )
            .is_err()
        );
        assert!(
            compare_definitions("column", &live(&[("a", "DEFINE a")]), &expected, true).is_err()
        );
        let with_extra = live(&[("a", "DEFINE a"), ("b", "DEFINE b"), ("c", "DEFINE c")]);
        assert!(compare_definitions("column", &with_extra, &expected, true).is_err());
        assert!(compare_definitions("index", &with_extra, &expected, false).is_ok());
    }
}
