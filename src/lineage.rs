//! Lineage encoding: rule sets, optimizer traces, and derivation edges.
//!
//! # What a trace is evidence of, and what it is not
//!
//! An optimizer run records the steps that took a starting morphism to the
//! cheapest one it reached: for each step, which rule fired and which hyperedges
//! of the state it fired on. Upstream calls that a *witness* rather than a
//! summary — replaying it re-derives the state it describes — and the store
//! keeps it in that spirit: the step rows are the engine's own indices,
//! unmodified.
//!
//! The store replays one as well. [`RunRecord::replay`] turns the stored step
//! rows back into upstream's own `RewriteStep` values and hands them to
//! `rewrite::replay`, which re-derives every step's assignment against the
//! state it has actually reached — so a trace that is not a legal derivation of
//! the start it is given, under the rules it is given, comes back as an error
//! rather than as an endpoint. A step binds a rule *index*, so the rules have
//! to be rebuilt in the order they were stored.
//!
//! The `replayable` column says which side of that a row was written on: `true`
//! for runs this build records, `false` for rows written before the
//! reconstruction existed.
//!
//! # `cost_model` is mandatory, and the reason is arithmetic
//!
//! A run's costs are sums of a per-generator weighting, and that weighting
//! arrives as a closure — unpersistable by construction. Two runs measured under
//! different weightings produce numbers that look comparable and are not: "12
//! down to 7" and "12 down to 7" can describe opposite outcomes. So the column
//! is `string`, non-optional, and carries whatever name the consumer gives its
//! weighting. The store takes no view on what a good name is; it only refuses to
//! store numbers that do not say what they measured.
//!
//! # Rule sets round-trip through the constructor, never around it
//!
//! A rule set is stored as the canonical JSON of its equation pairs, and loading
//! rebuilds every rule through `RewriteRule::new`. That is not a convenience:
//! `RewriteRule::new` is where the four conditions a rewrite site relies on are
//! checked — parallel sides, well-formed arities and words re-derived rather
//! than trusted, a non-empty left-hand side, and a mono left interface. Serde
//! checks none of them, and a rule that skipped them would fail at a match site
//! instead of at the boundary, or not fail at all and match on label equality
//! where it should not have matched.
//!
//! Its rejections arrive as `CatgraphError::Rewrite(RewriteRejection)` and are
//! surfaced unchanged as [`StoreError::Catgraph`], because upstream's message
//! names the violated condition and nothing here can say it better.
//!
//! # Integers cross the boundary checked
//!
//! Rule indices, edge indices, costs, and counts are all `usize` or `u64` in
//! memory and `int` in the database, and the SDK's own conversion is an
//! **unchecked** `as i64`. Every one of them is narrowed through a checked
//! conversion here instead, for the same reason terms are opaque strings: a
//! saturating sentinel silently becoming `-1` is the failure mode this store
//! exists to not have.

use catgraph_applied::prop::PropSignature;
use catgraph_applied::prop::colored::ColoredExpr;
use catgraph_applied::prop::presentation::content::is_arity_well_formed;
use catgraph_applied::prop::presentation::rewrite::{
    self, RewriteOutcome, RewriteRule, RewriteStep,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::addr::{DerivationAddr, RuleSetAddr, RunAddr, TermAddr};
use crate::error::{Result, RevalidationStage, StoreError};
use crate::schema::{DERIVES_TABLE, REWRITE_RUN_TABLE, RULE_SET_TABLE};
use crate::term::{self, MAX_TERM_DEPTH, TermRecord, term_depth};

/// The version tag stored in every rule set's `codec` column.
///
/// It versions the JSON shape of the equation pairs and the digest algorithm.
pub const RULE_SET_CODEC: &str = "cgs1";

/// The version tag stored in every rewrite run's `codec` column.
///
/// It versions the trace encoding and the way a run's content address is
/// derived from its columns.
pub const REWRITE_RUN_CODEC: &str = "cgt1";

/// The version tag stored in every derivation edge's `codec` column.
///
/// `cge2` (pre-release, no `cge1` data ever left a test process): the edge's
/// content address now covers its denormalized `rule_set` and `cost_model`
/// columns as well as `(parent, child, run)`. Under `cge1` those two were the
/// only stored columns in the tier that revalidation could not re-derive, so a
/// tampered or restored edge could misattribute a derivation to a weighting the
/// run never used and still verify. Folding them into the digest closes that,
/// at the cost of a new address for every edge — which is what the tag is for.
pub const DERIVATION_CODEC: &str = "cge2";

// ------------------------------------------------------------------ rule sets

/// A set of rewrite rules as it is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleSetRecord {
    // Crate-visible so the row conversion can MOVE the encoding string rather
    // than clone it per write; the public surface stays the getters.
    pub(crate) addr: RuleSetAddr,
    pub(crate) codec: String,
    pub(crate) rules_json: String,
    pub(crate) rule_count: i64,
}

impl RuleSetRecord {
    /// Rebuild a record from columns read back out of the database.
    ///
    /// Nothing here is trusted: [`Self::revalidate`] re-derives the content
    /// address, re-parses the encoding, and rebuilds every rule through its
    /// constructor.
    #[must_use]
    pub(crate) fn from_columns(
        addr: RuleSetAddr,
        codec: String,
        rules_json: String,
        rule_count: i64,
    ) -> Self {
        Self {
            addr,
            codec,
            rules_json,
            rule_count,
        }
    }

    /// The rule set's content address, which is also its record id.
    #[must_use]
    pub fn addr(&self) -> &RuleSetAddr {
        &self.addr
    }

    /// The encoding version this record was written under.
    #[must_use]
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// The canonical JSON encoding of the equation pairs.
    #[must_use]
    pub fn rules_json(&self) -> &str {
        &self.rules_json
    }

    /// How many rules the set holds.
    #[must_use]
    pub fn rule_count(&self) -> i64 {
        self.rule_count
    }

    /// Rebuild the rules, re-checking every one.
    ///
    /// # The order, and why it is fixed
    ///
    /// 1. **Codec.** A record written under an encoding this build does not know
    ///    is refused rather than decoded under the current rules.
    /// 2. **Address.** The encoding is re-digested and compared against the id
    ///    the record was filed under. Cheapest check, and it fully decides
    ///    byte-tampering, so it runs before anything interprets a byte.
    /// 3. **Parse.** The JSON parser's own container limit fires here, which is
    ///    the effective depth ceiling — see [`crate::term`].
    /// 4. **Screen, then construct.** Both sides of every pair go through the
    ///    depth and arity screens before `RewriteRule::new` sees them. The arity
    ///    screen is not optional: the constructor's own path into
    ///    `content_of_colored` aborts rather than errors on arities that
    ///    saturated, and the screen is what keeps a corrupt document an error.
    ///
    /// # Errors
    ///
    /// [`StoreError::TypeMismatch`] for an unknown codec or a `rule_count` that
    /// disagrees, [`StoreError::Corrupt`] for a record filed under an id that is
    /// not its own content address, [`StoreError::Codec`] if the encoding does
    /// not parse, [`StoreError::Revalidation`] if a side fails a screen, and
    /// [`StoreError::Catgraph`] if a rule is not one — naming the condition it
    /// violated.
    pub fn revalidate<G>(&self) -> Result<Vec<RewriteRule<G>>>
    where
        G: PropSignature + Serialize + DeserializeOwned,
        G::Color: Serialize + DeserializeOwned,
    {
        if self.codec != RULE_SET_CODEC {
            return Err(StoreError::TypeMismatch {
                field: "codec".to_owned(),
                expected: RULE_SET_CODEC.to_owned(),
                actual: self.codec.clone(),
            });
        }

        let addr = rule_set_address(&self.rules_json)?;
        if addr != self.addr {
            return Err(StoreError::Corrupt {
                context: format!("{RULE_SET_TABLE}:{}", self.addr),
                detail: format!("content address of the stored encoding is `{addr}`"),
            });
        }

        let pairs: Vec<(ColoredExpr<G>, ColoredExpr<G>)> = serde_json::from_str(&self.rules_json)?;
        let count = to_column("rule_count", pairs.len())?;
        if count != self.rule_count {
            return Err(StoreError::TypeMismatch {
                field: "rule_count".to_owned(),
                expected: self.rule_count.to_string(),
                actual: count.to_string(),
            });
        }

        let mut rules = Vec::with_capacity(pairs.len());
        for (lhs, rhs) in pairs {
            screen(&lhs)?;
            screen(&rhs)?;
            rules.push(RewriteRule::new(lhs, rhs)?);
        }
        Ok(rules)
    }
}

/// Encode a rule set into the record that will be stored.
///
/// The pairs are stored, not the compiled rules: `RewriteRule` holds a
/// content-level span with no serde representation, and re-compiling on load is
/// what re-runs the constructor's checks. An empty set is refused — a rule set
/// that can rewrite nothing is a mistake wearing a valid shape.
///
/// # Errors
///
/// [`StoreError::Revalidation`] if a side fails the depth or arity screen or the
/// set is empty, [`StoreError::Catgraph`] if a pair is not a well-formed rule,
/// and [`StoreError::Codec`] if the pairs do not serialize or do not parse back.
pub fn encode_rule_set<G>(rules: &[(ColoredExpr<G>, ColoredExpr<G>)]) -> Result<RuleSetRecord>
where
    G: PropSignature + Serialize + DeserializeOwned,
    G::Color: Serialize + DeserializeOwned,
{
    if rules.is_empty() {
        return Err(StoreError::Revalidation {
            stage: RevalidationStage::Check,
            detail: "a rule set with no rules rewrites nothing".to_owned(),
        });
    }
    for (lhs, rhs) in rules {
        screen(lhs)?;
        screen(rhs)?;
        // Compiled and discarded: the point is to refuse a set that would fail
        // to load, at the boundary where the caller still has the rules in hand.
        RewriteRule::new(lhs.clone(), rhs.clone())?;
    }

    let rules_json = serde_json::to_string(rules)?;
    // Write only what can be read back — the parser's container limit is
    // stricter than the structural one the screen applies.
    let parsed: Vec<(ColoredExpr<G>, ColoredExpr<G>)> = serde_json::from_str(&rules_json)?;
    debug_assert_eq!(
        parsed.len(),
        rules.len(),
        "invariant: a round-tripped rule set holds the same number of pairs"
    );

    Ok(RuleSetRecord {
        addr: rule_set_address(&rules_json)?,
        codec: RULE_SET_CODEC.to_owned(),
        rule_count: to_column("rule_count", rules.len())?,
        rules_json,
    })
}

// --------------------------------------------------------------- rewrite runs

/// One step of a stored trace: which rule fired, and on which hyperedges.
///
/// The edge indices are the state's own, in the rule's internal left-hand-side
/// edge order — the engine's numbering, kept as it was found. They are `i64`
/// rather than `usize` because this type mirrors the row; the conversion happens
/// once, in [`encode_run`], where it is checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceStep {
    pub(crate) rule: i64,
    pub(crate) matched_edges: Vec<i64>,
}

impl TraceStep {
    /// Rebuild a step from a row read back out of the database.
    #[must_use]
    pub(crate) fn from_columns(rule: i64, matched_edges: Vec<i64>) -> Self {
        Self {
            rule,
            matched_edges,
        }
    }

    /// Index of the fired rule, in the rule set the run names.
    #[must_use]
    pub fn rule(&self) -> i64 {
        self.rule
    }

    /// The hyperedges of the state this step matched.
    #[must_use]
    pub fn matched_edges(&self) -> &[i64] {
        &self.matched_edges
    }
}

/// The field layout `RewriteStep` deserializes from.
#[derive(Serialize)]
struct StepWire {
    rule: usize,
    matched_edges: Vec<usize>,
}

/// An optimizer run as it is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunRecord {
    pub(crate) addr: RunAddr,
    pub(crate) codec: String,
    pub(crate) rule_set: RuleSetAddr,
    pub(crate) start: TermAddr,
    pub(crate) best: TermAddr,
    pub(crate) cost_model: String,
    pub(crate) initial_cost: i64,
    pub(crate) best_cost: i64,
    pub(crate) fuel_exhausted: bool,
    pub(crate) states_explored: i64,
    pub(crate) step_count: i64,
    pub(crate) steps: Vec<TraceStep>,
    pub(crate) replayable: bool,
}

impl RunRecord {
    /// Rebuild a record from columns read back out of the database.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_columns(
        addr: RunAddr,
        codec: String,
        rule_set: RuleSetAddr,
        start: TermAddr,
        best: TermAddr,
        cost_model: String,
        initial_cost: i64,
        best_cost: i64,
        fuel_exhausted: bool,
        states_explored: i64,
        step_count: i64,
        steps: Vec<TraceStep>,
        replayable: bool,
    ) -> Self {
        Self {
            addr,
            codec,
            rule_set,
            start,
            best,
            cost_model,
            initial_cost,
            best_cost,
            fuel_exhausted,
            states_explored,
            step_count,
            steps,
            replayable,
        }
    }

    /// The run's content address, which is also its record id.
    #[must_use]
    pub fn addr(&self) -> &RunAddr {
        &self.addr
    }

    /// The encoding version this record was written under.
    #[must_use]
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// The rule set the run's step indices are relative to.
    ///
    /// A trace binds rule *indices*, not rule identities, so a step is only
    /// meaningful against this set. That is what makes the address a pin rather
    /// than a label: replaying a trace against a different set of rules may
    /// still be a legal derivation, just not this one.
    #[must_use]
    pub fn rule_set(&self) -> &RuleSetAddr {
        &self.rule_set
    }

    /// The morphism the run started from.
    #[must_use]
    pub fn start(&self) -> &TermAddr {
        &self.start
    }

    /// The cheapest morphism the run reached.
    ///
    /// Best **found under fuel**: not a normal form, not canonical, and not
    /// stable under a change of budget.
    #[must_use]
    pub fn best(&self) -> &TermAddr {
        &self.best
    }

    /// The name of the per-generator weighting the costs were measured under.
    #[must_use]
    pub fn cost_model(&self) -> &str {
        &self.cost_model
    }

    /// The starting morphism's cost.
    #[must_use]
    pub fn initial_cost(&self) -> i64 {
        self.initial_cost
    }

    /// The best morphism's cost. Never above [`Self::initial_cost`] — the start
    /// is itself a candidate.
    #[must_use]
    pub fn best_cost(&self) -> i64 {
        self.best_cost
    }

    /// Whether the budget ran out with applicable matches still unexplored.
    #[must_use]
    pub fn fuel_exhausted(&self) -> bool {
        self.fuel_exhausted
    }

    /// How many distinct states the search saw, including the start.
    #[must_use]
    pub fn states_explored(&self) -> i64 {
        self.states_explored
    }

    /// How many steps the trace holds.
    ///
    /// Stored beside the steps rather than derived from them so that a trace
    /// truncated on the way in or out disagrees with its own column instead of
    /// looking like a shorter run.
    #[must_use]
    pub fn step_count(&self) -> i64 {
        self.step_count
    }

    /// The steps from the start to the best, in application order.
    #[must_use]
    pub fn steps(&self) -> &[TraceStep] {
        &self.steps
    }

    /// Whether the build that recorded this run wrote a stored form
    /// [`Self::replay`] accepts.
    ///
    /// `true` for every run this build records; `false` for a row written
    /// before the reconstruction existed.
    ///
    /// It is deliberately **not** part of the run's content address. The flag
    /// describes the writer, not what the run was, so folding it in would give
    /// the same run two addresses either side of the change. It is also
    /// deliberately not re-derived on load: a build that can replay must be able
    /// to read a run an older build wrote, and a build that cannot must not
    /// silently rewrite the claim. The column is `READONLY`, so an older build
    /// re-writing a newer build's run is refused loudly rather than downgrading
    /// it.
    #[must_use]
    pub fn replayable(&self) -> bool {
        self.replayable
    }

    /// Re-derive the morphism this run's trace describes.
    ///
    /// `start` is the morphism the run began from and `rules` the rule set it
    /// ran under, rebuilt through `RewriteRule::new` **in stored order**: a step
    /// binds a rule *index*, so a differently ordered slice replays the same
    /// steps to a different endpoint or to none at all.
    ///
    /// Every step is re-derived against the state the replay has reached, so
    /// the stored rows are checked rather than trusted.
    ///
    /// # Errors
    ///
    /// [`StoreError::Corrupt`] for a step column that is not an index,
    /// [`StoreError::Codec`] if a step does not cross into upstream's own step
    /// type, and [`StoreError::Catgraph`] if `start`, `rules` and the steps are
    /// not a derivation — naming the step that failed.
    pub fn replay<G: PropSignature>(
        &self,
        start: &ColoredExpr<G>,
        rules: &[RewriteRule<G>],
    ) -> Result<ColoredExpr<G>> {
        Ok(rewrite::replay(start, rules, &self.rewrite_steps()?)?)
    }

    /// The stored steps as upstream's own [`RewriteStep`] values.
    ///
    /// The crossing goes through `RewriteStep`'s serde derive, its fields being
    /// private and its only other constructor taking a live match site. The
    /// wire shape is therefore a coupling between the two crates, and
    /// `tests/golden.rs` pins it.
    fn rewrite_steps(&self) -> Result<Vec<RewriteStep>> {
        self.steps
            .iter()
            .enumerate()
            .map(|(position, step)| {
                let wire = StepWire {
                    rule: to_index(position, "rule", step.rule)?,
                    matched_edges: step
                        .matched_edges
                        .iter()
                        .map(|edge| to_index(position, "matched_edges", *edge))
                        .collect::<Result<Vec<usize>>>()?,
                };
                Ok(serde_json::from_value(serde_json::to_value(wire)?)?)
            })
            .collect()
    }

    /// Re-derive everything derivable from the stored columns.
    ///
    /// The step *contents* cannot be checked against anything — they are the
    /// engine's own indices into a state this store does not reconstruct — so
    /// what is checked is everything that is: the codec, the content address
    /// over the whole record, and the step count against the steps themselves.
    ///
    /// # Errors
    ///
    /// [`StoreError::TypeMismatch`] for an unknown codec or a `step_count` that
    /// disagrees with the stored steps, and [`StoreError::Corrupt`] for a record
    /// filed under an id that is not its own content address.
    pub fn revalidate(&self) -> Result<()> {
        if self.codec != REWRITE_RUN_CODEC {
            return Err(StoreError::TypeMismatch {
                field: "codec".to_owned(),
                expected: REWRITE_RUN_CODEC.to_owned(),
                actual: self.codec.clone(),
            });
        }
        let count = to_column("step_count", self.steps.len())?;
        if count != self.step_count {
            return Err(StoreError::TypeMismatch {
                field: "step_count".to_owned(),
                expected: self.step_count.to_string(),
                actual: count.to_string(),
            });
        }
        let addr = run_address(self)?;
        if addr != self.addr {
            return Err(StoreError::Corrupt {
                context: format!("{REWRITE_RUN_TABLE}:{}", self.addr),
                detail: format!("content address of the stored columns is `{addr}`"),
            });
        }
        Ok(())
    }
}

/// Encode an optimizer outcome into the record that will be stored.
///
/// `cost_model` names the per-generator weighting the run was measured under and
/// may not be empty — see the [module documentation](self).
///
/// # Errors
///
/// [`StoreError::Revalidation`] if `cost_model` is empty, [`StoreError::Codec`]
/// if the endpoints do not encode, and [`StoreError::TypeMismatch`] if a count,
/// cost, or index does not fit the database's integer column.
pub fn encode_run<G>(
    rule_set: &RuleSetAddr,
    start: &ColoredExpr<G>,
    outcome: &RewriteOutcome<G>,
    cost_model: &str,
) -> Result<RunRecord>
where
    G: PropSignature + Serialize + DeserializeOwned,
    G::Color: Serialize + DeserializeOwned,
{
    encode_run_with_endpoints(rule_set, start, outcome, cost_model).map(|(run, _)| run)
}

/// [`encode_run`], keeping the endpoint terms it had to encode anyway.
///
/// A run's address covers its two endpoints, so both are encoded here whatever
/// the caller does with them — and encoding a term is the expensive half of this
/// call, since it derives a normal form before it digests anything. The
/// repository then has to *store* those same two terms, so handing them back is
/// the difference between encoding two terms and encoding four.
///
/// The endpoints are returned in the order they are stored: the run's start,
/// then its best.
///
/// # Errors
///
/// As [`encode_run`].
pub(crate) fn encode_run_with_endpoints<G>(
    rule_set: &RuleSetAddr,
    start: &ColoredExpr<G>,
    outcome: &RewriteOutcome<G>,
    cost_model: &str,
) -> Result<(RunRecord, [TermRecord; 2])>
where
    G: PropSignature + Serialize + DeserializeOwned,
    G::Color: Serialize + DeserializeOwned,
{
    if cost_model.is_empty() {
        return Err(StoreError::Revalidation {
            stage: RevalidationStage::Check,
            detail: "a run must name the cost model its costs were measured under".to_owned(),
        });
    }

    let steps = outcome
        .steps()
        .iter()
        .map(|step| {
            Ok(TraceStep {
                rule: to_column("steps.rule", step.rule())?,
                matched_edges: step
                    .matched_edges()
                    .iter()
                    .map(|edge| to_column("steps.matched_edges", *edge))
                    .collect::<Result<Vec<i64>>>()?,
            })
        })
        .collect::<Result<Vec<TraceStep>>>()?;

    let endpoints = [term::encode(start)?, term::encode(outcome.best())?];

    let mut record = RunRecord {
        // Filled in below, once every other column is settled: the address is a
        // digest over them.
        addr: RunAddr::from_digest(&"0".repeat(64)).expect("invariant: 64 zeroes is a digest"),
        codec: REWRITE_RUN_CODEC.to_owned(),
        rule_set: rule_set.clone(),
        start: endpoints[0].addr().clone(),
        best: endpoints[1].addr().clone(),
        cost_model: cost_model.to_owned(),
        initial_cost: cost_column("initial_cost", outcome.initial_cost())?,
        best_cost: cost_column("best_cost", outcome.best_cost())?,
        fuel_exhausted: outcome.fuel_exhausted(),
        states_explored: to_column("states_explored", outcome.states_explored())?,
        step_count: to_column("step_count", steps.len())?,
        steps,
        // This build reconstructs the stored steps; see `RunRecord::replayable`.
        replayable: true,
    };
    record.addr = run_address(&record)?;
    Ok((record, endpoints))
}

/// The content address of a run: a digest over every column that describes what
/// the run *was*.
///
/// `replayable` is excluded deliberately — see [`RunRecord::replayable`].
fn run_address(record: &RunRecord) -> Result<RunAddr> {
    let canonical = serde_json::to_string(&(
        REWRITE_RUN_CODEC,
        record.rule_set.as_str(),
        record.start.as_str(),
        record.best.as_str(),
        &record.cost_model,
        record.initial_cost,
        record.best_cost,
        record.fuel_exhausted,
        record.states_explored,
        record
            .steps
            .iter()
            .map(|step| (step.rule, &step.matched_edges))
            .collect::<Vec<_>>(),
    ))?;
    let digest = address(REWRITE_RUN_TABLE, &canonical);
    RunAddr::from_digest(&digest).ok_or_else(|| StoreError::Corrupt {
        context: REWRITE_RUN_TABLE.to_owned(),
        detail: "the derived digest is not well formed".to_owned(),
    })
}

// --------------------------------------------------------------- derivations

/// A derivation edge as it is stored: one morphism, the morphism a run derived
/// from it, and the run that did.
///
/// The edge carries a single hop — the run's start to the run's best — rather
/// than one hop per step, because the intermediate states are not persisted:
/// a step records which rule fired on which edges, not the morphism it produced.
/// The steps are on the run; the graph is what a consumer traverses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivationRecord {
    pub(crate) addr: DerivationAddr,
    pub(crate) codec: String,
    pub(crate) parent: TermAddr,
    pub(crate) child: TermAddr,
    pub(crate) run: RunAddr,
    pub(crate) rule_set: RuleSetAddr,
    pub(crate) cost_model: String,
}

impl DerivationRecord {
    /// Rebuild a record from columns read back out of the database.
    #[must_use]
    pub(crate) fn from_columns(
        addr: DerivationAddr,
        codec: String,
        parent: TermAddr,
        child: TermAddr,
        run: RunAddr,
        rule_set: RuleSetAddr,
        cost_model: String,
    ) -> Self {
        Self {
            addr,
            codec,
            parent,
            child,
            run,
            rule_set,
            cost_model,
        }
    }

    /// The edge's content address, which is also its record id.
    #[must_use]
    pub fn addr(&self) -> &DerivationAddr {
        &self.addr
    }

    /// The encoding version this record was written under.
    #[must_use]
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// The morphism the derivation started from.
    #[must_use]
    pub fn parent(&self) -> &TermAddr {
        &self.parent
    }

    /// The morphism it reached.
    #[must_use]
    pub fn child(&self) -> &TermAddr {
        &self.child
    }

    /// The run that recorded it.
    #[must_use]
    pub fn run(&self) -> &RunAddr {
        &self.run
    }

    /// The rule set the run used.
    #[must_use]
    pub fn rule_set(&self) -> &RuleSetAddr {
        &self.rule_set
    }

    /// The weighting the run's costs were measured under.
    #[must_use]
    pub fn cost_model(&self) -> &str {
        &self.cost_model
    }

    /// Re-derive the content address and compare it against the id this edge was
    /// filed under.
    ///
    /// # Errors
    ///
    /// [`StoreError::TypeMismatch`] for an unknown codec, [`StoreError::Corrupt`]
    /// for an edge filed under an id that is not its own content address.
    pub fn revalidate(&self) -> Result<()> {
        if self.codec != DERIVATION_CODEC {
            return Err(StoreError::TypeMismatch {
                field: "codec".to_owned(),
                expected: DERIVATION_CODEC.to_owned(),
                actual: self.codec.clone(),
            });
        }
        let addr = derivation_address(
            &self.parent,
            &self.child,
            &self.run,
            &self.rule_set,
            &self.cost_model,
        )?;
        if addr != self.addr {
            return Err(StoreError::Corrupt {
                context: format!("{DERIVES_TABLE}:{}", self.addr),
                detail: format!("content address of the stored columns is `{addr}`"),
            });
        }
        Ok(())
    }
}

/// Build the edge a run's endpoints imply.
///
/// # Errors
///
/// [`StoreError::Codec`] if the tuple does not encode.
pub fn encode_derivation(run: &RunRecord) -> Result<DerivationRecord> {
    Ok(DerivationRecord {
        addr: derivation_address(
            &run.start,
            &run.best,
            &run.addr,
            &run.rule_set,
            &run.cost_model,
        )?,
        codec: DERIVATION_CODEC.to_owned(),
        parent: run.start.clone(),
        child: run.best.clone(),
        run: run.addr.clone(),
        rule_set: run.rule_set.clone(),
        cost_model: run.cost_model.clone(),
    })
}

/// The content address of a derivation edge: a digest over **every** column it
/// stores.
///
/// This is what makes writing an edge idempotent without a second uniqueness
/// mechanism. The same columns always land on the same record id, so
/// re-recording a run rewrites the same row with the same values — no change, no
/// read-only refusal, no duplicate. A *different* payload under that id would be
/// a changed value, which the `READONLY` columns refuse.
///
/// `rule_set` and `cost_model` are in the pre-image even though both are
/// *denormalized* copies of columns the run already carries, and that is the
/// point: revalidation re-derives an address and compares it, so a column
/// outside the pre-image is a column nothing checks. Under `cge1` those two were
/// exactly that — the only stored values in the tier a tampered or restored edge
/// could change while still verifying, which would attribute a derivation to a
/// weighting the run never ran under. Reading them off the run instead was not
/// an option: the edge is what a traversal reads, and following it to the run to
/// find out what its numbers mean defeats storing them on it.
///
/// The alternative — a generated edge id plus a unique index over the tuple —
/// was not taken. It needs two mechanisms to say one thing, it puts a record-id
/// column in a unique index (where key normalisation is a standing hazard), and
/// the collision it detects arrives as an index refusal that has to be turned
/// back into "you already recorded this", which is not an error at all.
///
/// # Errors
///
/// [`StoreError::Codec`] if the tuple does not encode.
fn derivation_address(
    parent: &TermAddr,
    child: &TermAddr,
    run: &RunAddr,
    rule_set: &RuleSetAddr,
    cost_model: &str,
) -> Result<DerivationAddr> {
    let canonical = serde_json::to_string(&(
        DERIVATION_CODEC,
        parent.as_str(),
        child.as_str(),
        run.as_str(),
        rule_set.as_str(),
        cost_model,
    ))?;
    let digest = address(DERIVES_TABLE, &canonical);
    DerivationAddr::from_digest(&digest).ok_or_else(|| StoreError::Corrupt {
        context: DERIVES_TABLE.to_owned(),
        detail: "the derived digest is not well formed".to_owned(),
    })
}

// -------------------------------------------------------------------- shared

/// The content address of a rule set's encoding.
fn rule_set_address(rules_json: &str) -> Result<RuleSetAddr> {
    let digest = address(RULE_SET_TABLE, rules_json);
    RuleSetAddr::from_digest(&digest).ok_or_else(|| StoreError::Corrupt {
        context: RULE_SET_TABLE.to_owned(),
        detail: "the derived digest is not well formed".to_owned(),
    })
}

/// A domain-separated digest, as lowercase hex.
///
/// The table name is folded in so that two tiers digesting identical bytes get
/// different addresses. Nothing currently depends on that — the tiers encode
/// different shapes — but an address that means "this rule set" in one table and
/// "this run" in another is a coincidence waiting to be relied on.
fn address(context: &str, canonical: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(context.as_bytes());
    hasher.update(b"\0");
    hasher.update(canonical.as_bytes());
    hasher.finalize().to_hex().to_string()
}

/// Run the two screens a colored morphism must pass before anything interprets
/// it.
///
/// Depth first, then arity: the arity screen is what stands between a corrupt
/// document and an abort deep inside the content pass, and the depth screen is
/// what stands between a pathological one and a stack overflow on the way there.
fn screen<G: PropSignature>(expr: &ColoredExpr<G>) -> Result<()> {
    let depth = term_depth(expr.expr());
    if depth > MAX_TERM_DEPTH {
        return Err(StoreError::Revalidation {
            stage: RevalidationStage::Depth,
            detail: format!("nesting depth {depth} exceeds limit {MAX_TERM_DEPTH}"),
        });
    }
    if !is_arity_well_formed(expr.expr()) {
        return Err(StoreError::Revalidation {
            stage: RevalidationStage::Arity,
            detail: "the term's arities are not well-formed".to_owned(),
        });
    }
    Ok(())
}

/// Widen a stored step column back into an index.
///
/// The column is a signed integer read off disk, so a negative entry is
/// possible and is a corrupt document rather than a rewrite failure.
fn to_index(step: usize, field: &str, value: i64) -> Result<usize> {
    usize::try_from(value).map_err(|_| StoreError::Corrupt {
        context: format!("{REWRITE_RUN_TABLE}.steps"),
        detail: format!("step {step} holds `{field}` = {value}, which is not an index"),
    })
}

/// Narrow a count or index into the database's integer lane.
///
/// The SDK's own `usize` conversion is an unchecked `as i64`, and catgraph
/// deliberately produces saturating `usize::MAX` sentinels — so a cast here
/// would turn one of those into `-1` with nothing raised anywhere.
fn to_column(field: &str, value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| StoreError::TypeMismatch {
        field: field.to_owned(),
        expected: "int".to_owned(),
        actual: format!("{value} (too large for a 64-bit signed integer)"),
    })
}

/// Narrow a cost into the database's integer lane.
///
/// Costs are `u64` and saturate rather than overflow upstream, so the top half
/// of the range is reachable in principle and unrepresentable here.
fn cost_column(field: &str, value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| StoreError::TypeMismatch {
        field: field.to_owned(),
        expected: "int".to_owned(),
        actual: format!("{value} (too large for a 64-bit signed integer)"),
    })
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use catgraph::errors::{CatgraphError, RewriteRejection};
    use catgraph_applied::prop::Free;
    use catgraph_applied::prop::presentation::rewrite::optimize;
    use serde::Deserialize;

    use super::*;

    /// Two generators that can cancel: `Copy ; Discard ⇒ id₁` is a rule with a
    /// non-empty left-hand side, a mono interface, and parallel sides.
    #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
    enum Gen {
        /// `Δ : 1 → 2`
        Copy,
        /// `μ : 2 → 1`
        Add,
    }

    /// One variant byte: `Copy` 0, `Add` 1.
    impl catgraph::CanonicalEncode for Gen {
        fn encode_canonical(&self, out: &mut Vec<u8>) {
            out.push(match self {
                Self::Copy => 0,
                Self::Add => 1,
            });
        }
    }

    impl PropSignature for Gen {
        type Color = ();

        fn source_word(&self) -> Cow<'_, [()]> {
            Cow::Owned(match self {
                Self::Copy => vec![()],
                Self::Add => vec![(), ()],
            })
        }

        fn target_word(&self) -> Cow<'_, [()]> {
            Cow::Owned(match self {
                Self::Copy => vec![(), ()],
                Self::Add => vec![()],
            })
        }
    }

    /// `Δ ; μ : 1 → 1`.
    fn copy_then_add() -> ColoredExpr<Gen> {
        let expr = Free::compose(Free::generator(Gen::Copy), Free::<Gen>::generator(Gen::Add))
            .expect("Δ ; μ composes");
        ColoredExpr::new(vec![()], expr).expect("Δ ; μ type-checks")
    }

    /// `id₁ : 1 → 1`.
    fn identity() -> ColoredExpr<Gen> {
        ColoredExpr::new(vec![()], Free::<Gen>::identity(1)).expect("id₁ type-checks")
    }

    fn rules() -> Vec<(ColoredExpr<Gen>, ColoredExpr<Gen>)> {
        vec![(copy_then_add(), identity())]
    }

    #[test]
    fn a_rule_set_round_trips_through_its_constructor() {
        let record = encode_rule_set(&rules()).expect("a well-formed rule set encodes");
        assert_eq!(record.codec(), RULE_SET_CODEC);
        assert_eq!(record.rule_count(), 1);
        let rebuilt: Vec<RewriteRule<Gen>> =
            record.revalidate().expect("its own encoding revalidates");
        assert_eq!(rebuilt.len(), 1);
    }

    #[test]
    fn rule_set_encoding_is_deterministic() {
        let first = encode_rule_set(&rules()).expect("encodes");
        let second = encode_rule_set(&rules()).expect("encodes");
        assert_eq!(first.addr(), second.addr());
        assert_eq!(first.rules_json(), second.rules_json());
    }

    /// A rule set that can rewrite nothing is a mistake wearing a valid shape.
    #[test]
    fn an_empty_rule_set_is_refused() {
        let err = encode_rule_set::<Gen>(&[]).expect_err("an empty rule set is refused");
        assert!(
            matches!(
                err,
                StoreError::Revalidation {
                    stage: RevalidationStage::Check,
                    ..
                }
            ),
            "{err}"
        );
    }

    /// Upstream's constructor is where a rule's four conditions are checked, and
    /// its rejection has to reach the caller as the catgraph error it is —
    /// naming the violated condition, not flattened into something vaguer.
    #[test]
    fn a_pair_that_is_not_a_rule_is_refused_by_the_constructor() {
        // Non-parallel: `Δ : 1 → 2` against `id₁ : 1 → 1`.
        let copy = ColoredExpr::new(vec![()], Free::<Gen>::generator(Gen::Copy))
            .expect("Δ type-checks at one wire");
        let err = encode_rule_set(&[(copy, identity())])
            .expect_err("non-parallel sides are not a rewrite rule");
        assert!(matches!(err, StoreError::Catgraph(_)), "{err}");
    }

    /// A left-hand side with no generator matches everywhere, which upstream
    /// refuses — and the refusal has to happen at the boundary rather than at a
    /// match site.
    #[test]
    fn a_rule_matching_everywhere_is_refused() {
        let err = encode_rule_set(&[(identity(), identity())])
            .expect_err("an edge-free lhs matches everywhere");
        assert!(matches!(err, StoreError::Catgraph(_)), "{err}");
    }

    #[test]
    fn a_tampered_rule_set_encoding_is_corrupt() {
        let record = encode_rule_set(&rules()).expect("encodes");
        let tampered = RuleSetRecord::from_columns(
            record.addr().clone(),
            record.codec().to_owned(),
            record.rules_json().replace("Copy", "Add"),
            record.rule_count(),
        );
        let err = tampered
            .revalidate::<Gen>()
            .expect_err("a swapped-in encoding is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    #[test]
    fn an_unknown_rule_set_codec_is_refused() {
        let record = encode_rule_set(&rules()).expect("encodes");
        let tampered = RuleSetRecord::from_columns(
            record.addr().clone(),
            "cgs99".to_owned(),
            record.rules_json().to_owned(),
            record.rule_count(),
        );
        match tampered.revalidate::<Gen>() {
            Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "codec"),
            other => panic!("expected a codec mismatch, got {other:?}"),
        }
    }

    fn an_outcome() -> (RuleSetRecord, ColoredExpr<Gen>, RewriteOutcome<Gen>) {
        let record = encode_rule_set(&rules()).expect("encodes");
        let compiled: Vec<RewriteRule<Gen>> = record.revalidate().expect("revalidates");
        let start = copy_then_add();
        let outcome = optimize(&start, &compiled, 16, |_| 1).expect("the search runs");
        (record, start, outcome)
    }

    #[test]
    fn a_run_records_the_trace_and_round_trips() {
        let (rule_set, start, outcome) = an_outcome();
        let record = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("a run encodes");
        assert_eq!(record.codec(), REWRITE_RUN_CODEC);
        assert_eq!(record.cost_model(), "unit");
        assert_eq!(record.rule_set(), rule_set.addr());
        assert_eq!(record.step_count() as usize, record.steps().len());
        assert!(record.replayable(), "this build reconstructs stored steps");
        record.revalidate().expect("its own columns revalidate");
    }

    /// The rule fires, so the trace is not empty and the cost comes down — which
    /// is what makes the round trip test above test anything.
    #[test]
    fn the_fixture_run_actually_rewrites() {
        let (_, _, outcome) = an_outcome();
        assert!(!outcome.steps().is_empty());
        assert!(outcome.best_cost() < outcome.initial_cost());
    }

    /// Costs measured under different weightings are not comparable, so a run
    /// that does not say which weighting it used is refused rather than stored
    /// with a blank.
    #[test]
    fn a_run_without_a_cost_model_is_refused() {
        let (rule_set, start, outcome) = an_outcome();
        let err = encode_run(rule_set.addr(), &start, &outcome, "")
            .expect_err("a run must name its cost model");
        assert!(
            matches!(
                err,
                StoreError::Revalidation {
                    stage: RevalidationStage::Check,
                    ..
                }
            ),
            "{err}"
        );
    }

    /// And two runs differing only in the weighting are different records, which
    /// is the storage-level consequence of the same fact.
    #[test]
    fn the_cost_model_separates_otherwise_identical_runs() {
        let (rule_set, start, outcome) = an_outcome();
        let unit = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");
        let weighted = encode_run(rule_set.addr(), &start, &outcome, "weighted").expect("encodes");
        assert_ne!(unit.addr(), weighted.addr());
    }

    #[test]
    fn a_run_with_a_tampered_step_count_is_refused() {
        let (rule_set, start, outcome) = an_outcome();
        let record = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");
        let tampered = RunRecord::from_columns(
            record.addr().clone(),
            record.codec().to_owned(),
            record.rule_set().clone(),
            record.start().clone(),
            record.best().clone(),
            record.cost_model().to_owned(),
            record.initial_cost(),
            record.best_cost(),
            record.fuel_exhausted(),
            record.states_explored(),
            record.step_count() + 1,
            record.steps().to_vec(),
            record.replayable(),
        );
        match tampered.revalidate() {
            Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "step_count"),
            other => panic!("expected a step-count mismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_run_with_a_tampered_cost_is_corrupt() {
        let (rule_set, start, outcome) = an_outcome();
        let record = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");
        let tampered = RunRecord::from_columns(
            record.addr().clone(),
            record.codec().to_owned(),
            record.rule_set().clone(),
            record.start().clone(),
            record.best().clone(),
            record.cost_model().to_owned(),
            record.initial_cost(),
            record.best_cost() - 1,
            record.fuel_exhausted(),
            record.states_explored(),
            record.step_count(),
            record.steps().to_vec(),
            record.replayable(),
        );
        let err = tampered
            .revalidate()
            .expect_err("a cost nobody measured is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    /// The flag describes what a reader can do, not what the run was, so it must
    /// not change the run's identity — otherwise the day it flips, every stored
    /// run gets a second address.
    #[test]
    fn the_replayable_flag_is_not_part_of_a_runs_identity() {
        let (rule_set, start, outcome) = an_outcome();
        let record = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");
        let flipped = RunRecord::from_columns(
            record.addr().clone(),
            record.codec().to_owned(),
            record.rule_set().clone(),
            record.start().clone(),
            record.best().clone(),
            record.cost_model().to_owned(),
            record.initial_cost(),
            record.best_cost(),
            record.fuel_exhausted(),
            record.states_explored(),
            record.step_count(),
            record.steps().to_vec(),
            !record.replayable(),
        );
        flipped
            .revalidate()
            .expect("a flipped flag leaves the address alone");
    }

    /// `μ ; Δ ; μ ; Δ : 2 → 2` — overlapping reducible sites for both rules.
    fn two_reducible_sites() -> ColoredExpr<Gen> {
        let add_copy = || {
            Free::compose(Free::generator(Gen::Add), Free::<Gen>::generator(Gen::Copy))
                .expect("μ ; Δ composes")
        };
        let twice = Free::compose(add_copy(), add_copy()).expect("μ ; Δ ; μ ; Δ composes");
        ColoredExpr::new(vec![(), ()], twice).expect("μ ; Δ ; μ ; Δ type-checks")
    }

    /// Steps are applied **in order**: each one's hyperedges index the state the
    /// previous ones left behind, so the recorded sequence read backwards is a
    /// different claim about a different state.
    ///
    /// `μ ; Δ ; μ ; Δ` under both rules records two *distinct* steps — rule 0 at
    /// edges `[1, 2]`, then rule 1 at `[0, 1]` — which is what makes reversing
    /// them observable at all; `an_outcome` and `a_two_rule_outcome` each record
    /// the one step `[(0, [0, 1])]`, which reverses to itself.
    #[test]
    fn a_multi_step_trace_replays_only_in_the_recorded_order() {
        let rule_set = encode_rule_set(&[
            (copy_then_add(), identity()),
            (add_then_copy(), identity_two()),
        ])
        .expect("both pairs are rules");
        let compiled: Vec<RewriteRule<Gen>> = rule_set.revalidate().expect("revalidates");
        let start = two_reducible_sites();
        let outcome = optimize(&start, &compiled, 64, |_| 1).expect("the search runs");
        let record = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");

        assert_eq!(
            record.steps().len(),
            2,
            "the order pin needs two steps, got {:?}",
            record.steps()
        );
        assert_ne!(
            record.steps()[0],
            record.steps()[1],
            "two identical steps reverse to themselves, pinning nothing"
        );

        let replayed = record
            .replay(&start, &compiled)
            .expect("the recorded order replays");
        assert_eq!(&replayed, outcome.best());

        let mut reversed = record.steps().to_vec();
        reversed.reverse();
        let out_of_order = RunRecord::from_columns(
            record.addr().clone(),
            record.codec().to_owned(),
            record.rule_set().clone(),
            record.start().clone(),
            record.best().clone(),
            record.cost_model().to_owned(),
            record.initial_cost(),
            record.best_cost(),
            record.fuel_exhausted(),
            record.states_explored(),
            record.step_count(),
            reversed,
            record.replayable(),
        );
        match out_of_order.replay(&start, &compiled) {
            Err(StoreError::Catgraph(CatgraphError::Rewrite(RewriteRejection::NotAMatch {
                step,
            }))) => assert_eq!(step, Some(1)),
            other => panic!("expected a non-match at reversed step 1, got {other:?}"),
        }
    }

    /// `μ ; Δ : 2 → 2`, whose left-hand side cannot fire on `Δ ; μ`.
    fn add_then_copy() -> ColoredExpr<Gen> {
        let expr = Free::compose(Free::generator(Gen::Add), Free::<Gen>::generator(Gen::Copy))
            .expect("μ ; Δ composes");
        ColoredExpr::new(vec![(), ()], expr).expect("μ ; Δ type-checks")
    }

    /// `id₂ : 2 → 2`.
    fn identity_two() -> ColoredExpr<Gen> {
        ColoredExpr::new(vec![(), ()], Free::<Gen>::identity(2)).expect("id₂ type-checks")
    }

    /// A run over **two** rules, so that reordering them is observable at all.
    fn a_two_rule_outcome() -> (RuleSetRecord, ColoredExpr<Gen>, RewriteOutcome<Gen>) {
        let record = encode_rule_set(&[
            (copy_then_add(), identity()),
            (add_then_copy(), identity_two()),
        ])
        .expect("both pairs are rules");
        let compiled: Vec<RewriteRule<Gen>> = record.revalidate().expect("revalidates");
        let start = copy_then_add();
        let outcome = optimize(&start, &compiled, 16, |_| 1).expect("the search runs");
        (record, start, outcome)
    }

    /// A recorded trace is a witness: replayed against the start and the rules
    /// it was recorded under, it re-derives the run's own `best`.
    #[test]
    fn a_recorded_run_replays_to_its_best() {
        let (rule_set, start, outcome) = an_outcome();
        let compiled: Vec<RewriteRule<Gen>> = rule_set.revalidate().expect("revalidates");
        let record = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");
        assert!(
            !record.steps().is_empty(),
            "an empty trace replays vacuously"
        );

        let replayed = record
            .replay(&start, &compiled)
            .expect("a recorded trace replays");
        assert_eq!(&replayed, outcome.best());
        let encoded = term::encode(&replayed).expect("the endpoint encodes");
        assert_eq!(encoded.addr(), record.best());
    }

    /// The stored hyperedges are re-derived against the state, not trusted: a
    /// row whose `matched_edges` were reordered is not a match there.
    #[test]
    fn a_tampered_matched_edges_row_does_not_replay() {
        let (rule_set, start, outcome) = an_outcome();
        let compiled: Vec<RewriteRule<Gen>> = rule_set.revalidate().expect("revalidates");
        let record = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");

        let mut steps = record.steps().to_vec();
        let first = steps.first().expect("the fixture run takes a step").clone();
        assert_eq!(
            first.matched_edges().len(),
            2,
            "the tamper below needs two edges to swap"
        );
        let mut reordered = first.matched_edges().to_vec();
        reordered.reverse();
        steps[0] = TraceStep::from_columns(first.rule(), reordered);

        let tampered = RunRecord::from_columns(
            record.addr().clone(),
            record.codec().to_owned(),
            record.rule_set().clone(),
            record.start().clone(),
            record.best().clone(),
            record.cost_model().to_owned(),
            record.initial_cost(),
            record.best_cost(),
            record.fuel_exhausted(),
            record.states_explored(),
            record.step_count(),
            steps,
            record.replayable(),
        );
        match tampered.replay(&start, &compiled) {
            Err(StoreError::Catgraph(CatgraphError::Rewrite(RewriteRejection::NotAMatch {
                step,
            }))) => assert_eq!(step, Some(0)),
            other => panic!("expected a step-0 non-match, got {other:?}"),
        }
    }

    /// A step binds a rule *index*, so the same trace read against a reversed
    /// rule slice names a different rule — here one whose left-hand side is not
    /// at those hyperedges.
    #[test]
    fn rules_rebuilt_in_reverse_order_do_not_replay_the_trace() {
        let (rule_set, start, outcome) = a_two_rule_outcome();
        let mut compiled: Vec<RewriteRule<Gen>> = rule_set.revalidate().expect("revalidates");
        let record = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");
        record
            .replay(&start, &compiled)
            .expect("stored order replays");

        compiled.reverse();
        match record.replay(&start, &compiled) {
            Err(StoreError::Catgraph(CatgraphError::Rewrite(RewriteRejection::NotAMatch {
                step,
            }))) => assert_eq!(step, Some(0)),
            other => panic!("expected a step-0 non-match under reversed rules, got {other:?}"),
        }
    }

    /// `Δ ; μ ; Δ ; μ : 1 → 1`.
    fn copy_add_twice() -> ColoredExpr<Gen> {
        let once = || {
            Free::compose(Free::generator(Gen::Copy), Free::<Gen>::generator(Gen::Add))
                .expect("Δ ; μ composes")
        };
        let twice = Free::compose(once(), once()).expect("Δ ; μ ; Δ ; μ composes");
        ColoredExpr::new(vec![()], twice).expect("Δ ; μ ; Δ ; μ type-checks")
    }

    /// The other branch a reversed rule slice reaches: two rules sharing a
    /// left-hand side both match at the recorded site, so the trace replays —
    /// to the endpoint the *other* rule rewrites to.
    #[test]
    fn rules_sharing_a_left_hand_side_replay_swapped_to_a_different_endpoint() {
        let rule_set = encode_rule_set(&[
            (copy_then_add(), identity()),
            (copy_then_add(), copy_add_twice()),
        ])
        .expect("both pairs are rules");
        let compiled: Vec<RewriteRule<Gen>> = rule_set.revalidate().expect("revalidates");
        let start = copy_then_add();
        let outcome = optimize(&start, &compiled, 16, |_| 1).expect("the search runs");
        let record = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");
        assert_eq!(
            record.steps().len(),
            1,
            "the pin needs the one-step trace, got {:?}",
            record.steps()
        );
        assert_eq!(
            record.steps()[0].rule(),
            0,
            "the pin needs the trace to name rule 0, got {:?}",
            record.steps()
        );
        record
            .replay(&start, &compiled)
            .expect("stored order replays");

        let swapped_set = encode_rule_set(&[
            (copy_then_add(), copy_add_twice()),
            (copy_then_add(), identity()),
        ])
        .expect("the swapped pairs are rules");
        let swapped: Vec<RewriteRule<Gen>> = swapped_set.revalidate().expect("revalidates");
        let replayed = record
            .replay(&start, &swapped)
            .expect("the shared left-hand side matches at the recorded site");
        let endpoint = term::encode(&replayed).expect("the endpoint encodes");
        assert_ne!(
            endpoint.addr(),
            record.best(),
            "swapped rules replayed to `{}`, against a stored best of `{}`",
            endpoint.addr(),
            record.best()
        );
    }

    /// The mirror a stored trace crosses back through renders the same bytes as
    /// upstream's own step, so a field renamed on either side is a failure here.
    #[test]
    fn the_step_wire_mirror_renders_upstreams_step_bytes() {
        let (_rule_set, _start, outcome) = an_outcome();
        let steps = outcome.steps();
        assert_eq!(steps.len(), 1, "the fixture run takes one step: {steps:?}");
        let wire = StepWire {
            rule: steps[0].rule(),
            matched_edges: steps[0].matched_edges().to_vec(),
        };
        assert_eq!(
            serde_json::to_string(&wire).expect("the mirror serializes"),
            serde_json::to_string(&steps[0]).expect("upstream's step serializes")
        );
    }

    /// A step column off disk is a signed integer, so it can be negative — and
    /// that is a corrupt document rather than a rewrite failure.
    #[test]
    fn a_negative_step_column_does_not_replay() {
        let (rule_set, start, outcome) = an_outcome();
        let compiled: Vec<RewriteRule<Gen>> = rule_set.revalidate().expect("revalidates");
        let record = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");

        let negative = RunRecord::from_columns(
            record.addr().clone(),
            record.codec().to_owned(),
            record.rule_set().clone(),
            record.start().clone(),
            record.best().clone(),
            record.cost_model().to_owned(),
            record.initial_cost(),
            record.best_cost(),
            record.fuel_exhausted(),
            record.states_explored(),
            record.step_count(),
            vec![TraceStep::from_columns(-1, vec![0])],
            record.replayable(),
        );
        let err = negative
            .replay(&start, &compiled)
            .expect_err("a negative rule index is not an index");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    #[test]
    fn a_derivation_edge_is_the_tuple_it_addresses() {
        let (rule_set, start, outcome) = an_outcome();
        let run = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");
        let edge = encode_derivation(&run).expect("the endpoints imply an edge");
        assert_eq!(edge.parent(), run.start());
        assert_eq!(edge.child(), run.best());
        assert_eq!(edge.run(), run.addr());
        edge.revalidate().expect("its own columns revalidate");

        // Deterministic, which is what makes the write idempotent.
        let again = encode_derivation(&run).expect("encodes");
        assert_eq!(edge.addr(), again.addr());
    }

    #[test]
    fn a_derivation_edge_filed_under_the_wrong_id_is_corrupt() {
        let (rule_set, start, outcome) = an_outcome();
        let run = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");
        let edge = encode_derivation(&run).expect("encodes");
        let tampered = DerivationRecord::from_columns(
            edge.addr().clone(),
            edge.codec().to_owned(),
            // A different parent under the same id.
            edge.child().clone(),
            edge.child().clone(),
            edge.run().clone(),
            edge.rule_set().clone(),
            edge.cost_model().to_owned(),
        );
        let err = tampered
            .revalidate()
            .expect_err("an edge filed under another tuple's id is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    /// The denormalized columns are inside the pre-image, so a re-labelled edge
    /// is corrupt rather than merely wrong.
    ///
    /// Under `cge1` `rule_set` and `cost_model` were the only stored values in
    /// the whole tier that revalidation could not re-derive: an edge whose
    /// weighting had been swapped — by a hand edit, or by a restore replaying a
    /// doctored dump — verified cleanly while misattributing every derivation it
    /// described to numbers measured under something else.
    #[test]
    fn an_edge_whose_denormalized_columns_were_swapped_is_corrupt() {
        let (rule_set, start, outcome) = an_outcome();
        let run = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");
        let edge = encode_derivation(&run).expect("encodes");

        let relabelled = DerivationRecord::from_columns(
            edge.addr().clone(),
            edge.codec().to_owned(),
            edge.parent().clone(),
            edge.child().clone(),
            edge.run().clone(),
            edge.rule_set().clone(),
            "weighted".to_owned(),
        );
        let err = relabelled
            .revalidate()
            .expect_err("a swapped weighting is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");

        let repointed = DerivationRecord::from_columns(
            edge.addr().clone(),
            edge.codec().to_owned(),
            edge.parent().clone(),
            edge.child().clone(),
            edge.run().clone(),
            RuleSetAddr::from_digest(&"d".repeat(64)).expect("a valid digest"),
            edge.cost_model().to_owned(),
        );
        let err = repointed
            .revalidate()
            .expect_err("a swapped rule set is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    /// And the same fact stated positively, over the address function directly.
    ///
    /// Going through `encode_run` would prove less than it looks: two runs under
    /// different weightings already have different *run* addresses, so their
    /// edges would differ even under `cge1`. Holding `(parent, child, run)` fixed
    /// and varying only a denormalized column is what isolates the widening.
    #[test]
    fn the_denormalized_columns_move_the_address_on_their_own() {
        let (rule_set, start, outcome) = an_outcome();
        let run = encode_run(rule_set.addr(), &start, &outcome, "unit").expect("encodes");
        let edge = encode_derivation(&run).expect("encodes");
        let at = |rules: &RuleSetAddr, cost_model: &str| {
            derivation_address(edge.parent(), edge.child(), edge.run(), rules, cost_model)
                .expect("a tuple is always addressable")
        };

        let base = at(edge.rule_set(), "unit");
        assert_eq!(&base, edge.addr());
        assert_ne!(base, at(edge.rule_set(), "weighted"));
        assert_ne!(
            base,
            at(
                &RuleSetAddr::from_digest(&"d".repeat(64)).expect("a valid digest"),
                "unit"
            )
        );
    }

    /// Two tiers digesting identical bytes must not agree on an address.
    #[test]
    fn addresses_are_domain_separated_by_table() {
        let left = address("rule_set", "same bytes");
        let right = address("rewrite_run", "same bytes");
        assert_ne!(left, right);
    }
}
