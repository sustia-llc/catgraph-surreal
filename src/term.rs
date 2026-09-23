//! Term encoding, content addressing, and the revalidation-on-load discipline.
//!
//! # Terms are opaque JSON strings, never native objects
//!
//! A stored term is one `string` column holding its canonical JSON encoding.
//! Expanding it into a native nested object would be the obvious thing to do and
//! is wrong twice over:
//!
//! - **`usize` does not survive the value layer.** The SDK writes `usize`
//!   through an *unchecked* `as i64` cast, and catgraph deliberately produces
//!   `usize::MAX` saturation sentinels for arities that would otherwise
//!   overflow. Round-tripping one of those through a native object turns it into
//!   `-1` with nothing raised anywhere.
//! - **Content addressing needs byte-exactness.** An address is the digest of an
//!   encoding, so the encoding has to be the thing that is stored — not a
//!   re-serialization of a decomposed value whose key order, number
//!   representation, and container shapes are the database's business rather
//!   than ours.
//!
//! The columns beside it (`signature`, the arities, `depth`, the generator
//! count, `nf_class`) are all *derived* from that string. They exist to be
//! queried and indexed, and every one of them is re-derived and compared on
//! load — so a hand-edited column is caught rather than believed.
//!
//! # The `G` contract: `Serialize` must be deterministic — and equality-faithful
//!
//! Content addresses are only stable if a generator serializes to the same bytes
//! every time. In practice that means **no `HashMap` or `HashSet` in a
//! generator's serde representation**: their iteration order is unspecified, so
//! two runs would produce two encodings of the same term, two addresses, and two
//! stored records for one genome.
//!
//! The subtler half of the contract: **values the consumer considers equal must
//! serialize to equal bytes**. Floating-point colors are the standing hazard —
//! `-0.0` and `0.0` compare equal but encode as different JSON, so the "same"
//! morphism built through two arithmetic paths gets two addresses and two
//! `nf_class` buckets; a `NaN` color fails to encode at all, with a generic
//! [`Codec`](crate::StoreError::Codec) error that never names the float. Prefer
//! integral or otherwise canonical color and generator representations; if a
//! float must appear, canonicalize it (`-0.0 → 0.0`, no `NaN`) before it
//! reaches serde.
//!
//! [`encode`] cannot check this in general, but it can catch the deterministic
//! half: in debug builds it round-trips its own output (encode → decode →
//! re-encode) and asserts the bytes reproduce. A failure there means the
//! consumer's `G` serde is non-deterministic; it is not a bug in this crate.
//!
//! # The JSON recursion cap is the effective depth limit
//!
//! `PropExpr` serializes externally tagged, so each `Compose` or `Tensor` level
//! costs **two** JSON containers (the variant object plus its payload array).
//! `serde_json`'s parser refuses to nest more than 128 containers, which puts
//! the real ceiling at roughly **64 nesting levels** — comfortably below
//! [`MAX_TERM_DEPTH`], the structural limit catgraph's own interpreters enforce.
//!
//! That is a safety property (a pathological term cannot drive the parser into a
//! stack overflow) *and* a round-trip cap, and the second half is the one that
//! bites: a term deep enough to encode but too deep to parse would be written
//! and then be permanently unreadable. [`encode`] therefore parses its own
//! output back before returning, in every build profile. What this store writes,
//! it can read.
//!
//! # Revalidation: deserialization is not validation
//!
//! `ColoredExpr`'s own documentation is explicit that its `Deserialize` does not
//! re-run `check`, and prescribes rebuilding through `ColoredExpr::new` when
//! ingesting untrusted documents. [`TermRecord::revalidate`] is that discipline,
//! in four steps whose **order is load-bearing** — see its documentation.

use catgraph_applied::prop::colored::ColoredExpr;
use catgraph_applied::prop::presentation::Presentation;
use catgraph_applied::prop::presentation::content::is_arity_well_formed;
use catgraph_applied::prop::presentation::smc_nf::{Atom, nf};
use catgraph_applied::prop::{PropExpr, PropSignature};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::addr::TermAddr;
use crate::error::{Result, RevalidationStage, StoreError};

/// The version tag stored in every term's `codec` column.
///
/// It versions the *whole* encoding — the JSON shape, the signature encoding,
/// the digest algorithm, and the way `nf_class` is derived. A record whose codec
/// this build does not recognise is refused on load rather than interpreted
/// under the current rules, which is the difference between a loud upgrade and a
/// silent misreading.
///
/// `cgj2` (pre-release, no `cgj1` data ever left a test process): `nf_class`
/// hashes the normalized diagram's **layer structure directly** rather than the
/// re-associated expression — see [`nf_class`] for why the older derivation
/// aborted the process on wide terms.
pub const TERM_CODEC: &str = "cgj2";

/// The maximum structural nesting depth a stored term may have.
///
/// Mirrors `catgraph-syntax`'s `MAX_TERM_DEPTH`, which is the limit its
/// interpreters enforce and the limit its parser produces. This crate does not
/// depend on `catgraph-syntax` at run time — it persists terms rather than
/// interpreting them — so the constant and the walk below are reimplemented
/// here. **They must track upstream**: a store that accepted deeper terms than
/// the interpreters do would be handing its consumers documents they cannot
/// use. The mirror is *enforced*, not hoped for: `catgraph-syntax` is a
/// dev-dependency, and a test asserts this constant and [`term_depth`] agree
/// with the upstream originals — an upstream change fails CI here rather than
/// silently diverging.
///
/// In practice the JSON parser's own limit bites first (see the [module
/// documentation](self)); this is the structural backstop, not the effective
/// ceiling.
pub const MAX_TERM_DEPTH: usize = 256;

/// The structural nesting depth of `expr`: the longest root-to-leaf path through
/// `Compose`/`Tensor` nodes, counting a leaf as depth `1`.
///
/// Computed **iteratively**, with an explicit heap stack. That is the whole
/// point: measuring an arbitrarily deep term must never itself overflow on the
/// way to reporting that the term is too deep.
#[must_use]
pub fn term_depth<G: PropSignature>(expr: &PropExpr<G>) -> usize {
    let mut stack = vec![(expr, 1usize)];
    let mut max_depth = 0usize;
    while let Some((node, depth)) = stack.pop() {
        max_depth = max_depth.max(depth);
        match node {
            PropExpr::Compose(f, g) | PropExpr::Tensor(f, g) => {
                stack.push((f, depth + 1));
                stack.push((g, depth + 1));
            }
            PropExpr::Identity(_) | PropExpr::Braid(_, _) | PropExpr::Generator(_) => {}
        }
    }
    max_depth
}

/// How many `Generator` leaves `expr` has.
///
/// Iterative for the same reason as [`term_depth`].
#[must_use]
pub fn generator_count<G: PropSignature>(expr: &PropExpr<G>) -> usize {
    let mut stack = vec![expr];
    let mut count = 0usize;
    while let Some(node) = stack.pop() {
        match node {
            PropExpr::Compose(f, g) | PropExpr::Tensor(f, g) => {
                stack.push(f);
                stack.push(g);
            }
            PropExpr::Generator(_) => count += 1,
            PropExpr::Identity(_) | PropExpr::Braid(_, _) => {}
        }
    }
    count
}

/// The boundary signature of a term: its source and target words, canonically
/// encoded.
///
/// This is what the fourth revalidation step compares against. Re-running the
/// type check derives a target word from the expression itself; if it disagrees
/// with the stored signature, the document is lying about what morphism it is.
///
/// # Errors
///
/// Fails with [`StoreError::Codec`] if the color type does not serialize.
pub fn signature<G>(term: &ColoredExpr<G>) -> Result<String>
where
    G: PropSignature + Serialize,
    G::Color: Serialize,
{
    Ok(serde_json::to_string(&(
        term.source_word(),
        term.target_word(),
    ))?)
}

/// The semantic bucket a term belongs to.
///
/// # What this is and is not
///
/// It is a **sound bucket, not a complete one**. `nf` applies only rewrites that
/// are valid in the free symmetric monoidal category, so two terms with equal
/// `nf_class` really are equal there — the bucket never conflates distinct
/// morphisms. Canonicality is open in general, so two *equal* morphisms may
/// still land in different buckets; the cost of that is a duplicate, never a
/// false identification. Complete semantic deduplication is an in-process
/// concern over a working set, not something to persist.
///
/// # Why the boundary words are folded in
///
/// The class covers the source and target words as well as the normalized
/// diagram. Without them the soundness claim would not hold for colored
/// terms: two expressions can share an underlying `PropExpr` while typing at
/// different words, and those are different morphisms. Folding the words in
/// keeps "equal class implies equal morphism" true rather than nearly true.
///
/// # Why the diagram is hashed directly
///
/// The hash input is the normalized diagram's **layer structure**, walked
/// flatly — never the diagram folded back into an expression. The fold
/// (`from_string_diagram`) right-associates the normal form into a chain whose
/// nesting equals the generator count, and both `serde_json`'s recursive
/// `Serialize` and the chain's own recursive drop glue then abort the process
/// on wide terms — a depth screen bounds nesting, not width, so a term of a few
/// thousand *parallel* generators sails through every guard and dies here. The
/// layer walk nests a fixed handful of JSON containers no matter how wide the
/// diagram is, and it keys the same equivalence: the fold is a deterministic
/// function of the diagram, so equal diagrams and equal folds coincide.
///
/// The atom tags (`I`/`B`/`G`) are part of the encoding [`TERM_CODEC`]
/// versions.
///
/// # Errors
///
/// Returns [`StoreError::Revalidation`] at the arity stage if the term's arities
/// are not well-formed — `nf` sizes collections from those arities and would
/// abort rather than answer. Returns [`StoreError::Codec`] if the term does not
/// serialize.
pub fn nf_class<G>(term: &ColoredExpr<G>) -> Result<String>
where
    G: PropSignature + Serialize,
    G::Color: Serialize,
{
    // The screen is not optional: `nf` rejects an overflowing arity by aborting,
    // in both build profiles.
    if !is_arity_well_formed(term.expr()) {
        return Err(arity_failure());
    }

    /// One atom of the canonical hash input. Borrowing and shallow: a
    /// diagram of any width serializes at constant nesting depth.
    #[derive(Serialize)]
    enum AtomRepr<'a, S> {
        I(usize),
        B(usize, usize),
        G(&'a S),
    }

    let diagram = nf(term.expr());
    let layers: Vec<Vec<AtomRepr<'_, G>>> = diagram
        .layers
        .iter()
        .map(|layer| {
            layer
                .atoms
                .iter()
                .map(|atom| match atom {
                    Atom::Identity(width) => AtomRepr::I(*width),
                    Atom::Braid(m, n) => AtomRepr::B(*m, *n),
                    Atom::Generator(generator) => AtomRepr::G(generator),
                })
                .collect()
        })
        .collect();
    let canonical = serde_json::to_string(&(term.source_word(), term.target_word(), &layers))?;
    Ok(digest(canonical.as_bytes()))
}

/// A term as it is stored: the canonical encoding plus every column derived from
/// it.
///
/// The counts are `i64` rather than `usize` on purpose — this type mirrors the
/// row, so nothing is silently narrowed or widened between here and the
/// database. The conversion happens once, in [`encode`], where it is checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermRecord {
    // Crate-visible so the row conversion can MOVE the encoding string rather
    // than clone it per write; the public surface stays the getters.
    pub(crate) addr: TermAddr,
    pub(crate) codec: String,
    pub(crate) term_json: String,
    pub(crate) signature: String,
    pub(crate) source_arity: i64,
    pub(crate) target_arity: i64,
    pub(crate) depth: i64,
    pub(crate) generator_count: i64,
    pub(crate) nf_class: String,
}

impl TermRecord {
    /// Rebuild a record from columns read back out of the database.
    ///
    /// Nothing here is trusted: [`Self::revalidate`] re-derives every one of
    /// these values from `term_json` and compares.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_columns(
        addr: TermAddr,
        codec: String,
        term_json: String,
        signature: String,
        source_arity: i64,
        target_arity: i64,
        depth: i64,
        generator_count: i64,
        nf_class: String,
    ) -> Self {
        Self {
            addr,
            codec,
            term_json,
            signature,
            source_arity,
            target_arity,
            depth,
            generator_count,
            nf_class,
        }
    }

    /// The term's content address, which is also its record id.
    #[must_use]
    pub fn addr(&self) -> &TermAddr {
        &self.addr
    }

    /// The encoding version this record was written under.
    #[must_use]
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// The canonical JSON encoding — the only column that is not derived.
    #[must_use]
    pub fn term_json(&self) -> &str {
        &self.term_json
    }

    /// The canonically encoded boundary words.
    #[must_use]
    pub fn signature(&self) -> &str {
        &self.signature
    }

    /// The length of the source word.
    #[must_use]
    pub fn source_arity(&self) -> i64 {
        self.source_arity
    }

    /// The length of the target word.
    #[must_use]
    pub fn target_arity(&self) -> i64 {
        self.target_arity
    }

    /// The structural nesting depth.
    #[must_use]
    pub fn depth(&self) -> i64 {
        self.depth
    }

    /// How many generator leaves the term has.
    #[must_use]
    pub fn generator_count(&self) -> i64 {
        self.generator_count
    }

    /// The semantic bucket. See [`nf_class`].
    #[must_use]
    pub fn nf_class(&self) -> &str {
        &self.nf_class
    }

    /// Re-derive everything from `term_json` and hand back a term that has
    /// actually been checked.
    ///
    /// # The order, and why it is fixed
    ///
    /// **Before anything interprets the document, the content address is
    /// verified**: `term_json` is re-digested and compared against the record
    /// id it was filed under. It needs only the raw bytes, it is the cheapest
    /// check here, and it fully decides byte-tampering — running it first means
    /// a swapped-in encoding is rejected at hash-of-bytes cost instead of after
    /// the whole pipeline below has run on hostile input.
    ///
    /// Then the four interpretation steps:
    ///
    /// 1. **Parse.** `serde_json` refuses over-deep nesting before anything else
    ///    gets a chance to look at the document, so its own recursion limit is
    ///    the first line of defence and reports as [`StoreError::Codec`].
    /// 2. **Depth.** The structural walk, against [`MAX_TERM_DEPTH`] and against
    ///    the stored column.
    /// 3. **Arity.** The well-formedness screen. **Skipping this makes the next
    ///    step abort rather than return an error** — the content pass sizes
    ///    collections from arities that may have saturated at `usize::MAX`.
    /// 4. **Check.** Rebuild through `ColoredExpr::new`, which re-runs the type
    ///    check and re-derives the target word, then compare the derived
    ///    signature against the stored one — and the remaining derived columns
    ///    with it.
    ///
    /// # Cost
    ///
    /// Every derived column is re-derived, and that includes `nf_class` — which
    /// means the normal-form pipeline runs on **every load**, not just on write.
    /// It is the most expensive thing here by a wide margin, and it is kept
    /// because leaving it out would make the bucket's guarantee conditional on
    /// the database: a `nf_class` corrupted into agreement with another row's
    /// would make a bucket query claim two unequal terms are equal, which is the
    /// one thing a *sound* bucket promises never to do. A read path that needs
    /// to be cheaper than this needs a measurement first, not a shortcut.
    ///
    /// # Errors
    ///
    /// - [`StoreError::TypeMismatch`] if the record was written under a codec
    ///   version this build does not know.
    /// - [`StoreError::Codec`] if the encoding does not parse.
    /// - [`StoreError::Revalidation`] naming the stage that rejected it.
    /// - [`StoreError::Corrupt`] if the record is filed under an id that is not
    ///   its own content address.
    pub fn revalidate<G>(&self) -> Result<ColoredExpr<G>>
    where
        G: PropSignature + Serialize + DeserializeOwned,
        G::Color: Serialize + DeserializeOwned,
    {
        if self.codec != TERM_CODEC {
            return Err(StoreError::TypeMismatch {
                field: "codec".to_owned(),
                expected: TERM_CODEC.to_owned(),
                actual: self.codec.clone(),
            });
        }

        // The record id is the content address; verify it before interpreting a
        // single byte. Cheapest check, fully decides tampering.
        let addr = address_of(&self.term_json)?;
        if addr != self.addr {
            return Err(StoreError::Corrupt {
                context: format!("{}:{}", crate::schema::TERM_TABLE, self.addr),
                detail: format!("content address of the stored encoding is `{addr}`"),
            });
        }

        // Step 1 — parse. The parser's container limit fires here.
        let parsed: ColoredExpr<G> = serde_json::from_str(&self.term_json)?;

        // Step 2 — depth.
        let depth = term_depth(parsed.expr());
        if depth > MAX_TERM_DEPTH {
            return Err(StoreError::Revalidation {
                stage: RevalidationStage::Depth,
                detail: format!("nesting depth {depth} exceeds limit {MAX_TERM_DEPTH}"),
            });
        }
        expect_column(RevalidationStage::Depth, "depth", self.depth, depth)?;

        // Step 3 — arity. Nothing below may run before this passes.
        if !is_arity_well_formed(parsed.expr()) {
            return Err(arity_failure());
        }

        // Step 4 — check. Rebuilding is what re-runs the type check: the value
        // serde produced carries a target word nobody verified.
        let (source_word, stored_target, expr) = parsed.into_inner();
        let rebuilt =
            ColoredExpr::new(source_word, expr).map_err(|e| StoreError::Revalidation {
                stage: RevalidationStage::Check,
                detail: e.to_string(),
            })?;
        if rebuilt.target_word() != stored_target.as_slice() {
            return Err(StoreError::Revalidation {
                stage: RevalidationStage::Check,
                detail: "the encoded target word is not the one the expression derives".to_owned(),
            });
        }

        expect_column(
            RevalidationStage::Arity,
            "source_arity",
            self.source_arity,
            rebuilt.source_word().len(),
        )?;
        expect_column(
            RevalidationStage::Arity,
            "target_arity",
            self.target_arity,
            rebuilt.target_word().len(),
        )?;
        expect_column(
            RevalidationStage::Check,
            "generator_count",
            self.generator_count,
            generator_count(rebuilt.expr()),
        )?;

        let derived = signature(&rebuilt)?;
        if derived != self.signature {
            return Err(StoreError::Revalidation {
                stage: RevalidationStage::Check,
                detail: format!(
                    "stored signature `{}` disagrees with the derived `{derived}`",
                    self.signature
                ),
            });
        }

        let derived = nf_class(&rebuilt)?;
        if derived != self.nf_class {
            return Err(StoreError::Revalidation {
                stage: RevalidationStage::Check,
                detail: format!(
                    "stored nf_class `{}` disagrees with the derived `{derived}`",
                    self.nf_class
                ),
            });
        }

        Ok(rebuilt)
    }
}

/// Encode a term into the record that will be stored.
///
/// # What this refuses, and why
///
/// - A term deeper than [`MAX_TERM_DEPTH`], because the loader would refuse it.
/// - A term whose arities are not well-formed, because the derived columns
///   cannot be computed from one.
/// - A term whose own encoding does not parse back. This is the JSON recursion
///   cap made loud: without the check, a term between the parser's ceiling and
///   [`MAX_TERM_DEPTH`] would be written successfully and then be unreadable
///   forever. Parsing back costs one pass over the encoding, which a database
///   write dwarfs.
///
/// In debug builds the parsed value is re-encoded and the bytes compared, which
/// is the [`G` determinism](self#the-g-contract-serialize-must-be-deterministic)
/// assertion.
///
/// # Errors
///
/// [`StoreError::Revalidation`] for a term the loader would reject,
/// [`StoreError::Codec`] if it does not serialize or does not parse back, and
/// [`StoreError::TypeMismatch`] in the theoretical case of a count too large for
/// the database's integer column.
///
/// # Panics
///
/// Never in a release build. In a debug build, panics if re-encoding a
/// round-tripped term does not reproduce the original bytes — see the module
/// documentation; that indicates non-deterministic serde on `G`, not a fault
/// here.
pub fn encode<G>(term: &ColoredExpr<G>) -> Result<TermRecord>
where
    G: PropSignature + Serialize + DeserializeOwned,
    G::Color: Serialize + DeserializeOwned,
{
    let depth = term_depth(term.expr());
    if depth > MAX_TERM_DEPTH {
        return Err(StoreError::Revalidation {
            stage: RevalidationStage::Depth,
            detail: format!("nesting depth {depth} exceeds limit {MAX_TERM_DEPTH}"),
        });
    }
    if !is_arity_well_formed(term.expr()) {
        return Err(arity_failure());
    }

    let term_json = serde_json::to_string(term)?;

    // Write only what can be read back. The parser's container limit is stricter
    // than the structural one above, and this is where that difference surfaces.
    let parsed: ColoredExpr<G> = serde_json::from_str(&term_json)?;
    assert_deterministic(&parsed, &term_json)?;

    Ok(TermRecord {
        addr: address_of(&term_json)?,
        codec: TERM_CODEC.to_owned(),
        signature: signature(term)?,
        source_arity: to_column("source_arity", term.source_word().len())?,
        target_arity: to_column("target_arity", term.target_word().len())?,
        depth: to_column("depth", depth)?,
        generator_count: to_column("generator_count", generator_count(term.expr()))?,
        nf_class: nf_class(term)?,
        term_json,
    })
}

/// Rebuild a presentation from stored equation pairs, re-checking each one.
///
/// `Presentation`'s `Deserialize` bypasses `add_equation`'s check exactly as
/// `ColoredExpr`'s bypasses `check`, so a presentation that came off disk has
/// not been validated. This rebuilds it the only way that does validate: an
/// empty presentation plus one `add_equation` per pair.
///
/// Both sides of every equation go through the depth and arity screens first,
/// for the same reason terms do — the equality machinery downstream recurses
/// over these expressions and sizes collections from their arities.
///
/// # What does not survive the rebuild
///
/// A presentation also carries a rewrite-depth bound and an engine selector.
/// Neither is stored, so both come back at the constructor default. Both have
/// public accessors (`rewrite_depth`, `engine`), so a caller holding the
/// original can carry them: `set_engine` restores the selector, and a
/// non-default bound needs `Presentation::with_depth` in place of this
/// function.
///
/// # Errors
///
/// [`StoreError::Revalidation`] naming the stage that rejected an equation.
pub fn presentation_from_equations<G: PropSignature>(
    equations: impl IntoIterator<Item = (PropExpr<G>, PropExpr<G>)>,
) -> Result<Presentation<G>> {
    let mut presentation = Presentation::new();
    for (lhs, rhs) in equations {
        for side in [&lhs, &rhs] {
            let depth = term_depth(side);
            if depth > MAX_TERM_DEPTH {
                return Err(StoreError::Revalidation {
                    stage: RevalidationStage::Depth,
                    detail: format!("nesting depth {depth} exceeds limit {MAX_TERM_DEPTH}"),
                });
            }
            if !is_arity_well_formed(side) {
                return Err(arity_failure());
            }
        }
        presentation
            .add_equation(lhs, rhs)
            .map_err(|e| StoreError::Revalidation {
                stage: RevalidationStage::Check,
                detail: e.to_string(),
            })?;
    }
    Ok(presentation)
}

/// The `G`-determinism assertion: re-encoding a round-tripped term must
/// reproduce the original bytes.
///
/// Debug builds only. It doubles the encoding cost, and a non-deterministic `G`
/// announces itself on the very first write rather than gradually — so paying
/// for it in a release build would buy nothing a development run has not already
/// found.
///
/// # Panics
///
/// Panics on a mismatch. That is deliberate: a `G` whose `Serialize` is not
/// deterministic makes every content address in the store meaningless, and there
/// is no partial recovery from it worth offering a caller.
#[cfg(debug_assertions)]
fn assert_deterministic<G>(parsed: &ColoredExpr<G>, term_json: &str) -> Result<()>
where
    G: PropSignature + Serialize,
    G::Color: Serialize,
{
    assert_eq!(
        serde_json::to_string(parsed)?,
        term_json,
        "the generator type's `Serialize` is not deterministic: re-encoding a \
         round-tripped term produced different bytes. Content addresses are \
         unstable under such a `G` — the usual cause is a `HashMap` or `HashSet` \
         in the generator's serde representation"
    );
    Ok(())
}

/// The release-build counterpart of the determinism assertion: nothing.
///
/// The *parse* it accompanies still runs in release — that is the readability
/// guarantee, not a step toward this assertion.
#[cfg(not(debug_assertions))]
fn assert_deterministic<G>(_parsed: &ColoredExpr<G>, _term_json: &str) -> Result<()>
where
    G: PropSignature + Serialize,
    G::Color: Serialize,
{
    Ok(())
}

/// The content address of an encoding.
fn address_of(term_json: &str) -> Result<TermAddr> {
    let digest = digest(term_json.as_bytes());
    TermAddr::from_digest(&digest).ok_or_else(|| StoreError::Corrupt {
        context: crate::schema::TERM_TABLE.to_owned(),
        detail: format!("`{digest}` is not a well-formed digest"),
    })
}

/// A 256-bit BLAKE3 digest, lowercase hex.
fn digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// The shared arity-stage failure.
///
/// Deliberately one message: the arity screen is a single predicate upstream,
/// and inventing a finer taxonomy here would be describing detail this crate
/// does not actually have.
fn arity_failure() -> StoreError {
    StoreError::Revalidation {
        stage: RevalidationStage::Arity,
        detail: "the term's arities are not well-formed".to_owned(),
    }
}

/// Narrow a derived count into the database's integer lane.
///
/// Unreachable in practice — these are lengths of in-memory collections — but
/// narrowing silently is exactly the failure mode that makes terms opaque
/// strings in the first place, so it is checked rather than cast.
fn to_column(field: &str, value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| StoreError::TypeMismatch {
        field: field.to_owned(),
        expected: "int".to_owned(),
        actual: format!("{value} (too large for a 64-bit signed integer)"),
    })
}

/// Compare a stored derived column against the value re-derived on load.
fn expect_column(stage: RevalidationStage, field: &str, stored: i64, derived: usize) -> Result<()> {
    let derived = to_column(field, derived)?;
    if stored == derived {
        Ok(())
    } else {
        Err(StoreError::Revalidation {
            stage,
            detail: format!("stored `{field}` is {stored}, but the encoding derives {derived}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use catgraph_applied::prop::Free;
    use serde::Deserialize;

    use super::*;

    /// A minimal well-behaved signature: four generators, one color, derived
    /// serde with no map or set anywhere in it.
    #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
    enum Gen {
        /// `Δ : 1 → 2`
        Copy,
        /// `! : 1 → 0`
        Discard,
        /// `μ : 2 → 1`
        Add,
        /// `η : 0 → 1`
        Zero,
    }

    /// One variant byte: `Copy` 0, `Discard` 1, `Add` 2, `Zero` 3.
    impl catgraph::CanonicalEncode for Gen {
        fn encode_canonical(&self, out: &mut Vec<u8>) {
            out.push(match self {
                Self::Copy => 0,
                Self::Discard => 1,
                Self::Add => 2,
                Self::Zero => 3,
            });
        }
    }

    impl PropSignature for Gen {
        type Color = ();

        fn source_word(&self) -> Cow<'_, [()]> {
            Cow::Owned(match self {
                Self::Copy | Self::Discard => vec![()],
                Self::Add => vec![(), ()],
                Self::Zero => vec![],
            })
        }

        fn target_word(&self) -> Cow<'_, [()]> {
            Cow::Owned(match self {
                Self::Copy => vec![(), ()],
                Self::Discard => vec![],
                Self::Add | Self::Zero => vec![()],
            })
        }
    }

    fn copy_then_add() -> ColoredExpr<Gen> {
        let expr = Free::compose(Free::generator(Gen::Copy), Free::<Gen>::generator(Gen::Add))
            .expect("Δ ; μ composes: 1 → 2 → 1");
        ColoredExpr::new(vec![()], expr).expect("Δ ; μ type-checks at one wire")
    }

    /// A left-nested `id(1) ; … ; id(1)` chain of structural depth `d`, built
    /// iteratively so construction does not recurse.
    fn deep_chain(d: usize) -> PropExpr<Gen> {
        let mut expr = Free::<Gen>::identity(1);
        for _ in 1..d {
            expr = Free::compose(expr, Free::<Gen>::identity(1)).expect("identities compose");
        }
        expr
    }

    #[test]
    fn depth_of_a_leaf_is_one_and_nesting_counts_the_longest_path() {
        assert_eq!(term_depth(&Free::<Gen>::identity(1)), 1);
        assert_eq!(term_depth(&deep_chain(5)), 5);
        let wide = Free::tensor(deep_chain(4), Free::<Gen>::identity(1));
        assert_eq!(term_depth(&wide), 5);
    }

    #[test]
    fn generators_are_counted_and_identities_are_not() {
        assert_eq!(generator_count(&Free::<Gen>::identity(3)), 0);
        assert_eq!(generator_count(&deep_chain(9)), 0);
        assert_eq!(generator_count(copy_then_add().expr()), 2);
    }

    /// The property the whole store rests on: the same term encodes to the same
    /// bytes, so it gets the same address, so it is one record.
    #[test]
    fn encoding_is_deterministic() {
        let first = encode(&copy_then_add()).expect("a well-formed term encodes");
        let second = encode(&copy_then_add()).expect("a well-formed term encodes");
        assert_eq!(first.term_json(), second.term_json());
        assert_eq!(first.addr(), second.addr());
        assert_eq!(first.signature(), second.signature());
        assert_eq!(first.nf_class(), second.nf_class());
    }

    #[test]
    fn derived_columns_describe_the_term() {
        let record = encode(&copy_then_add()).expect("a well-formed term encodes");
        assert_eq!(record.codec(), TERM_CODEC);
        assert_eq!(record.source_arity(), 1);
        assert_eq!(record.target_arity(), 1);
        assert_eq!(record.depth(), 2);
        assert_eq!(record.generator_count(), 2);
    }

    #[test]
    fn a_record_round_trips_through_revalidation() {
        let term = copy_then_add();
        let record = encode(&term).expect("a well-formed term encodes");
        let loaded: ColoredExpr<Gen> = record.revalidate().expect("its own encoding revalidates");
        assert_eq!(loaded, term);
    }

    /// Distinct syntax is distinct identity — a population store wants that —
    /// while the semantic bucket sees through the interchange law.
    #[test]
    fn distinct_writings_of_one_morphism_share_a_bucket_but_not_an_address() {
        let left = ColoredExpr::new(
            vec![(), ()],
            Free::tensor(
                Free::generator(Gen::Discard),
                Free::<Gen>::generator(Gen::Discard),
            ),
        )
        .expect("! ⊗ ! type-checks at two wires");
        let right = ColoredExpr::new(
            vec![(), ()],
            Free::compose(
                Free::tensor(Free::generator(Gen::Discard), Free::<Gen>::identity(1)),
                Free::generator(Gen::Discard),
            )
            .expect("(! ⊗ id) ; ! composes"),
        )
        .expect("(! ⊗ id) ; ! type-checks at two wires");

        let left = encode(&left).expect("a well-formed term encodes");
        let right = encode(&right).expect("a well-formed term encodes");
        assert_ne!(left.addr(), right.addr());
        assert_eq!(left.nf_class(), right.nf_class());
    }

    /// Terms typing at different words are different morphisms even when the
    /// underlying expression is the same, so they must not share a bucket.
    #[test]
    fn boundary_words_separate_buckets() {
        let one = ColoredExpr::new(vec![()], Free::<Gen>::identity(1)).expect("id₁ checks");
        let two = ColoredExpr::new(vec![(), ()], Free::<Gen>::identity(2)).expect("id₂ checks");
        assert_ne!(
            nf_class(&one).expect("id₁ has a class"),
            nf_class(&two).expect("id₂ has a class")
        );
    }

    #[test]
    fn a_term_at_the_depth_limit_encodes_and_one_past_it_does_not() {
        // The parser's container limit is the tighter of the two, so the
        // structural limit is exercised well below `MAX_TERM_DEPTH`.
        let at_limit = ColoredExpr::new(vec![()], deep_chain(40)).expect("identities check");
        assert!(encode(&at_limit).is_ok());

        let over = ColoredExpr::new(vec![()], deep_chain(MAX_TERM_DEPTH + 1))
            .expect("identities check at any depth");
        let err = encode(&over).expect_err("a term past the structural limit is refused");
        assert!(
            matches!(
                err,
                StoreError::Revalidation {
                    stage: RevalidationStage::Depth,
                    ..
                }
            ),
            "{err}"
        );
    }

    /// The round-trip cap made loud. A term between the parser's ceiling and
    /// `MAX_TERM_DEPTH` is refused at encode rather than written and then found
    /// to be unreadable.
    #[test]
    fn a_term_too_deep_for_the_parser_is_refused_at_encode() {
        let deep = ColoredExpr::new(vec![()], deep_chain(100)).expect("identities check");
        let err = encode(&deep).expect_err("the parser cannot read this back");
        assert!(matches!(err, StoreError::Codec(_)), "{err}");
    }

    #[test]
    fn presentations_rebuild_by_re_checking_every_equation() {
        let equations = vec![(Free::<Gen>::identity(2), Free::<Gen>::braid(1, 1))];
        let presentation =
            presentation_from_equations(equations).expect("a parallel pair is acceptable");
        assert_eq!(presentation.equations().len(), 1);
    }

    #[test]
    fn a_presentation_equation_that_does_not_check_is_rejected() {
        let equations = vec![(Free::<Gen>::identity(1), Free::<Gen>::identity(2))];
        let err = presentation_from_equations(equations)
            .expect_err("id₁ and id₂ are not parallel morphisms");
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

    /// A balanced tensor tree of `width` parallel `Discard` generators:
    /// structural depth ~log₂(width), but as wide as asked.
    fn wide_discards(width: usize) -> ColoredExpr<Gen> {
        assert!(width >= 1);
        let mut nodes: Vec<PropExpr<Gen>> =
            (0..width).map(|_| Free::generator(Gen::Discard)).collect();
        while nodes.len() > 1 {
            nodes = nodes
                .chunks(2)
                .map(|pair| match pair {
                    [only] => only.clone(),
                    [left, right] => Free::tensor(left.clone(), right.clone()),
                    _ => unreachable!("chunks(2) yields one or two"),
                })
                .collect();
        }
        let expr = nodes.pop().expect("invariant: width >= 1 leaves one root");
        ColoredExpr::new(vec![(); width], expr).expect("parallel discards type-check")
    }

    /// Regression for the wide-term process abort: `nf_class` used to fold the
    /// normalized diagram back into a right-associated expression whose nesting
    /// equals the generator count, and serializing (or even dropping) that
    /// chain blew the stack — SIGABRT, not an error. The depth screens bound
    /// nesting, never width, so this term passes every guard. Run on a
    /// deliberately small stack so the regression cannot hide behind a roomy
    /// main thread.
    #[test]
    fn a_wide_term_encodes_and_revalidates_without_exhausting_the_stack() {
        std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(|| {
                let wide = wide_discards(2048);
                let record = encode(&wide).expect("a wide, shallow term encodes");
                let loaded: ColoredExpr<Gen> = record.revalidate().expect("and revalidates");
                assert_eq!(loaded, wide);
            })
            .expect("invariant: spawning a test thread succeeds")
            .join()
            .expect("the wide-term thread must not abort or panic");
    }

    /// The depth limit and walk are mirrors of `catgraph-syntax`'s originals,
    /// and this is the enforcement: an upstream change fails here instead of
    /// silently diverging. The walk is compared on shapes that exercise every
    /// arm — leaf, deep chain, wide tensor, braid.
    #[test]
    fn depth_limit_and_walk_track_catgraph_syntax() {
        assert_eq!(MAX_TERM_DEPTH, catgraph_syntax::depth::MAX_TERM_DEPTH);
        let samples: Vec<PropExpr<Gen>> = vec![
            Free::identity(1),
            Free::braid(2, 3),
            deep_chain(37),
            wide_discards(64).into_inner().2,
            Free::tensor(deep_chain(5), Free::generator(Gen::Zero)),
        ];
        for expr in &samples {
            assert_eq!(
                term_depth(expr),
                catgraph_syntax::depth::term_depth(expr),
                "depth walks disagree on {expr:?}"
            );
        }
    }

    #[test]
    fn an_over_deep_presentation_side_is_rejected_before_it_is_checked() {
        let equations = vec![(deep_chain(MAX_TERM_DEPTH + 1), Free::<Gen>::identity(1))];
        let err =
            presentation_from_equations(equations).expect_err("the side is past the depth limit");
        assert!(
            matches!(
                err,
                StoreError::Revalidation {
                    stage: RevalidationStage::Depth,
                    ..
                }
            ),
            "{err}"
        );
    }
}
