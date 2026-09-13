//! Cospan encoding, canonical keys, and the bounds-checking discipline.
//!
//! # Legs are arrays, not edges
//!
//! A cospan's two legs are *ordered functions* from the boundary into the apex.
//! Stored as arrays, that ordering is structural — position `i` of `left` is
//! where boundary node `i` lands, and nothing outside the array can change it.
//! Stored as graph edges it would not be: every read would need an index column
//! and an `ORDER BY` to recover the order, and an edge lost anywhere in that
//! pipeline would silently change which morphism the record describes rather
//! than failing. So the legs are flattened into the row.
//!
//! # Two identities, and the difference between them
//!
//! - The **record id** is a content address of the *presentation*: the two leg
//!   arrays and the labelled apex, exactly as built. Two spellings of one
//!   morphism get two addresses.
//! - The **canonical key** ([`canon_key`]) identifies the *morphism*. Two
//!   parallel cospans are equal as morphisms exactly when some bijection of
//!   their apexes commutes with both legs, and catgraph's `CospanCanon` decides
//!   that. The key is a digest of the same equivalence data, so equal keys mean
//!   equal morphisms and vice versa — **a complete invariant**, unlike the
//!   term tier's `nf_class`, which is only sound. That is why the key column is
//!   `UNIQUE` where `nf_class` is not.
//!
//! # Why the store re-derives the canonical form instead of using `CospanCanon`
//!
//! `CospanCanon` is `Eq + Hash` but carries no serializable form: its `classes`
//! field is private, and standard-library hash output is not stable across
//! processes, so it cannot be persisted. It does not need to be. The equivalence
//! data is *fully recomputable* from the public accessors
//! (`left_to_middle`/`right_to_middle`/`middle`), and this module recomputes it,
//! encodes it deterministically, and hashes that. The re-derivation is guarded
//! by a proptest asserting the equivalence in **both** directions against
//! `Cospan::canonical_form` — see `tests/cospan_canon.rs`.
//!
//! Two facts make the re-encoding faithful:
//!
//! - Sorting canonicalises a *multiset*, and any deterministic total order does
//!   that job. This module sorts on the **encoded** class tuples rather than on
//!   `Lambda`'s own `Ord`, which is a different order — but sorting a multiset
//!   yields one sequence whichever order is used, so equal equivalence data
//!   still yields equal bytes.
//! - The reverse direction needs [`LabelCodec::encode`](crate::LabelCodec) to be
//!   *injective*, which its round-trip contract gives: `decode(encode(x))` is
//!   `Some(x)`, so two labels cannot share an encoding. Without that, two
//!   distinct morphisms could share a key.
//!
//! # The store bounds-checks legs itself
//!
//! `Cospan::new` checks both legs against the apex in every build profile.
//! `Cospan::new_unchecked` and `Cospan::add_boundary_node_unchecked` check only
//! under `debug_assert!`, so a release build accepts an out-of-bounds leg
//! through either and defers its failure to whatever indexes it later.
//!
//! The class derivation covers that. `canonical_classes` indexes the apex by
//! every entry of both legs, so a leg entry at or past the apex length is
//! [`StoreError::Corrupt`] — and both [`encode`] and
//! [`CospanRecord::revalidate`] derive the classes: the write side for a value
//! that reached this store through an unchecked constructor, the load side for
//! a column that arrives off disk, where rebuilding through `Cospan::new`
//! checks the same thing again.

use catgraph::cospan::Cospan;
use serde::Serialize;

use crate::addr::{self, CospanAddr};
use crate::codec::LabelCodec;
use crate::error::{Result, StoreError};
use crate::schema::COSPAN_TABLE;

/// The version tag stored in every cospan's `codec` column.
///
/// It versions the *whole* encoding — the structural encoding the record id
/// addresses, the canonical encoding [`canon_key`] digests, and the label
/// codec's own output. A record whose codec this build does not recognise is
/// refused on load rather than interpreted under the current rules.
pub const COSPAN_CODEC: &str = "cgc1";

/// One apex vertex's signature: its label, the boundary indices of the domain
/// leg that land on it, and those of the codomain leg.
type Class = (String, Vec<usize>, Vec<usize>);

/// The equivalence data `CospanCanon` holds, recomputed over already-encoded
/// labels: `(domain size, codomain size, sorted apex signatures)`.
///
/// Mirrors `Cospan::canonical_form` step for step. Sorting is what makes the
/// value invariant under any relabelling of apex vertices.
///
/// Taking *encoded* labels rather than a `Cospan<L>` is deliberate: the write
/// path encodes the apex exactly once and reuses the strings here, and the
/// load path reuses the **stored** strings — which it may, because revalidation
/// has already required each one to be the canonical spelling of the label it
/// decodes to.
///
/// # Errors
///
/// [`StoreError::Corrupt`] for a leg entry at or past `apex.len()`. Every entry
/// of both legs indexes `apex` here, so an out-of-bounds leg is refused here on
/// the write path and on the load path alike.
fn canonical_classes(
    dom_leg: &[usize],
    cod_leg: &[usize],
    apex: &[String],
) -> Result<(usize, usize, Vec<Class>)> {
    let mut classes: Vec<Class> = apex
        .iter()
        .map(|encoded| (encoded.clone(), Vec::new(), Vec::new()))
        .collect();

    // Boundary indices are pushed in ascending order, so each preimage vector is
    // sorted already.
    for (side, leg) in [(Side::Domain, dom_leg), (Side::Codomain, cod_leg)] {
        for (boundary, &apex_index) in leg.iter().enumerate() {
            let class = classes
                .get_mut(apex_index)
                .ok_or_else(|| out_of_bounds(side, boundary, apex_index, apex.len()))?;
            match side {
                Side::Domain => class.1.push(boundary),
                Side::Codomain => class.2.push(boundary),
            }
        }
    }

    classes.sort();
    Ok((dom_leg.len(), cod_leg.len(), classes))
}

/// Encode a cospan's apex once, in order.
fn encoded_apex<L: LabelCodec>(cospan: &Cospan<L>) -> Vec<String> {
    cospan.middle().iter().map(LabelCodec::encode).collect()
}

/// Which leg a bounds failure was found on.
#[derive(Debug, Clone, Copy)]
enum Side {
    Domain,
    Codomain,
}

impl Side {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Domain => "domain",
            Self::Codomain => "codomain",
        }
    }
}

/// The uniform out-of-bounds report.
fn out_of_bounds(side: Side, boundary: usize, apex: usize, apex_len: usize) -> StoreError {
    StoreError::Corrupt {
        context: COSPAN_TABLE.to_owned(),
        detail: format!(
            "{} leg maps boundary node {boundary} to apex index {apex}, but the apex has {apex_len} vertices",
            side.as_str()
        ),
    }
}

/// The canonical key of a cospan: the digest of its apex-isomorphism class.
///
/// Two parallel cospans share a key **exactly** when they are equal as
/// morphisms — see the [module documentation](self) for why this is a complete
/// invariant and how the equivalence is guarded.
///
/// # Errors
///
/// [`StoreError::Corrupt`] if a leg index is out of bounds for the apex;
/// [`StoreError::Codec`] if the encoding fails.
pub fn canon_key<L: LabelCodec>(cospan: &Cospan<L>) -> Result<String> {
    let apex = encoded_apex(cospan);
    let (dom_len, cod_len, classes) =
        canonical_classes(cospan.left_to_middle(), cospan.right_to_middle(), &apex)?;
    Ok(canon_key_of(dom_len, cod_len, &classes)?.0)
}

/// The canonical key and the scalar count, from already-computed classes.
///
/// Returns both because the scalar count falls out of the same walk: an apex
/// vertex hit by neither leg is a **scalar** (a closed bubble), and cospans keep
/// scalars — `k` bubbles are a different morphism from `k - 1`. The column
/// exists so that fact is queryable without re-deriving the class.
fn canon_key_of(dom_len: usize, cod_len: usize, classes: &[Class]) -> Result<(String, usize)> {
    let canonical = serde_json::to_string(&(dom_len, cod_len, classes))?;
    let scalars = classes
        .iter()
        .filter(|(_, dom, cod)| dom.is_empty() && cod.is_empty())
        .count();
    Ok((addr::digest_key(canonical.as_bytes()), scalars))
}

/// A cospan as it is stored: the presentation, plus every column derived from
/// it.
///
/// The integer columns are `i64` rather than `usize` on purpose — this type
/// mirrors the row, so nothing is silently narrowed or widened between here and
/// the database. The conversion happens once, in [`encode`], where it is
/// checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CospanRecord {
    // Crate-visible so the row conversion can MOVE the payload vectors rather
    // than deep-clone them per write; the public surface stays the getters.
    pub(crate) addr: CospanAddr,
    pub(crate) codec: String,
    pub(crate) dom_leg: Vec<i64>,
    pub(crate) cod_leg: Vec<i64>,
    pub(crate) apex: Vec<String>,
    pub(crate) dom_len: i64,
    pub(crate) cod_len: i64,
    pub(crate) apex_len: i64,
    pub(crate) scalar_count: i64,
    pub(crate) canon_key: String,
}

impl CospanRecord {
    /// Rebuild a record from columns read back out of the database.
    ///
    /// Nothing here is trusted: [`Self::revalidate`] re-derives every derived
    /// value from the presentation columns and compares.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_columns(
        addr: CospanAddr,
        codec: String,
        dom_leg: Vec<i64>,
        cod_leg: Vec<i64>,
        apex: Vec<String>,
        dom_len: i64,
        cod_len: i64,
        apex_len: i64,
        scalar_count: i64,
        canon_key: String,
    ) -> Self {
        Self {
            addr,
            codec,
            dom_leg,
            cod_leg,
            apex,
            dom_len,
            cod_len,
            apex_len,
            scalar_count,
            canon_key,
        }
    }

    /// The cospan's content address, which is also its record id.
    #[must_use]
    pub fn addr(&self) -> &CospanAddr {
        &self.addr
    }

    /// The encoding version this record was written under.
    #[must_use]
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// The domain leg: apex index per domain boundary node.
    #[must_use]
    pub fn dom_leg(&self) -> &[i64] {
        &self.dom_leg
    }

    /// The codomain leg: apex index per codomain boundary node.
    #[must_use]
    pub fn cod_leg(&self) -> &[i64] {
        &self.cod_leg
    }

    /// The apex, as encoded labels.
    #[must_use]
    pub fn apex(&self) -> &[String] {
        &self.apex
    }

    /// The domain size.
    #[must_use]
    pub fn dom_len(&self) -> i64 {
        self.dom_len
    }

    /// The codomain size.
    #[must_use]
    pub fn cod_len(&self) -> i64 {
        self.cod_len
    }

    /// The number of apex vertices.
    #[must_use]
    pub fn apex_len(&self) -> i64 {
        self.apex_len
    }

    /// How many apex vertices are hit by neither leg — the scalars.
    #[must_use]
    pub fn scalar_count(&self) -> i64 {
        self.scalar_count
    }

    /// The canonical key: the morphism's identity. See [`canon_key`].
    #[must_use]
    pub fn canon_key(&self) -> &str {
        &self.canon_key
    }

    /// Re-derive everything from the presentation columns and hand back a cospan
    /// that has actually been checked.
    ///
    /// # The order, and why it is fixed
    ///
    /// 1. **Codec.** A record written under an encoding this build does not know
    ///    is refused rather than interpreted under the current rules.
    /// 2. **Content address.** The presentation is re-encoded and re-digested,
    ///    and compared against the record id it was filed under. It is the
    ///    cheapest check that fully decides tampering of the leg or apex
    ///    columns, so it runs before anything interprets them.
    /// 3. **Widening.** Every leg entry must widen to an index. The column is a
    ///    signed integer read off disk, so a negative entry is possible, and it
    ///    is refused here; an entry at or past the apex length is refused in
    ///    step 5.
    /// 4. **Labels.** Each apex label is decoded — and then **re-encoded and
    ///    compared against the stored string**. Decoding alone is not enough:
    ///    [`LabelCodec::decode`] is free to accept non-canonical spellings, and
    ///    a decodable-but-non-canonical apex would let a forged presentation
    ///    squat the *canonical* spelling's `canon_key` under a different
    ///    address, breaking the two-identity discipline outright. Requiring
    ///    `encode(decode(s)) == s` pins every stored label to its one canonical
    ///    spelling — and it is what entitles step 5 to reuse the stored strings
    ///    instead of re-encoding.
    /// 5. **Derived columns.** The sizes, the scalar count, and the canonical
    ///    key are all re-derived from the presentation and compared, and the
    ///    class derivation they are computed from is where a leg entry at or
    ///    past the apex length is refused. A hand-edited `canon_key` in
    ///    particular must be caught: it is `UNIQUE` and complete, so a
    ///    corrupted one would make a key lookup claim two unequal morphisms are
    ///    equal.
    ///
    /// # Errors
    ///
    /// [`StoreError::TypeMismatch`] for an unknown codec, and
    /// [`StoreError::Corrupt`] naming what disagreed for everything else.
    pub fn revalidate<L: LabelCodec>(&self) -> Result<Cospan<L>> {
        if self.codec != COSPAN_CODEC {
            return Err(StoreError::TypeMismatch {
                field: "codec".to_owned(),
                expected: COSPAN_CODEC.to_owned(),
                actual: self.codec.clone(),
            });
        }

        let addr = address_of(&self.dom_leg, &self.cod_leg, &self.apex)?;
        if addr != self.addr {
            return Err(corrupt(&format!(
                "record `{}` holds a presentation whose content address is `{addr}`",
                self.addr
            )));
        }

        let left = to_leg(Side::Domain, &self.dom_leg)?;
        let right = to_leg(Side::Codomain, &self.cod_leg)?;

        let middle = self
            .apex
            .iter()
            .enumerate()
            .map(|(index, encoded)| {
                let label = L::decode(encoded).ok_or_else(|| {
                    corrupt(&format!(
                        "apex vertex {index} carries `{encoded}`, which is not a label of this type"
                    ))
                })?;
                let canonical = label.encode();
                if &canonical != encoded {
                    return Err(corrupt(&format!(
                        "apex vertex {index} carries `{encoded}`, a non-canonical spelling of \
                         `{canonical}`"
                    )));
                }
                Ok(label)
            })
            .collect::<Result<Vec<L>>>()?;

        // The stored strings are reusable here precisely because step 4 pinned
        // each one as the canonical spelling of its label.
        let derived = derive(&left, &right, &self.apex)?;
        let cospan = Cospan::new(left, right, middle).map_err(|error| {
            corrupt(&format!("the stored presentation is not a cospan: {error}"))
        })?;

        expect_column("dom_len", self.dom_len, derived.dom_len)?;
        expect_column("cod_len", self.cod_len, derived.cod_len)?;
        expect_column("apex_len", self.apex_len, derived.apex_len)?;
        expect_column("scalar_count", self.scalar_count, derived.scalar_count)?;
        if self.canon_key != derived.canon_key {
            return Err(corrupt(&format!(
                "stored canon_key `{}` disagrees with the derived `{}`",
                self.canon_key, derived.canon_key
            )));
        }

        Ok(cospan)
    }
}

/// Everything a cospan's derived columns hold.
struct Derived {
    dom_len: i64,
    cod_len: i64,
    apex_len: i64,
    scalar_count: i64,
    canon_key: String,
}

/// Compute the derived columns from a presentation's encoded pieces.
fn derive(dom_leg: &[usize], cod_leg: &[usize], apex: &[String]) -> Result<Derived> {
    let (dom_len, cod_len, classes) = canonical_classes(dom_leg, cod_leg, apex)?;
    let (canon_key, scalar_count) = canon_key_of(dom_len, cod_len, &classes)?;
    Ok(Derived {
        dom_len: to_column("dom_len", dom_len)?,
        cod_len: to_column("cod_len", cod_len)?,
        apex_len: to_column("apex_len", classes.len())?,
        scalar_count: to_column("scalar_count", scalar_count)?,
        canon_key,
    })
}

/// Encode a cospan into the record that will be stored.
///
/// # What this refuses
///
/// A cospan whose legs point outside its apex, whatever constructor built it.
/// `Cospan::new` rejects one in every build profile; `Cospan::new_unchecked`
/// and `Cospan::add_boundary_node_unchecked` accept one in a release build, and
/// deriving the canonical classes here is what catches a value built through
/// either.
///
/// # Errors
///
/// [`StoreError::Corrupt`] for an out-of-bounds leg, [`StoreError::Codec`] if
/// the encoding fails, and [`StoreError::TypeMismatch`] in the theoretical case
/// of a size too large for the database's integer column.
pub fn encode<L: LabelCodec>(cospan: &Cospan<L>) -> Result<CospanRecord> {
    let dom_leg = from_leg(Side::Domain, cospan.left_to_middle())?;
    let cod_leg = from_leg(Side::Codomain, cospan.right_to_middle())?;
    // Encoded exactly once; `derive` reuses these strings.
    let apex = encoded_apex(cospan);

    let derived = derive(cospan.left_to_middle(), cospan.right_to_middle(), &apex)?;

    Ok(CospanRecord {
        addr: address_of(&dom_leg, &cod_leg, &apex)?,
        codec: COSPAN_CODEC.to_owned(),
        dom_leg,
        cod_leg,
        apex,
        dom_len: derived.dom_len,
        cod_len: derived.cod_len,
        apex_len: derived.apex_len,
        scalar_count: derived.scalar_count,
        canon_key: derived.canon_key,
    })
}

/// The canonical encoding of a presentation — what the record id addresses.
fn structural_encoding(dom_leg: &[i64], cod_leg: &[i64], apex: &[String]) -> Result<String> {
    #[derive(Serialize)]
    struct Presentation<'a> {
        dom_leg: &'a [i64],
        cod_leg: &'a [i64],
        apex: &'a [String],
    }
    Ok(serde_json::to_string(&Presentation {
        dom_leg,
        cod_leg,
        apex,
    })?)
}

/// The content address of a presentation, as stored.
///
/// Public because a restore has to **re-verify record ids rather than trust the
/// replay**: the id `ASSERT` is skipped under `OPTION IMPORT`, so a dump can
/// carry ids this store would never have written, and the columns are exactly
/// what a restore has in hand. Ordinary reads never need this — loading
/// re-derives and compares the address for every row it returns.
///
/// # Errors
///
/// [`StoreError::Codec`] if the presentation does not encode.
pub fn address_of(dom_leg: &[i64], cod_leg: &[i64], apex: &[String]) -> Result<CospanAddr> {
    let encoding = structural_encoding(dom_leg, cod_leg, apex)?;
    let key = addr::digest_key(encoding.as_bytes());
    CospanAddr::parse(&key).ok_or_else(|| corrupt(&format!("`{key}` is not a well-formed address")))
}

/// Narrow an in-memory leg into the database's integer lane.
fn from_leg(side: Side, leg: &[usize]) -> Result<Vec<i64>> {
    leg.iter()
        .map(|&apex| to_column(side.as_str(), apex))
        .collect()
}

/// Widen a stored leg back into indices.
///
/// A leg entry off disk is a signed integer, so it can be negative; an entry at
/// or past the apex length is left to the class derivation.
fn to_leg(side: Side, leg: &[i64]) -> Result<Vec<usize>> {
    leg.iter()
        .enumerate()
        .map(|(boundary, &apex)| {
            usize::try_from(apex).map_err(|_| {
                corrupt(&format!(
                    "{} leg maps boundary node {boundary} to {apex}, which is not an index",
                    side.as_str()
                ))
            })
        })
        .collect()
}

/// Narrow a derived size into the database's integer lane.
///
/// Unreachable in practice — these are lengths of in-memory collections — but
/// narrowing silently is how a corrupt document is born, so it is checked rather
/// than cast.
fn to_column(field: &str, value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| StoreError::TypeMismatch {
        field: field.to_owned(),
        expected: "int".to_owned(),
        actual: format!("{value} (too large for a 64-bit signed integer)"),
    })
}

/// Compare a stored derived column against the value re-derived on load.
fn expect_column(field: &str, stored: i64, derived: i64) -> Result<()> {
    if stored == derived {
        Ok(())
    } else {
        Err(corrupt(&format!(
            "stored `{field}` is {stored}, but the presentation derives {derived}"
        )))
    }
}

/// The uniform corrupt-document error for this tier.
fn corrupt(detail: &str) -> StoreError {
    StoreError::Corrupt {
        context: COSPAN_TABLE.to_owned(),
        detail: detail.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `id₂`, with each wire on its own apex vertex.
    fn id2() -> Cospan<usize> {
        Cospan::new(vec![0, 1], vec![0, 1], vec![7, 7]).expect("id₂'s legs are in bounds")
    }

    /// The same morphism with the apex vertices swapped.
    fn id2_swapped() -> Cospan<usize> {
        Cospan::new(vec![1, 0], vec![1, 0], vec![7, 7]).expect("id₂'s legs are in bounds")
    }

    /// The braid on two wires — a genuinely different morphism.
    fn braid() -> Cospan<usize> {
        Cospan::new(vec![0, 1], vec![1, 0], vec![7, 7]).expect("the braid's legs are in bounds")
    }

    #[test]
    fn encoding_is_deterministic() {
        let first = encode(&id2()).expect("a well-formed cospan encodes");
        let second = encode(&id2()).expect("a well-formed cospan encodes");
        assert_eq!(first, second);
    }

    #[test]
    fn derived_columns_describe_the_cospan() {
        // μ-shape: two domain wires and one codomain wire, all on one apex
        // vertex, plus a spare vertex nobody touches — a scalar.
        let mu = Cospan::new(vec![0, 0], vec![0], vec![3usize, 9]).expect("μ's legs are in bounds");
        let record = encode(&mu).expect("a well-formed cospan encodes");
        assert_eq!(record.codec(), COSPAN_CODEC);
        assert_eq!(record.dom_len(), 2);
        assert_eq!(record.cod_len(), 1);
        assert_eq!(record.apex_len(), 2);
        assert_eq!(record.scalar_count(), 1);
        assert_eq!(record.dom_leg(), [0, 0]);
        assert_eq!(record.cod_leg(), [0]);
        assert_eq!(record.apex(), ["3".to_owned(), "9".to_owned()]);
    }

    #[test]
    fn a_record_round_trips_through_revalidation() {
        let cospan =
            Cospan::new(vec![0, 0], vec![1], vec![3usize, 9]).expect("the legs are in bounds");
        let record = encode(&cospan).expect("a well-formed cospan encodes");
        let loaded: Cospan<usize> = record.revalidate().expect("its own encoding revalidates");
        assert_eq!(loaded, cospan);
    }

    /// The presentation is what the address identifies, so an apex reordering is
    /// a *different record* — while the canonical key says they are one
    /// morphism. That split is the whole design of this tier.
    #[test]
    fn apex_reordering_changes_the_address_but_not_the_key() {
        let straight = encode(&id2()).expect("encodes");
        let swapped = encode(&id2_swapped()).expect("encodes");
        assert_ne!(straight.addr(), swapped.addr());
        assert_eq!(straight.canon_key(), swapped.canon_key());
    }

    #[test]
    fn a_different_morphism_gets_a_different_key() {
        let identity = encode(&id2()).expect("encodes");
        let braided = encode(&braid()).expect("encodes");
        assert_ne!(identity.canon_key(), braided.canon_key());
    }

    /// Scalars are kept, not collapsed: `k` closed bubbles are a different
    /// morphism from `k - 1`.
    #[test]
    fn scalars_are_counted_not_collapsed() {
        let none = Cospan::<usize>::new(vec![], vec![], vec![]).expect("no legs to check");
        let one = Cospan::new(vec![], vec![], vec![1usize]).expect("no legs to check");
        let two = Cospan::new(vec![], vec![], vec![1usize, 1]).expect("no legs to check");

        let keys: Vec<String> = [&none, &one, &two]
            .into_iter()
            .map(|c| canon_key(c).expect("a bubble cospan has a key"))
            .collect();
        assert_ne!(keys[0], keys[1]);
        assert_ne!(keys[1], keys[2]);

        assert_eq!(encode(&none).expect("encodes").scalar_count(), 0);
        assert_eq!(encode(&one).expect("encodes").scalar_count(), 1);
        assert_eq!(encode(&two).expect("encodes").scalar_count(), 2);
    }

    /// Labels are part of the morphism: the same wiring over different apex
    /// labels is not the same cospan.
    #[test]
    fn labels_separate_keys() {
        let a = Cospan::new(vec![0], vec![0], vec![1usize]).expect("the legs are in bounds");
        let b = Cospan::new(vec![0], vec![0], vec![2usize]).expect("the legs are in bounds");
        assert_ne!(
            canon_key(&a).expect("a has a key"),
            canon_key(&b).expect("b has a key")
        );
    }

    /// The recomputation must agree with catgraph's own accessors, or the key is
    /// keying something other than the documented equivalence.
    #[test]
    fn derived_columns_agree_with_catgraph_canonical_form() {
        let samples = [
            id2(),
            id2_swapped(),
            braid(),
            Cospan::new(vec![0, 0], vec![0], vec![3usize, 9]).expect("μ's legs are in bounds"),
            Cospan::new(vec![], vec![], vec![1usize, 1, 2]).expect("no legs to check"),
            Cospan::<usize>::new(vec![], vec![], vec![]).expect("no legs to check"),
        ];
        for cospan in &samples {
            let record = encode(cospan).expect("encodes");
            let canon = cospan.canonical_form();
            assert_eq!(record.dom_len(), canon.dom_len() as i64, "{cospan:?}");
            assert_eq!(record.cod_len(), canon.cod_len() as i64, "{cospan:?}");
            assert_eq!(record.apex_len(), canon.apex_len() as i64, "{cospan:?}");
            assert_eq!(
                record.scalar_count(),
                canon.scalar_count() as i64,
                "{cospan:?}"
            );
        }
    }

    /// [`encode`] refuses a leg index at or past the apex length with
    /// [`StoreError::Corrupt`], driven end to end rather than through the
    /// helper that finds it.
    ///
    /// `Cospan::new_unchecked` is what builds the value, and it `debug_assert!`s
    /// the leg bounds; `Cargo.toml` turns `debug-assertions` off for the
    /// `catgraph` package in the test profile so this fixture exists.
    #[test]
    fn an_out_of_bounds_leg_is_refused_on_the_write_path() {
        let out_of_bounds = Cospan::new_unchecked(vec![5], vec![], vec![1usize]);
        let err = encode(&out_of_bounds)
            .expect_err("a leg pointing outside the apex must not become a record");
        match err {
            StoreError::Corrupt { detail, .. } => {
                assert!(detail.contains("apex index 5"), "{detail}");
                assert!(detail.contains("1 vertices"), "{detail}");
            }
            other => panic!("expected an out-of-bounds corruption, got {other:?}"),
        }
    }

    /// A record's columns, so a test can change exactly one and rebuild.
    ///
    /// Every read-side case below is one field's difference from a faithful
    /// record; spelling out all ten arguments each time hides which one that is.
    struct Columns {
        addr: CospanAddr,
        codec: String,
        dom_leg: Vec<i64>,
        cod_leg: Vec<i64>,
        apex: Vec<String>,
        dom_len: i64,
        cod_len: i64,
        apex_len: i64,
        scalar_count: i64,
        canon_key: String,
    }

    impl Columns {
        fn of(record: &CospanRecord) -> Self {
            Self {
                addr: record.addr().clone(),
                codec: record.codec().to_owned(),
                dom_leg: record.dom_leg().to_vec(),
                cod_leg: record.cod_leg().to_vec(),
                apex: record.apex().to_vec(),
                dom_len: record.dom_len(),
                cod_len: record.cod_len(),
                apex_len: record.apex_len(),
                scalar_count: record.scalar_count(),
                canon_key: record.canon_key().to_owned(),
            }
        }

        /// Re-file the row under the content address of whatever presentation it
        /// now holds, so the address check passes and the *later* stage fires.
        fn refile(mut self) -> Self {
            self.addr = address_of(&self.dom_leg, &self.cod_leg, &self.apex)
                .expect("a presentation is always addressable");
            self
        }

        fn build(self) -> CospanRecord {
            CospanRecord::from_columns(
                self.addr,
                self.codec,
                self.dom_leg,
                self.cod_leg,
                self.apex,
                self.dom_len,
                self.cod_len,
                self.apex_len,
                self.scalar_count,
                self.canon_key,
            )
        }
    }

    /// And the same invariant on the read side, where it arrives as a tampered
    /// column.
    #[test]
    fn an_out_of_bounds_leg_is_refused_on_load() {
        let original = Cospan::new(vec![0], vec![0], vec![1usize]).expect("the legs are in bounds");
        let record = encode(&original).expect("the well-formed original encodes");
        let mut columns = Columns::of(&record);
        columns.dom_leg = vec![5];
        let err = columns
            .refile()
            .build()
            .revalidate::<usize>()
            .expect_err("a leg pointing outside the apex is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    #[test]
    fn an_unknown_codec_is_refused() {
        let record = encode(&id2()).expect("encodes");
        let mut columns = Columns::of(&record);
        columns.codec = "cgc99".to_owned();
        match columns.build().revalidate::<usize>() {
            Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "codec"),
            other => panic!("expected a codec type mismatch, got {other:?}"),
        }
    }

    /// A label type whose `decode` accepts spellings its `encode` never
    /// produces. The integer codecs refuse those outright, so a lenient codec is
    /// what the store's own canonical-spelling comparison ranges over.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Lenient(usize);

    impl LabelCodec for Lenient {
        fn encode(&self) -> String {
            self.0.to_string()
        }

        fn decode(raw: &str) -> Option<Self> {
            raw.parse().ok().map(Self)
        }
    }

    /// A spelling that decodes but does not re-encode to itself is corrupt:
    /// accepting it would let a forged presentation squat the canonical
    /// spelling's `canon_key` under a different address.
    #[test]
    fn a_non_canonical_label_spelling_is_corrupt() {
        let original =
            Cospan::new(vec![0], vec![0], vec![Lenient(7)]).expect("the legs are in bounds");
        let record = encode(&original).expect("encodes");
        let mut columns = Columns::of(&record);
        columns.apex = vec!["007".to_owned()];
        assert_eq!(
            Lenient::decode("007"),
            Some(Lenient(7)),
            "the fixture codec has to accept the spelling for the next check to be reached"
        );
        match columns.refile().build().revalidate::<Lenient>() {
            Err(StoreError::Corrupt { detail, .. }) => {
                assert!(detail.contains("non-canonical"), "{detail}");
            }
            other => panic!("expected a non-canonical-spelling rejection, got {other:?}"),
        }
    }

    /// And at an integer label, where `decode` refuses the spelling one stage
    /// earlier — the same forgery, refused for a different stated reason.
    #[test]
    fn a_zero_padded_integer_label_is_corrupt() {
        let original = Cospan::new(vec![0], vec![0], vec![7usize]).expect("the legs are in bounds");
        let record = encode(&original).expect("encodes");
        let mut columns = Columns::of(&record);
        columns.apex = vec!["007".to_owned()];
        match columns.refile().build().revalidate::<usize>() {
            Err(StoreError::Corrupt { detail, .. }) => {
                assert!(detail.contains("`007`"), "{detail}");
                // Which stage refused, not merely that one did: the
                // canonical-spelling comparison carries `007` too.
                assert!(detail.contains("not a label of this type"), "{detail}");
            }
            other => panic!("expected `007` to be refused, got {other:?}"),
        }
    }

    #[test]
    fn an_undecodable_label_is_corrupt() {
        let record = encode(&id2()).expect("encodes");
        let mut columns = Columns::of(&record);
        columns.apex = vec!["not-a-number".to_owned(), "7".to_owned()];
        let err = columns
            .refile()
            .build()
            .revalidate::<usize>()
            .expect_err("an undecodable label is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    /// A negative leg entry cannot come from an in-memory cospan, but it can
    /// come off disk — the column is a signed integer.
    #[test]
    fn a_negative_leg_entry_is_corrupt() {
        let record = encode(&id2()).expect("encodes");
        let mut columns = Columns::of(&record);
        columns.dom_leg = vec![-1, 1];
        let err = columns
            .refile()
            .build()
            .revalidate::<usize>()
            .expect_err("a negative index is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    /// The canonical key is `UNIQUE` and complete, so a corrupted one would make
    /// a key lookup claim two unequal morphisms are equal. It has to be
    /// re-derived, not believed.
    #[test]
    fn a_tampered_canonical_key_is_corrupt() {
        let record = encode(&id2()).expect("encodes");
        let mut columns = Columns::of(&record);
        columns.canon_key = canon_key(&braid()).expect("the braid has a key");
        let err = columns
            .build()
            .revalidate::<usize>()
            .expect_err("a canon_key that is not the derived one is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    /// A presentation swapped in under someone else's address: the cheapest
    /// check decides it, before anything interprets a leg.
    #[test]
    fn a_row_filed_under_the_wrong_address_is_corrupt() {
        let record = encode(&id2()).expect("encodes");
        let mut columns = Columns::of(&record);
        columns.addr = CospanAddr::from_digest(&"c".repeat(64)).expect("a valid digest");
        let err = columns
            .build()
            .revalidate::<usize>()
            .expect_err("a misfiled presentation is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    /// Each derived size is re-derived and compared, so a hand-edited one is
    /// caught rather than served.
    #[test]
    fn lying_size_columns_are_corrupt() {
        let record = encode(&id2()).expect("encodes");
        for mutate in [
            (|c: &mut Columns| c.dom_len = 9) as fn(&mut Columns),
            |c: &mut Columns| c.cod_len = 9,
            |c: &mut Columns| c.apex_len = 9,
            |c: &mut Columns| c.scalar_count = 9,
        ] {
            let mut columns = Columns::of(&record);
            mutate(&mut columns);
            let err = columns
                .build()
                .revalidate::<usize>()
                .expect_err("a lying size column is corrupt");
            assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
        }
    }
}
