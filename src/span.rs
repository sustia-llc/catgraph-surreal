//! Span encoding, canonical keys, and the bounds-checking discipline.
//!
//! # Columns
//!
//! A span's apex maps into both boundaries, so apex element `i` is a pair
//! `(mid_dom[i], mid_cod[i])` of boundary indices. The pairs are stored as two
//! parallel integer arrays beside the two boundaries' encoded labels, `dom` and
//! `cod`.
//!
//! # Two identities
//!
//! - The **record id** is a content address of the *presentation*
//!   `(dom, cod, mid_dom, mid_cod)`, exactly as stored. Two orderings of one
//!   span's middle pairs get two addresses.
//! - The **canonical key** ([`canon_key`]) identifies the *morphism*. It is the
//!   digest of the encoded boundaries and the multiset of middle pairs, sorted
//!   ascending. A span's apex carries no labels of its own, so two parallel
//!   spans are related by a bijection of apexes commuting with both legs exactly
//!   when their sorted pair lists are equal: equal keys mean equal morphisms and
//!   vice versa. `tests/span_canon.rs` checks that equivalence against a
//!   brute-force bijection search.
//!
//! The reverse direction also needs [`LabelCodec::encode`](crate::LabelCodec)
//! to be injective, which its round-trip law gives.
//!
//! # The store checks middle pairs itself
//!
//! `Span::new` checks every middle pair's bounds and label agreement in every
//! build profile; `Span::new_unchecked` checks them only under `debug_assert!`.
//! Both [`encode`] and [`SpanRecord::revalidate`] check every pair: a component
//! at or past its boundary's length, or a pair whose two labels differ, is
//! [`StoreError::Corrupt`] on the write path and on the load path alike.

use catgraph::span::Span;
use serde::Serialize;

use crate::addr::{self, SpanAddr};
use crate::codec::LabelCodec;
use crate::error::{Result, StoreError};
use crate::schema::SPAN_TABLE;

/// The version tag stored in every span's `codec` column.
///
/// It versions the whole encoding: the presentation the record id addresses,
/// the canonical encoding [`canon_key`] digests, and the label codec's output. A
/// record whose codec this build does not recognise is refused on load.
pub const SPAN_CODEC: &str = "cgsp1";

/// One middle pair: `(domain index, codomain index)`.
type Pair = (usize, usize);

/// Which boundary a failure was found on.
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

/// Check every middle pair against the encoded boundaries, and return the pairs
/// sorted ascending.
///
/// Labels are compared as encoded strings. On the write path those are fresh
/// `encode` output; on the load path they are stored strings already required to
/// be canonical spellings. Either way string equality is label equality.
///
/// # Errors
///
/// [`StoreError::Corrupt`] for a pair component at or past its boundary's
/// length, or a pair whose two labels differ. Pairs are checked in position
/// order, the domain component before the codomain component before the labels.
fn canonical_pairs(dom: &[String], cod: &[String], pairs: &[Pair]) -> Result<Vec<Pair>> {
    for (position, &(d, c)) in pairs.iter().enumerate() {
        let left = dom
            .get(d)
            .ok_or_else(|| out_of_bounds(Side::Domain, position, d, dom.len()))?;
        let right = cod
            .get(c)
            .ok_or_else(|| out_of_bounds(Side::Codomain, position, c, cod.len()))?;
        if left != right {
            return Err(corrupt(&format!(
                "middle pair {position} links domain node {d} (`{left}`) to codomain node {c} \
                 (`{right}`), whose labels differ"
            )));
        }
    }
    let mut sorted = pairs.to_vec();
    sorted.sort_unstable();
    Ok(sorted)
}

/// The uniform out-of-bounds report.
fn out_of_bounds(side: Side, position: usize, target: usize, len: usize) -> StoreError {
    corrupt(&format!(
        "middle pair {position} maps to {side} node {target}, but the {side} boundary has {len} \
         nodes",
        side = side.as_str()
    ))
}

/// Encode a boundary's labels, in order.
fn encoded_boundary<L: LabelCodec>(labels: &[L]) -> Vec<String> {
    labels.iter().map(LabelCodec::encode).collect()
}

/// The canonical encoding `(dom, cod, sorted pairs)` — what the key digests.
fn canonical_encoding(dom: &[String], cod: &[String], sorted: &[Pair]) -> Result<String> {
    Ok(serde_json::to_string(&(dom, cod, sorted))?)
}

/// The digest of [`canonical_encoding`].
fn canon_key_of(dom: &[String], cod: &[String], sorted: &[Pair]) -> Result<String> {
    let canonical = canonical_encoding(dom, cod, sorted)?;
    Ok(addr::digest_key(canonical.as_bytes()))
}

/// The canonical key of a span: the digest of its apex-isomorphism class.
///
/// Two parallel spans share a key exactly when they are equal as morphisms —
/// see the [module documentation](self).
///
/// # Errors
///
/// [`StoreError::Corrupt`] if a middle pair is out of bounds or names two
/// different labels; [`StoreError::Codec`] if the encoding fails.
pub fn canon_key<L: LabelCodec>(span: &Span<L>) -> Result<String> {
    let dom = encoded_boundary(span.left());
    let cod = encoded_boundary(span.right());
    let sorted = canonical_pairs(&dom, &cod, span.middle_pairs())?;
    canon_key_of(&dom, &cod, &sorted)
}

/// A span as it is stored: the presentation, plus every column derived from it.
///
/// The integer columns are `i64`, mirroring the row; [`encode`] converts from
/// `usize` with a checked narrowing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanRecord {
    pub(crate) addr: SpanAddr,
    pub(crate) codec: String,
    pub(crate) dom: Vec<String>,
    pub(crate) cod: Vec<String>,
    pub(crate) mid_dom: Vec<i64>,
    pub(crate) mid_cod: Vec<i64>,
    pub(crate) dom_len: i64,
    pub(crate) cod_len: i64,
    pub(crate) apex_len: i64,
    pub(crate) canon_key: String,
}

impl SpanRecord {
    /// Rebuild a record from columns read back out of the database, unchecked;
    /// [`Self::revalidate`] checks them.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_columns(
        addr: SpanAddr,
        codec: String,
        dom: Vec<String>,
        cod: Vec<String>,
        mid_dom: Vec<i64>,
        mid_cod: Vec<i64>,
        dom_len: i64,
        cod_len: i64,
        apex_len: i64,
        canon_key: String,
    ) -> Self {
        Self {
            addr,
            codec,
            dom,
            cod,
            mid_dom,
            mid_cod,
            dom_len,
            cod_len,
            apex_len,
            canon_key,
        }
    }

    /// The span's content address, which is also its record id.
    #[must_use]
    pub fn addr(&self) -> &SpanAddr {
        &self.addr
    }

    /// The encoding version this record was written under.
    #[must_use]
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// The domain boundary, as encoded labels.
    #[must_use]
    pub fn dom(&self) -> &[String] {
        &self.dom
    }

    /// The codomain boundary, as encoded labels.
    #[must_use]
    pub fn cod(&self) -> &[String] {
        &self.cod
    }

    /// The domain index of each apex element, in presentation order.
    #[must_use]
    pub fn mid_dom(&self) -> &[i64] {
        &self.mid_dom
    }

    /// The codomain index of each apex element, in presentation order.
    #[must_use]
    pub fn mid_cod(&self) -> &[i64] {
        &self.mid_cod
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

    /// The number of apex elements — middle pairs.
    #[must_use]
    pub fn apex_len(&self) -> i64 {
        self.apex_len
    }

    /// The canonical key: the morphism's identity. See [`canon_key`].
    #[must_use]
    pub fn canon_key(&self) -> &str {
        &self.canon_key
    }

    /// Re-derive everything from the presentation columns and return a span
    /// built through `Span::new`.
    ///
    /// The checks run in this order, and the first failure is returned:
    ///
    /// 1. **Codec.** The `codec` column must be [`SPAN_CODEC`].
    /// 2. **Content address.** The presentation's digest must equal the record
    ///    id.
    /// 3. **Widening.** Every `mid_dom`/`mid_cod` entry must be non-negative,
    ///    and the two columns must have equal length.
    /// 4. **Labels.** Every `dom`/`cod` string must decode, and must re-encode
    ///    to itself (`encode(decode(s)) == s`).
    /// 5. **Derived columns.** Every middle pair must be in bounds and name two
    ///    equal labels; the sizes and the canonical key are re-derived and
    ///    compared.
    ///
    /// # Errors
    ///
    /// [`StoreError::TypeMismatch`] for an unknown codec, and
    /// [`StoreError::Corrupt`] naming what disagreed for everything else.
    pub fn revalidate<L: LabelCodec>(&self) -> Result<Span<L>> {
        if self.codec != SPAN_CODEC {
            return Err(StoreError::TypeMismatch {
                field: "codec".to_owned(),
                expected: SPAN_CODEC.to_owned(),
                actual: self.codec.clone(),
            });
        }

        let addr = address_of(&self.dom, &self.cod, &self.mid_dom, &self.mid_cod)?;
        if addr != self.addr {
            return Err(corrupt(&format!(
                "record `{}` holds a presentation whose content address is `{addr}`",
                self.addr
            )));
        }

        let mid_dom = to_indices(Side::Domain, &self.mid_dom)?;
        let mid_cod = to_indices(Side::Codomain, &self.mid_cod)?;
        if mid_dom.len() != mid_cod.len() {
            return Err(corrupt(&format!(
                "mid_dom has {} entries but mid_cod has {}",
                mid_dom.len(),
                mid_cod.len()
            )));
        }
        let pairs: Vec<Pair> = mid_dom.into_iter().zip(mid_cod).collect();

        let left = decode_boundary::<L>(Side::Domain, &self.dom)?;
        let right = decode_boundary::<L>(Side::Codomain, &self.cod)?;

        let derived = derive(&self.dom, &self.cod, &pairs)?;
        expect_column("dom_len", self.dom_len, derived.dom_len)?;
        expect_column("cod_len", self.cod_len, derived.cod_len)?;
        expect_column("apex_len", self.apex_len, derived.apex_len)?;
        if self.canon_key != derived.canon_key {
            return Err(corrupt(&format!(
                "stored canon_key `{}` disagrees with the derived `{}`",
                self.canon_key, derived.canon_key
            )));
        }

        Span::new(left, right, pairs)
            .map_err(|error| corrupt(&format!("the stored presentation is not a span: {error}")))
    }
}

/// Everything a span's derived columns hold.
struct Derived {
    dom_len: i64,
    cod_len: i64,
    apex_len: i64,
    canon_key: String,
}

/// Check the middle pairs and compute the derived columns from a presentation's
/// encoded boundaries.
fn derive(dom: &[String], cod: &[String], pairs: &[Pair]) -> Result<Derived> {
    let sorted = canonical_pairs(dom, cod, pairs)?;
    Ok(Derived {
        dom_len: to_column("dom_len", dom.len())?,
        cod_len: to_column("cod_len", cod.len())?,
        apex_len: to_column("apex_len", pairs.len())?,
        canon_key: canon_key_of(dom, cod, &sorted)?,
    })
}

/// Encode a span into the record that will be stored.
///
/// Refuses a span whose middle pairs are out of bounds or name two different
/// labels, whichever constructor built it.
///
/// # Errors
///
/// [`StoreError::Corrupt`] for an out-of-bounds or label-disagreeing middle
/// pair, [`StoreError::Codec`] if the encoding fails, and
/// [`StoreError::TypeMismatch`] for a size or index too large for a 64-bit
/// signed integer.
pub fn encode<L: LabelCodec>(span: &Span<L>) -> Result<SpanRecord> {
    let dom = encoded_boundary(span.left());
    let cod = encoded_boundary(span.right());
    let pairs = span.middle_pairs();

    let derived = derive(&dom, &cod, pairs)?;

    let mid_dom = from_indices("mid_dom", pairs.iter().map(|&(d, _)| d))?;
    let mid_cod = from_indices("mid_cod", pairs.iter().map(|&(_, c)| c))?;

    Ok(SpanRecord {
        addr: address_of(&dom, &cod, &mid_dom, &mid_cod)?,
        codec: SPAN_CODEC.to_owned(),
        dom,
        cod,
        mid_dom,
        mid_cod,
        dom_len: derived.dom_len,
        cod_len: derived.cod_len,
        apex_len: derived.apex_len,
        canon_key: derived.canon_key,
    })
}

/// The canonical encoding of a presentation — what the record id addresses.
fn structural_encoding(
    dom: &[String],
    cod: &[String],
    mid_dom: &[i64],
    mid_cod: &[i64],
) -> Result<String> {
    #[derive(Serialize)]
    struct Presentation<'a> {
        dom: &'a [String],
        cod: &'a [String],
        mid_dom: &'a [i64],
        mid_cod: &'a [i64],
    }
    Ok(serde_json::to_string(&Presentation {
        dom,
        cod,
        mid_dom,
        mid_cod,
    })?)
}

/// The content address of a presentation, as stored.
///
/// Public so a restore can re-verify record ids from the columns it holds.
///
/// # Errors
///
/// [`StoreError::Codec`] if the presentation does not encode.
pub fn address_of(
    dom: &[String],
    cod: &[String],
    mid_dom: &[i64],
    mid_cod: &[i64],
) -> Result<SpanAddr> {
    let encoding = structural_encoding(dom, cod, mid_dom, mid_cod)?;
    let key = addr::digest_key(encoding.as_bytes());
    SpanAddr::parse(&key).ok_or_else(|| corrupt(&format!("`{key}` is not a well-formed address")))
}

/// Narrow in-memory indices into the database's integer lane.
fn from_indices(field: &str, indices: impl Iterator<Item = usize>) -> Result<Vec<i64>> {
    indices.map(|index| to_column(field, index)).collect()
}

/// Widen a stored index column back into indices; a negative entry is corrupt.
fn to_indices(side: Side, column: &[i64]) -> Result<Vec<usize>> {
    column
        .iter()
        .enumerate()
        .map(|(position, &index)| {
            usize::try_from(index).map_err(|_| {
                corrupt(&format!(
                    "middle pair {position} has {} index {index}, which is not an index",
                    side.as_str()
                ))
            })
        })
        .collect()
}

/// Decode a stored boundary, requiring each string to be its label's canonical
/// spelling.
fn decode_boundary<L: LabelCodec>(side: Side, column: &[String]) -> Result<Vec<L>> {
    column
        .iter()
        .enumerate()
        .map(|(index, encoded)| {
            let label = L::decode(encoded).ok_or_else(|| {
                corrupt(&format!(
                    "{} node {index} carries `{encoded}`, which is not a label of this type",
                    side.as_str()
                ))
            })?;
            let canonical = label.encode();
            if &canonical != encoded {
                return Err(corrupt(&format!(
                    "{} node {index} carries `{encoded}`, a non-canonical spelling of \
                     `{canonical}`",
                    side.as_str()
                )));
            }
            Ok(label)
        })
        .collect()
}

/// Narrow a size or index into the database's integer lane.
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
        context: SPAN_TABLE.to_owned(),
        detail: detail.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A span's observable parts, for comparison — `Span` implements neither
    /// `PartialEq` nor `Debug`.
    fn parts<L: LabelCodec>(span: &Span<L>) -> (Vec<L>, Vec<L>, Vec<Pair>) {
        (
            span.left().to_vec(),
            span.right().to_vec(),
            span.middle_pairs().to_vec(),
        )
    }

    /// `id₂` as a span: two apex elements, each linking node `i` to node `i`.
    fn id2() -> Span<usize> {
        Span::new(vec![7, 7], vec![7, 7], vec![(0, 0), (1, 1)]).expect("id₂'s pairs are valid")
    }

    /// The same morphism with the middle pairs listed in the other order.
    fn id2_swapped() -> Span<usize> {
        Span::new(vec![7, 7], vec![7, 7], vec![(1, 1), (0, 0)]).expect("id₂'s pairs are valid")
    }

    /// The braid on two wires — a different morphism.
    fn braid() -> Span<usize> {
        Span::new(vec![7, 7], vec![7, 7], vec![(0, 1), (1, 0)])
            .expect("the braid's pairs are valid")
    }

    #[test]
    fn encoding_is_deterministic() {
        let first = encode(&id2()).expect("a well-formed span encodes");
        let second = encode(&id2()).expect("a well-formed span encodes");
        assert_eq!(first, second);
    }

    #[test]
    fn derived_columns_describe_the_span() {
        let span =
            Span::new(vec![3usize, 3], vec![3], vec![(1, 0), (0, 0)]).expect("the pairs are valid");
        let record = encode(&span).expect("a well-formed span encodes");
        assert_eq!(record.codec(), SPAN_CODEC);
        assert_eq!(record.dom(), ["3".to_owned(), "3".to_owned()]);
        assert_eq!(record.cod(), ["3".to_owned()]);
        assert_eq!(record.mid_dom(), [1, 0]);
        assert_eq!(record.mid_cod(), [0, 0]);
        assert_eq!(record.dom_len(), 2);
        assert_eq!(record.cod_len(), 1);
        assert_eq!(record.apex_len(), 2);
    }

    /// Both hash pre-images, pinned to exact bytes for the golden fixture in
    /// `tests/golden.rs`.
    #[test]
    fn the_pre_images_are_exact() {
        let dom = vec!["3".to_owned(), "3".to_owned()];
        let cod = vec!["3".to_owned()];
        assert_eq!(
            structural_encoding(&dom, &cod, &[1, 0], &[0, 0]).expect("encodes"),
            r#"{"dom":["3","3"],"cod":["3"],"mid_dom":[1,0],"mid_cod":[0,0]}"#
        );
        let sorted = canonical_pairs(&dom, &cod, &[(1, 0), (0, 0)]).expect("valid pairs");
        assert_eq!(
            canonical_encoding(&dom, &cod, &sorted).expect("encodes"),
            r#"[["3","3"],["3"],[[0,0],[1,0]]]"#
        );
    }

    #[test]
    fn a_record_round_trips_through_revalidation() {
        let span =
            Span::new(vec![3usize, 9], vec![9, 3], vec![(1, 0), (0, 1), (1, 0)]).expect("valid");
        let record = encode(&span).expect("a well-formed span encodes");
        let loaded: Span<usize> = record
            .revalidate()
            .map_err(|e| e.to_string())
            .expect("its own encoding revalidates");
        assert_eq!(parts(&loaded), parts(&span));
    }

    /// The presentation is what the address identifies, so a pair reordering is
    /// a different record, while the canonical key says they are one morphism.
    #[test]
    fn pair_reordering_changes_the_address_but_not_the_key() {
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

    /// Middle pairs form a multiset: a repeated pair is another apex element, so
    /// `k` copies are a different morphism from `k - 1`.
    #[test]
    fn duplicate_pair_multiplicity_changes_the_key() {
        let key = |pairs: Vec<Pair>| {
            canon_key(&Span::new(vec![1usize], vec![1], pairs).expect("the pairs are valid"))
                .expect("a well-formed span has a key")
        };
        let none = key(vec![]);
        let one = key(vec![(0, 0)]);
        let two = key(vec![(0, 0), (0, 0)]);
        assert_ne!(none, one);
        assert_ne!(one, two);
        assert_ne!(none, two);
    }

    /// Labels are part of the morphism: the same wiring over different boundary
    /// labels is not the same span.
    #[test]
    fn labels_separate_keys() {
        let a = Span::new(vec![1usize], vec![1], vec![(0, 0)]).expect("valid");
        let b = Span::new(vec![2usize], vec![2], vec![(0, 0)]).expect("valid");
        assert_ne!(
            canon_key(&a).expect("a has a key"),
            canon_key(&b).expect("b has a key")
        );
    }

    /// A span that is not its own dagger gets a different key from its dagger,
    /// over boundaries of equal size; the double dagger gets the original key.
    #[test]
    fn the_dagger_of_a_non_symmetric_span_has_a_different_key() {
        let span = Span::new(vec![1usize, 1], vec![1, 1], vec![(0, 0), (0, 1)]).expect("valid");
        let dagger = span.dagger();
        assert_eq!(dagger.middle_pairs(), [(0, 0), (1, 0)]);
        let key = canon_key(&span).expect("a key");
        assert_ne!(key, canon_key(&dagger).expect("a key"));
        assert_eq!(key, canon_key(&dagger.dagger()).expect("a key"));
    }

    /// The size columns agree with catgraph's own accessors.
    #[test]
    fn derived_columns_agree_with_catgraph_accessors() {
        let samples = [
            id2(),
            id2_swapped(),
            braid(),
            Span::new(vec![3usize, 3], vec![3], vec![(1, 0), (0, 0)]).expect("valid"),
            Span::<usize>::new(vec![], vec![], vec![]).expect("no pairs to check"),
            Span::new(vec![1usize, 2], vec![], vec![]).expect("no pairs to check"),
        ];
        for span in &samples {
            let record = encode(span).expect("encodes");
            assert_eq!(record.dom_len(), span.left().len() as i64);
            assert_eq!(record.cod_len(), span.right().len() as i64);
            assert_eq!(record.apex_len(), span.middle_pairs().len() as i64);
        }
    }

    /// [`encode`] refuses a pair component at or past its boundary's length.
    ///
    /// `Span::new_unchecked` builds the value; `Cargo.toml` turns
    /// `debug-assertions` off for the `catgraph` package in the test profile, so
    /// its `debug_assert!`s do not fire first.
    #[test]
    fn an_out_of_bounds_pair_is_refused_on_the_write_path() {
        let out_of_bounds = Span::new_unchecked(vec![1usize], vec![1], vec![(5, 0)]);
        match encode(&out_of_bounds) {
            Err(StoreError::Corrupt { detail, .. }) => {
                assert!(detail.contains("domain node 5"), "{detail}");
                assert!(detail.contains("has 1 nodes"), "{detail}");
            }
            other => panic!("expected an out-of-bounds corruption, got {other:?}"),
        }
    }

    /// [`encode`] refuses a pair whose two labels differ.
    #[test]
    fn a_label_disagreement_is_refused_on_the_write_path() {
        let mismatched = Span::new_unchecked(vec![1usize], vec![2], vec![(0, 0)]);
        match encode(&mismatched) {
            Err(StoreError::Corrupt { detail, .. }) => {
                assert!(detail.contains("labels differ"), "{detail}");
            }
            other => panic!("expected a label-disagreement corruption, got {other:?}"),
        }
    }

    /// A record's columns, so a test can change exactly one and rebuild.
    struct Columns {
        addr: SpanAddr,
        codec: String,
        dom: Vec<String>,
        cod: Vec<String>,
        mid_dom: Vec<i64>,
        mid_cod: Vec<i64>,
        dom_len: i64,
        cod_len: i64,
        apex_len: i64,
        canon_key: String,
    }

    impl Columns {
        fn of(record: &SpanRecord) -> Self {
            Self {
                addr: record.addr().clone(),
                codec: record.codec().to_owned(),
                dom: record.dom().to_vec(),
                cod: record.cod().to_vec(),
                mid_dom: record.mid_dom().to_vec(),
                mid_cod: record.mid_cod().to_vec(),
                dom_len: record.dom_len(),
                cod_len: record.cod_len(),
                apex_len: record.apex_len(),
                canon_key: record.canon_key().to_owned(),
            }
        }

        /// Re-file the row under the content address of the presentation it now
        /// holds, so the address check passes and a later stage fires.
        fn refile(mut self) -> Self {
            self.addr = address_of(&self.dom, &self.cod, &self.mid_dom, &self.mid_cod)
                .expect("a presentation is always addressable");
            self
        }

        fn build(self) -> SpanRecord {
            SpanRecord::from_columns(
                self.addr,
                self.codec,
                self.dom,
                self.cod,
                self.mid_dom,
                self.mid_cod,
                self.dom_len,
                self.cod_len,
                self.apex_len,
                self.canon_key,
            )
        }
    }

    /// Revalidate and keep only the error, since `Span` has no `Debug`.
    fn load_err<L: LabelCodec>(columns: Columns) -> StoreError {
        match columns.build().revalidate::<L>() {
            Ok(_) => panic!("the tampered record revalidated"),
            Err(error) => error,
        }
    }

    fn expect_corrupt_containing(error: StoreError, needle: &str) {
        match error {
            StoreError::Corrupt { detail, .. } => {
                assert!(detail.contains(needle), "`{needle}` not in: {detail}");
            }
            other => panic!("expected a corrupt-document failure, got {other:?}"),
        }
    }

    #[test]
    fn an_out_of_bounds_pair_is_refused_on_load() {
        let mut columns = Columns::of(&encode(&id2()).expect("encodes"));
        columns.mid_dom = vec![0, 5];
        expect_corrupt_containing(load_err::<usize>(columns.refile()), "domain node 5");
    }

    #[test]
    fn a_label_disagreement_is_refused_on_load() {
        let mut columns = Columns::of(&encode(&id2()).expect("encodes"));
        columns.cod = vec!["7".to_owned(), "8".to_owned()];
        expect_corrupt_containing(load_err::<usize>(columns.refile()), "labels differ");
    }

    #[test]
    fn an_unknown_codec_is_refused() {
        let mut columns = Columns::of(&encode(&id2()).expect("encodes"));
        columns.codec = "cgs99".to_owned();
        match load_err::<usize>(columns) {
            StoreError::TypeMismatch { field, .. } => assert_eq!(field, "codec"),
            other => panic!("expected a codec type mismatch, got {other:?}"),
        }
    }

    /// A label type whose `decode` accepts spellings its `encode` never
    /// produces, so the store's canonical-spelling comparison is reachable.
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

    /// A spelling that decodes but does not re-encode to itself is corrupt.
    #[test]
    fn a_non_canonical_label_spelling_is_corrupt() {
        let original = Span::new(vec![Lenient(7)], vec![Lenient(7)], vec![(0, 0)]).expect("valid");
        let mut columns = Columns::of(&encode(&original).expect("encodes"));
        columns.dom = vec!["007".to_owned()];
        assert_eq!(
            Lenient::decode("007"),
            Some(Lenient(7)),
            "the fixture codec has to accept the spelling for the next check to be reached"
        );
        expect_corrupt_containing(load_err::<Lenient>(columns.refile()), "non-canonical");
    }

    /// A non-canonical row whose derived columns are all consistent with its own
    /// stored strings is refused by the canonical-spelling check alone.
    #[test]
    fn a_self_consistent_non_canonical_row_is_corrupt() {
        let original = Span::new(vec![Lenient(7)], vec![Lenient(7)], vec![(0, 0)]).expect("valid");
        let mut columns = Columns::of(&encode(&original).expect("encodes"));
        columns.dom = vec!["007".to_owned()];
        columns.cod = vec!["007".to_owned()];
        columns.canon_key =
            canon_key_of(&columns.dom, &columns.cod, &[(0, 0)]).expect("a key over the forgery");
        expect_corrupt_containing(load_err::<Lenient>(columns.refile()), "non-canonical");
    }

    /// At an integer label, `decode` refuses the spelling one stage earlier.
    #[test]
    fn a_zero_padded_integer_label_is_corrupt() {
        let mut columns = Columns::of(&encode(&id2()).expect("encodes"));
        columns.cod = vec!["007".to_owned(), "7".to_owned()];
        let error = load_err::<usize>(columns.refile());
        expect_corrupt_containing(error, "`007`, which is not a label of this type");
    }

    #[test]
    fn an_undecodable_label_is_corrupt() {
        let mut columns = Columns::of(&encode(&id2()).expect("encodes"));
        columns.dom = vec!["not-a-number".to_owned(), "7".to_owned()];
        expect_corrupt_containing(
            load_err::<usize>(columns.refile()),
            "not a label of this type",
        );
    }

    /// A negative index cannot come from an in-memory span, but it can come off
    /// disk — the column is a signed integer.
    #[test]
    fn a_negative_pair_index_is_corrupt() {
        let mut columns = Columns::of(&encode(&id2()).expect("encodes"));
        columns.mid_cod = vec![0, -1];
        expect_corrupt_containing(
            load_err::<usize>(columns.refile()),
            "codomain index -1, which is not an index",
        );
    }

    #[test]
    fn mismatched_middle_column_lengths_are_corrupt() {
        let mut columns = Columns::of(&encode(&id2()).expect("encodes"));
        columns.mid_cod = vec![0];
        expect_corrupt_containing(
            load_err::<usize>(columns.refile()),
            "mid_dom has 2 entries but mid_cod has 1",
        );
    }

    /// The canonical key is re-derived on load, not believed.
    #[test]
    fn a_tampered_canonical_key_is_corrupt() {
        let mut columns = Columns::of(&encode(&id2()).expect("encodes"));
        columns.canon_key = canon_key(&braid()).expect("the braid has a key");
        expect_corrupt_containing(load_err::<usize>(columns), "canon_key");
    }

    #[test]
    fn a_row_filed_under_the_wrong_address_is_corrupt() {
        let mut columns = Columns::of(&encode(&id2()).expect("encodes"));
        columns.addr = SpanAddr::from_digest(&"c".repeat(64)).expect("a valid digest");
        expect_corrupt_containing(load_err::<usize>(columns), "content address");
    }

    #[test]
    fn lying_size_columns_are_corrupt() {
        let record = encode(&id2()).expect("encodes");
        for (field, mutate) in [
            (
                "dom_len",
                (|c: &mut Columns| c.dom_len = 9) as fn(&mut Columns),
            ),
            ("cod_len", |c: &mut Columns| c.cod_len = 9),
            ("apex_len", |c: &mut Columns| c.apex_len = 9),
        ] {
            let mut columns = Columns::of(&record);
            mutate(&mut columns);
            expect_corrupt_containing(load_err::<usize>(columns), field);
        }
    }
}
