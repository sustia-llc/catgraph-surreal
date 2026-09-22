//! Bridging catgraph's label types to their stored form.

use std::fmt::Debug;
use std::hash::Hash;

/// A label type that can cross the storage boundary.
///
/// catgraph's structures are generic over their wire labels (`Lambda`) and, for
/// named cospans, over port-name types. Persisting them means agreeing on a
/// stable encoding for those labels — that is all this trait is for. It is
/// deliberately the *only* shared trait in the store: the repositories
/// themselves have genuinely different contracts (four-step revalidation for
/// terms, bounds-checking for cospans, bit-exact bytes for weights, caller-shaped
/// documents for the generic tier) and forcing them into one signature would buy
/// nothing.
///
/// # Why these bounds
///
/// Each bound is load-bearing, and the set is deliberately stricter than
/// `Clone`:
///
/// - `Eq` + `Hash` — catgraph's canonicalization puts labels in hash sets and
///   maps. Without both, a label cannot participate in `canonical_form`.
/// - `Ord` — canonicalization needs a *deterministic order*, not just equality.
///   Hashing alone leaves iteration order unpinned, and an unpinned order means
///   a canonical encoding that differs between runs, which defeats
///   content-addressing.
/// - `Copy` — labels are index-like scalars sitting in tight inner loops.
///   Requiring `Copy` (rather than `Clone`) keeps those loops allocation-free
///   and documents that a label is a small value, not an owned buffer.
/// - `Debug` — corrupt-document errors quote the offending label; without
///   `Debug` those messages degrade to "something was wrong".
///
/// # The round-trip law (load-bearing, not advisory)
///
/// Every implementation must satisfy
///
/// ```text
/// decode(encode(x)) == Some(x)      for every label x
/// ```
///
/// This is not a politeness: it makes `encode` **injective** — two distinct
/// labels can never share an encoding — and injectivity is exactly what the
/// cospan and span tiers' completeness arguments stand on. Their canonical keys
/// are built from *encoded* labels, and "equal keys ⇒ equal morphisms" holds
/// only if distinct labels stay distinct after encoding. An implementation that
/// violates the law (say, two enum variants encoding to one string) makes the
/// `UNIQUE` key column refuse genuinely new morphisms as duplicates and makes
/// `find_by_canon` answer with unrelated morphisms — with no error anywhere
/// naming this trait. Pin the law with a round-trip test over your label type,
/// the way the integer implementations below do.
///
/// The store enforces the *storage-side* corollary itself: a stored label must
/// be the **canonical spelling** — `encode(decode(s)) == s` — and a row whose
/// stored string decodes but re-encodes differently is refused as corrupt on
/// load. `decode` is free to be lenient about spellings; what reaches disk is
/// not.
///
/// # Stability
///
/// [`Self::encode`] output is persisted and may be content-addressed, so it is
/// part of the on-disk format. Changing an existing implementation's encoding
/// changes the identity of everything already stored under it; treat it as a
/// schema migration, not a refactor.
pub trait LabelCodec: Eq + Copy + Debug + Ord + Hash {
    /// Encode this label into its stored form.
    ///
    /// Must be deterministic: the same label encodes to the same string on every
    /// process, build, and run. In particular, never derive this from a standard
    /// library hash — `DefaultHasher` output is not stable across processes.
    fn encode(&self) -> String;

    /// Recover a label from its stored form.
    ///
    /// Returns `None` when the string is not a valid encoding, which the caller
    /// reports as a corrupt document rather than treating as a missing value.
    fn decode(raw: &str) -> Option<Self>;
}

/// Implement [`LabelCodec`] for the primitive integers, one contract for all of
/// them: the decimal spelling out, exactly that spelling back.
macro_rules! integer_codec {
    ($($ty:ty),+ $(,)?) => {
        $(impl LabelCodec for $ty {
            /// The label's decimal spelling, as `to_string` renders it.
            fn encode(&self) -> String {
                self.to_string()
            }

            /// The label this decimal spelling names, and `None` for every
            /// other string — a spelling this type parses but would never
            /// produce (`"+1"`, `"01"`, and `"-0"` at the signed types)
            /// included, alongside the ones it does not parse at all (`" 1"`,
            /// `""`, a value out of range, and `"-0"` at the unsigned types).
            fn decode(raw: &str) -> Option<Self> {
                raw.parse::<Self>()
                    .ok()
                    .filter(|label| label.to_string() == raw)
            }
        })+
    };
}

integer_codec!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Expand `$case!(ty)` once for every type [`integer_codec`] implements the
    /// trait for, so a rule below is stated once and checked over all twelve.
    macro_rules! over_the_family {
        ($case:ident) => {
            $case!(u8);
            $case!(u16);
            $case!(u32);
            $case!(u64);
            $case!(u128);
            $case!(usize);
            $case!(i8);
            $case!(i16);
            $case!(i32);
            $case!(i64);
            $case!(i128);
            $case!(isize);
        };
    }

    /// The round-trip law, at each type's extremes and around zero.
    #[test]
    fn every_integer_label_round_trips() {
        macro_rules! case {
            ($ty:ty) => {{
                for label in [<$ty>::MIN, 0, 1, <$ty>::MAX] {
                    let encoded = label.encode();
                    assert_eq!(
                        <$ty>::decode(&encoded),
                        Some(label),
                        "round trip for {} as {}",
                        label,
                        stringify!($ty)
                    );
                }
            }};
        }
        over_the_family!(case);
    }

    /// A leading `+` parses in Rust and is not a spelling `encode` produces.
    #[test]
    fn an_explicitly_signed_spelling_is_refused() {
        macro_rules! case {
            ($ty:ty) => {{
                assert_eq!(<$ty>::decode("+1"), None, "{}", stringify!($ty));
            }};
        }
        over_the_family!(case);
    }

    /// Leading zeros parse and are not a spelling `encode` produces.
    #[test]
    fn a_zero_padded_spelling_is_refused() {
        macro_rules! case {
            ($ty:ty) => {{
                assert_eq!(<$ty>::decode("01"), None, "{}", stringify!($ty));
                assert_eq!(<$ty>::decode("007"), None, "{}", stringify!($ty));
            }};
        }
        over_the_family!(case);
    }

    /// `-0` parses to zero on the signed types and is not zero's spelling.
    #[test]
    fn negative_zero_is_refused() {
        macro_rules! case {
            ($ty:ty) => {{
                assert_eq!(<$ty>::decode("-0"), None, "{}", stringify!($ty));
            }};
        }
        over_the_family!(case);
    }

    /// Surrounding whitespace is not part of any spelling.
    #[test]
    fn a_padded_spelling_is_refused() {
        macro_rules! case {
            ($ty:ty) => {{
                assert_eq!(<$ty>::decode(" 1"), None, "{}", stringify!($ty));
                assert_eq!(<$ty>::decode("1 "), None, "{}", stringify!($ty));
            }};
        }
        over_the_family!(case);
    }

    /// The empty string is a missing label, and it is reported as undecodable
    /// rather than defaulted to zero.
    #[test]
    fn an_empty_spelling_is_refused() {
        macro_rules! case {
            ($ty:ty) => {{
                assert_eq!(<$ty>::decode(""), None, "{}", stringify!($ty));
                assert_eq!(<$ty>::decode("not-a-number"), None, "{}", stringify!($ty));
            }};
        }
        over_the_family!(case);
    }

    /// A minus sign is a spelling of the signed types only, so the same string
    /// is a label at one type and undecodable at another.
    #[test]
    fn a_negative_spelling_belongs_to_the_signed_types() {
        assert_eq!(i32::decode("-1"), Some(-1));
        assert_eq!(isize::decode("-1"), Some(-1));
        assert_eq!(usize::decode("-1"), None);
        assert_eq!(u8::decode("-1"), None);
    }

    /// A spelling outside a type's range is undecodable at that type, which is
    /// what keeps a widened label from silently narrowing.
    #[test]
    fn a_spelling_out_of_range_is_refused() {
        assert_eq!(u8::decode("256"), None);
        assert_eq!(u8::decode(&255u8.encode()), Some(255));
        assert_eq!(i8::decode("128"), None);
        assert_eq!(i8::decode(&(-128i8).encode()), Some(-128));
    }

    /// Encoding must not depend on run-to-run state; a content address computed
    /// from it has to be reproducible.
    #[test]
    fn encoding_is_deterministic() {
        macro_rules! case {
            ($ty:ty) => {{
                assert_eq!(<$ty>::encode(&7), <$ty>::encode(&7));
                assert_eq!(<$ty>::encode(&7), "7", "{}", stringify!($ty));
            }};
        }
        over_the_family!(case);
    }
}
