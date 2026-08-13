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

impl LabelCodec for usize {
    fn encode(&self) -> String {
        self.to_string()
    }

    fn decode(raw: &str) -> Option<Self> {
        raw.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usize_labels_round_trip() {
        for label in [0usize, 1, 42, usize::MAX] {
            let encoded = label.encode();
            assert_eq!(
                usize::decode(&encoded),
                Some(label),
                "round trip for {label}"
            );
        }
    }

    #[test]
    fn undecodable_labels_are_rejected_rather_than_defaulted() {
        assert_eq!(usize::decode(""), None);
        assert_eq!(usize::decode("not-a-number"), None);
        assert_eq!(usize::decode("-1"), None);
    }

    /// Encoding must not depend on run-to-run state; a content address computed
    /// from it has to be reproducible.
    #[test]
    fn encoding_is_deterministic() {
        assert_eq!(7usize.encode(), 7usize.encode());
        assert_eq!(7usize.encode(), "7");
    }
}
