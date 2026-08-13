//! Opaque handles to stored records.
//!
//! # One shape, several types
//!
//! Every content address and every store-derived key in this crate is the same
//! string: a fixed alphabetic prefix followed by a full 256-bit BLAKE3 digest in
//! lowercase hex. The shape is generated once, by a macro, so a second address
//! type cannot drift from the first — and the types stay *distinct*, so a term
//! address cannot be handed to a cospan read.
//!
//! The same shape is reused for derived key *columns* — a cospan's canonical
//! key, a weight row's derived record key. That is deliberate: one prefix, one
//! digest, one place to change when either moves. Those columns need no separate
//! format validator, because every load re-derives them and compares, which
//! decides the format along with everything else.

use std::fmt;

/// The prefix every address and derived key carries.
///
/// Two properties depend on it, and both would break if it were dropped:
///
/// - **Rendering is unconditional.** Record key escaping is *value-dependent*:
///   an all-digit key renders wrapped in backticks while a key containing a
///   letter renders bare. A fixed alphabetic prefix puts every address on the
///   same side of that rule, so no caller ever meets a surprise-escaped key.
/// - **The address self-describes its hash.** The digest algorithm is visible in
///   the address itself, so a future change of hash is a change of prefix rather
///   than a silent reinterpretation of existing addresses.
const DIGEST_PREFIX: &str = "b3_";

/// Whether `digest` is exactly 64 lowercase hex characters — a full 256-bit
/// digest.
///
/// Truncation is refused rather than accommodated: a shortened digest weakens
/// collision resistance, and a collision here means two distinct structures
/// silently becoming one stored record.
fn is_full_digest(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'))
}

/// The prefixed BLAKE3 digest of `bytes`.
///
/// This is the derived-key form: the same shape as an address, but for a
/// *column* rather than a record id. Used for keys the store computes and then
/// re-computes on load to compare.
pub(crate) fn digest_key(bytes: &[u8]) -> String {
    format!("{DIGEST_PREFIX}{}", blake3::hash(bytes).to_hex())
}

/// Define a content-address newtype.
///
/// Every address type in this crate is generated here rather than written out,
/// so their validation, prefix, and rendering cannot diverge from one another.
macro_rules! content_address {
    (
        $(#[$meta:meta])*
        $name:ident
    ) => {
        $(#[$meta])*
        ///
        /// # Opaque on purpose
        ///
        /// The concrete encoding is an internal, versioned detail. There is no
        /// public field and no public constructor taking a pre-formatted
        /// string, so callers cannot come to depend on the layout — which
        /// leaves the store free to change digest or prefix later without
        /// breaking them. Callers that need the stored form use [`AsRef<str>`]
        /// or [`Display`](fmt::Display); callers holding a value read back out
        /// of the database use `parse`, which validates.
        ///
        /// Never build assumptions on the *rendered* form of a record id —
        /// escaping depends on the key's own characters. Interpolating an
        /// address into query text is always wrong; bind it as a parameter.
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            /// Build an address from a hex digest.
            ///
            /// Returns `None` unless `digest` is exactly 64 lowercase hex
            /// characters — a full 256-bit digest.
            #[must_use]
            pub fn from_digest(digest: &str) -> Option<Self> {
                if is_full_digest(digest) {
                    Some(Self(format!("{DIGEST_PREFIX}{digest}")))
                } else {
                    None
                }
            }

            /// Recover an address from its stored form.
            ///
            /// Returns `None` for anything that is not a well-formed address,
            /// so a corrupt or foreign record id is rejected at the boundary
            /// rather than propagating inward as a plausible-looking handle.
            #[must_use]
            pub fn parse(raw: &str) -> Option<Self> {
                let digest = raw.strip_prefix(DIGEST_PREFIX)?;
                Self::from_digest(digest)
            }

            /// The stored form, as a string slice.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// The digest portion, without the prefix.
            #[must_use]
            pub fn digest(&self) -> &str {
                self.0.strip_prefix(DIGEST_PREFIX).expect(concat!(
                    "invariant: every ",
                    stringify!($name),
                    " is constructed with the prefix and the field is private"
                ))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

content_address! {
    /// An opaque handle to a stored term.
    ///
    /// The address is a content address: it is derived from the term's
    /// canonical encoding, so identical encodings share an address and
    /// different encodings never collide in practice.
    TermAddr
}

content_address! {
    /// An opaque handle to a stored set of rewrite rules.
    ///
    /// The address is a content address of the rule set's canonical encoding, so
    /// a rule pin — a rule-set address plus the equation indices and
    /// orientations drawn from it — names an exact set of rules rather than
    /// whatever a mutable "current rules" row happens to hold.
    RuleSetAddr
}

content_address! {
    /// An opaque handle to a stored optimizer trace.
    ///
    /// The address is a content address of everything about the run that is
    /// reproducible: its rule set, its endpoints, its cost model, its costs, and
    /// its steps. Recording the same run twice is therefore idempotent, and two
    /// runs differing only in which weighting produced their numbers are
    /// different records — which is the point of `cost_model` being mandatory.
    RunAddr
}

content_address! {
    /// An opaque handle to a stored derivation edge.
    ///
    /// The address is a content address of the tuple the edge *is* — parent,
    /// child, and the run that derived one from the other — which is what makes
    /// writing the edge idempotent without a second uniqueness mechanism.
    DerivationAddr
}

content_address! {
    /// An opaque handle to a stored bus event.
    ///
    /// The address is a content address of the `(stream, seq)` pair, so a
    /// sequence number that was somehow handed out twice collides on the record
    /// id and the second write is refused rather than silently overwriting the
    /// first.
    BusAddr
}

content_address! {
    /// An opaque handle to a stored cospan.
    ///
    /// The address is a content address of the cospan's *presentation* — its
    /// two leg maps and its labelled apex, exactly as the caller built them.
    /// Two presentations of the same morphism therefore have different
    /// addresses; what identifies them as one morphism is the canonical key
    /// (see [`crate::cospan`]), which is a separate, uniquely-indexed column.
    CospanAddr
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn digest_becomes_a_prefixed_address() {
        let addr = TermAddr::from_digest(DIGEST).expect("64 lowercase hex chars is a valid digest");
        assert_eq!(addr.as_str(), format!("b3_{DIGEST}"));
        assert_eq!(addr.to_string(), format!("b3_{DIGEST}"));
        assert_eq!(addr.as_ref() as &str, format!("b3_{DIGEST}"));
        assert_eq!(addr.digest(), DIGEST);
    }

    #[test]
    fn stored_form_round_trips_through_parse() {
        let addr = TermAddr::from_digest(DIGEST).expect("valid digest");
        let reparsed = TermAddr::parse(addr.as_str()).expect("an address parses back");
        assert_eq!(addr, reparsed);
    }

    /// Truncation is the dangerous input: it looks fine and silently weakens
    /// collision resistance, so it must be refused outright.
    #[test]
    fn truncated_digests_are_refused() {
        assert_eq!(TermAddr::from_digest(&DIGEST[..32]), None);
        assert_eq!(TermAddr::from_digest(&DIGEST[..63]), None);
        assert_eq!(TermAddr::from_digest(""), None);
    }

    #[test]
    fn non_hex_and_uppercase_digests_are_refused() {
        assert_eq!(TermAddr::from_digest(&DIGEST.to_uppercase()), None);
        let with_g = format!("g{}", &DIGEST[1..]);
        assert_eq!(TermAddr::from_digest(&with_g), None);
        let with_dash = format!("-{}", &DIGEST[1..]);
        assert_eq!(TermAddr::from_digest(&with_dash), None);
    }

    #[test]
    fn foreign_or_unprefixed_ids_do_not_parse() {
        assert_eq!(TermAddr::parse(DIGEST), None);
        assert_eq!(TermAddr::parse("term:whatever"), None);
        assert_eq!(TermAddr::parse(&format!("sha_{DIGEST}")), None);
        assert_eq!(TermAddr::parse(""), None);
    }

    /// The prefix exists so rendering never depends on the digest's characters.
    /// An all-digit digest is the case that would otherwise render escaped.
    #[test]
    fn all_digit_digests_still_begin_with_the_alphabetic_prefix() {
        let numeric = "0".repeat(64);
        let addr = TermAddr::from_digest(&numeric).expect("all-digit digest is valid hex");
        assert!(addr.as_str().starts_with("b3_"), "{addr}");
        assert!(
            addr.as_str().starts_with(|c: char| c.is_ascii_alphabetic()),
            "{addr}"
        );
    }

    /// The second address type is generated from the same macro, so it must
    /// validate identically — the point of generating it rather than writing it
    /// out is that these cannot drift apart.
    #[test]
    fn cospan_addresses_validate_exactly_as_term_addresses_do() {
        let addr =
            CospanAddr::from_digest(DIGEST).expect("64 lowercase hex chars is a valid digest");
        assert_eq!(addr.as_str(), format!("b3_{DIGEST}"));
        assert_eq!(addr.digest(), DIGEST);
        assert_eq!(CospanAddr::parse(addr.as_str()), Some(addr));

        assert_eq!(CospanAddr::from_digest(&DIGEST[..63]), None);
        assert_eq!(CospanAddr::from_digest(&DIGEST.to_uppercase()), None);
        assert_eq!(CospanAddr::parse(DIGEST), None);
        assert_eq!(CospanAddr::parse(""), None);
    }

    /// Derived key columns share the address shape exactly, so a key can be
    /// parsed as an address (which is how a cospan's record id is recovered).
    #[test]
    fn derived_keys_share_the_address_shape() {
        let key = digest_key(b"whatever");
        assert!(key.starts_with("b3_"), "{key}");
        assert_eq!(key.len(), 3 + 64);
        assert!(CospanAddr::parse(&key).is_some());
        // Deterministic, which is the property a key column lives or dies by.
        assert_eq!(key, digest_key(b"whatever"));
        assert_ne!(key, digest_key(b"something else"));
    }
}
