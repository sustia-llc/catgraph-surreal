//! Opaque handles to stored records.

use std::fmt;

/// The prefix every term address carries.
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
const TERM_ADDR_PREFIX: &str = "b3_";

/// An opaque handle to a stored term.
///
/// The address is a content address: it is derived from the term's canonical
/// encoding, so identical encodings share an address and different encodings
/// never collide in practice.
///
/// # Opaque on purpose
///
/// The concrete encoding is an internal, versioned detail. There is no public
/// field and no public constructor taking a pre-formatted string, so callers
/// cannot come to depend on the layout — which leaves the store free to change
/// digest or prefix later without breaking them. Callers that need the stored
/// form use [`AsRef<str>`] or [`Display`](fmt::Display); callers holding a value
/// read back out of the database use [`TermAddr::parse`], which validates.
///
/// Never build assumptions on the *rendered* form of a record id — escaping
/// depends on the key's own characters. Interpolating an address into query text
/// is always wrong; bind it as a parameter.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TermAddr(String);

impl TermAddr {
    /// Build an address from a hex digest.
    ///
    /// Returns `None` unless `digest` is exactly 64 lowercase hex characters —
    /// a full 256-bit digest. Truncation is refused rather than accommodated:
    /// a shortened digest weakens collision resistance, and a collision here
    /// means two distinct terms silently becoming one stored record.
    #[must_use]
    pub fn from_digest(digest: &str) -> Option<Self> {
        let is_full_length = digest.len() == 64;
        let is_lower_hex = digest
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'));
        if is_full_length && is_lower_hex {
            Some(Self(format!("{TERM_ADDR_PREFIX}{digest}")))
        } else {
            None
        }
    }

    /// Recover an address from its stored form.
    ///
    /// Returns `None` for anything that is not a well-formed address, so a
    /// corrupt or foreign record id is rejected at the boundary rather than
    /// propagating inward as a plausible-looking handle.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        let digest = raw.strip_prefix(TERM_ADDR_PREFIX)?;
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
        self.0.strip_prefix(TERM_ADDR_PREFIX).expect(
            "invariant: every TermAddr is constructed with the prefix and the field is private",
        )
    }
}

impl fmt::Display for TermAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for TermAddr {
    fn as_ref(&self) -> &str {
        &self.0
    }
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
}
