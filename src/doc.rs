//! Document encoding for the consumer-shaped tier.
//!
//! # This tier stores shapes the store does not know
//!
//! Everything else in this crate persists a catgraph structure and knows exactly
//! what it is looking at. This tier is the opposite: a consumer hands over its
//! own serde type — a solver state, a novelty-archive entry, a registration
//! manifest — and the store keeps it without depending on the crate that defines
//! it. What it can still do is guarantee that what comes back is what went in,
//! and that is what the digest column is for.
//!
//! # `FLEXIBLE`, and what happens without it
//!
//! The payload column is `object FLEXIBLE`. In a `SCHEMAFULL` table an `object`
//! column without `FLEXIBLE` accepts only the keys the schema declares and
//! refuses any write carrying another — `Found field 'payload.whatever', but no
//! such field exists for table 'document'`. For a tier whose whole purpose is
//! shapes the store does not know, that is not a limitation but a table that
//! cannot hold anything, which is why the round trip is pinned by a test
//! comparing the *full key set* rather than a value or two.
//!
//! (Verified at SurrealDB 3.2.4: the refusal is loud. It is worth stating
//! because the opposite — a silent drop, a document coming back smaller than it
//! went in — would be the far more dangerous behaviour, and is what one might
//! reasonably fear from a schema that declares nothing about the inside of a
//! column.)
//!
//! # This is a JSON lane, and floats are the price
//!
//! A payload crosses through `serde_json::Value`, which has no representation
//! for a non-finite float: `NaN` and the infinities become `null` on the way in,
//! silently and irreversibly. That is a real limit rather than a bug to route
//! around here — the digest is computed *after* the conversion, so a round trip
//! is still exact with respect to what was stored, but what was stored is not
//! what was handed over.
//!
//! Consumers with bit-exactness requirements on float arrays want
//! [`crate::weight`] instead, whose byte lane exists for exactly this and
//! survives an export as well as a load.
//!
//! # Key order is decided here, not by the consumer's type
//!
//! The digest is taken over the payload *after* it becomes a `serde_json::Value`,
//! whose object representation is an ordered map. So a consumer type holding a
//! `HashMap` still digests deterministically: the conversion sorts the keys
//! before anything hashes them. Serializing the consumer type straight to a
//! string would not have that property, and the difference is the whole reason
//! the conversion happens first.

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::addr;
use crate::error::{Result, StoreError};

/// The version tag stored in every document's `codec` column.
///
/// It versions the payload encoding (a JSON object) and the way the digest is
/// derived from it.
pub const DOCUMENT_CODEC: &str = "cgd1";

/// A consumer document as it is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentRecord {
    // Crate-visible so the row conversion can MOVE the payload rather than clone
    // it per write; the public surface stays the getters.
    pub(crate) id: String,
    pub(crate) codec: String,
    pub(crate) kind: String,
    pub(crate) digest: String,
    pub(crate) payload: serde_json::Value,
}

impl DocumentRecord {
    /// Rebuild a record from columns read back out of the database.
    ///
    /// Nothing here is trusted: [`Self::revalidate`] re-derives the digest from
    /// the payload and compares before deserializing anything.
    #[must_use]
    pub(crate) fn from_columns(
        id: String,
        codec: String,
        kind: String,
        digest: String,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            id,
            codec,
            kind,
            digest,
            payload,
        }
    }

    /// The caller-supplied id this document is filed under.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The encoding version this record was written under.
    #[must_use]
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// The caller-supplied class tag.
    ///
    /// Indexed, so "every registration manifest" is a query rather than a scan.
    /// The store attaches no meaning to it beyond that.
    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// The digest of the stored payload.
    ///
    /// Re-derived and compared on every load, and the reason a restore can
    /// verify what it replayed: the database-side guards are all void under
    /// `OPTION IMPORT`, so after an import the only thing that says a manifest
    /// is the manifest that was registered is this column agreeing with its own
    /// payload.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// The stored payload, as the JSON value it round-trips through.
    #[must_use]
    pub fn payload(&self) -> &serde_json::Value {
        &self.payload
    }

    /// Re-derive the digest, then deserialize the payload into the consumer's
    /// type.
    ///
    /// # The order, and why it is fixed
    ///
    /// 1. **Codec.** A record written under an encoding this build does not know
    ///    is refused rather than decoded under the current rules.
    /// 2. **Digest.** Re-derived from the payload and compared. This runs before
    ///    deserialization because it needs nothing but the bytes and it fully
    ///    decides tampering — a payload that was edited in place is rejected at
    ///    hash cost rather than after a consumer type has been built from it.
    /// 3. **Deserialize.** Into `T`, reported as [`StoreError::Codec`] if the
    ///    stored shape is not the one `T` expects.
    ///
    /// The store cannot check anything *about* `T` — only the consumer knows
    /// what a valid one of those is. What it guarantees is that the bytes are
    /// the bytes that were written.
    ///
    /// # Errors
    ///
    /// [`StoreError::TypeMismatch`] for an unknown codec, [`StoreError::Corrupt`]
    /// if the payload does not match its digest, and [`StoreError::Codec`] if it
    /// does not deserialize into `T`.
    pub fn revalidate<T: DeserializeOwned>(&self) -> Result<T> {
        if self.codec != DOCUMENT_CODEC {
            return Err(StoreError::TypeMismatch {
                field: "codec".to_owned(),
                expected: DOCUMENT_CODEC.to_owned(),
                actual: self.codec.clone(),
            });
        }
        let digest = digest_of(&self.payload)?;
        if digest != self.digest {
            return Err(StoreError::Corrupt {
                context: format!("document `{}`", self.id),
                detail: format!(
                    "stored digest `{}` disagrees with the payload's `{digest}`",
                    self.digest
                ),
            });
        }
        Ok(serde_json::from_value(self.payload.clone())?)
    }
}

/// Encode a consumer value into the record that will be stored.
///
/// `id` is a caller-supplied opaque stable string and may not be empty: it is
/// the record key, and identity here genuinely belongs to the caller — the store
/// cannot derive a stable one for a type it does not know. Deriving it, and
/// guaranteeing it is stable across processes and builds, stays with the caller.
/// (Standard-library hash output in particular is not stable across processes
/// and must never become a key.)
///
/// The payload must encode to a JSON **object**. That is what the column's type
/// says, and refusing a bare scalar or array here rather than at the database is
/// what turns a modelling mistake into a message naming the problem.
///
/// # Errors
///
/// [`StoreError::Revalidation`] if `id` is empty, [`StoreError::TypeMismatch`]
/// if the payload is not an object, and [`StoreError::Codec`] if it does not
/// serialize.
pub fn encode<T: Serialize>(id: &str, kind: &str, payload: &T) -> Result<DocumentRecord> {
    if id.is_empty() {
        return Err(StoreError::Revalidation {
            stage: crate::error::RevalidationStage::Check,
            detail: "a document id may not be empty".to_owned(),
        });
    }
    let payload = serde_json::to_value(payload)?;
    if !payload.is_object() {
        return Err(StoreError::TypeMismatch {
            field: "payload".to_owned(),
            expected: "a JSON object".to_owned(),
            actual: json_shape(&payload).to_owned(),
        });
    }
    Ok(DocumentRecord {
        id: id.to_owned(),
        codec: DOCUMENT_CODEC.to_owned(),
        kind: kind.to_owned(),
        digest: digest_of(&payload)?,
        payload,
    })
}

/// The digest of a payload.
///
/// Taken over the payload's canonical JSON rendering, which sorts object keys —
/// see the [module documentation](self) for why that matters even when the
/// consumer's type does not.
///
/// # Errors
///
/// [`StoreError::Codec`] if the value does not render.
pub fn digest_of(payload: &serde_json::Value) -> Result<String> {
    let canonical = serde_json::to_string(payload)?;
    Ok(addr::digest_key(canonical.as_bytes()))
}

/// A short name for a JSON value's shape, for the "expected an object" message.
fn json_shape(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde::Deserialize;

    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct State {
        beliefs: Vec<f64>,
        trial_boundary: usize,
        labels: HashMap<String, i32>,
    }

    fn state() -> State {
        let mut labels = HashMap::new();
        labels.insert("b".to_owned(), 2);
        labels.insert("a".to_owned(), 1);
        labels.insert("c".to_owned(), 3);
        State {
            beliefs: vec![0.25, 0.5, 0.25],
            trial_boundary: 7,
            labels,
        }
    }

    #[test]
    fn a_consumer_value_round_trips() {
        let record = encode("state-1", "solver", &state()).expect("a serde value encodes");
        assert_eq!(record.id(), "state-1");
        assert_eq!(record.kind(), "solver");
        assert_eq!(record.codec(), DOCUMENT_CODEC);
        let loaded: State = record.revalidate().expect("its own payload revalidates");
        assert_eq!(loaded, state());
    }

    /// A `HashMap` in the consumer's type would make a naive digest
    /// non-deterministic. Going through the ordered JSON representation first is
    /// what makes the digest a property of the value rather than of the run.
    #[test]
    fn the_digest_is_deterministic_across_hash_map_orderings() {
        let first = encode("state-1", "solver", &state()).expect("encodes");
        let second = encode("state-1", "solver", &state()).expect("encodes");
        assert_eq!(first.digest(), second.digest());
        assert_eq!(first.payload(), second.payload());
    }

    #[test]
    fn a_tampered_payload_is_corrupt() {
        let record = encode("state-1", "solver", &state()).expect("encodes");
        let mut payload = record.payload().clone();
        payload["trial_boundary"] = serde_json::json!(8);
        let tampered = DocumentRecord::from_columns(
            record.id().to_owned(),
            record.codec().to_owned(),
            record.kind().to_owned(),
            record.digest().to_owned(),
            payload,
        );
        let err = tampered
            .revalidate::<State>()
            .expect_err("an edited payload is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    #[test]
    fn an_unknown_codec_is_refused() {
        let record = encode("state-1", "solver", &state()).expect("encodes");
        let tampered = DocumentRecord::from_columns(
            record.id().to_owned(),
            "cgd99".to_owned(),
            record.kind().to_owned(),
            record.digest().to_owned(),
            record.payload().clone(),
        );
        match tampered.revalidate::<State>() {
            Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "codec"),
            other => panic!("expected a codec mismatch, got {other:?}"),
        }
    }

    /// A payload whose stored shape is not the one the consumer's type expects
    /// is a codec failure, not a silent default.
    #[test]
    fn a_payload_of_the_wrong_shape_does_not_deserialize() {
        let record = encode("x", "k", &serde_json::json!({ "unrelated": true })).expect("encodes");
        let err = record
            .revalidate::<State>()
            .expect_err("an unrelated shape is not a State");
        assert!(matches!(err, StoreError::Codec(_)), "{err}");
    }

    #[test]
    fn a_non_object_payload_is_refused_with_its_shape_named() {
        for payload in [
            serde_json::json!(1),
            serde_json::json!("text"),
            serde_json::json!([1, 2, 3]),
            serde_json::json!(null),
        ] {
            match encode("x", "k", &payload) {
                Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "payload"),
                other => panic!("expected a payload shape mismatch, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_empty_id_is_refused() {
        let err = encode("", "k", &state()).expect_err("an empty id is not an id");
        assert!(matches!(err, StoreError::Revalidation { .. }), "{err}");
    }

    /// The JSON lane's honest limit, pinned so nobody discovers it in
    /// production: a non-finite float becomes `null` on the way in. The weight
    /// tier's byte lane is the answer, not a workaround here.
    #[test]
    fn non_finite_floats_become_null_on_this_lane() {
        let record = encode("x", "k", &serde_json::json!({ "v": 1.0 })).expect("encodes");
        assert_eq!(record.payload()["v"], serde_json::json!(1.0));

        #[derive(Serialize)]
        struct Awkward {
            v: f64,
        }
        let record = encode("y", "k", &Awkward { v: f64::NAN }).expect("encodes");
        assert!(
            record.payload()["v"].is_null(),
            "the JSON lane cannot hold a NaN: {:?}",
            record.payload()
        );
    }

    /// Nested shapes have to survive intact, digest included — the column is
    /// `FLEXIBLE` precisely so the store never sees the inside of one.
    #[test]
    fn deeply_nested_unknown_shapes_survive() {
        let payload = serde_json::json!({
            "outer": { "inner": [1, 2, { "deep": "value" }], "flag": true },
            "empty": {},
            "list": [],
        });
        let record = encode("x", "k", &payload).expect("encodes");
        let loaded: serde_json::Value = record.revalidate().expect("revalidates");
        assert_eq!(loaded, payload);
    }
}
