//! Bus event encoding, and the shape of a stored notification.
//!
//! # A bus event is a durable row first and a notification second
//!
//! Live queries on this database are a **wakeup**, not a delivery mechanism.
//! Notifications are best-effort and at-most-once: they flush only after a
//! commit, they are never replayed, nothing detects a gap, and each one is
//! dispatched on its own task, so their relative order is not structurally
//! guaranteed. A consumer that treats an arriving notification as the event has
//! built something that silently loses events under load and after any
//! disconnect.
//!
//! So the row is the event. The notification says "look again", and everything
//! about correctness — what happened, in what order, and whether anything was
//! missed — is decided by reading rows.
//!
//! # `seq` is what makes a gap detectable
//!
//! Each stream numbers its events from zero, and the numbers are contiguous by
//! construction: a publisher allocates from a per-stream counter inside the same
//! transaction that writes the event. A reader that sees 7 after 5 therefore
//! knows something is missing — which is the one thing change-capture retention
//! will not tell it, since expiry is silent.
//!
//! The record id is the digest of `(stream, seq)`, which turns a duplicated
//! sequence number into a refused write rather than an overwrite: the second
//! event to claim a number collides on the id.

use serde::Serialize;
use serde::de::DeserializeOwned;

use surrealdb::types::{Kind, RecordId, RecordIdKey, SerializationError, SurrealValue, Value};

use crate::addr::BusAddr;
use crate::addr::digest_key;
use crate::error::{Result, RevalidationStage, StoreError};
use crate::schema::BUS_TABLE;

/// The version tag stored in every bus event's `codec` column.
pub const BUS_CODEC: &str = "cgb1";

/// A published event, as it is stored and as it is delivered.
///
/// The payload stays a `serde_json::Value` rather than a type parameter, because
/// one bus routinely carries several kinds of event and a reader draining a
/// batch should not have to know which stream produced which shape before it can
/// look. [`Self::decode`] is where a consumer commits to a type, per event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BusEvent {
    pub(crate) addr: BusAddr,
    pub(crate) codec: String,
    pub(crate) stream: String,
    pub(crate) seq: i64,
    pub(crate) payload: serde_json::Value,
}

impl BusEvent {
    /// Rebuild an event from columns read back out of the database, or off a
    /// live notification.
    #[must_use]
    pub(crate) fn from_columns(
        addr: BusAddr,
        codec: String,
        stream: String,
        seq: i64,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            addr,
            codec,
            stream,
            seq,
            payload,
        }
    }

    /// The event's content address, which is also its record id.
    #[must_use]
    pub fn addr(&self) -> &BusAddr {
        &self.addr
    }

    /// The encoding version this event was written under.
    #[must_use]
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// The stream it was published to.
    #[must_use]
    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// Its sequence number within that stream, counting from zero.
    #[must_use]
    pub fn seq(&self) -> i64 {
        self.seq
    }

    /// The payload, as the JSON value it round-trips through.
    #[must_use]
    pub fn payload(&self) -> &serde_json::Value {
        &self.payload
    }

    /// Deserialize the payload into a consumer type.
    ///
    /// # Errors
    ///
    /// [`StoreError::Codec`] if the stored shape is not the one `T` expects.
    pub fn decode<T: DeserializeOwned>(&self) -> Result<T> {
        Ok(serde_json::from_value(self.payload.clone())?)
    }

    /// Re-derive the address from `(stream, seq)` and compare it against the id
    /// the event was filed under.
    ///
    /// Cheap, and it decides the one thing that matters about a bus row's
    /// identity: that its sequence number is the one it is filed under. A row
    /// whose `seq` was edited would otherwise make a reader's contiguity check
    /// report a gap that never happened, or hide one that did.
    ///
    /// # Errors
    ///
    /// [`StoreError::TypeMismatch`] for an unknown codec or a negative sequence
    /// number, and [`StoreError::Corrupt`] for an event filed under an id that
    /// is not its own.
    pub fn revalidate(&self) -> Result<()> {
        if self.codec != BUS_CODEC {
            return Err(StoreError::TypeMismatch {
                field: "codec".to_owned(),
                expected: BUS_CODEC.to_owned(),
                actual: self.codec.clone(),
            });
        }
        if self.seq < 0 {
            return Err(StoreError::TypeMismatch {
                field: "seq".to_owned(),
                expected: "a non-negative sequence number".to_owned(),
                actual: self.seq.to_string(),
            });
        }
        let addr = event_address(&self.stream, self.seq);
        if addr != self.addr {
            return Err(StoreError::Corrupt {
                context: format!("{BUS_TABLE}:{}", self.addr),
                detail: format!("address of `({}, {})` is `{addr}`", self.stream, self.seq),
            });
        }
        Ok(())
    }
}

/// One row of the `bus` table, in the shape the SDK's value layer speaks.
///
/// Private, and the only reason [`BusEvent`] does not simply derive
/// `SurrealValue`: an event's public identity is a [`BusAddr`], while a row's is
/// a record id, and the conversion between them can fail. The derive has nowhere
/// to put that.
#[derive(Debug, Clone, SurrealValue)]
struct BusRow {
    id: RecordId,
    codec: String,
    stream: String,
    seq: i64,
    payload: serde_json::Value,
}

/// The value layer's view of an event.
///
/// **Structural only.** `from_value` recovers the address from the record id and
/// stops there; it does not check that the address agrees with `(stream, seq)`.
/// That check is [`BusEvent::revalidate`], and it is separate on purpose: the
/// durable read paths call it and report a mismatch as
/// [`StoreError::Corrupt`], with the typed detail intact, which a conversion
/// error returning the SDK's error type could not carry.
///
/// A live notification is *not* revalidated by arriving. It is a wakeup, and
/// nothing about it is load-bearing; a consumer treating a notification's
/// payload as truth should call [`BusEvent::revalidate`] itself — or better,
/// read the row.
impl SurrealValue for BusEvent {
    fn kind_of() -> Kind {
        BusRow::kind_of()
    }

    fn is_value(value: &Value) -> bool {
        BusRow::is_value(value)
    }

    fn into_value(self) -> Value {
        BusRow {
            id: RecordId::new(BUS_TABLE, self.addr.as_str()),
            codec: self.codec,
            stream: self.stream,
            seq: self.seq,
            payload: self.payload,
        }
        .into_value()
    }

    fn from_value(value: Value) -> std::result::Result<Self, surrealdb::Error> {
        let row = BusRow::from_value(value)?;
        let RecordIdKey::String(key) = &row.id.key else {
            return Err(surrealdb::Error::serialization(
                "a bus row's record id key is not a string".to_owned(),
                SerializationError::Deserialization,
            ));
        };
        let addr = BusAddr::parse(key).ok_or_else(|| {
            surrealdb::Error::serialization(
                format!("`{key}` is not a bus address"),
                SerializationError::Deserialization,
            )
        })?;
        Ok(Self::from_columns(
            addr,
            row.codec,
            row.stream,
            row.seq,
            row.payload,
        ))
    }
}

/// A payload prepared for publication, before a sequence number is allocated.
///
/// The split exists because the number is assigned by the database: everything
/// that can be validated up front — that the stream is named, that the payload
/// is an object, that it serializes — happens here, so a publish that is going
/// to fail does so before it takes a lock on the stream's counter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingEvent {
    stream: String,
    payload: serde_json::Value,
}

impl PendingEvent {
    /// The stream it will be published to.
    #[must_use]
    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// The payload, as the JSON object it round-trips through.
    #[must_use]
    pub fn payload(&self) -> &serde_json::Value {
        &self.payload
    }
}

/// Prepare a payload for publication to `stream`.
///
/// The payload must encode to a JSON **object**, matching the column's type — a
/// bare scalar is refused here rather than at the database, where the message
/// would name a type instead of the mistake.
///
/// This is the same lane [`crate::doc`] uses, with the same limit: a non-finite
/// float becomes `null` on the way in. Bus payloads are notifications, so that
/// is rarely the wrong trade; a consumer that needs bit-exact floats should
/// publish a *reference* to a weight row rather than the numbers themselves.
///
/// # Errors
///
/// [`StoreError::Revalidation`] if `stream` is empty,
/// [`StoreError::TypeMismatch`] if the payload is not an object, and
/// [`StoreError::Codec`] if it does not serialize.
pub fn prepare<T: Serialize>(stream: &str, payload: &T) -> Result<PendingEvent> {
    if stream.is_empty() {
        return Err(StoreError::Revalidation {
            stage: RevalidationStage::Check,
            detail: "a stream name may not be empty".to_owned(),
        });
    }
    let payload = serde_json::to_value(payload)?;
    if !payload.is_object() {
        return Err(StoreError::TypeMismatch {
            field: "payload".to_owned(),
            expected: "a JSON object".to_owned(),
            actual: "a value that is not an object".to_owned(),
        });
    }
    Ok(PendingEvent {
        stream: stream.to_owned(),
        payload,
    })
}

/// The record id an event on `stream` with sequence number `seq` is filed under.
///
/// A digest of the pair rather than the pair itself, for the same reason every
/// other key in this crate is: it renders unconditionally bare, it is one fixed
/// width, and it cannot be confused with a caller-supplied string. The pair is
/// encoded as JSON before hashing rather than concatenated, because
/// concatenation is ambiguous — `("ab", 1)` and `("a", 11)` would be one key if
/// the parts were run together.
#[must_use]
pub fn event_address(stream: &str, seq: i64) -> BusAddr {
    let canonical = serde_json::to_string(&(BUS_CODEC, stream, seq))
        .expect("invariant: a string-and-integer tuple always encodes");
    let key = digest_key(canonical.as_bytes());
    BusAddr::parse(&key).expect("invariant: a derived key has the address shape")
}

/// The record key a stream's sequence allocator is filed under.
///
/// Derived rather than the raw stream name, so a stream named with something the
/// record-id layer would escape still lands on a plain key.
#[must_use]
pub fn allocator_key(stream: &str) -> String {
    let canonical = serde_json::to_string(&("bus_seq", stream))
        .expect("invariant: a string tuple always encodes");
    digest_key(canonical.as_bytes())
}

/// The record key a consumer's catch-up cursor is filed under.
#[must_use]
pub fn cursor_key(consumer: &str) -> String {
    let canonical = serde_json::to_string(&("bus_mark", consumer))
        .expect("invariant: a string tuple always encodes");
    digest_key(canonical.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_event_address_is_a_function_of_the_pair() {
        let addr = event_address("goals", 3);
        assert_eq!(addr, event_address("goals", 3));
        assert_ne!(addr, event_address("goals", 4));
        assert_ne!(addr, event_address("ticks", 3));
    }

    /// Concatenating the pair would make these two collide. They must not.
    #[test]
    fn the_pair_encoding_is_unambiguous() {
        assert_ne!(event_address("ab", 1), event_address("a", 11));
    }

    #[test]
    fn an_event_revalidates_against_its_own_pair() {
        let event = BusEvent::from_columns(
            event_address("goals", 3),
            BUS_CODEC.to_owned(),
            "goals".to_owned(),
            3,
            serde_json::json!({ "reached": true }),
        );
        event.revalidate().expect("its own pair revalidates");
        assert_eq!(event.stream(), "goals");
        assert_eq!(event.seq(), 3);

        let decoded: serde_json::Value = event.decode().expect("the payload decodes");
        assert_eq!(decoded, serde_json::json!({ "reached": true }));
    }

    /// The check that matters: an edited sequence number would make a reader's
    /// contiguity assertion report a gap that never happened, or miss one that
    /// did.
    #[test]
    fn an_event_filed_under_another_pairs_id_is_corrupt() {
        let event = BusEvent::from_columns(
            event_address("goals", 3),
            BUS_CODEC.to_owned(),
            "goals".to_owned(),
            4,
            serde_json::json!({}),
        );
        let err = event
            .revalidate()
            .expect_err("a rewritten sequence number is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    #[test]
    fn a_negative_sequence_number_is_refused() {
        let event = BusEvent::from_columns(
            event_address("goals", -1),
            BUS_CODEC.to_owned(),
            "goals".to_owned(),
            -1,
            serde_json::json!({}),
        );
        match event.revalidate() {
            Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "seq"),
            other => panic!("expected a sequence-number mismatch, got {other:?}"),
        }
    }

    #[test]
    fn an_unknown_codec_is_refused() {
        let event = BusEvent::from_columns(
            event_address("goals", 0),
            "cgb99".to_owned(),
            "goals".to_owned(),
            0,
            serde_json::json!({}),
        );
        match event.revalidate() {
            Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "codec"),
            other => panic!("expected a codec mismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_payload_that_is_not_an_object_is_refused() {
        match prepare("goals", &serde_json::json!(42)) {
            Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "payload"),
            other => panic!("expected a payload shape mismatch, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_stream_name_is_refused() {
        let err = prepare("", &serde_json::json!({})).expect_err("a stream needs a name");
        assert!(matches!(err, StoreError::Revalidation { .. }), "{err}");
    }

    /// Allocator and cursor keys share the address shape, so neither can
    /// surprise a caller with an escaped record id.
    #[test]
    fn derived_keys_render_unconditionally_bare() {
        for key in [allocator_key("goals"), cursor_key("worker-1")] {
            assert!(key.starts_with("b3_"), "{key}");
            assert!(key.starts_with(|c: char| c.is_ascii_alphabetic()), "{key}");
        }
        assert_ne!(allocator_key("a"), allocator_key("b"));
        assert_ne!(cursor_key("a"), cursor_key("b"));
        // The two namespaces are separate: one stream and one consumer with the
        // same name are different rows in different tables, and must not share a
        // derived key either.
        assert_ne!(allocator_key("x"), cursor_key("x"));
    }
}
