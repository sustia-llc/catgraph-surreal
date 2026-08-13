//! Weight encoding: IEEE-754 coordinates on a byte lane.
//!
//! # Why bytes and not a float array
//!
//! SurrealDB's native value layer does round-trip non-finite floats bit-exactly
//! — `NaN` payloads, `±∞`, and `-0.0` all survive a store and a load. The
//! *export* lane does not: writing a dump renders every `NaN`, whatever its
//! sign or payload, as the bare literal `NaN`. So a checkpoint of a native float
//! column is lossy precisely where a training run is most likely to want the
//! evidence — the moment the weights went non-finite.
//!
//! Storing the raw IEEE-754 bytes sidesteps both the export lane and the
//! serde-JSON lane (which destroys non-finites outright, turning them into
//! `null` on write and failing to read that back). Bytes export as `b"<HEX>"`
//! and come back byte-identical.
//!
//! Two further properties follow, and both matter:
//!
//! - **Bytes are index-safe.** Float index keys go through a lexicographic
//!   decimal encoding that collapses `±NaN` onto one key and `-0.0` onto `0.0`,
//!   so a float column can never be a key where binary exactness matters. Byte
//!   index keys are the raw byte order.
//! - **Equality is bit equality.** Two coordinate vectors that differ only in a
//!   `NaN` payload or a sign of zero are different documents here, which is what
//!   a "did this checkpoint change?" question needs.
//!
//! The encoding is **little-endian**, eight bytes per coordinate, in coordinate
//! order. `dim` is stored beside it and the two are checked against each other
//! on every load.
//!
//! # `Vec<u8>` is not the SDK's byte type
//!
//! The field type on the Rust side must be [`surrealdb::types::Bytes`].
//! `Vec<u8>` deliberately carries no `SurrealValue` implementation, so that
//! choosing between "an array of small integers" and "binary data" has to be
//! made explicitly rather than by inference.
//!
//! # What the store checks, and what stays with the caller
//!
//! `RModule`'s own documentation is explicit that deserialization checks
//! nothing: a loaded module's `dim()` is whatever the payload said, and only
//! `add` rejects a dimension mismatch. This module closes the half it can see —
//! a loaded module's dimension always equals its stored `dim` column, because
//! the byte length is checked against it before the coordinates are decoded.
//! The half it cannot see stays with the caller: whether that dimension is the
//! one the *architecture* expects is a question only the caller can ask, and it
//! should ask it once, at its own entry point.

use catgraph_dl::para::RModule;

use crate::addr;
use crate::error::{Result, StoreError};
use crate::schema::WEIGHT_TABLE;

/// The version tag stored in every weight row's `codec` column.
///
/// It versions the coordinate encoding (little-endian IEEE-754, eight bytes per
/// coordinate) and the way the record key is derived from `(genome, gen_key)`.
pub const WEIGHT_CODEC: &str = "cgw1";

/// Bytes per stored coordinate.
const COORDINATE_WIDTH: usize = 8;

/// The record key a `(genome, gen_key)` pair is filed under.
///
/// The pair is encoded as JSON before hashing rather than concatenated, and that
/// is not decoration: concatenation is ambiguous — `("ab", "c")` and
/// `("a", "bc")` would produce one key for two different weight slots. JSON
/// quotes and escapes both halves, so the encoding is injective.
///
/// The key shape is an internal, versioned detail; the public identity of a
/// weight row is the `(genome, gen_key)` pair, which is what the unique index
/// enforces. It is exposed for one reason — a restore has to **re-verify record
/// ids rather than trust the replay**, since the id `ASSERT` is skipped under
/// `OPTION IMPORT` — and ordinary reads never need it, because every load
/// re-derives and compares the key.
///
/// # Errors
///
/// [`StoreError::Codec`] if the pair does not encode.
pub fn record_key(genome: &str, gen_key: &str) -> Result<String> {
    let encoding = serde_json::to_string(&(genome, gen_key))?;
    Ok(addr::digest_key(encoding.as_bytes()))
}

/// A weight vector as it is stored.
///
/// `dim` is `i64` rather than `usize` because this type mirrors the row; the
/// conversion happens once, in [`encode`], where it is checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightRecord {
    // Crate-visible so the row conversion can MOVE the coordinate buffer rather
    // than memcpy it a second time per write; the public surface stays the
    // getters.
    pub(crate) key: String,
    pub(crate) codec: String,
    pub(crate) genome: String,
    pub(crate) gen_key: String,
    pub(crate) dim: i64,
    pub(crate) coordinates: Vec<u8>,
    pub(crate) finite: bool,
}

impl WeightRecord {
    /// Rebuild a record from columns read back out of the database.
    ///
    /// Nothing here is trusted: [`Self::revalidate`] re-derives the record key,
    /// checks the coordinate width against `dim`, and re-derives `finite`.
    #[must_use]
    pub(crate) fn from_columns(
        key: String,
        codec: String,
        genome: String,
        gen_key: String,
        dim: i64,
        coordinates: Vec<u8>,
        finite: bool,
    ) -> Self {
        Self {
            key,
            codec,
            genome,
            gen_key,
            dim,
            coordinates,
            finite,
        }
    }

    /// The record key this row is filed under.
    ///
    /// Derived from the `(genome, gen_key)` pair — see [`record_key`] — and
    /// re-derived and compared on every load, so it is never trusted as read.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The encoding version this record was written under.
    #[must_use]
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// The caller-supplied genome key.
    #[must_use]
    pub fn genome(&self) -> &str {
        &self.genome
    }

    /// The caller-supplied per-genome key.
    #[must_use]
    pub fn gen_key(&self) -> &str {
        &self.gen_key
    }

    /// The number of coordinates.
    #[must_use]
    pub fn dim(&self) -> i64 {
        self.dim
    }

    /// The raw coordinate bytes: `dim × 8`, little-endian, in coordinate order.
    #[must_use]
    pub fn coordinates(&self) -> &[u8] {
        &self.coordinates
    }

    /// Whether every coordinate is finite.
    ///
    /// Derived, indexed, and re-derived on load. It exists so "which checkpoints
    /// went non-finite?" is a query rather than a scan — the store takes no view
    /// on whether a non-finite weight is acceptable, which is a question about
    /// the model, not about storage.
    #[must_use]
    pub fn finite(&self) -> bool {
        self.finite
    }

    /// Re-derive everything from the stored columns and hand back a module whose
    /// dimension is the one the row claims.
    ///
    /// # The order, and why it is fixed
    ///
    /// 1. **Codec.** A row written under an encoding this build does not know is
    ///    refused rather than decoded under the current rules.
    /// 2. **Record key.** Re-derived from `(genome, gen_key)` and compared
    ///    against the id the row was filed under. Cheap, and it fully decides
    ///    whether the row is where its own key says it belongs — a row filed
    ///    elsewhere was written by something other than this store.
    /// 3. **Width.** `dim` must be non-negative and the coordinate bytes must be
    ///    exactly `dim × 8` long. This is reported as
    ///    [`StoreError::TypeMismatch`] rather than
    ///    [`StoreError::Corrupt`]: the failure is a
    ///    *shape* disagreement between two columns, which is what that variant
    ///    describes, and its `expected`/`actual` fields say exactly how the two
    ///    disagreed. `Corrupt` is reserved here for an invariant that a
    ///    constructor was supposed to hold.
    /// 4. **Coordinates.** Decoded little-endian, eight bytes at a time.
    /// 5. **Finiteness.** Re-derived and compared against the stored flag, which
    ///    is otherwise a claim nobody checked.
    ///
    /// # Errors
    ///
    /// [`StoreError::TypeMismatch`] for an unknown codec or a width
    /// disagreement, [`StoreError::Corrupt`] for a row filed under the wrong key
    /// or carrying a `finite` flag that does not match its coordinates.
    pub fn revalidate(&self) -> Result<RModule<f64>> {
        if self.codec != WEIGHT_CODEC {
            return Err(StoreError::TypeMismatch {
                field: "codec".to_owned(),
                expected: WEIGHT_CODEC.to_owned(),
                actual: self.codec.clone(),
            });
        }

        let key = record_key(&self.genome, &self.gen_key)?;
        if key != self.key {
            return Err(corrupt(&format!(
                "record `{}` holds the key pair filed under `{key}`",
                self.key
            )));
        }

        let dim = usize::try_from(self.dim).map_err(|_| StoreError::TypeMismatch {
            field: "dim".to_owned(),
            expected: "a non-negative dimension".to_owned(),
            actual: self.dim.to_string(),
        })?;
        let expected_width =
            dim.checked_mul(COORDINATE_WIDTH)
                .ok_or_else(|| StoreError::TypeMismatch {
                    field: "dim".to_owned(),
                    expected: "a dimension whose byte width is representable".to_owned(),
                    actual: self.dim.to_string(),
                })?;
        if self.coordinates.len() != expected_width {
            return Err(StoreError::TypeMismatch {
                field: "coordinates".to_owned(),
                expected: format!("{expected_width} bytes for dim {dim}"),
                actual: format!("{} bytes", self.coordinates.len()),
            });
        }

        let coordinates = decode_coordinates(&self.coordinates);
        let finite = all_finite(&coordinates);
        if finite != self.finite {
            return Err(corrupt(&format!(
                "stored `finite` is {}, but the coordinates are {}finite",
                self.finite,
                if finite { "" } else { "not " }
            )));
        }

        Ok(RModule::new(coordinates))
    }
}

/// Encode a weight vector into the record that will be stored.
///
/// # Errors
///
/// [`StoreError::Codec`] if the key pair does not encode, and
/// [`StoreError::TypeMismatch`] in the theoretical case of a dimension too large
/// for the database's integer column.
pub fn encode(genome: &str, gen_key: &str, weights: &RModule<f64>) -> Result<WeightRecord> {
    let coordinates = weights.as_slice();
    Ok(WeightRecord {
        key: record_key(genome, gen_key)?,
        codec: WEIGHT_CODEC.to_owned(),
        genome: genome.to_owned(),
        gen_key: gen_key.to_owned(),
        dim: to_column("dim", weights.dim())?,
        coordinates: encode_coordinates(coordinates),
        finite: all_finite(coordinates),
    })
}

/// Coordinates to bytes: little-endian, eight per coordinate, in order.
fn encode_coordinates(coordinates: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(coordinates.len() * COORDINATE_WIDTH);
    for coordinate in coordinates {
        bytes.extend_from_slice(&coordinate.to_le_bytes());
    }
    bytes
}

/// Bytes back to coordinates.
///
/// The caller has already checked that the length is a whole number of
/// coordinates, so any trailing partial coordinate is dropped rather than
/// guessed at — there is no reachable path that produces one.
fn decode_coordinates(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(COORDINATE_WIDTH)
        .map(|chunk| {
            let mut word = [0u8; COORDINATE_WIDTH];
            word.copy_from_slice(chunk);
            f64::from_le_bytes(word)
        })
        .collect()
}

/// Whether every coordinate is finite. Vacuously true for the zero-dimensional
/// module.
fn all_finite(coordinates: &[f64]) -> bool {
    coordinates.iter().all(|c| c.is_finite())
}

/// Narrow a dimension into the database's integer lane.
fn to_column(field: &str, value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| StoreError::TypeMismatch {
        field: field.to_owned(),
        expected: "int".to_owned(),
        actual: format!("{value} (too large for a 64-bit signed integer)"),
    })
}

/// The uniform corrupt-document error for this tier.
fn corrupt(detail: &str) -> StoreError {
    StoreError::Corrupt {
        context: WEIGHT_TABLE.to_owned(),
        detail: detail.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `NaN` with a payload that a naive round-trip would flatten, and a
    /// negative zero, which compares equal to positive zero but is a different
    /// bit pattern.
    fn awkward() -> RModule<f64> {
        RModule::new(vec![
            f64::from_bits(0x7FF8_0000_DEAD_BEEF),
            -0.0,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.5,
        ])
    }

    fn bits(module: &RModule<f64>) -> Vec<u64> {
        module.as_slice().iter().map(|c| c.to_bits()).collect()
    }

    #[test]
    fn coordinates_round_trip_bit_exactly() {
        let weights = awkward();
        let record = encode("genome", "layer-0", &weights).expect("a module encodes");
        let loaded = record.revalidate().expect("its own encoding revalidates");
        assert_eq!(bits(&loaded), bits(&weights));
    }

    /// The point of the byte lane, stated as a bit pattern: the `NaN` payload
    /// and the sign of zero both survive, and neither is visible to `==`.
    #[test]
    fn nan_payloads_and_signed_zeros_survive() {
        let record = encode("g", "k", &awkward()).expect("encodes");
        let loaded = record.revalidate().expect("revalidates");
        assert_eq!(loaded.as_slice()[0].to_bits(), 0x7FF8_0000_DEAD_BEEF);
        assert_eq!(loaded.as_slice()[1].to_bits(), (-0.0f64).to_bits());
        assert_eq!(loaded.as_slice()[2].to_bits(), 0.0f64.to_bits());
        assert_ne!(
            loaded.as_slice()[1].to_bits(),
            loaded.as_slice()[2].to_bits()
        );
    }

    #[test]
    fn the_encoding_is_little_endian_and_eight_bytes_wide() {
        let record = encode("g", "k", &RModule::new(vec![1.0f64])).expect("encodes");
        assert_eq!(record.coordinates().len(), COORDINATE_WIDTH);
        assert_eq!(record.coordinates(), 1.0f64.to_le_bytes());
    }

    #[test]
    fn the_zero_dimensional_module_round_trips() {
        let record = encode("g", "k", &RModule::new(Vec::new())).expect("encodes");
        assert_eq!(record.dim(), 0);
        assert!(record.coordinates().is_empty());
        assert!(record.finite(), "an empty module is vacuously finite");
        let loaded = record.revalidate().expect("revalidates");
        assert_eq!(loaded.dim(), 0);
    }

    #[test]
    fn the_finite_flag_tracks_the_coordinates() {
        assert!(
            encode("g", "k", &RModule::new(vec![1.0, -2.5]))
                .expect("encodes")
                .finite()
        );
        assert!(
            !encode("g", "k", &RModule::new(vec![1.0, f64::NAN]))
                .expect("encodes")
                .finite()
        );
        assert!(
            !encode("g", "k", &RModule::new(vec![f64::INFINITY]))
                .expect("encodes")
                .finite()
        );
    }

    /// Concatenating the key pair would make these two rows collide. They must
    /// not.
    #[test]
    fn the_key_pair_encoding_is_unambiguous() {
        let left = record_key("ab", "c").expect("a key pair encodes");
        let right = record_key("a", "bc").expect("a key pair encodes");
        assert_ne!(left, right);
        assert_eq!(left, record_key("ab", "c").expect("deterministic"));
    }

    #[test]
    fn a_width_that_disagrees_with_dim_is_a_type_mismatch() {
        let record = encode("g", "k", &RModule::new(vec![1.0, 2.0])).expect("encodes");
        let mut coordinates = record.coordinates().to_vec();
        coordinates.pop();
        let tampered = WeightRecord::from_columns(
            record.key().to_owned(),
            record.codec().to_owned(),
            record.genome().to_owned(),
            record.gen_key().to_owned(),
            record.dim(),
            coordinates,
            record.finite(),
        );
        match tampered.revalidate() {
            Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "coordinates"),
            other => panic!("expected a coordinate width mismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_negative_dimension_is_a_type_mismatch() {
        let record = encode("g", "k", &RModule::new(vec![1.0])).expect("encodes");
        let tampered = WeightRecord::from_columns(
            record.key().to_owned(),
            record.codec().to_owned(),
            record.genome().to_owned(),
            record.gen_key().to_owned(),
            -1,
            record.coordinates().to_vec(),
            record.finite(),
        );
        match tampered.revalidate() {
            Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "dim"),
            other => panic!("expected a dimension mismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_lying_finite_flag_is_corrupt() {
        let record = encode("g", "k", &RModule::new(vec![f64::NAN])).expect("encodes");
        let tampered = WeightRecord::from_columns(
            record.key().to_owned(),
            record.codec().to_owned(),
            record.genome().to_owned(),
            record.gen_key().to_owned(),
            record.dim(),
            record.coordinates().to_vec(),
            true,
        );
        let err = tampered
            .revalidate()
            .expect_err("a `finite` flag that lies is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    /// A row filed under a key that is not its own pair's — written by
    /// something other than this store, or moved by hand.
    #[test]
    fn a_row_filed_under_the_wrong_key_is_corrupt() {
        let record = encode("g", "k", &RModule::new(vec![1.0])).expect("encodes");
        let tampered = WeightRecord::from_columns(
            record.key().to_owned(),
            record.codec().to_owned(),
            "someone else".to_owned(),
            record.gen_key().to_owned(),
            record.dim(),
            record.coordinates().to_vec(),
            record.finite(),
        );
        let err = tampered
            .revalidate()
            .expect_err("a misfiled row is corrupt");
        assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    }

    #[test]
    fn an_unknown_codec_is_refused() {
        let record = encode("g", "k", &RModule::new(vec![1.0])).expect("encodes");
        let tampered = WeightRecord::from_columns(
            record.key().to_owned(),
            "cgw99".to_owned(),
            record.genome().to_owned(),
            record.gen_key().to_owned(),
            record.dim(),
            record.coordinates().to_vec(),
            record.finite(),
        );
        match tampered.revalidate() {
            Err(StoreError::TypeMismatch { field, .. }) => assert_eq!(field, "codec"),
            other => panic!("expected a codec mismatch, got {other:?}"),
        }
    }
}
