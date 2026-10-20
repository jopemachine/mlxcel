// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Load-time guards against checkpoints whose config and tensors disagree.
//!
//! Each hostile case goes through the real loader and is paired with the
//! honest fixture loading through the same entry point.

use serde_json::json;

use super::parity_tests::{config, fixture, weights};
use super::ple::{NgramHasher, PleLayer};

/// The heads must fit the table the checkpoint stores: one row short of the
/// derived size is refused, the exact size loads.
#[test]
fn ngram_prime_search_stops_at_the_stored_table() {
    let fixture = fixture("ple");
    let config = config(&fixture, None);
    let unbounded = NgramHasher::new(&config, 0, u64::MAX).unwrap();
    let rows = (unbounded.head_offsets.last().unwrap() + unbounded.head_vocab_sizes.last().unwrap())
        as u64;

    assert_eq!(NgramHasher::new(&config, 0, rows).unwrap(), unbounded);
    let error = NgramHasher::new(&config, 0, rows - 1).unwrap_err();
    assert!(error.contains("rows the checkpoint stores"), "{error}");
}

/// A base the stored table cannot hold is refused before any candidate past
/// the table is tested for primality; near 2^62 each test is a trial
/// division up to 2^31.
#[test]
fn ple_layer_refuses_an_ngram_base_the_stored_table_cannot_hold() {
    let fixture = fixture("ple");
    let weights = weights(&fixture);
    PleLayer::from_weights(&weights, &config(&fixture, None), "ple", 0)
        .expect("the fixture's own n-gram base loads");
    for base in [10_000u64, 1 << 62] {
        let config = config(&fixture, Some(&json!({ "ngram_vocab_size_base": base })));
        let error = PleLayer::from_weights(&weights, &config, "ple", 0)
            .err()
            .unwrap_or_else(|| panic!("ngram_vocab_size_base {base} must be refused"));
        assert!(
            error.contains("rows the checkpoint stores"),
            "base {base}: {error}"
        );
    }
}

/// A stored hash tensor is checked against the config by value, so its shape
/// is checked first: reading back an arbitrarily large tensor is not needed
/// to refuse it.
#[test]
fn stored_hash_tensors_of_the_wrong_shape_are_refused() {
    let fixture = fixture("ple");
    let config = config(&fixture, None);
    let hasher = NgramHasher::new(&config, 0, u64::MAX).unwrap();
    let multipliers = &hasher.multipliers;
    let stored = |shape: &[i32]| {
        let mut weights = weights(&fixture);
        weights.insert(
            "ple.ple_embedding.layer_multipliers".to_string(),
            mlxcel_core::from_slice_i64(multipliers, shape),
        );
        PleLayer::from_weights(&weights, &config, "ple", 0)
    };

    stored(&[multipliers.len() as i32]).expect("matching stored multipliers load");
    let error = stored(&[1, multipliers.len() as i32])
        .err()
        .expect("a 2-D multiplier tensor must be refused");
    assert!(error.contains("has shape [1, 3]"), "{error}");
}
