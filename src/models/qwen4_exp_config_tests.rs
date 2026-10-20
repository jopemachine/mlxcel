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

use serde_json::{Value, json};

use super::*;

/// The released checkpoint's text config, minus `layer_types` (derived from
/// `full_attention_interval`) and the per-module quantization overrides.
fn released() -> Value {
    json!({
        "model_type": "qwen4_exp_text",
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_value_heads": 48,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "shared_expert_intermediate_size": 640,
        "moe_intermediate_size": 640,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "hc_count": 4,
        "hc_lowrank": 320,
        "ple_layer_ids": [2],
        "ple_embed_dim": 2560,
        "eos_token_id": 248044,
        "output_gate_type": "sigmoid",
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default"
        },
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "model.layers.0.mlp.switch_mlp.gate_proj": {"group_size": 64, "bits": 2, "mode": "affine"},
            "model.layers.0.attn_hyper_connection.input_mix_weight_down": {"group_size": 64, "bits": 8, "mode": "affine"},
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_7": {"group_size": 32, "bits": 4, "mode": "affine"}
        }
    })
}

fn resolve(patch: Value) -> Result<ResolvedQwen4ExpConfig, Qwen4ExpConfigError> {
    let mut raw = released();
    for (key, value) in patch.as_object().expect("patch is an object") {
        raw[key] = value.clone();
    }
    let config: Qwen4ExpConfig = serde_json::from_value(raw).expect("config deserializes");
    config.resolve()
}

#[test]
fn released_config_resolves_to_the_published_geometry() {
    let config = resolve(json!({})).unwrap();
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.rope_dims(), 64);
    assert_eq!(config.ngram_heads(), 16);
    assert_eq!(config.eos_token_id, Some(248044));
    assert_eq!(config.output_gate, OutputGate::Sigmoid);
    let attention: Vec<usize> = (0..48).filter(|&i| !config.is_linear_layer(i)).collect();
    assert_eq!(attention, (3..48).step_by(4).collect::<Vec<_>>());
    assert_eq!(config.ple_index(1), Some(0));
    assert_eq!(config.ple_index(2), None);
}

#[test]
fn quantization_overrides_are_resolved_per_module() {
    let config = resolve(json!({})).unwrap();
    let table = &config.quantization;
    assert_eq!(
        table.for_module("model.layers.0.mlp.switch_mlp.gate_proj"),
        (64, 2)
    );
    assert_eq!(
        table.for_module("model.layers.0.attn_hyper_connection.input_mix_weight_down"),
        (64, 8)
    );
    assert_eq!(
        table.for_module("model.layers.1.ple.ple_embedding.ngram_embedding.shards.7"),
        (32, 4),
        "override keys are canonicalized like weight keys"
    );
    assert_eq!(table.for_module("model.layers.5.self_attn.q_proj"), (64, 4));
}

#[test]
fn full_attention_layer_types_are_renamed_to_qwen_sparse_attention() {
    let types: Vec<&str> = (0..48)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let config = resolve(json!({ "layer_types": types })).unwrap();
    assert_eq!(config.layer_types[3], QWEN_SPARSE_ATTENTION);
    assert_eq!(config.layer_types[0], LINEAR_ATTENTION);
}

#[test]
fn eos_list_resolves_to_its_first_entry() {
    let config = resolve(json!({ "eos_token_id": [248044, 248046] })).unwrap();
    assert_eq!(config.eos_token_id, Some(248044));
}

/// One test per rejection, each patching one field of the released config.
macro_rules! rejects {
    ($($name:ident: $patch:expr => $expected:expr;)+) => {
        $(
            #[test]
            fn $name() {
                assert_eq!(resolve($patch).unwrap_err(), $expected);
            }
        )+
    };
}

rejects! {
    rejects_layer_types_of_the_wrong_length:
        json!({ "layer_types": ["linear_attention", "full_attention"] })
        => Qwen4ExpConfigError::LayerTypesLength { got: 2, expected: 48 };
    rejects_unknown_layer_types:
        json!({ "num_hidden_layers": 4, "ple_layer_ids": [], "layer_types": ["linear_attention", "sliding_attention", "linear_attention", "full_attention"] })
        => Qwen4ExpConfigError::UnsupportedLayerTypes(vec!["sliding_attention".to_string()]);
    rejects_a_single_hyper_connection_stream:
        json!({ "hc_count": 1 })
        => Qwen4ExpConfigError::HcCount(1);
    rejects_an_unknown_output_gate:
        json!({ "output_gate_type": "relu" })
        => Qwen4ExpConfigError::OutputGateType(Some("relu".to_string()));
    rejects_a_null_output_gate:
        json!({ "output_gate_type": null })
        => Qwen4ExpConfigError::OutputGateType(None);
    rejects_more_than_one_indexer_key_head:
        json!({ "indexer_kv_heads": 2 })
        => Qwen4ExpConfigError::IndexerKvHeads(2);
    rejects_a_zero_indexer_compress_ratio:
        json!({ "indexer_compress_ratio": 0 })
        => Qwen4ExpConfigError::IndexerCompressRatio;
    rejects_an_indexer_budget_that_is_not_a_whole_number_of_blocks:
        json!({ "indexer_budget": 2050 })
        => Qwen4ExpConfigError::IndexerBudget { budget: 2050, ratio: 4 };
    rejects_a_ple_width_not_divisible_by_the_ngram_heads:
        json!({ "ple_embed_dim": 2561 })
        => Qwen4ExpConfigError::PleEmbedDim { ple_embed_dim: 2561, ngram_heads: 16 };
    rejects_unigram_only_ngrams:
        json!({ "ngram_size": 1 })
        => Qwen4ExpConfigError::PleEmbedDim { ple_embed_dim: 2560, ngram_heads: 0 };
    rejects_zero_indexed_ple_layers:
        json!({ "ple_layer_ids": [0] })
        => Qwen4ExpConfigError::PleLayerIds { invalid: vec![0], num_hidden_layers: 48 };
    rejects_ple_layers_past_the_last_layer:
        json!({ "ple_layer_ids": [49] })
        => Qwen4ExpConfigError::PleLayerIds { invalid: vec![49], num_hidden_layers: 48 };
    rejects_ple_on_an_attention_layer:
        json!({ "ple_layer_ids": [4] })
        => Qwen4ExpConfigError::PleOnNonLinearLayer(vec![4]);
    rejects_ple_without_an_eos_token:
        json!({ "eos_token_id": null })
        => Qwen4ExpConfigError::MissingEosForPle;
    rejects_a_non_integer_eos_token:
        json!({ "eos_token_id": "248044" })
        => Qwen4ExpConfigError::EosTokenId("\"248044\"".to_string());
    rejects_rope_parameters_missing_required_keys:
        json!({ "rope_parameters": { "rope_theta": 10000000 } })
        => Qwen4ExpConfigError::RopeParameters(vec![
            "mrope_section".to_string(),
            "type".to_string(),
            "partial_rotary_factor".to_string(),
        ]);
    rejects_a_malformed_mrope_section:
        json!({ "rope_parameters": { "type": "default", "mrope_section": "11,11,10", "rope_theta": 10000000, "partial_rotary_factor": 0.25 } })
        => Qwen4ExpConfigError::RopeParameterType { field: "mrope_section", value: "\"11,11,10\"".to_string() };
    rejects_external_ple_storage:
        json!({ "ple_storage": { "manifest": "ple/manifest.json" } })
        => Qwen4ExpConfigError::PleStorage;
    rejects_a_zero_hidden_size:
        json!({ "hidden_size": 0 })
        => Qwen4ExpConfigError::Zero { field: "hidden_size" };
}

/// A pair reaches `gather_qmm` / `quantized_matmul` either as the default the
/// text config inherits from the top-level block or as one module's override.
#[test]
fn hostile_quantization_pairs_are_refused_as_default_and_as_override() {
    use crate::models::switch_layers::HOSTILE_QUANT_PARAMS;

    let mut text = released();
    text.as_object_mut()
        .expect("released config is an object")
        .remove("quantization");
    let full = |quantization: Value| json!({ "text_config": text, "quantization": quantization });
    text_config_from_full(&full(json!({ "group_size": 64, "bits": 4 })))
        .expect("an honest inherited quantization block resolves");

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        for (module, quantization) in [
            (
                "<default>",
                json!({ "group_size": group_size, "bits": bits }),
            ),
            (
                "model.layers.0.mlp.switch_mlp.up_proj",
                json!({
                    "group_size": 64,
                    "bits": 4,
                    "model.layers.0.mlp.switch_mlp.up_proj": { "group_size": group_size, "bits": bits }
                }),
            ),
        ] {
            let error = text_config_from_full(&full(quantization)).expect_err(&format!(
                "{module} (group_size {group_size}, bits {bits}) must be refused"
            ));
            assert!(
                error.contains(&format!("quantization.{module} declares"))
                    && error.contains(&format!("quantization.{field} (")),
                "{module} (group_size {group_size}, bits {bits}) must be blamed on {field}: {error}"
            );
        }
    }
}
