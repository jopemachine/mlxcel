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

use serde_json::json;

use super::reject_deepstack;

#[test]
fn deepstack_indexes_are_refused_by_name() {
    let error = reject_deepstack(&json!({"vision_config": {"deepstack_visual_indexes": [8, 16]}}))
        .expect_err("a non-empty deepstack list must not load");
    assert!(
        error.to_string().contains("deepstack_visual_indexes"),
        "{error}"
    );
}

#[test]
fn absent_or_empty_deepstack_indexes_load() {
    for config in [
        json!({"vision_config": {}}),
        json!({"vision_config": {"deepstack_visual_indexes": []}}),
        json!({"vision_config": {"deepstack_visual_indexes": null}}),
    ] {
        reject_deepstack(&config).unwrap_or_else(|error| panic!("{config}: {error}"));
    }
}
