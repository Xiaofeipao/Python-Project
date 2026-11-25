# Configuration Roles

- `tabular.example.json`, `text.example.json`, `image.example.json`: input-schema examples.
- `released.example.json`, `unseen.example.json`: formal evaluation task inputs.
- `final_policy.json`: algorithm/resource policy used when creating a frozen bundle.
- `final_policy.llm.example.json`: explicit LLM policy example.
- `model_registry.json`, `model_registry.flash.json`, `model_registry.pro.json`: model identity, eligibility and fixed token-price definitions.
- `evidence/`: lossless publisher evidence archives, checked against decoded-byte hashes.

Paths are resolved relative to the containing configuration file. These files remain at their established locations to preserve documented commands and evidence references. Put machine-specific task inputs outside frozen bundles; credentials belong outside this directory and outside Git.

Model-result pointers belong to docs/development/current_results.json, not these runtime configurations. They must not cause task-name dispatch or test-score-based selection inside the agent.
