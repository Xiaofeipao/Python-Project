# Pretrained Model Cache

Prepare every downloadable model registered in this project before an unattended
run. This setup step downloads general model resources only: it does not train,
read task data, call the planner API or add LLM accounting entries.

From the repository root, in the environment containing `requirements-gpu.txt`:

```powershell
python scripts/setup/download_models.py
python scripts/setup/download_models.py --offline
```

The first command prepares all seven pinned text encoders, their tokenizer and
configuration files, and both explicit ImageNet weight files. It reuses prior
materialized text snapshots and Torch hub weights where available. The offline
command checks local configuration/tokenizer loading and image weight hashes.
Neither command performs model fitting or task inference.

## Default directories

Paths are relative to the project checkout, not the current shell directory:

```text
.pretrained_cache/
  manifest.json
  text/
    sentence-transformers--all-MiniLM-L6-v2-1110a243fdf4/
    sentence-transformers--all-MiniLM-L12-v2-a50ef00143b4/
    sentence-transformers--all-mpnet-base-v2-e8c3b32edf54/
    BAAI--bge-base-en-v1.5-a5beb1e3e68b/
    intfloat--e5-base-v2-f52bf8ec8c71/
    thenlper--gte-base-c078288308d8/
    answerdotai--ModernBERT-base-8949b909ec90/
  torch/
    hub/checkpoints/
      resnet18-f37072fd.pth
      resnet34-b627a593.pth
```

On the current checkout the cache root is `D:\Python Project\.pretrained_cache`.
`manifest.json` records the resolved absolute paths, fixed text revisions,
official source URLs, downloaded file sizes, SHA256 values and setup failures.
It is a local resource inventory, not an experiment result or a cost ledger.
Read `complete` and `errors` before considering preparation successful. Retrying
the setup command reuses completed files. Partial preparation is not reported
as complete.

Text encoder, semantic blend and Transformer fine-tuning branches share these
seven snapshots. ResNet feature extraction and fine-tuning share the two ImageNet
weight files. TF-IDF, tree models and randomly initialized neural networks do
not require pretrained weight downloads. The remote Qwen/DeepSeek planner uses
its configured API and has no local model weights in this cache.

## Relocation and frozen source snapshots

Optional overrides must be absolute paths shared across the controller and its
workers. For example, to use a copied cache:

```powershell
$env:SUPERAGENT_MODEL_CACHE = 'D:\Python Project\.pretrained_cache\text'
$env:TORCH_HOME = 'D:\Python Project\.pretrained_cache\torch'
python scripts/setup/download_models.py --offline
```

The controller passes the resolved paths to isolated workers, including workers
started from recorded source snapshots. Thus a worker does not create an empty
cache beneath its own source snapshot. When running a frozen bundle directly,
set these variables to the prepared cache, or copy `.pretrained_cache/` beside
that bundle's `main.py`. This does not change the bundle's algorithm policy.

Copy the complete directories when moving machines. Some Transformer prediction
branches reconstruct tokenizers/model objects and need the local snapshots even
though they do not call the planner API. A successful cache check is not proof
of compatibility for every optional training branch.

The entire `.pretrained_cache/` is ignored by Git. GitHub receives code and setup
instructions, not multi-gigabyte model weights. A fresh clone must run the setup
command or receive a separately copied cache. Incomplete or absent resources
retain the existing runtime download/fallback behavior.
