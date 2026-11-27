# Reproducing the execution environment

Install requirements.txt in a clean Python environment. Neural and pretrained
image candidates additionally need requirements-gpu.txt, which includes both
torch and torchvision. CPU execution also supports these candidates when selected.

Use a matched torch and torchvision build from the official PyTorch installation
selector at https://pytorch.org/get-started/locally/. Select the operating system
and CPU or CUDA build appropriate to the execution machine. Preserve an existing
working matched build rather than independently upgrading torchvision.

Verify the matched build before starting an unattended run:

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python main.py doctor
```

The inspected development environment has Python 3.11, torch 2.12.1+cu132 and
torchvision 0.27.1+cu132. This is an observed pair, not a portable CUDA requirement.
The exact installed distribution snapshot is in environment_versions.json beside
this guide. Install exact package versions with access to the corresponding wheel
indexes; CUDA build suffixes are platform specific and may require the official
PyTorch wheel index. A package snapshot alone does not install a GPU driver.

Each new training run writes environment.json and requirements.snapshot.txt.
These bind actual Python, library versions and devices to its source snapshot.
Resume requires the original environment and source. Use the matching source for
saved models. The snapshot contains package names and versions, not credentials.

Pretrained image models use ResNet18_Weights.IMAGENET1K_V1 or
ResNet34_Weights.IMAGENET1K_V1 explicitly. On the initial download, torchvision
checks the publisher hash prefix. The cached file is checked again and its full
SHA256, source URL and weight enum are recorded in training_resources. Prediction
reconstructs the architecture with no download and loads the saved fitted weights.
Download failures fall back to a randomly initialized backbone and record the
reason. Worker time limits bound unavailable downloads. No downloaded weights
are claimed when fallback occurs. Optional packages are checked before planning;
dependency installation is an environment setup step, not an unlogged runtime action.

The frozen_image_logistic candidate also supports CPU. It streams images through
a fixed ResNet18 backbone and trains a logistic classification head. Backbone
weights are included in the fitted artifact and prediction needs no download.
Failed initial weight loading rejects this optional candidate, leaving the basic
image models available. Image fine-tuning candidates include augmentation on/off
choices so that horizontal flips are not assumed beneficial for every dataset.

Run tests with Python UTF-8 mode on Windows:

```powershell
python -X utf8 -m unittest discover -s tests/models -p "test_*.py"
```

Test access is blocked in the controller and training workers throughout search
and refitting. progressive_output records checkpoints only. Final prediction or
evaluation starts after selection_completed. No test statistic chooses a model.

## Prepare model resources

Before an unattended pretrained-model run, prepare the pinned weights and
tokenizers using the [model cache guide](model_cache.md). The default directories
are `.pretrained_cache/text/` and `.pretrained_cache/torch/` inside the checkout.
The local manifest lists exact paths, revisions, sizes and hashes. Cache files
are ignored by Git; a new machine must prepare or copy them separately.
