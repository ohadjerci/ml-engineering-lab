# PCB image classification

MNIST remains the default. Select `experiment=pcb` to use the nested PCB manifest.
Each image is a sample, including images labelled OK inside NG directories. Only
`anomaly_class` determines the target. Masks and original metadata are retained
for analysis and never used as classifier inputs.

## Commands

Install `just` 1.21 or newer and run `just` to list recipes. Project Python always
runs through Docker Compose. GPU requests remain enabled in Compose even when
selecting `trainer=cpu`, so the host still needs NVIDIA Container Toolkit.

```bash
just train                         # Existing MNIST GPU training
just train trainer=cpu trainer.max_epochs=2
just train-pcb
just train-pcb trainer=cpu trainer.max_epochs=2 data.num_workers=0
just validate-pcb /workspace/logs/train/runs/<run>/checkpoints/<best>.ckpt trainer=cpu
just eval-pcb /workspace/logs/train/runs/<run>/checkpoints/<best>.ckpt trainer=cpu logger=mlflow
just test-pcb
just test                          # Excludes sweeps
just test-cpu                      # Hosted CPU selection
just check                         # Ruff lint, format check, core MyPy, pytest
```

PCB uses the YAML defaults under `data/REALIAD/`, containing `pcb.json` and `pcb/`.
Training and debugging share the `./data:/workspace/data` mount with MNIST.
The train service also mounts host `logs/train` and
`logs/eval`; checkpoint arguments use their corresponding `/workspace/logs/...`
container paths. Validation outputs are retained under host `logs/eval/validate/runs/`.
Standalone validation does not run the test split or publish a model. Evaluation
publishes only when MLflow logging and registry publication are enabled.

Equivalent training without just:

```bash
docker compose run --build --rm \
  train uv run --locked python src/train.py experiment=pcb
```

Hydra overrides can change either path independently; the image root points
directly at `pcb/`, so the manifest's `meta.prefix` is preserved but not prepended.
Use `data.split.assignments_path=/workspace/logs/train/runs/<run>/results/splits.json`
to explicitly reuse a saved split. Checkpoints also preserve split assignments.

## Configuration

`configs/data/pcb.yaml` controls paths, ordered target-to-anomaly mapping,
preprocessing, split proportions/seed/grouping, batch budget, sampling target,
workers, and pinning. The default mapping is OK → 0 and HS/QS/YW/ZW → defective
→ 1. Unknown or overlapping labels fail clearly. Class count is derived from this
mapping; model channels reference preprocessing channels.

The baseline uses RGB resized to 256×256 with bilinear interpolation, mean/std
0.5 per channel, and no random augmentation. Grayscale is supported with
channel-matched normalization. Optional `preprocessing.train_transforms` contains
Hydra-instantiated torchvision transforms applied after resize and before tensor
conversion. Validation/test preprocessing is deterministic. Resizing can lose
small defects; tune image size using validation results.

The model inherits the existing ResNet, Adam, and ReduceLROnPlateau settings.
PCB selects a 32-channel stem and 50 epochs. Model and trainer settings remain
overridable, e.g. `model.optimizer.lr=0.0001`,
`data.preprocessing.image_size=[512,512]`, or `trainer.max_epochs=100`.

## Splits and leakage

Default proportions are 70/15/15 with seed 42 and `board_group` isolation.
Canonical ordering, seeded tie-breaking, and a greedy objective approximate
requested image and class counts while keeping groups intact. Achieved counts
are recorded; exact proportions are not guaranteed. Each partition must contain
both classes. Other grouping fields are accepted only if resulting assignments
also preserve every S group and capture group.

S identifiers are directory-derived, not verified physical-board or design IDs.
Disjoint S groups therefore reduce known leakage without proving physical-board
isolation. Manifest fingerprints are independent of input record ordering and
cover metadata, not image file contents. Saved splits/checkpoints reject changed
manifest metadata, class order/mapping, split settings, and preprocessing.

## Complete-group batches

Splitting and sampling are separate. The sampler sees only training records.
Each epoch uses every training S group once, with all its images in one batch.
Several groups can share a batch. The default maximum is 64 images and target
defective fraction is 0.5. At each step the sampler selects a fitting group that
moves the current batch nearest that target, with seeded tie-breaking.

Batch sizes vary. Mixed-label, indivisible groups make exact per-batch balance
impossible in general. Exhaustive coverage preserves the natural epoch ratio;
there is no oversampling or omission. Final partial batches are kept. Oversized
groups fail by default; `data.sampling.oversized=singleton` emits them intact,
exceeding the configured batch budget. Tune the budget for available GPU memory.
Changing the target changes packing, not the epoch's total class counts.

PCB supports one device/node. Distributed settings are rejected, since ordinary
distributed image sampling would split groups and can duplicate evaluation
samples. MNIST's existing DDP behavior is preserved. Validation/test use natural
distributions with every image evaluated once unless a deliberate debug limit
is supplied.

## Results, publication, and CI

Training uses validation loss for scheduling, early stopping, and best-checkpoint
selection, then tests that checkpoint. PCB saves its best checkpoint and last
checkpoint. Epoch loss is weighted by image count. Existing metrics are retained
alongside defective-class precision/recall/F1. `results/` includes split assignments,
data summaries, per-epoch batch balance/coverage diagnostics, confusion matrices,
and per-image test predictions with probabilities and original metadata.
Debug limits can omit images; sampling diagnostics report actual omissions.

MLflow uploads results and checkpoints. TorchScript export derives the input
shape from preprocessing and records the ordered labels, mapping, and full
preprocessing settings. It is self-contained and does not import training code
when loaded. External inference must implement the recorded PCB preprocessing;
this change does not modify the separate inference application.

PCB publishes under `PCB`, separately from `MNIST`. As requested, both retain the
existing strict test-loss promotion rule for `best_model`; ties keep the incumbent.
Repeated registry selection therefore uses test results. See the documented
model-selection mismatch in architecture.md. Publication errors remain visible.

The existing GitHub Actions CPU job discovers synthetic PCB tests automatically,
excludes sweeps, runs independent Ruff lint/format checks, and explicitly checks
the shared classifier, PCB data code, results callback, and validation entry point
with MyPy. Coverage floors, four adjacent badges, seven-day report artifacts, and
trusted default-branch PR comment publishing remain in place. Real PCB data is
not required by CI. Local commands use Compose; hosted checks use the runner's uv.
