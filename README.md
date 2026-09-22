# Computer Vision Object Detection Utilities

Dependency-free building blocks for inspecting object-detection output: bounding-box validation, intersection over union, class-aware non-max suppression, and one-to-one precision/recall matching.

```bash
python -m pip install -e ".[dev]"
python example_metrics.py
pytest -q
```

## Why these utilities exist

Model demos often stop at drawing boxes. This repository isolates the less visible evaluation decisions: rejecting invalid coordinates, preventing one truth box from matching multiple predictions, preserving overlapping boxes from different classes, and making thresholds explicit.

It is an evaluation utility rather than a training pipeline. The repository intentionally contains no employer or thesis dataset, imagery, credentials, or trained weights.
