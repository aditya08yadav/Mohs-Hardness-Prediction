MATERIAL HARDNESS PREDICTION - COMPLETE CODEBASE

Start with project.ipynb. It contains all training and evaluation code,
explanations, charts, final predictions, and the supplementary PCA experiment.

CONTENTS
  project.ipynb              Complete notebook, with saved executed outputs.
  data/                     Three input CSV files used in the recorded run.
  results/                  Metrics, predictions and PCA diagnostics.
  scripts/run_notebook.py    Runs the notebook in a fresh kernel.
  scripts/pca_comparison.py  Optional standalone PCA experiment.
  requirements.txt          Pinned analysis and execution dependencies.
  DATA_SOURCES.txt           Data provenance and implementation provenance.
  data_checksums.json       SHA-256 hashes of the included input files.

QUICK START (Python 3.12)
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python scripts/run_notebook.py

Windows activation:
  .venv\Scripts\activate

To study interactively, open project.ipynb in an existing Jupyter or VS Code
installation, select the installed environment, restart the kernel and run all.
Run from the project folder; retain data/ beside the notebook. Data is included,
so the analysis itself needs no network access. Installing packages does.

EXPECTED REFERENCE RESULTS
  Four original model families; five-fold development comparison.
  Selected estimator: histogram gradient boosting.
  Original boosting CV MedAE: approximately 0.5418.
  Original holdout MedAE: approximately 0.5292, baseline 1.0000.
  PCA boosting development CV MedAE: approximately 0.5928.
  Final prediction file: 6,939 rows, id and Hardness columns.

EXPERIMENT BOUNDARIES
  The reserved holdout evaluates the original selected configuration.
  PCA was a later development-only experiment. It does not change the original
  selected model or its prediction file. No PCA holdout score is reported.
  The original workflow has 23 supervised fits; PCA adds ten, making 33 total.

ORIGIN AND PORTFOLIO USE
  This implementation was developed for this project with AI assistance using
  standard pandas/scikit-learn APIs. It is not a renamed copy of the reference
  author's notebook. Ordinary library calls naturally resemble other projects.
  The input data is externally sourced and remains attributed in DATA_SOURCES.txt
  and the notebook. Do not imply that this project collected the data or invented
  the underlying algorithms. Understand and reproduce the decisions before
  presenting the work; do not claim sole unaided authorship where that matters.

The existing professional study guide is a separate deliverable, project.pdf.
