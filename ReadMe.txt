COSC 6324.W01 - MINIMALIST 6-D IMAGE CAPTIONING
================================================

REQUIREMENTS
- Windows 10/11, macOS, or Linux
- Git
- Python 3.11 or 3.12
- uv (https://docs.astral.sh/uv/)
- Internet access for initial dependencies, Flickr8k, NLTK WordNet, and ResNet weights

WINDOWS / VS CODE SETUP
1. Open PowerShell in the project root.
2. Run: uv sync --frozen --extra dev
3. In VS Code select .venv\Scripts\python.exe as the Python interpreter.
4. Run: uv run minimal-caption download-data --config configs/default.yaml
5. Run: uv run minimal-caption setup-nltk
6. Run: uv run minimal-caption validate-data --config configs/default.yaml
7. Run: uv run minimal-caption architecture --config configs/default.yaml
8. Run: uv run minimal-caption submission-check --config configs/default.yaml

SHORT PIPELINE CHECK (NOT FINAL RESULTS)
uv run minimal-caption train --config configs/quick.yaml
uv run minimal-caption analyze --config configs/quick.yaml --checkpoint checkpoints/quick/best.pt

FINAL COMPLETE EXPERIMENT
uv run minimal-caption run-all --config configs/default.yaml

RESUME TRAINING
uv run minimal-caption train --config configs/default.yaml --resume checkpoints/last.pt

RESUME THE COMPLETE EXPERIMENT
uv run minimal-caption run-all --config configs/default.yaml --resume checkpoints/last.pt

INDIVIDUAL FINAL COMMANDS
uv run minimal-caption train --config configs/default.yaml
uv run minimal-caption analyze --config configs/default.yaml --checkpoint checkpoints/best.pt
uv run minimal-caption caption --config configs/default.yaml --checkpoint checkpoints/best.pt --image data/flickr8k/Images/IMAGE_NAME.jpg
uv run minimal-caption evidence --config configs/default.yaml
uv run minimal-caption submission-check --config configs/default.yaml --require-generated

IMPORTANT OUTPUTS
- outputs/dataset_structure.txt
- outputs/dataset_structure.png
- outputs/architecture_diagram.png
- outputs/architecture_summary.txt
- outputs/training_curve.png
- outputs/training_history.csv
- outputs/caption_metrics.json
- outputs/test_predictions.csv
- outputs/latent_vectors.csv
- outputs/tsne_latent_space.png
- outputs/sensitivity_results.csv
- outputs/sensitivity_results.json
- outputs/sensitivity_analysis.png
- outputs/caption_result.png
- outputs/caption_result.json
- screenshots/project_evidence.png
- checkpoints/best.pt
- checkpoints/last.pt

ARCHITECTURE CONTRACT
DIP preprocessing -> frozen pre-trained ResNet-18 -> 512-D feature -> one
Linear(512, 6) bottleneck -> one-layer GRU with hidden_size=6 and the 6-D
latent as h0 -> caption tokens.

METRICS AND ANALYSES
BLEU-1, BLEU-4, METEOR, t-SNE of the 6-D latent space, Gaussian noise,
and salt-and-pepper noise are all implemented.

See README.md for every command, expected data structure, output description,
troubleshooting, tests, source attribution, and the professor's submission checklist.
See REFERENCES.md for dataset, method, metric, and software citations.

The presentation slides and 8-10 page report are intentionally deferred.
