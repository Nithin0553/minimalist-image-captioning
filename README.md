# Minimalist 6-D Image Captioning

COSC 6324.W01 Digital Image Processing course project, Summer 2026.

This repository tests whether a deliberately severe six-dimensional image bottleneck can retain enough global semantic information to generate coherent Flickr8k captions. The implementation follows the assignment's flat, non-hierarchical architecture exactly.

## Professor requirement mapping

| Assignment requirement | Implementation |
|---|---|
| DIP preprocessing | Configurable Sobel edge emphasis (default) or Gaussian smoothing in `preprocessing.py` |
| Fixed pre-trained ResNet-18 | ImageNet ResNet-18 with its classifier removed and every encoder parameter frozen |
| 512-D encoder feature | Runtime shape check requires `[batch, 512]` |
| Single direct 512 -> 6 map | Exactly one `torch.nn.Linear(512, 6)`, with no intermediate layer or activation |
| GRU receives the 6-D vector as `h0` | One-layer GRU with `hidden_size=6`; initial state shape is `[1, batch, 6]` |
| Flickr8k | Kaggle download command plus support for both common Flickr8k caption-file formats |
| BLEU-1, BLEU-4, METEOR | Standard NLTK corpus metrics, saved as JSON |
| t-SNE of 6-D latent space | Test-set t-SNE colored by scene labels inferred from the human references |
| Gaussian and salt-and-pepper sensitivity | Caption metrics, mean latent L2 shift, and latent cosine similarity at every configured noise level |
| Screenshots of structure and outputs | Architecture, caption, t-SNE, sensitivity, and evidence-montage image generators |
| ReadMe.txt | Included at repository root |

The code rejects a configuration whose latent dimension is not six or whose experiment encoder is not pre-trained. `assert_required_architecture()` also fails if the frozen encoder, direct projection, GRU hidden size, or GRU layer count changes.

## Project layout

```text
minimalist-image-captioning/
|-- configs/                 # Full and quick experiment settings
|-- data/flickr8k/           # Local Flickr8k files; ignored by Git
|   |-- Images/              # 8,000+ JPEG images
|   `-- captions.txt         # Five captions per image
|-- src/minimal_captioning/  # Dataset, DIP, model, train, evaluation, and CLI code
|-- tests/                   # Offline unit/integration tests
|-- artifacts/               # Generated splits and vocabulary; ignored by Git
|-- checkpoints/             # best.pt and last.pt; ignored by Git
|-- outputs/                 # Metrics, predictions, plots; ignored by Git
|-- screenshots/             # Submission evidence montage; ignored by Git
|-- ReadMe.txt               # Professor-requested run instructions
`-- pyproject.toml           # Dependencies and quality-tool configuration
```

## 1. Windows and VS Code prerequisites

Install these once:

1. [Git for Windows](https://git-scm.com/download/win).
2. [Visual Studio Code](https://code.visualstudio.com/) with Microsoft's **Python** extension.
3. Python 3.11 or 3.12. The project is tested with Python 3.12.
4. `uv`, the Python environment/package manager. In a normal PowerShell window:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

Close and reopen PowerShell if `uv` is not recognized immediately.

## 2. Clone and open the project

```powershell
git clone https://github.com/Nithin0553/minimalist-image-captioning.git
cd minimalist-image-captioning
code .
```

In VS Code, open **Terminal -> New Terminal**, then install the exact environment:

```powershell
uv sync --extra dev
```

Select the interpreter if VS Code asks: press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, then choose `.venv\Scripts\python.exe`.

Confirm PyTorch and the available device:

```powershell
uv run python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

A CUDA GPU is helpful but not required. `device: auto` selects CUDA, then Apple MPS, then CPU.

## 3. Download Flickr8k

The automated command downloads the public [Flickr8k Kaggle dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) and arranges it under `data/flickr8k`:

```powershell
uv run minimal-caption download-data --config configs/default.yaml
```

If Kaggle asks you to authenticate, sign in to Kaggle in your browser and follow the current Kaggle API credential prompt. Do not commit credentials.

Manual alternative: download and extract Flickr8k, then make this exact structure:

```text
data/flickr8k/Images/1000268201_693b08cb0e.jpg
data/flickr8k/Images/...other images...
data/flickr8k/captions.txt
```

The loader also recognizes `Flickr8k.token.txt` and nested Kaggle folders.

## 4. Install the METEOR lexical resource

Standard METEOR uses WordNet. Download it once:

```powershell
uv run minimal-caption setup-nltk
```

## 5. Validate data and architecture

```powershell
uv run minimal-caption validate-data --config configs/default.yaml
uv run minimal-caption architecture --config configs/default.yaml
```

These commands create deterministic 80/10/10 image-level splits (seed 42), build the vocabulary from training captions only, verify all captioned image files, and render the required model diagram. Splitting by image prevents captions for the same image from leaking across train and test sets.

## 6. Run a short end-to-end check first

The quick configuration uses 256 train images, 64 validation images, 64 test images, two epochs, and one level of each noise type. These results are only a pipeline check, not the final course results.

```powershell
uv run minimal-caption train --config configs/quick.yaml
uv run minimal-caption analyze --config configs/quick.yaml --checkpoint checkpoints/quick/best.pt
uv run minimal-caption architecture --config configs/quick.yaml
```

Caption one test image by replacing the filename with any Flickr8k image:

```powershell
uv run minimal-caption caption --config configs/quick.yaml --checkpoint checkpoints/quick/best.pt --image data/flickr8k/Images/REPLACE_WITH_IMAGE_NAME.jpg
uv run minimal-caption evidence --config configs/quick.yaml
```

Quick outputs are under `outputs/quick`, `checkpoints/quick`, and `screenshots/quick`.

## 7. Run the final experiment

The single command below validates the full dataset, trains up to 30 epochs with early stopping, evaluates clean captions, creates t-SNE, runs every required noise condition, captions a test image, and creates the screenshot montage:

```powershell
uv run minimal-caption run-all --config configs/default.yaml
```

Keep the terminal open. On CPU this can take several hours; runtime depends on the computer. The best and latest checkpoints are saved after every epoch, so interruption does not erase completed epochs.

Resume an interrupted run:

```powershell
uv run minimal-caption train --config configs/default.yaml --resume checkpoints/last.pt
```

You can also run the final stages separately:

```powershell
uv run minimal-caption train --config configs/default.yaml
uv run minimal-caption evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt
uv run minimal-caption analyze --config configs/default.yaml --checkpoint checkpoints/best.pt
uv run minimal-caption evidence --config configs/default.yaml
```

## 8. Final output files

| File | Purpose |
|---|---|
| `outputs/dataset_structure.txt` | Dataset counts, paths, split sizes, and image fingerprint |
| `outputs/architecture_summary.txt` | Exact dimensions and frozen/trainable contract |
| `outputs/architecture_diagram.png` | Pipeline screenshot |
| `outputs/training_history.csv` | Per-epoch train and validation loss |
| `outputs/training_curve.png` | Training plot |
| `outputs/caption_metrics.json` | BLEU-1, BLEU-4, METEOR, image count |
| `outputs/test_predictions.csv` | Each test prediction and up to five human references |
| `outputs/latent_vectors.csv` | Six latent coordinates, scene label, and t-SNE coordinates |
| `outputs/tsne_latent_space.png` | Semantic clustering visualization |
| `outputs/sensitivity_results.csv` | Metrics and latent robustness for clean/noisy inputs |
| `outputs/sensitivity_analysis.png` | Gaussian and salt-and-pepper comparison plot |
| `outputs/caption_result.png` | Original image, DIP image, and generated caption |
| `screenshots/project_evidence.png` | Combined screenshot-ready submission evidence |
| `checkpoints/best.pt` | Best validation-loss model |
| `checkpoints/last.pt` | Most recent resumable model |

Dataset files and generated checkpoints are intentionally excluded from Git because they are large. Submit the source and selected output screenshots through Canvas as directed by the professor.

## 9. Automated code checks

These tests do not download Flickr8k or ResNet weights:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run basedpyright
uv run pytest --cov=minimal_captioning --cov-branch --cov-report=term-missing
```

## 10. Common fixes

- **`uv` not recognized:** close/reopen VS Code, or restart Windows after installing `uv`.
- **Flickr8k root not found:** confirm `data/flickr8k/Images` and `data/flickr8k/captions.txt`, then run `validate-data`.
- **WordNet/METEOR error:** run `uv run minimal-caption setup-nltk`.
- **CUDA out of memory:** reduce `training.batch_size` and `evaluation.batch_size` in the YAML file, or set `training.device: cpu`.
- **CUDA is false:** the system still runs on CPU. For GPU acceleration, install the PyTorch build recommended by the [official PyTorch selector](https://pytorch.org/get-started/locally/) for your NVIDIA driver.
- **ResNet download blocked:** connect to the internet once; TorchVision caches the official pre-trained weights afterward.
- **Empty/poor quick captions:** two epochs on 256 images only verify execution. Use the complete dataset and default configuration for reportable results.

## Course submission reminders from the assignment

- This is a group project for two or three students.
- It is worth 30 points and is due August 2, 2026.
- Only the group leader submits once through the Canvas project-submission entry.
- Required submission items: `ReadMe.txt`, source code, screenshots of data structure and outputs, presentation slides, and the final report.
- The 8-10 page report must later cover: introduction/hypothesis, theory of the D -> 6 map, DIP/model methodology, experiments/metrics/visualizations, information loss versus semantic sufficiency, and conclusion.
- Oral presentation dates listed by the professor are August 3 (optional depending on group count) and August 5.

Presentation slides and the 8-10 page report are intentionally deferred and are not included in this source-code milestone.

## Technical references

- [TorchVision ResNet-18 and official weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [NLTK translation metrics](https://www.nltk.org/api/nltk.translate.html)
- [Flickr8k dataset page](https://www.kaggle.com/datasets/adityajn105/flickr8k)
