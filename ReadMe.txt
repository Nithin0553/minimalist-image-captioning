MINIMALIST 6-D IMAGE CAPTIONING
COSC 6324.W01 - DIGITAL IMAGE PROCESSING

HOW TO RUN THE SYSTEM

This file explains how to run the project from the submitted ZIP folder. The
project was developed and tested with Python 3.12 and uv on Windows. The same
steps also work with Python 3.11.

1. SOFTWARE REQUIRED

Install the following before running the project:

- Visual Studio Code
- Python 3.11 or Python 3.12
- The Python extension for Visual Studio Code
- uv, which creates the environment and installs the packages
- An internet connection for the first setup

To install uv, open PowerShell and run:

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

Close and reopen PowerShell after the installation. Then check that uv is
available:

uv --version

2. EXTRACT AND OPEN THE ZIP FILE

Extract the submitted ZIP file to a normal folder on the computer. Do not run
the project from inside the compressed ZIP window.

Open Visual Studio Code, select File -> Open Folder, and choose the extracted
minimalist-image-captioning folder. The correct project folder contains these
items:

configs
src
tests
pyproject.toml
uv.lock
ReadMe.txt

Open Terminal -> New Terminal in Visual Studio Code. All commands below should
be run from this project folder.

3. CREATE THE PYTHON ENVIRONMENT

Run:

uv sync --frozen --extra dev

This creates a local .venv folder and installs PyTorch, TorchVision, NLTK,
scikit-learn, Matplotlib, Pillow, and the other packages used by the project.

If Visual Studio Code asks for the Python interpreter, press Ctrl+Shift+P,
search for "Python: Select Interpreter", and choose:

.venv\Scripts\python.exe

The PyTorch version and available device can be checked with:

uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

The project works on either CPU or GPU. A GPU makes training faster, but it is
not required.

4. DOWNLOAD THE FLICKR8K DATASET

The Flickr8k dataset is not included in the submitted ZIP because it is large.
Download and arrange it by running:

uv run minimal-caption download-data --config configs/default.yaml

After the download, the important dataset paths should be:

data/flickr8k/Images
data/flickr8k/captions.txt

If Kaggle asks for authentication, sign in to Kaggle and complete its API
setup. Do not place Kaggle credentials inside the project folder or the
submitted ZIP.

Flickr8k can also be downloaded manually. For a manual setup, place the image
files and caption file in the paths shown above.

5. DOWNLOAD THE NLTK RESOURCE

METEOR uses NLTK WordNet. Download it once by running:

uv run minimal-caption setup-nltk

6. VALIDATE THE DATA AND ARCHITECTURE

Run these commands before training:

uv run minimal-caption validate-data --config configs/default.yaml
uv run minimal-caption architecture --config configs/default.yaml
uv run minimal-caption submission-check --config configs/default.yaml

The first command checks the Flickr8k images and captions and creates the
training, validation, and test splits. The second command verifies the model
architecture and creates its diagram. The third command checks the required
source files.

The required model pipeline is:

DIP preprocessing -> frozen pretrained ResNet-18 -> 512-D feature vector ->
one Linear(512, 6) layer -> one-layer GRU with hidden size 6 -> caption

7. RUN A QUICK TEST

We recommend running the quick configuration before starting the complete
experiment. It uses a small part of Flickr8k and only two training epochs, so
it is only a pipeline test.

Run:

uv run minimal-caption train --config configs/quick.yaml
uv run minimal-caption analyze --config configs/quick.yaml --checkpoint checkpoints/quick/best.pt
uv run minimal-caption architecture --config configs/quick.yaml

To caption one image, replace IMAGE_NAME.jpg with the name of an image from
data/flickr8k/Images:

uv run minimal-caption caption --config configs/quick.yaml --checkpoint checkpoints/quick/best.pt --image data/flickr8k/Images/IMAGE_NAME.jpg

Create the quick evidence image with:

uv run minimal-caption evidence --config configs/quick.yaml

The quick files are saved under outputs/quick, checkpoints/quick, and
screenshots/quick. Quick-test results are not the final project results.

8. RUN THE COMPLETE EXPERIMENT

After the quick test succeeds, run:

uv run minimal-caption run-all --config configs/default.yaml

This one command performs the complete project workflow. It validates Flickr8k,
creates the dataset structure output, verifies the model architecture, trains
the model for up to 30 epochs, calculates BLEU-1, BLEU-4, and METEOR, creates
the t-SNE visualization, runs Gaussian and salt-and-pepper sensitivity tests,
generates a sample caption, and creates the final evidence image.

The complete run can take several hours, especially on CPU. Keep the terminal
and computer running until it finishes.

9. RESUME AN INTERRUPTED RUN

The project saves the latest checkpoint after every completed epoch. If the
complete experiment is interrupted, resume it with:

uv run minimal-caption run-all --config configs/default.yaml --resume checkpoints/last.pt

This continues from the saved epoch and then finishes the remaining evaluation,
visualization, sensitivity, caption, and evidence steps.

10. MAIN OUTPUT FILES

The main generated files are:

outputs/dataset_structure.txt
outputs/dataset_structure.png
outputs/architecture_summary.txt
outputs/architecture_diagram.png
outputs/training_history.csv
outputs/training_curve.png
outputs/caption_metrics.json
outputs/test_predictions.csv
outputs/latent_vectors.csv
outputs/tsne_latent_space.png
outputs/sensitivity_results.csv
outputs/sensitivity_results.json
outputs/sensitivity_analysis.png
outputs/caption_result.png
outputs/caption_result.json
screenshots/project_evidence.png
checkpoints/best.pt
checkpoints/last.pt

If selected output screenshots are already included in the submitted ZIP, they
can be opened directly without running the experiment again. The commands above
can be used to reproduce all outputs from the source code.

11. FINAL PROJECT CHECK

After the complete experiment finishes, run:

uv run minimal-caption submission-check --config configs/default.yaml --require-generated

A successful check should show:

Submission check: PASS (source and generated artifacts; 41 files checked)

12. OPTIONAL CODE CHECKS

The following commands check the formatting, source code, types, tests,
coverage, and package build:

uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run basedpyright
uv run pytest --cov=minimal_captioning --cov-branch --cov-report=term-missing
uv build

13. COMMON PROBLEMS

If uv is not recognized:
Close and reopen Visual Studio Code or PowerShell, then run uv --version again.

If the terminal is in the wrong folder:
Use cd to enter the extracted folder that contains pyproject.toml.

If Flickr8k cannot be found:
Check that data/flickr8k/Images contains the image files and that
data/flickr8k/captions.txt exists.

If METEOR reports a WordNet error:
Run uv run minimal-caption setup-nltk again.

If CUDA is not available:
The project automatically runs on CPU. It will work, but training will take
longer.

If training is interrupted:
Use the resume command with checkpoints/last.pt instead of restarting.

If the quick captions are poor:
This is expected because the quick test uses a small subset and only two
epochs. Use configs/default.yaml for the final results.
