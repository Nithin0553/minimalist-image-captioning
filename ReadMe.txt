MINIMALIST 6-D IMAGE CAPTIONING
COSC 6324.W01 - DIGITAL IMAGE PROCESSING

HOW TO RUN THE SYSTEM

This project was developed and tested with Python 3.12 and uv on Windows. The
commands below are the same commands we used for the project. Run them from
PowerShell in the project folder.

1. SOFTWARE REQUIRED

Before starting, install the following:

- Git
- Visual Studio Code
- Python 3.11 or Python 3.12
- The Python extension for Visual Studio Code
- uv, which is used to create the environment and install the packages
- An internet connection for the first setup

To install uv in PowerShell, run:

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

Close and reopen PowerShell after the installation, and then check it with:

uv --version

2. DOWNLOAD AND OPEN THE PROJECT

Clone the repository and open it in Visual Studio Code:

git clone https://github.com/Nithin0553/minimalist-image-captioning.git
cd minimalist-image-captioning
code .

In Visual Studio Code, open Terminal -> New Terminal. All remaining commands
should be run from the project root, where pyproject.toml is located.

3. CREATE THE PYTHON ENVIRONMENT

Run:

uv sync --frozen --extra dev

This creates a local .venv folder and installs PyTorch, TorchVision, NLTK,
scikit-learn, Matplotlib, Pillow, and the other required packages.

If Visual Studio Code asks for a Python interpreter, press Ctrl+Shift+P, search
for "Python: Select Interpreter", and select:

.venv\Scripts\python.exe

The available PyTorch device can be checked with:

uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

The project runs on either CPU or GPU. A GPU makes the full training run much
faster, but it is not required.

4. DOWNLOAD THE FLICKR8K DATASET

Run:

uv run minimal-caption download-data --config configs/default.yaml

The command downloads Flickr8k and arranges it under:

data/flickr8k/Images
data/flickr8k/captions.txt

If the automatic download asks for Kaggle authentication, sign in to Kaggle
and complete its API setup. Kaggle credentials should never be added to Git.

The dataset can also be downloaded manually. If it is downloaded manually,
make sure that the image folder and caption file use the paths shown above.

5. DOWNLOAD THE NLTK RESOURCE

METEOR uses the WordNet resource from NLTK. Install it once by running:

uv run minimal-caption setup-nltk

6. CHECK THE DATA AND MODEL ARCHITECTURE

Before training, run these commands:

uv run minimal-caption validate-data --config configs/default.yaml
uv run minimal-caption architecture --config configs/default.yaml
uv run minimal-caption submission-check --config configs/default.yaml

The first command checks the Flickr8k images and captions and creates the
training, validation, and test splits. The second command verifies the required
model architecture and creates its diagram. The third command checks the
required source files.

The expected model pipeline is:

DIP preprocessing -> frozen pretrained ResNet-18 -> 512-D feature vector ->
one Linear(512, 6) layer -> one-layer GRU with hidden size 6 -> caption

7. RUN THE QUICK TEST

We recommend running the quick configuration before the full experiment. It
uses a small part of Flickr8k and only two epochs, so it is only meant to check
that the full pipeline works.

Run:

uv run minimal-caption train --config configs/quick.yaml
uv run minimal-caption analyze --config configs/quick.yaml --checkpoint checkpoints/quick/best.pt
uv run minimal-caption architecture --config configs/quick.yaml

To generate a caption for one image, replace IMAGE_NAME.jpg with a file from
the Flickr8k Images folder:

uv run minimal-caption caption --config configs/quick.yaml --checkpoint checkpoints/quick/best.pt --image data/flickr8k/Images/IMAGE_NAME.jpg

Create the quick evidence image with:

uv run minimal-caption evidence --config configs/quick.yaml

Quick-test files are saved in outputs/quick, checkpoints/quick, and
screenshots/quick. These results should not be used as the final project
results.

8. RUN THE COMPLETE EXPERIMENT

After the quick test succeeds, run:

uv run minimal-caption run-all --config configs/default.yaml

This command performs the complete workflow:

- validates Flickr8k
- creates the dataset structure output
- verifies the model architecture
- trains the model for up to 30 epochs
- saves the best and latest checkpoints
- calculates BLEU-1, BLEU-4, and METEOR
- creates the t-SNE visualization of the 6-D latent vectors
- runs Gaussian and salt-and-pepper noise tests
- generates a sample caption result
- creates the final evidence image
- checks all required generated files

The full experiment can take several hours, especially when it runs on CPU.
The terminal should be kept open while training is running.

9. RESUME AN INTERRUPTED RUN

The latest checkpoint is saved after each completed epoch. If the complete
experiment is interrupted, resume it with:

uv run minimal-caption run-all --config configs/default.yaml --resume checkpoints/last.pt

This continues training from the saved epoch and then completes evaluation,
visualization, sensitivity testing, caption generation, and the final artifact
check.

10. MAIN OUTPUT FILES

The most important generated files are:

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

The dataset, checkpoints, and generated results are stored locally and are not
pushed to GitHub because they can be large.

11. FINAL OUTPUT CHECK

After the complete experiment finishes, run:

uv run minimal-caption submission-check --config configs/default.yaml --require-generated

A successful result should look similar to:

Submission check: PASS (source and generated artifacts; 41 files checked)

12. OPTIONAL CODE CHECKS

The following commands check formatting, source code, types, tests, coverage,
and package building:

uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run basedpyright
uv run pytest --cov=minimal_captioning --cov-branch --cov-report=term-missing
uv build

13. COMMON PROBLEMS

If uv is not recognized:
Close and reopen Visual Studio Code or PowerShell, and run uv --version again.

If Flickr8k cannot be found:
Check that data/flickr8k/Images contains the image files and that
data/flickr8k/captions.txt exists.

If METEOR reports a WordNet error:
Run uv run minimal-caption setup-nltk again.

If CUDA is not available:
The project will use the CPU automatically. This is slower but still works.

If training stops before completion:
Use the resume command with checkpoints/last.pt instead of starting again.

If the quick captions are poor:
This is normal because the quick test uses only a small dataset and two epochs.
Use configs/default.yaml for the final results.
