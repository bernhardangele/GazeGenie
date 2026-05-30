# Installation and launch

## Option 1: GitHub Codespaces

This is the easiest browser-based setup.

1. Open the repository in GitHub Codespaces.
2. Wait for the development container to finish building.
3. The workspace is configured to start Streamlit on port `8501`.
4. Open the forwarded application URL when Codespaces shows that the app is running.
5. Use the file explorer to retrieve outputs from the `results/` folder.

Why this works:

- the devcontainer uses the published GazeGenie image,
- port `8501` is forwarded automatically,
- and `results/` is mounted so generated files are easy to access.

## Option 2: Docker

Use Docker if you want a reproducible local setup without managing Python packages yourself.

### Quick run

1. Create a results folder in the repository root:
   ```bash
   mkdir results
   ```
2. Start the published image:
   ```bash
   docker run --name gazegenie_app -p 8501:8501 -v $(pwd)/results:/app/results bangele1/gazegenie:latest
   ```
   On Windows Command Prompt, replace `$(pwd)` with `%cd%`.
3. Open `http://localhost:8501` in your browser.
4. When you stop the container, restart it later with:
   ```bash
   docker start -a gazegenie_app
   ```

### Build locally with docker compose

If you want to build from the checked-out source:

```bash
docker compose up --build
```

This mounts:

- `./results` to persist outputs,
- `./testfiles` for examples,
- `./logs` for logs.

The Docker image installs:

- Python dependencies,
- `tesseract-ocr`,
- and the Spanish OCR language pack currently requested by the OCR helper.

That makes Docker the safest local option if you plan to use image-based stimulus extraction.

## Option 3: Local Python environment

Use this if you want to modify the code or run directly from source.

### Basic setup

1. Install Miniforge or another Conda-compatible Python distribution.
2. Create an environment:
   ```bash
   conda create -n eye python=3.11 -y
   ```
3. Activate it:
   ```bash
   conda activate eye
   ```
4. Install Cairo support:
   ```bash
   conda install conda-forge::cairo
   ```
5. Install Python requirements from the repository root:
   ```bash
   pip install -r requirements.txt
   ```
6. Start the app:
   ```bash
   streamlit run app.py
   ```
7. Open `http://localhost:8501`.

### Extra requirement for OCR/image workflows

If you want to upload stimulus images instead of prepared stimulus tables, local source installs also need a working **Tesseract OCR** installation. The OCR helper looks for common installations such as:

- `/usr/bin/tesseract` on Linux,
- `C:/Program Files/Tesseract-OCR/tesseract.exe` on Windows.

If Tesseract is not installed, image-based stimulus extraction will not work.

## Option 4: Hosted Hugging Face Space

A hosted deployment is linked from the repository README. This is convenient for quick access, but keep the manuscript's caveats in mind:

- uploads and downloads can take time,
- available compute is limited,
- and browser-hosted sessions are less suitable for large batch jobs.

## Where outputs are saved

Depending on how you run the tool, outputs are either offered for download in the browser, written to `results/`, or both.

Typical output locations and artifacts include:

- zipped batch results,
- corrected fixation CSV files,
- combined fixations/saccades/word/sentence tables,
- eyekit JSON exports,
- OCR-derived stimulus tables,
- plots,
- OCR cache files in `results/ocr_cache`.
