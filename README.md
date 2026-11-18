# Class-Conditional Image Studio using BigGAN

An academic Streamlit application that maps natural-language prompts to one of 1,000 ImageNet classes and generates a 128 × 128 image with pretrained BigGAN-deep-128.

**BigGAN is class-conditional image generation, not unrestricted text-to-image generation.** The generator receives an ImageNet class ID and latent noise; it does not follow arbitrary prompt details such as attributes, backgrounds, poses, or multiple objects. Groq maps a prompt to a supported class but does not change this limitation.

Try the app here: https://class-conditional-image-studio-hc6pztkn42kuutvbbpquwb.streamlit.app/

## Features

- Groq prompt-to-ImageNet-class mapping, with local keyword and manual-class fallbacks.
- Single Image Studio with reproducible seed and truncation controls.
- 5 × 3 comparison grid across three truncation values; generation is sequential.
- DistilBERT prompt safety screening.
- Optional, on-demand CLIP ViT-B/32 image-text similarity evaluation.
- Per-image PNG downloads and reproducible generation metadata.
- CPU inference; model checkpoints are loaded only when their features are used.

## Technology stack

Python 3.10–3.12, Streamlit, CPU-only PyTorch, Hugging Face Transformers and Hub, SciPy, NumPy, Pillow, pandas, psutil (benchmark only), and the Groq API.

## Architecture and workflow

```text
Prompt
  -> DistilBERT safety screening
  -> Groq class mapping (or local/manual fallback)
  -> validated ImageNet-1K class ID
  -> BigGAN-deep-128 + sampled latent vector
  -> 128 × 128 RGB image
  -> optional user-triggered CLIP text-image similarity
```

In the comparison workflow, prompts are screened and mapped before BigGAN is loaded. The safety classifier is then released, and images are generated one at a time. CLIP is loaded only after the user requests an evaluation; BigGAN and the safety classifier are released first, and CLIP is released after evaluation. These measures reduce overlap between model instances but do not guarantee that the app fits any hosting memory limit. No diffusion model is loaded.

## Local installation

Use Python 3.10, 3.11, or 3.12.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements select the CPU-only PyTorch 2.7.1 wheel. The BigGAN checkpoint and enabled evaluation models download from Hugging Face when first used and are cached outside the repository. Network access is required for initial downloads.

## Groq secrets configuration

For local development, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the placeholder:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
GROQ_MODEL = "openai/gpt-oss-20b"
```

Alternatively, set `GROQ_API_KEY` and optionally `GROQ_MODEL` as environment variables. Without a configured key, prompt mapping falls back to local taxonomy search and manual class selection remains available. Do not commit `.streamlit/secrets.toml`, paste credentials into source code, or include them in logs.

## Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The first use of safety screening or image generation may download its model checkpoint.

## Tests and benchmark

Run the unit and integration test suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Run syntax validation:

```bash
python -m compileall -q app.py benchmark.py models utils tests
```

`python benchmark.py` records local measurements to `docs/BENCHMARK_RESULTS.csv`; it may call Groq if configured and should be run only when a new benchmark is desired. The recorded 2,012.59 MB peak RSS is a local Windows measurement from a run that loaded BigGAN, DistilBERT, and CLIP in one process. It is not a Streamlit Community Cloud measurement or a hosting-fit guarantee.

## Streamlit Community Cloud deployment

1. Prepare a GitHub repository from the project source after reviewing the files. Do not upload `.venv/`, `weights/`, model caches, generated output, or `.streamlit/secrets.toml`; `.gitignore` excludes these paths. The example secrets file contains placeholders only.
2. In Streamlit Community Cloud, create an app from the repository and select `app.py` as the entry point.
3. Select Python 3.12 in the app's advanced settings.
4. In the app settings, add the following under **Secrets**, replacing the key placeholder with your own key:

   ```toml
   GROQ_API_KEY = "YOUR_GROQ_API_KEY"
   GROQ_MODEL = "openai/gpt-oss-20b"
   ```

5. Deploy and wait for the dependency installation and first-use model downloads to complete. The model weights are fetched from Hugging Face at runtime and are not stored in this repository.
6. Smoke-test manual class generation first, then Groq mapping, safety screening, comparison-grid generation, and optional CLIP evaluation. Monitor app restarts, load times, and memory use on the hosted instance.

The measured local peak was about 2.0 GB before the model lifecycle changes. Sequential model release is intended to lower model overlap, but memory allocators, Streamlit, PyTorch, model downloads, and host limits vary. **No hosted memory or deployment success is claimed**; a Community Cloud smoke test is required to determine whether the application can run reliably on the selected instance.

## Known limitations and responsible use

- ImageNet-1K coverage is limited; prompt mapping may choose an imperfect nearest class.
- BigGAN does not guarantee prompt attributes, scene composition, or output quality.
- The safety classifier can produce false positives and false negatives. It is not a complete moderation system; see the documented failure cases in [docs/EVALUATION.md](docs/EVALUATION.md).
- CLIP similarity measures embedding alignment, not image accuracy, truth, or prompt completeness.
- CPU generation, model downloads, and a 15-image grid may be slow or exceed a host's resource limits.
- ImageNet, checkpoint, and model licensing and use terms may impose restrictions; verify them before redistribution or public/commercial use.

## Model attribution and licensing

- BigGAN-deep-128 checkpoint: [`osanseviero/BigGAN-deep-128`](https://huggingface.co/osanseviero/BigGAN-deep-128).
- Safety classifier: [`martin-ha/toxic-comment-model`](https://huggingface.co/martin-ha/toxic-comment-model).
- CLIP: [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32).
- ImageNet: review the applicable dataset terms separately.
- Groq: review current service and configured-model terms.

The project does not independently assert or grant licenses for these checkpoints or datasets. Verify each official model card, checkpoint license, and service term before public release. Further technical and responsible-AI details are in [docs/MODEL_CARD.md](docs/MODEL_CARD.md), [docs/EVALUATION.md](docs/EVALUATION.md), and [PROJECT_PLAN.md](PROJECT_PLAN.md).
