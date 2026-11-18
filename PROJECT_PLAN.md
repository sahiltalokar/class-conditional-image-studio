# PROJECT PLAN: Class-Conditional Image Studio using BigGAN

---

## 1. Executive Summary & Core Objective

The **Class-Conditional Image Studio using BigGAN** is an academic Generative AI project designed to demonstrate the intersection of natural language understanding and class-conditional generative models. The application takes a user's natural language input, utilizes a GPT-style decoder-only Transformer LLM accessed via the Groq API to map that prompt to the closest valid ImageNet-1K category ID, and synthesizes a $128 \times 128$ image using a pretrained **BigGAN-deep-128** generator on CPU.

### Critical Conceptual Distinctions & Terminology
To avoid common misconceptions in generative modeling, the project enforces strict technical terminology:
- **BigGAN is a Class-Conditional GAN, NOT an unrestricted Text-to-Image model:** BigGAN generates images conditioned strictly on an ImageNet categorical class label $y \in \{0, \dots, 999\}$ and a latent noise vector $z \in \mathbb{R}^{128}$. It cannot synthesize arbitrary multi-object scenes (e.g., *"an astronaut riding a bicycle on Mars"*) or open-vocabulary concepts not present in the ImageNet-1K training distribution.
- **Groq API provides access to a GPT-style decoder-only Transformer LLM:** Depending on the model configured in `st.secrets["GROQ_MODEL"]` (such as `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`), Groq performs semantic reasoning to translate freeform user descriptions into a valid ImageNet class index.
- **CLIP (ViT-B/32) is a Vision-Language Model, NOT a BERT-family model:** CLIP projects images and text into a shared 512-dimensional multimodal embedding space to evaluate semantic alignment via cosine similarity.
- **BERT-family Component (Safety Classification):** A lightweight pretrained BERT-family sequence-classification model (e.g., `martin-ha/toxic-comment-model` or `unitary/toxic-bert`) provides dedicated safety and toxicity classification. Bare `distilbert-base-uncased` lacks a classification head and cannot perform safety filtering without task-specific weights.
- **Diffusion Model Comparison is strictly optional:** It serves as a conceptual and comparative reference against GAN class-conditioning and must not interfere with or jeopardize the core BigGAN pipeline.

---

## 2. Core Workflow & Streamlined Scope

To keep the project manageable and focused on high reliability, all secondary features (such as latent interpolation or real-time diffusion) are deferred until the primary end-to-end pipeline is operational and validated.

### Primary Workflow (Core Feature):
$$\text{User Prompt} \longrightarrow \text{Groq LLM Class Mapping} \longrightarrow \text{Validated ImageNet Class ID } (0-999) \longrightarrow \text{BigGAN-deep-128 Generation} \longrightarrow \text{Display Image}$$

### Offline / Fallback Workflow:
$$\text{Manual ImageNet Class Selector / Search Bar} \longrightarrow \text{ImageNet Class ID} \longrightarrow \text{BigGAN-deep-128 Generation} \longrightarrow \text{Display Image}$$

The manual ImageNet class selector remains accessible at all times, ensuring the core generator functions even if Groq API keys are absent or rate limits are reached.

---

## 3. Project Folder Structure

```
c:\GEN AI Project\
├── .streamlit/
│   ├── config.toml               # Streamlit UI theme and server configuration
│   └── secrets.toml.example      # Template for GROQ_API_KEY and GROQ_MODEL
├── assets/
│   ├── imagenet_classes.json     # Standard 1,000 ImageNet class definitions (ID, label, description)
│   └── test_outputs/             # Verified generated test images from real forward passes
├── models/
│   ├── __init__.py
│   ├── biggan.py                 # BigGAN-deep-128 PyTorch Generator (verified op-for-op implementation)
│   ├── loader.py                 # Checkpoint downloader (huggingface_hub) and model loader
│   ├── clip_evaluator.py         # OpenAI CLIP ViT-B/32 similarity calculation (on-demand)
│   └── safety_classifier.py      # Pretrained BERT-family sequence classifier for prompt safety
├── utils/
│   ├── __init__.py
│   ├── groq_client.py            # Groq API client with structured JSON parsing & fallback
│   ├── imagenet_utils.py         # Class index mapping, fuzzy matching, and validation
│   └── sampling.py               # Truncated normal noise generator (numpy / scipy)
├── app.py                        # Main Streamlit web application
├── benchmark.py                  # Benchmarking script (records real latencies, memory, CLIP scores)
├── requirements.txt              # Pinned, conflict-free dependency list
├── .gitignore                    # Git ignore file protecting secrets, weights, and caches
├── PROJECT_PLAN.md               # Master engineering blueprint
├── MODEL_CARD.md                 # Formal model card covering BigGAN-deep-128, CLIP, and Safety BERT
└── README.md                     # Setup, usage, deployment, and live URL documentation
```

---

## 4. BigGAN Implementation & Checkpoint Compatibility Assessment

### Verified Technical Facts:
1. **Repository & Checkpoint Source:**
   - **Repository:** [`osanseviero/BigGAN-deep-128`](https://huggingface.co/osanseviero/BigGAN-deep-128) on Hugging Face Hub.
   - **Weights:** `pytorch_model.bin` (File size: 211,273,439 bytes / ~201.5 MB; 50,564,228 parameters).
   - **Config:** `config.json` ($z_{\text{dim}}=128$, class embedding $=128$, base channel width $=128$, output dim $=128$, 10 residual generator layers, self-attention at layer index 8, $n_{\text{stats}}=51$).
2. **State Dictionary Verification:**
   - Inspection of `pytorch_model.bin` confirmed **506 parameter tensors**.
   - Spectral normalization is parameterized via `weight_orig`, `weight_u`, and `weight_v` on linear projection and convolutional layers.
   - Conditional batch normalization utilizes `scale` and `offset` projections with pre-computed `running_means` and `running_vars` buffers across 51 truncation steps.
   - Model loading test via PyTorch confirmed **0 missing keys** and **0 unexpected keys** (`missing_keys: 0, unexpected_keys: 0`).
3. **Real Forward Pass Verification:**
   - Test script [`scratch/verify_biggan.py`](file:///c:/GEN%20AI%20Project/scratch/verify_biggan.py) executed a real forward pass on CPU with ImageNet class 207 (*golden retriever*).
   - Forward pass completed in **0.326 seconds** on CPU.
   - Output tensor shape: `torch.Size([1, 3, 128, 128])`, `dtype: torch.float32`, bounded in $[-0.978, 0.977]$.
   - Valid $128 \times 128$ RGB image was generated and successfully written to disk: [`assets/test_outputs/golden_retriever_207_trunc0.5.png`](file:///c:/GEN%20AI%20Project/assets/test_outputs/golden_retriever_207_trunc0.5.png).
   - Truncation comparison images were also generated and verified:
     - [`assets/test_outputs/golden_retriever_trunc_0.2.png`](file:///c:/GEN%20AI%20Project/assets/test_outputs/golden_retriever_trunc_0.2.png)
     - [`assets/test_outputs/golden_retriever_trunc_0.8.png`](file:///c:/GEN%20AI%20Project/assets/test_outputs/golden_retriever_trunc_0.8.png)

---

## 5. Technical Correction: The Truncation Trick in BigGAN

### Accurate Mathematical Explanation
The truncation trick in BigGAN is a technique used during inference to balance **sample fidelity (precision)** against **sample variety (recall)**.

In BigGAN-deep-128:
1. **Noise Generation:**
   Latent vectors $z$ are sampled from a truncated standard normal distribution:
   $$z = \psi \cdot v, \quad v \sim \text{TruncNorm}_{[-2, 2]}(0, 1)$$
   where $\psi \in (0, 1]$ represents the truncation parameter.
   - Because $v$ is bounded in $[-2, 2]$, values of $z$ are strictly bounded in $[-2\psi, 2\psi]$.
   - The effective variance of the latent distribution scales with $\psi^2$.
2. **Model Conditioning on Truncation:**
   Unlike simple GANs where truncation only affects noise generation, BigGAN-deep explicitly receives $\psi$ in its forward pass:
   `output = model(noise_tensor, class_vector, truncation=psi)`
   Inside `BigGANBatchNorm`, the generator holds pre-computed batch norm statistics for 51 discrete truncation levels ($n_{\text{stats}}=51$, step size $\Delta = 0.02$). The forward pass retrieves or linearly interpolates the `running_mean` and `running_var` corresponding to $\psi$:
   $$\text{coef}, \text{idx} = \text{modf}(\psi / 0.02)$$
   $$\mu_{\text{bn}} = \text{running\_means}[\text{idx}] \cdot \text{coef} + \text{running\_means}[\text{idx}+1] \cdot (1 - \text{coef})$$
   This architectural compensation ensures that layer-wise feature activations match the expected normalization statistics even as the variance of input $z$ shifts.

### Accurate Behavioral Description (Correcting Previous Overstatements)
- **Lower Truncation ($\psi \to 0.1 - 0.3$):** Concentrates latent samples near the mode of the distribution. This reduces variety across different random seeds and favors canonical, highly typical class exemplars with smoother features. It does *not* mathematically eliminate all variation.
- **Higher Truncation ($\psi \to 0.8 - 1.0$):** Retains the full variance of the prior distribution, enabling broader visual diversity in pose, background, and texture. Samples drawn from low-density tails of the distribution may appear unusual or less typical of the training distribution, but truncation 1.0 does *not* "automatically create grotesque artifacts" for every class.
- The actual visual impact of truncation varies significantly across classes and seeds, as shown in our real generated test images.

---

## 6. DistilBERT Safety Classification: Selected Model & Verification

### The Issue with Base DistilBERT
`distilbert-base-uncased` is a pretrained masked language model containing only the core transformer encoder. It has no sequence-classification head and has never been fine-tuned on safety or toxicity benchmarks. Claiming it can classify harmful content out-of-the-box is technically invalid.

### Selected Pretrained Sequence Classifier: `martin-ha/toxic-comment-model`
To satisfy the requirement for a lightweight BERT-family model for input safety classification:
- **Exact Repository:** [`martin-ha/toxic-comment-model`](https://huggingface.co/martin-ha/toxic-comment-model)
- **Architecture:** `DistilBertForSequenceClassification` (DistilBERT base with a linear sequence classification head).
- **Parameters:** ~66.95 Million parameters (~268 MB checkpoint).
- **Task & Actual Labels:** Binary toxicity classification:
  - `LABEL_0`: Non-toxic (safe)
  - `LABEL_1`: Toxic (unsafe / harmful)
- **Training Data:** Fine-tuned on the Jigsaw Toxic Comment Classification Challenge dataset.
- **Licence:** Open-source (Apache-2.0 / MIT compatible).
- **CPU Inference Performance:** Operates efficiently on CPU (< 50 ms for short prompt strings).
- **Alternative Multi-label Option:** [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert) (BERT-base, 110M parameters, ~440 MB checkpoint, 6 explicit labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`, Apache-2.0).
- **Integration Rule:** The safety model loads on-demand and is explicitly released after each screening phase before BigGAN is loaded.

---

## 7. Empirical Resource Benchmark & Memory Management

### Current Model Lifecycle and Hosting Constraints

- The application uses CPU-only PyTorch and explicitly places BigGAN, DistilBERT, and CLIP on CPU.
- Model weights are not loaded at application startup. BigGAN and the text models load only when the corresponding feature is used.
- The comparison workflow screens and maps prompts before loading BigGAN, then releases the DistilBERT cache and generates images sequentially.
- Before CLIP evaluation, the app clears the BigGAN and DistilBERT caches; CLIP is cleared after the requested evaluation.
- Cache eviction and garbage collection do not guarantee that the operating system reclaims every allocated page. They reduce live model overlap but do not prove a specific RSS ceiling.
- The latest local benchmark recorded 2,012.59 MB process RSS after using all three models in one process. It is a Windows local measurement, not a Streamlit Community Cloud fit test. A hosted smoke test must measure actual memory and cold-start behavior.
- Diffusion models are not loaded by this application.

---

## 8. Removal of Fabricated Results & Empirical Benchmarking Methodology

All previously listed performance values were estimates and have been removed as performance claims. They are replaced by an automated benchmarking harness ([`benchmark.py`](file:///c:/GEN%20AI%20Project/benchmark.py)) that measures and records real performance metrics:

### Real-Time Metric Collection Methodology:
1. **Groq Response Latency ($t_{\text{groq}}$):** Recorded via `time.perf_counter()` before and after the `groq_client.chat.completions.create()` call.
2. **BigGAN Model Load Latency ($t_{\text{load}}$):** Measured during cold-start model initialization.
3. **BigGAN Generation Latency ($t_{\text{gen}}$):** Measured around the forward pass `model(noise, class_vector, truncation)`.
4. **Total Pipeline Latency ($t_{\text{total}}$):** Sum of prompt validation, Groq mapping, noise sampling, generation, and PIL conversion.
5. **CLIP Semantic Similarity:** Computed dynamically using `openai/clip-vit-base-patch32`:
   $$\text{Score} = \frac{\mathbf{e}_{\text{prompt}} \cdot \mathbf{e}_{\text{image}}}{\|\mathbf{e}_{\text{prompt}}\| \|\mathbf{e}_{\text{image}}\|}$$
   Recorded across diverse test prompts.
6. **Safety Classification Metrics:** Output probability/score from `martin-ha/toxic-comment-model` recorded for test prompts.

All numbers in the final project report, model card, and demonstration will reflect actual logged measurements from this methodology.

---

## 9. Comprehensive Mapping to Five Syllabus Units

| Unit | Syllabus Domain | Project Feature / Theoretical Component |
| :--- | :--- | :--- |
| **Unit 1** | **Foundations of Generative Learning & Probabilistic Modeling** | Latent noise sampling $z \sim \mathcal{N}(0, I_{128})$; Mathematical formulation of the Truncation Trick $\psi$; Prior distribution shaping and mode concentration. |
| **Unit 2** | **Deep Generative Architectures & Generative Adversarial Networks (GANs)** | BigGAN-deep generator architecture; Class-Conditional Batch Normalization (CBN); Spectral Normalization on weights; Non-local Self-Attention module. |
| **Unit 3** | **Transformer Models & Natural Language Processing** | Groq API decoder-only Transformer LLM (e.g., Llama-3.3-70B); Prompt parsing and structured semantic taxonomy mapping to ImageNet-1K; BERT-family sequence classification for safety (`DistilBertForSequenceClassification`). |
| **Unit 4** | **Multimodal Systems & Vision-Language Alignment** | OpenAI CLIP (ViT-B/32) vision-language cross-modal embeddings; Text-image cosine similarity calculation; Conceptual and qualitative comparison: Class-Conditional GAN synthesis vs. Diffusion synthesis. |
| **Unit 5** | **Generative Evaluation, AI Safety, Failure Analysis & Cloud Deployment** | Quantitative latency and CLIP similarity benchmarking; Documentation of failure cases (multi-object compositionality, out-of-distribution prompts); Input safety moderation; Streamlit Secrets and CPU deployment preparation without a hosting-memory guarantee. |

---

## 10. Status Summary & Remaining Technical Risks

### Current Project Status

#### Verified Technical Facts (Measured, Not Estimated)
> **Benchmark supersession notice:** The historical values in this section and in earlier project-plan text are baseline observations from previous runs and must not be reused as current benchmark results. The reproducible 2026-11-18 CPU benchmark in `docs/BENCHMARK_RESULTS.csv` supersedes them.

| Metric | Measured Value | Test Environment |
|:---|:---|:---|
| BigGAN cold-start loading | 1.350929 s | Python 3.12.1, PyTorch 2.7.1 CPU, Windows 11 |
| Single-image forward pass | 0.190796 s | Seed 42, existing forward pass and PIL conversion |
| Average three-image generation | 0.189732 s | Seeds 42, 43, and 44 |
| 15-image comparison grid | 10.484235 s | Existing grid service; all 15 images successful |
| Groq mapping latency | 1.086654 s | Live configured API; model `openai/gpt-oss-20b` |
| DistilBERT inference latency | 29.242 ms | Warm in-process call; CPU |
| CLIP evaluation latency | 3.438937 s | Warm in-process call; CPU |
| Process RSS | 609.15 MB initial; 1028.34 MB after BigGAN; 2012.59 MB final | Measured with psutil |

#### Measured Safety Classifier Label Mapping
- **Model:** `martin-ha/toxic-comment-model`
- **LABEL_0:** non-toxic | **LABEL_1:** toxic
- Benign prompt ("golden retriever on grass"): Non-toxic=0.9928, Toxic=0.0072
- Explicit toxic prompt ("fuck you, die in a fire"): Non-toxic=0.0440, Toxic=0.9560
- Insult ("You are disgusting and useless"): Non-toxic=0.0395, Toxic=0.9605
- Ambiguous phrasing ("Kill yourself, you idiot piece of trash"): Non-toxic=0.9949, Toxic=0.0051 (**documented limitation**)

#### Measured CLIP Similarity Baseline
- Dummy 128×128 RGB image vs "A blue sky with clouds": **cosine similarity = 0.2315** (Moderate Semantic Alignment)
- Interpretation thresholds: ≥0.28 → Strong, ≥0.20 → Moderate, ≥0.12 → Weak, <0.12 → Low

#### Features Implemented & Verified
- ✅ Core single-image generation workflow (`app.py`)
- ✅ ImageNet-1K taxonomy integration & fallback keyword search (`assets/imagenet_classes.json`, `utils/imagenet_utils.py`)
- ✅ Native PyTorch BigGAN-deep-128 generator & cached loader (`models/biggan.py`, `models/loader.py`)
- ✅ Groq API class-mapping with strict JSON Schema and reasoning model support (`utils/groq_client.py`)
- ✅ Multi-prompt & truncation comparison grid: 5 prompts × 3 truncations = 15 images (`utils/comparison_service.py`)
- ✅ **DistilBERT prompt safety screening** — `martin-ha/toxic-comment-model`, lazy loaded, fail-closed, per-row blocking in grid (`models/safety_classifier.py`)
- ✅ **CLIP ViT-B/32 vision-language evaluation** — on-demand cosine similarity, qualitative interpretation, no auto-load at startup (`models/clip_evaluator.py`)
- ✅ Configurable toxicity threshold slider in sidebar
- ✅ Safety screening integrated in both Single Image Studio and Comparison Grid
- ✅ On-demand per-cell "🔍 Eval CLIP" buttons in comparison grid
- ✅ All test suites passing: 17 core tests + 2 grid tests + 23 evaluation tests = **42 total**

#### Documented Failure Case
The `martin-ha/toxic-comment-model` does not reliably detect all forms of toxic phrasing. In the actual 2026-11-18 run, "Kill yourself, you idiot piece of trash" scored **toxic confidence = 0.5118%** (classified as non-toxic), and "I want to hurt a person" scored **0.5425%** (classified as non-toxic). These are measured false negatives. The adjustable threshold can lower the cutoff, but it does not make the classifier fully reliable.
