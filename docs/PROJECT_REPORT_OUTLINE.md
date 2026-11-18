# Project Report Outline

## 1. Title and abstract

State the project title, purpose, methodology, and distinction between actual experimental findings and theoretical discussion.

## 2. U1 - Foundations of Generative AI

- Generative vs discriminative models.
- Generative AI lifecycle diagram.
- Probability distributions and latent sampling.
- Zero-shot vs few-shot prompting.
- ImageNet class-conditioning limitation.

## 3. U2 - GANs and LLMs

- Transformer-based LLM fundamentals.
- Groq role in prompt-to-class mapping.
- BigGAN-deep architecture.
- Class-conditional batch normalization and truncation.
- Prompt-based learning vs fine-tuning.

## 4. U3 - GPT, BERT and Multimodal AI

- GPT-style decoder model role.
- DistilBERT safety classification role.
- CLIP vision-language role.
- CLIP not being a BERT model.
- Relationship between generation and evaluation.

## 5. U4 - Tools and Generation

- Python, PyTorch, Streamlit, Hugging Face, Groq.
- BigGAN workflow.
- Single image studio.
- 5 x 3 comparison grid.
- Truncation experiment setup and actual images.

## 6. U5 - Responsible AI

- Safety screening.
- Measured false positives/negatives.
- Known failure case.
- Bias and class imbalance.
- Licensing and attribution verification.
- Model card summary.

## 7. Implementation

- Repository structure.
- Existing application components.
- Data flow.
- Reproducibility and fallback behavior.

## 8. Experimental evaluation

- Test dataset.
- Actual mapping outputs.
- Unsupported prompt findings.
- Actual safety scores.
- Actual CLIP scores.
- Truncation examples.
- Actual benchmark CSV.

## 9. Results

Clearly mark each table or observation as measured or theoretical. Include screenshots and hardware/software environment.

## 10. Limitations

- Class-only generation.
- 128 x 128 resolution.
- Safety misses.
- CLIP limitations.
- Bias and licensing unknowns.
- API and hardware variance.

## 11. Conclusion and remaining work

Summarize findings, unresolved checks, and public-deployment requirements.
