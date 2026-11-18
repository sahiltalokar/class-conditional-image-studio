# Model Card: Class-Conditional Image Studio using BigGAN

## Project and model names

- Project: **Class-Conditional Image Studio using BigGAN**
- Generator: **BigGAN-deep-128**
- Checkpoint: **osanseviero/BigGAN-deep-128**
- Language mapper: **Groq-hosted configured model**
- Safety classifier: **martin-ha/toxic-comment-model**
- Vision-language evaluator: **openai/clip-vit-base-patch32**

## Model purpose

The application maps text to a supported ImageNet-1K class and synthesizes a 128 x 128 RGB image conditioned on that class. CLIP provides optional text-image cosine similarity. DistilBERT provides a pre-generation toxicity screen.

## Intended use

- Academic demonstration of class-conditional GANs.
- Controlled comparison of low, medium, and high truncation.
- Exploration of ImageNet-1K class generation.
- Demonstration of BERT-family safety classification and CLIP multimodal evaluation.
- Manual fallback for offline or API-unavailable operation.

## Out-of-scope use

- Open-vocabulary or unrestricted text-to-image generation.
- Medical, forensic, legal, biometric, or safety-critical synthetic image use.
- Depiction of real people or claims that a synthetic image represents actual events.
- Automated moderation as the only safety control.
- Production child-safety or high-stakes harm detection.
- Claims that CLIP similarity is classification accuracy.

## BigGAN architecture

The application uses the checked-in PyTorch implementation in `models/biggan.py`. The checkpoint repository is `osanseviero/BigGAN-deep-128`. The project expects 128-dimensional latent noise, a 128-dimensional class embedding, a 1,000-class one-hot condition, BigGAN-deep residual blocks, spectral normalization, conditional batch normalization, and a self-attention block. The exact training dataset, training hyperparameters, and internal training curves cannot be verified from this application alone and are not invented here.

## Groq model

The default configured model is `openai/gpt-oss-20b`. A model name may also be supplied through `GROQ_MODEL`. Groq access is used for prompt-to-ImageNet mapping, not image synthesis or safety decisions.

## DistilBERT model

`martin-ha/toxic-comment-model` is a DistilBERT-family `DistilBertForSequenceClassification` checkpoint. It is assumed to output `LABEL_0` as non-toxic and `LABEL_1` as toxic, as used by the application. Its Jigsaw training provenance is referenced in the project plan, but independent license/training-document verification must be completed before public release.

## CLIP model

`openai/clip-vit-base-patch32` is a multimodal vision-language model used to normalize image and text embeddings and report cosine similarity. It is not a BERT model and is not a measure of image accuracy or truth.

## Training and evaluation information

The project does not train the generator or either pretrained classifier. It uses the released checkpoints through their existing loaders. Training details and benchmark claims must come from checkpoint cards or papers because they are not verifiable from source code. Actual project measurements are in `BENCHMARK_RESULTS.csv` and actual test findings are in `EVALUATION.md`.

## Known limitations

- Prompt attributes and backgrounds are not explicitly retained by the class-one-hot condition.
- Multi-object prompts are reduced to one class.
- Unsupported concepts fall back to a potentially distant ImageNet class or manual selection.
- Truncated sampling can reduce diversity or introduce less typical features.
- The 128 x 128 resolution limits visual detail.
- Safety classification can miss harmful wording and can vary across paraphrases.
- CLIP scores can reward partial semantic overlap without guaranteeing full prompt correctness.

## Bias and fairness considerations

ImageNet-1K has uneven demographic, geographic, and semantic coverage; this can bias mappings and outputs toward frequent or stereotyped concepts. Pretraining corpora for Groq, DistilBERT, CLIP, and BigGAN may contain representation and cultural biases that have not been independently audited here. The project should provide representative prompts across classes and avoid using outputs for identity or demographic claims.

## Safety considerations

Prompts are screened before generation. Screening fails closed when inference or loading fails. The safety classifier is not a complete harm detector. Public deployments should add rate limiting, abuse monitoring, human review, and a mechanism to report false negatives.

## Licensing and attribution

- BigGAN checkpoint source: `osanseviero/BigGAN-deep-128` on Hugging Face; the exact license from the checkpoint repository must be verified before redistribution.
- Safety model: `martin-ha/toxic-comment-model`; license and model card must be verified directly.
- CLIP: `openai/clip-vit-base-patch32`; license and model card must be verified directly.
- ImageNet: its terms and dataset restrictions must be reviewed separately for derivative use.
- Groq service: verify current Groq terms and model usage terms for the configured model.

No license or royalty status is asserted beyond the official sources.
