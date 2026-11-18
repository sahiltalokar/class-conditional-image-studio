"""
Class-Conditional Image Studio using BigGAN
Streamlit Web Application
Features:
1. Single Image Studio (Core prompt-to-image workflow with DistilBERT safety screening & CLIP evaluation)
2. Prompt & Truncation Comparison (5 prompts x 3 truncations = 15 images grid with row-level safety and on-demand CLIP)
"""

import time
import io
import pandas as pd
import torch
import streamlit as st
from PIL import Image

from models.loader import get_cached_biggan_model, release_cached_biggan_model
from models.safety_classifier import (
    check_prompt_safety,
    DEFAULT_SAFETY_THRESHOLD,
    release_safety_classifier,
)
from models.clip_evaluator import (
    compute_clip_similarity,
    interpret_clip_score,
    release_clip_evaluator,
)
from utils.groq_client import map_prompt_to_imagenet, get_groq_config
from utils.imagenet_utils import (
    load_imagenet_classes,
    validate_class_id,
    get_class_by_id,
    fuzzy_search_imagenet,
)
from utils.sampling import (
    generate_truncated_noise,
    prepare_class_vector,
    tensor_to_pil,
)
from utils.comparison_service import run_comparison_grid, DEFAULT_TRUNCATIONS


def _show_action_error(action, error):
    message = str(error)
    if isinstance(error, MemoryError) or any(
        phrase in message.lower()
        for phrase in ("out of memory", "cannot allocate memory", "not enough memory")
    ):
        st.error(
            f"{action} could not complete because the app ran out of memory. "
            "Try again after restarting the app, and avoid using CLIP while generating a grid."
        )
    else:
        st.error(f"{action}: {message}")


def _evaluate_clip(prompt, image):
    release_cached_biggan_model()
    release_safety_classifier()
    try:
        return compute_clip_similarity(prompt, image)
    finally:
        release_clip_evaluator()


# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Class-Conditional Image Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.0rem;
    }
    .info-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 1.0rem;
        font-size: 0.9rem;
        color: #1E293B;
    }
    .class-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #3730A3;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .safety-badge-pass {
        display: inline-block;
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .safety-badge-block {
        display: inline-block;
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .clip-badge {
        display: inline-block;
        background-color: #F3E8FF;
        color: #6B21A8;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .row-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 1.2rem;
    }
    .col-header {
        text-align: center;
        font-weight: 700;
        color: #1E3A8A;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
        border-bottom: 2px solid #DBEAFE;
        padding-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Class-Conditional Image Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Synthesizing 128×128 images conditioned on ImageNet-1K using pretrained BigGAN-deep-128.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
    <strong>Architecture Notice:</strong> BigGAN is a <em>class-conditional Generative Adversarial Network</em>,
    not an open-ended text-to-image model (like Stable Diffusion). It generates images corresponding strictly to one of the
    <strong>1,000 ImageNet categories</strong>. The natural-language prompt is semantically mapped to the closest ImageNet
    class via a Transformer LLM on Groq before synthesis. User prompts are screened for toxicity with
    <strong>DistilBERT</strong>, and generated imagery is evaluated with <strong>CLIP ViT-B/32</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🧭 Navigation")
    app_mode = st.radio(
        "Choose Mode:",
        ["🖼️ Single Image Studio", "📊 Prompt & Truncation Comparison"],
        index=0,
        help="Switch between single image generation and the 5x3 multi-prompt truncation comparison grid.",
    )

    st.markdown("---")
    st.subheader("🛡️ Safety Settings")
    safety_threshold = st.slider(
        "DistilBERT Toxicity Cutoff",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_SAFETY_THRESHOLD,
        step=0.05,
        help="Prompts classified as toxic/harmful with confidence above this cutoff will be blocked by DistilBERT.",
    )
    st.caption(
        f"Classifier: `martin-ha/toxic-comment-model` (Cutoff: **{safety_threshold:.0%}**)"
    )

    st.markdown("---")
    st.subheader("🔑 Groq Status")
    api_key_configured, model_name = get_groq_config()

    if api_key_configured:
        st.success(f"Groq prompt mapping enabled\nModel: `{model_name}`")
    else:
        st.warning(
            "Groq is not configured. Add `GROQ_API_KEY` in Streamlit Secrets; "
            "local keyword fallback and manual class selection remain available."
        )

    st.markdown("---")
    st.markdown(
        "**Technical Stack:**\n"
        "- Generator: BigGAN-deep-128 (CPU)\n"
        "- LLM: Groq (Llama-3.3-70B)\n"
        "- Taxonomy: ImageNet-1K (1,000 classes)\n"
        "- Safety: DistilBERT (`martin-ha/toxic-comment-model`)\n"
        "- Evaluation: CLIP (`openai/clip-vit-base-patch32`)\n"
        "- Resolution: 128 × 128 RGB\n"
        "- Sequential CPU inference (host memory limits vary)"
    )

# Session state initialization for Single Image Studio
if "single_gen_data" not in st.session_state:
    st.session_state["single_gen_data"] = None


# =============================================================================
# MODE 1: SINGLE IMAGE STUDIO
# =============================================================================
if app_mode == "🖼️ Single Image Studio":
    st.markdown("### 🖼️ Single Image Generation")

    # Sidebar controls specific to Single Image Studio
    with st.sidebar:
        st.markdown("---")
        st.subheader("⚙️ Single Image Settings")
        truncation = st.slider(
            "Truncation Threshold (ψ)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Lower values (0.2-0.4) favor typical exemplars; higher values (0.7-1.0) increase diversity.",
        )
        use_random_seed = st.checkbox("Randomize seed on generate", value=True)
        if not use_random_seed:
            single_seed = st.number_input("Custom Random Seed", min_value=0, max_value=999999, value=42, step=1)
        else:
            single_seed = None

    tabs = st.tabs(["💬 Natural Language Prompt", "📋 Manual Class Selector"])

    # Tab 1: Natural Language Prompt via Groq (with DistilBERT screening)
    with tabs[0]:
        prompt_input = st.text_input(
            "Enter what you would like to generate:",
            value="",
            placeholder="e.g. A fluffy golden retriever puppy sitting on green grass",
            help="Describe an object, animal, or scene. Screened with DistilBERT before Groq class mapping.",
        )
        generate_btn = st.button("🚀 Generate Image", key="btn_prompt", type="primary")

        if generate_btn:
            cleaned_prompt = prompt_input.strip()
            if not cleaned_prompt:
                st.error("Please enter a text prompt to generate an image.")
            else:
                # 1. DistilBERT Safety Screening
                release_cached_biggan_model()
                with st.spinner("Screening prompt for harmful content (DistilBERT)..."):
                    t_safety_0 = time.perf_counter()
                    safety_res = check_prompt_safety(cleaned_prompt, threshold=safety_threshold)
                    t_safety = time.perf_counter() - t_safety_0
                release_safety_classifier()

                if not safety_res["is_safe"]:
                    st.error(f"🛑 **Generation Blocked by Safety Screening:** {safety_res['reason']}")
                    if safety_res.get("error"):
                        _show_action_error("Safety screening failed", safety_res["error"])
                    with st.expander("🛡️ Safety Screening Diagnostics", expanded=True):
                        st.markdown(
                            f"""
                            - **Prompt:** *"{cleaned_prompt}"*
                            - **Toxicity Confidence:** `{safety_res['toxic_score']:.1%}`
                            - **Non-Toxic Confidence:** `{safety_res['non_toxic_score']:.1%}`
                            - **Threshold Cutoff:** `{safety_res['threshold']:.1%}`
                            - **Safety Classifier:** `martin-ha/toxic-comment-model`
                            - **Screening Latency:** `{t_safety*1000:.1f} ms`
                            """
                        )
                    st.session_state["single_gen_data"] = None
                else:
                    # 2. Prompt is safe -> Proceed to Groq Class Mapping
                    with st.spinner("Mapping prompt to closest ImageNet category via Groq..."):
                        t_groq_start = time.perf_counter()
                        success, groq_res, err_msg = map_prompt_to_imagenet(cleaned_prompt)
                        t_groq = time.perf_counter() - t_groq_start

                    if success and groq_res:
                        selected_class_id = groq_res["class_id"]
                        mapping_explanation = groq_res["explanation"]
                        source_mode = f"Groq LLM ({groq_res.get('model_used', 'LLM')}) [{t_groq*1000:.0f} ms]"
                    else:
                        st.warning(f"⚠️ {err_msg}")
                        st.info("Falling back to local keyword search across ImageNet taxonomy...")
                        matches = fuzzy_search_imagenet(cleaned_prompt, top_k=3)
                        if matches:
                            best = matches[0]
                            selected_class_id = best["class_id"]
                            mapping_explanation = (
                                f"Offline heuristic fallback selected closest matching class "
                                f"'{best['name']}' based on token overlap."
                            )
                            source_mode = "Local Keyword Search (Fallback)"
                        else:
                            selected_class_id = None
                            st.error("Could not find a matching ImageNet class. Please use the Manual Class Selector tab.")

                    if selected_class_id is not None:
                        is_valid, class_data = validate_class_id(selected_class_id)
                        if is_valid:
                            try:
                                model = get_cached_biggan_model()
                            except Exception as e:
                                _show_action_error("Failed to load BigGAN checkpoint", e)
                                st.stop()

                            try:
                                with st.spinner("Synthesizing 128×128 image with BigGAN-deep..."):
                                    t_gen_start = time.perf_counter()
                                    effective_seed = single_seed if single_seed is not None else int(time.time() * 1000) % 1000000
                                    noise_tensor = generate_truncated_noise(
                                        batch_size=1, dim_z=128, truncation=truncation, seed=effective_seed
                                    )
                                    class_tensor = prepare_class_vector(selected_class_id)

                                    with torch.no_grad():
                                        output_tensor = model(noise_tensor, class_tensor, truncation)

                                    generated_image = tensor_to_pil(output_tensor)
                                    t_gen = time.perf_counter() - t_gen_start
                            except Exception as e:
                                _show_action_error("BigGAN image generation failed", e)
                                st.stop()

                            # Save to session state
                            st.session_state["single_gen_data"] = {
                                "image": generated_image,
                                "class_data": class_data,
                                "prompt": cleaned_prompt,
                                "mapping_explanation": mapping_explanation,
                                "source_mode": source_mode,
                                "truncation": truncation,
                                "seed": effective_seed,
                                "t_gen": t_gen,
                                "safety": safety_res,
                                "clip_result": None,
                            }

    # Tab 2: Manual Class Selection
    with tabs[1]:
        all_classes = load_imagenet_classes()
        class_options = [c["display_name"] for c in all_classes]

        search_term = st.text_input("Filter ImageNet classes by name:", placeholder="Type to filter, e.g. shark, dog, car")
        if search_term.strip():
            filtered_classes = fuzzy_search_imagenet(search_term, top_k=20)
            filtered_options = [c["display_name"] for c in filtered_classes]
        else:
            filtered_options = class_options

        selected_option = st.selectbox(
            "Choose an ImageNet-1K Class:",
            options=filtered_options,
            index=0 if filtered_options else 0,
            help="Select any of the 1,000 supported ImageNet categories directly.",
        )

        manual_generate_btn = st.button("🎨 Generate Selected Class", key="btn_manual", type="primary")

        if manual_generate_btn and selected_option:
            raw_id = int(selected_option.split(":")[0].replace("Class", "").strip())
            class_info = get_class_by_id(raw_id)
            mapping_explanation = f"Direct user manual selection of ImageNet category '{class_info['name']}'."
            source_mode = "Manual Class Selection"

            is_valid, class_data = validate_class_id(raw_id)
            if is_valid:
                try:
                    model = get_cached_biggan_model()
                except Exception as e:
                    _show_action_error("Failed to load BigGAN checkpoint", e)
                    st.stop()

                try:
                    with st.spinner("Synthesizing 128×128 image with BigGAN-deep..."):
                        t_gen_start = time.perf_counter()
                        effective_seed = single_seed if single_seed is not None else int(time.time() * 1000) % 1000000
                        noise_tensor = generate_truncated_noise(
                            batch_size=1, dim_z=128, truncation=truncation, seed=effective_seed
                        )
                        class_tensor = prepare_class_vector(raw_id)

                        with torch.no_grad():
                            output_tensor = model(noise_tensor, class_tensor, truncation)

                        generated_image = tensor_to_pil(output_tensor)
                        t_gen = time.perf_counter() - t_gen_start
                except Exception as e:
                    _show_action_error("BigGAN image generation failed", e)
                    st.stop()

                st.session_state["single_gen_data"] = {
                    "image": generated_image,
                    "class_data": class_data,
                    "prompt": class_data["name"],
                    "mapping_explanation": mapping_explanation,
                    "source_mode": source_mode,
                    "truncation": truncation,
                    "seed": effective_seed,
                    "t_gen": t_gen,
                    "safety": None,
                    "clip_result": None,
                }

    # Single Image Generation Display & CLIP Evaluation
    gen_data = st.session_state.get("single_gen_data")
    if gen_data is not None:
        c_data = gen_data["class_data"]
        st.markdown("---")
        st.subheader("🖼️ Generation Output")

        # Status Badges
        badge_html = f"""
        <span class="class-badge">ImageNet Class {c_data['class_id']}</span>
        <strong>{c_data['name'].title()}</strong> (WordNet: <code>{c_data['wnid']}</code>)
        """
        if gen_data["safety"]:
            s_info = gen_data["safety"]
            badge_html += f"""
            &nbsp;<span class="safety-badge-pass">🛡️ DistilBERT Safety Passed (Tox: {s_info['toxic_score']:.1%})</span>
            """
        else:
            badge_html += """
            &nbsp;<span class="safety-badge-pass">🛡️ Direct ImageNet Taxonomy Category</span>
            """
        st.markdown(badge_html, unsafe_allow_html=True)

        if gen_data["mapping_explanation"]:
            st.caption(f"**Mapping Reasoning:** {gen_data['mapping_explanation']} *(via {gen_data['source_mode']})*")

        col_img, col_details = st.columns([1, 1])

        with col_img:
            st.image(
                gen_data["image"],
                caption=f"{c_data['name'].title()} (128×128, ψ={gen_data['truncation']})",
                use_container_width=True,
            )

            buf = io.BytesIO()
            gen_data["image"].save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="💾 Download Image (PNG)",
                data=byte_im,
                file_name=f"biggan_{c_data['name'].replace(' ', '_')}_trunc{gen_data['truncation']}.png",
                mime="image/png",
                key="btn_dl_single",
            )

        with col_details:
            st.markdown("#### Generation Details")
            st.markdown(
                f"""
                - **Evaluated Prompt:** *"{gen_data['prompt']}"*
                - **Resolution:** 128 × 128 pixels (RGB)
                - **Truncation (ψ):** `{gen_data['truncation']}`
                - **Random Seed:** `{gen_data['seed']}`
                - **Inference Hardware:** CPU
                - **BigGAN Synthesis Time:** `{gen_data['t_gen']:.3f} s` ({gen_data['t_gen']*1000:.0f} ms)
                """
            )

            # CLIP Vision-Language Evaluation
            st.markdown("---")
            st.markdown("#### 🔍 CLIP Vision-Language Evaluation")
            st.caption(
                "Calculates cosine similarity between the evaluated text prompt and the synthesized image "
                "using pretrained **CLIP ViT-B/32**. Loaded lazily on-demand to conserve memory."
            )

            eval_clip_btn = st.button("📊 Compute CLIP Similarity Score", key="btn_eval_clip_single")

            if eval_clip_btn:
                with st.spinner("Computing vision-language alignment with CLIP ViT-B/32..."):
                    t_clip_0 = time.perf_counter()
                    clip_res = _evaluate_clip(gen_data["prompt"], gen_data["image"])
                    t_clip = time.perf_counter() - t_clip_0
                    clip_res["eval_latency"] = t_clip
                    gen_data["clip_result"] = clip_res
                    st.session_state["single_gen_data"] = gen_data

            if gen_data.get("clip_result"):
                c_res = gen_data["clip_result"]
                if c_res.get("error"):
                    _show_action_error("CLIP evaluation failed", c_res["error"])
                else:
                    score = c_res["similarity"]
                    interp = c_res["interpretation"]
                    st.metric(
                        label="CLIP Similarity Score (Cosine Similarity)",
                        value=f"{score:.4f}",
                        delta=interp,
                        delta_color="normal",
                    )
                    st.markdown(
                        f"""
                        <div class="info-card">
                        <strong>Alignment Label:</strong> <span class="clip-badge">{interp}</span><br>
                        <small>{c_res['explanation']}</small><br>
                        <small><em>Evaluated in {c_res.get('eval_latency', 0.0)*1000:.1f} ms on CPU.</em></small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Click **'Compute CLIP Similarity Score'** to screen vision-language alignment.")


# =============================================================================
# MODE 2: PROMPT & TRUNCATION COMPARISON (5 PROMPTS x 3 TRUNCATIONS)
# =============================================================================
elif app_mode == "📊 Prompt & Truncation Comparison":
    st.markdown("### 📊 Prompt & Truncation Comparison")
    st.markdown(
        "Enter **5 different natural-language prompts** to generate a **5 × 3 comparison grid** (15 images total) "
        "at truncation thresholds **ψ = 0.2**, **ψ = 0.4**, and **ψ = 0.8**. "
        "Each prompt is screened by **DistilBERT** before generation; toxic prompts are blocked at the row level. "
        "Images are synthesized sequentially on CPU to limit peak memory use."
    )

    # Pre-configured default prompts for immediate testing
    default_prompts = [
        "A golden retriever dog sitting on green grass",
        "A red sports car parked on a mountain road",
        "A great white shark swimming in ocean water",
        "A plate of delicious hot cheese pizza",
        "An espresso coffee mug resting on a wooden table",
    ]

    st.markdown("#### 1. Enter 5 Prompts")
    prompt_cols = st.columns(2)
    p_inputs = []

    with prompt_cols[0]:
        p0 = st.text_input("Prompt 1:", value=default_prompts[0], key="grid_p0")
        p1 = st.text_input("Prompt 2:", value=default_prompts[1], key="grid_p1")
        p2 = st.text_input("Prompt 3:", value=default_prompts[2], key="grid_p2")
        p_inputs.extend([p0, p1, p2])

    with prompt_cols[1]:
        p3 = st.text_input("Prompt 4:", value=default_prompts[3], key="grid_p3")
        p4 = st.text_input("Prompt 5:", value=default_prompts[4], key="grid_p4")
        base_seed_val = st.number_input(
            "Base Random Seed (Each image receives a distinct offset seed):",
            min_value=1,
            max_value=999999,
            value=42,
            step=1,
            key="grid_base_seed",
        )
        p_inputs.extend([p3, p4])

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        generate_grid_btn = st.button("🚀 Generate Comparison Grid", key="btn_grid", type="primary")

    # Session State initialization for Grid Results
    if "grid_results" not in st.session_state:
        st.session_state["grid_results"] = None

    # Handle Generation Trigger
    if generate_grid_btn:
        valid_prompts = [p.strip() for p in p_inputs if p.strip()]
        if len(valid_prompts) < 1:
            st.error("Please provide at least one valid prompt to generate the comparison grid.")
        else:
            # Progress Bar Setup
            prog_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(current, total, msg):
                prog_bar.progress(min(current / total, 1.0))
                status_text.caption(f"⏳ **[{current}/{total}]** {msg}")

            with st.spinner("Screening prompts and generating comparison images sequentially..."):
                release_cached_biggan_model()
                try:
                    results = run_comparison_grid(
                        model=None,
                        prompts=p_inputs,
                        truncations=[0.2, 0.4, 0.8],
                        base_seed=int(base_seed_val),
                        safety_threshold=safety_threshold,
                        progress_callback=update_progress,
                    )
                except Exception as e:
                    _show_action_error("Comparison grid generation failed", e)
                    st.stop()

            prog_bar.progress(1.0)
            status_text.success(
                f"✅ Comparison workflow finished! Synthesized {results['total_images']} images in {results['total_time']:.2f} seconds."
            )
            st.session_state["grid_results"] = results

    # Render Results Grid if available
    grid_data = st.session_state["grid_results"]

    if grid_data is not None:
        st.markdown("---")
        st.subheader("🖼️ Comparison Grid (5 Prompts × 3 Truncations)")

        # Grid Column Headers
        header_cols = st.columns([1.1, 1, 1, 1])
        with header_cols[0]:
            st.markdown("**Prompt & Class / Safety**")
        with header_cols[1]:
            st.markdown('<div class="col-header">ψ = 0.2<br><small>High Fidelity / Canonical</small></div>', unsafe_allow_html=True)
        with header_cols[2]:
            st.markdown('<div class="col-header">ψ = 0.4<br><small>Balanced Sampling</small></div>', unsafe_allow_html=True)
        with header_cols[3]:
            st.markdown('<div class="col-header">ψ = 0.8<br><small>High Diversity / Variance</small></div>', unsafe_allow_html=True)

        table_records = []

        for row in grid_data["rows"]:
            r_idx = row["row_index"]
            row_cols = st.columns([1.1, 1, 1, 1])

            # Column 0: Prompt Info, Safety Status & Selected Class
            with row_cols[0]:
                st.markdown(f"**Row {r_idx + 1}:** *\"{row['original_prompt']}\"*")

                # If blocked by safety screening
                if row.get("safety") and not row["safety"]["is_safe"]:
                    s_info = row["safety"]
                    st.markdown(
                        f"""
                        <span class="safety-badge-block">🛑 Harmful Content Blocked</span><br>
                        <small>DistilBERT Toxicity: <strong>{s_info['toxic_score']:.1%}</strong> (Cutoff: {s_info['threshold']:.1%})</small>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Reason: {s_info['reason']}")
                elif row["class_id"] is not None:
                    st.markdown(
                        f"""
                        <span class="class-badge">Class {row['class_id']}</span>
                        <strong>{row['class_name'].title()}</strong>
                        """,
                        unsafe_allow_html=True,
                    )
                    if row.get("safety"):
                        st.markdown(
                            f'<span class="safety-badge-pass">🛡️ DistilBERT Safe ({row["safety"]["toxic_score"]:.1%})</span>',
                            unsafe_allow_html=True,
                        )
                    st.caption(f"*{row.get('mapping_source', 'Mapping')}*")
                elif row["error"]:
                    st.error(f"⚠️ {row['error']}")

            # If row was blocked by safety, show blocked placeholders across the 3 columns
            if row.get("safety") and not row["safety"]["is_safe"]:
                for c_idx in range(3):
                    with row_cols[c_idx + 1]:
                        st.warning(f"Generation blocked\n(Toxicity: {row['safety']['toxic_score']:.1%})")
                st.markdown("<hr style='margin: 0.8rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
                continue

            # Columns 1, 2, 3: Truncation Images (0.2, 0.4, 0.8)
            for c_idx, item in enumerate(row["items"]):
                trunc_val = item["truncation"]
                col_target = row_cols[c_idx + 1]

                with col_target:
                    if item["image"] is not None:
                        st.image(
                            item["image"],
                            caption=f"ψ={trunc_val} | Seed: {item['seed']}",
                            use_container_width=True,
                        )

                        # Individual Download Button
                        img_buf = io.BytesIO()
                        item["image"].save(img_buf, format="PNG")
                        img_bytes = img_buf.getvalue()

                        st.download_button(
                            label=f"💾 Download (ψ={trunc_val})",
                            data=img_bytes,
                            file_name=f"biggan_r{r_idx+1}_{row['class_name']}_trunc{trunc_val}.png",
                            mime="image/png",
                            key=f"dl_r{r_idx}_c{c_idx}",
                        )

                        # On-demand CLIP evaluation per image cell
                        clip_key = f"clip_r{r_idx}_c{c_idx}"
                        if item.get("clip_score") is not None:
                            st.caption(
                                f"**CLIP Score:** `{item['clip_score']:.4f}` "
                                f"({interpret_clip_score(item['clip_score'])})"
                            )
                        else:
                            if st.button("🔍 Eval CLIP", key=f"btn_{clip_key}"):
                                with st.spinner("Evaluating with CLIP..."):
                                    c_eval = _evaluate_clip(row["original_prompt"], item["image"])
                                    if c_eval.get("similarity") is not None:
                                        item["clip_score"] = c_eval["similarity"]
                                        st.session_state["grid_results"] = grid_data
                                        st.rerun()
                                    elif c_eval.get("error"):
                                        _show_action_error("CLIP evaluation failed", c_eval["error"])

                        # Record for Summary Table
                        table_records.append({
                            "Row": r_idx + 1,
                            "User Prompt": row["original_prompt"],
                            "ImageNet Class": row["class_name"].title(),
                            "Class ID": row["class_id"],
                            "Toxicity (%)": f"{row['safety']['toxic_score']:.1%}" if row.get("safety") else "N/A",
                            "Truncation (ψ)": trunc_val,
                            "Seed": item["seed"],
                            "Synthesis Time (s)": round(item["generation_time"], 3),
                            "CLIP Similarity": f"{item['clip_score']:.4f}" if item.get("clip_score") is not None else "Not evaluated",
                        })
                    elif item["error"]:
                        _show_action_error("Image generation failed", item["error"])

            st.markdown("<hr style='margin: 0.8rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

        # Summary and Reproducibility Table
        st.markdown("#### 📋 Generation Metadata & Reproducibility Table")
        if table_records:
            df = pd.DataFrame(table_records)
            st.dataframe(df, use_container_width=True, hide_index=True)

            avg_time = df["Synthesis Time (s)"].mean()
            st.caption(
                f"**Performance Summary:** Total Images: **{len(df)}/15** | "
                f"Total Synthesis Time: **{grid_data['total_time']:.2f} s** | "
                f"Average CPU Inference per Image: **{avg_time:.3f} s** ({avg_time*1000:.0f} ms)"
            )
            st.caption(
                "💡 *Note:* CLIP similarity scores evaluate semantic vision-language alignment in joint ViT-B/32 embedding space. "
                "They are computed on-demand to conserve RAM."
            )
