from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from chronolens.pipeline import ChronoLensPipeline


pipeline = ChronoLensPipeline()
CURRENT_VIDEO = None


def index_video(video_path, sample_every):
    global CURRENT_VIDEO

    if not video_path:
        return "❌ Upload a video first."

    try:
        pipeline.index_video(
            video_path=video_path,
            sample_every=float(sample_every),
            batch_size=16,
        )

        CURRENT_VIDEO = video_path

        return (
            "✅ Video indexed successfully\n\n"
            f"**Video:** `{Path(video_path).name}`\n\n"
            f"**Sampling:** every {sample_every:.1f}s"
        )

    except Exception as exc:
        return f"❌ Indexing failed:\n\n`{exc}`"


def investigate(video_path, query, evidence_prompt, duration):
    global CURRENT_VIDEO

    if not video_path:
        raise gr.Error("Upload a video first.")

    if not query or not query.strip():
        raise gr.Error("Enter a query.")

    try:
        if CURRENT_VIDEO != video_path:
            gr.Info("Indexing video automatically...")

            pipeline.index_video(
                video_path=video_path,
                sample_every=1.0,
                batch_size=16,
            )

            CURRENT_VIDEO = video_path

        prompt = evidence_prompt.strip() if evidence_prompt else query.strip()

        result = pipeline.investigate(
            query=query.strip(),
            evidence_prompt=prompt,
            tracking_duration=float(duration),
            top_k=5,
        )

        rows = []

        for rank, hit in enumerate(
            result["retrieval_results"],
            start=1,
        ):
            rows.append(
                [
                    rank,
                    round(hit["timestamp"], 2),
                    round(hit["score"], 4),
                ]
            )

        summary = {
            "query": result["query"],
            "best_timestamp": round(
                result["best_timestamp"],
                2,
            ),
            "retrieval_score": round(
                result["retrieval_score"],
                4,
            ),
            "detected_object": result[
                "detection"
            ]["label"],
            "detection_score": round(
                result["detection"]["score"],
                4,
            ),
            "tracked_frames": result[
                "tracked_frames"
            ],
            "failed_frames": result[
                "failed_frames"
            ],
        }

        return (
            result["evidence_image"],
            result["tracking_video"],
            rows,
            json.dumps(summary, indent=2),
        )

    except Exception as exc:
        raise gr.Error(str(exc))


CSS = """
.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
}

.hero {
    padding: 26px;
    border-radius: 22px;
    background:
        linear-gradient(
            135deg,
            #101d33,
            #07101d
        );
    border: 1px solid rgba(255,255,255,.08);
    margin-bottom: 20px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 4px;
}

.hero-subtitle {
    font-size: 17px;
    opacity: .75;
}

.badge {
    color: #5df2c2;
    font-size: 12px;
    letter-spacing: .16em;
    font-weight: 800;
    margin-bottom: 8px;
}
"""


with gr.Blocks(
    title="ChronoLens",
    css=CSS,
    theme=gr.themes.Soft(),
) as demo:

    gr.HTML(
        """
        <div class="hero">
            <div class="badge">
                LONG-VIDEO INTELLIGENCE
            </div>

            <div class="hero-title">
                ChronoLens
            </div>

            <div class="hero-subtitle">
                Search video with natural language,
                retrieve exact moments,
                and visually track the evidence.
            </div>
        </div>
        """
    )

    with gr.Row():

        with gr.Column(scale=6):

            video = gr.Video(
                label="Video",
                sources=["upload"],
            )

        with gr.Column(scale=4):

            gr.Markdown(
                "## Step 1 — Index video"
            )

            sample_every = gr.Slider(
                minimum=0.5,
                maximum=5.0,
                value=1.0,
                step=0.5,
                label="Frame sampling interval",
            )

            index_button = gr.Button(
                "⚡ Build Video Index",
                variant="primary",
            )

            index_status = gr.Markdown(
                "Waiting for video."
            )

    gr.Markdown("---")

    gr.Markdown(
        "## Step 2 — Investigate the video"
    )

    query = gr.Textbox(
        label="Natural-language query",
        placeholder=(
            "Find where the black backpack appears"
        ),
    )

    evidence_prompt = gr.Textbox(
        label="Visual target",
        placeholder=(
            "black backpack"
        ),
        info=(
            "Object/person to visually detect and track."
        ),
    )

    duration = gr.Slider(
        minimum=1,
        maximum=10,
        value=4,
        step=1,
        label="Evidence tracking duration (seconds)",
    )

    search_button = gr.Button(
        "🔎 Investigate",
        variant="primary",
        size="lg",
    )

    gr.Markdown("---")

    gr.Markdown(
        "## ChronoLens Evidence"
    )

    with gr.Row():

        evidence_image = gr.Image(
            label="Grounded Evidence",
            type="filepath",
        )

        tracking_video = gr.Video(
            label="Tracked Evidence",
        )

    gr.Markdown(
        "### Retrieved Moments"
    )

    results_table = gr.Dataframe(
        headers=[
            "Rank",
            "Timestamp (s)",
            "Similarity",
        ],
        datatype=[
            "number",
            "number",
            "number",
        ],
        interactive=False,
    )

    gr.Markdown(
        "### Investigation Summary"
    )

    result_json = gr.Code(
        language="json",
        label="Result",
    )

    index_button.click(
        fn=index_video,
        inputs=[
            video,
            sample_every,
        ],
        outputs=[
            index_status,
        ],
    )

    search_button.click(
        fn=investigate,
        inputs=[
            video,
            query,
            evidence_prompt,
            duration,
        ],
        outputs=[
            evidence_image,
            tracking_video,
            results_table,
            result_json,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
