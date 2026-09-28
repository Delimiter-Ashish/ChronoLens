from __future__ import annotations

import json
import re
from pathlib import Path

from chronolens.retrieval import VideoRetriever
from chronolens.tracking import EvidenceTracker


class ChronoLensPipeline:
    def __init__(self):
        self.retriever = VideoRetriever()
        self.tracker = EvidenceTracker()
        self.indexed_video = None

    @staticmethod
    def _slug(text: str) -> str:
        text = re.sub(
            r"[^a-zA-Z0-9]+",
            "-",
            text.strip().lower(),
        )

        return text.strip("-") or "query"

    def index_video(
        self,
        video_path: str,
        sample_every: float = 1.0,
        batch_size: int = 16,
    ) -> None:

        print("\n[1/4] Building searchable video index...")

        self.retriever.build_index(
            video_path=video_path,
            sample_every=sample_every,
            batch_size=batch_size,
            output_dir="outputs/retrieval",
        )

        self.indexed_video = video_path

        print("Video index ready.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_gap: float = 2.0,
    ) -> list[dict]:

        if self.indexed_video is None:
            raise RuntimeError(
                "Index a video before searching."
            )

        print("\n[2/4] Searching video...")

        results = self.retriever.search(
            query=query,
            top_k=top_k,
            min_gap=min_gap,
        )

        if not results:
            raise RuntimeError(
                f"No retrieval results for query: {query}"
            )

        return results

    def investigate(
        self,
        query: str,
        evidence_prompt: str | None = None,
        tracking_duration: float = 4.0,
        top_k: int = 5,
    ) -> dict:

        results = self.search(
            query=query,
            top_k=top_k,
        )

        best = results[0]

        timestamp = float(
            best["timestamp"]
        )

        prompt = (
            evidence_prompt.strip()
            if evidence_prompt
            else query.strip()
        )

        query_dir = (
            Path("outputs/investigations")
            / self._slug(query)
        )

        query_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"\nBest moment: {timestamp:.2f}s "
            f"(score={best['score']:.4f})"
        )

        print("\n[3/4] Detecting visual evidence...")

        evidence_path, detections = (
            self.tracker.detector.save_evidence(
                video_path=self.indexed_video,
                timestamp=timestamp,
                prompt=prompt,
                output_path=str(
                    query_dir / "evidence.jpg"
                ),
            )
        )

        if not detections:
            raise RuntimeError(
                f"No visual evidence detected for: {prompt}"
            )

        print(
            f"Detected: {detections[0]['label']} "
            f"(score={detections[0]['score']:.3f})"
        )

        print("\n[4/4] Tracking evidence...")

        tracking = self.tracker.track(
            video_path=self.indexed_video,
            start_time=timestamp,
            prompt=prompt,
            duration=tracking_duration,
            output_path=str(
                query_dir / "tracking.mp4"
            ),
        )

        summary = {
            "query": query,
            "evidence_prompt": prompt,
            "best_timestamp": timestamp,
            "retrieval_score": float(
                best["score"]
            ),
            "retrieval_results": results,
            "evidence_image": evidence_path,
            "detection": detections[0],
            "tracking_video": tracking["output"],
            "tracked_frames": tracking[
                "tracked_frames"
            ],
            "failed_frames": tracking[
                "failed_frames"
            ],
        }

        summary_path = (
            query_dir / "result.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            )
        )

        return summary
