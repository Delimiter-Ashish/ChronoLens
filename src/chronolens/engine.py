from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from .evidence import VisualEvidence
from .media import cut_clip, format_timestamp, save_thumbnail, timeline_html
from .retrieval import ClipRetriever


class ChronoLensEngine:
    def __init__(self, output_root="outputs"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.retriever = ClipRetriever()
        self.evidence = VisualEvidence()
        self.video_path = None
        self.meta = None

    @property
    def device_label(self):
        if torch.cuda.is_available():
            return f"CUDA · {torch.cuda.get_device_name(0)}"
        return "CPU"

    def index_video(self, video_path, sample_every_s=2.0, batch_size=32):
        if not video_path:
            raise ValueError("Upload a video first.")
        self.video_path = video_path
        self.meta = self.retriever.build(
            video_path,
            output_root=self.output_root / "indexes",
            sample_every_s=sample_every_s,
            batch_size=batch_size,
        )
        return self.meta

    @staticmethod
    def _slug(text):
        text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
        return text[:48] or "query"

    def search(self, query, top_k=5, min_gap_s=5.0, evidence_prompt=None):
        if not self.video_path or not self.meta:
            raise RuntimeError("Index a video before searching.")

        hits = self.retriever.search(query, top_k=top_k, min_gap_s=min_gap_s)
        hit_dicts = self.retriever.hits_as_dicts(hits)

        query_dir = self.output_root / "queries" / self.meta["fingerprint"] / self._slug(query)
        query_dir.mkdir(parents=True, exist_ok=True)

        gallery, clip_paths = [], []
        for hit in hit_dicts:
            rank = hit["rank"]
            clip = cut_clip(
                self.video_path, hit["start"], hit["end"], query_dir / f"rank_{rank:02d}.mp4"
            )
            thumb = save_thumbnail(
                self.video_path, hit["timestamp"], query_dir / f"rank_{rank:02d}.jpg"
            )
            hit["clip"] = clip
            hit["thumbnail"] = thumb
            clip_paths.append(clip)
            gallery.append((
                thumb,
                f'#{rank} · {format_timestamp(hit["timestamp"])} · score {hit["score"]:.3f}'
            ))

        evidence_image, evidence_video, detections = None, None, []
        if hit_dicts:
            prompt = (evidence_prompt or query).strip()
            try:
                evidence_image, detections = self.evidence.render_frame(
                    self.video_path,
                    hit_dicts[0]["timestamp"],
                    prompt,
                    query_dir / "evidence_rank_01.jpg",
                )
                evidence_video = self.evidence.track_top_detection(
                    self.video_path,
                    hit_dicts[0]["timestamp"],
                    prompt,
                    query_dir / "evidence_track_rank_01.mp4",
                )
            except Exception as exc:
                evidence_image = hit_dicts[0]["thumbnail"]
                detections = [{"warning": str(exc)}]

        payload = {"query": query, "hits": hit_dicts, "detections": detections}
        (query_dir / "results.json").write_text(json.dumps(payload, indent=2))

        table = [[
            h["rank"],
            format_timestamp(h["timestamp"]),
            round(h["score"], 4),
            format_timestamp(h["start"]),
            format_timestamp(h["end"]),
        ] for h in hit_dicts]

        return {
            "top_video": clip_paths[0] if clip_paths else None,
            "evidence_image": evidence_image,
            "evidence_video": evidence_video,
            "table": table,
            "timeline": timeline_html(float(self.meta["duration"]), hit_dicts),
            "gallery": gallery,
            "hits": hit_dicts,
        }
