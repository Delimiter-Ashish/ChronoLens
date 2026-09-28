from __future__ import annotations

import json
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from chronolens.video import probe_video, read_frame


class VideoRetriever:
    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = None
        self.model = None
        self.index = None
        self.timestamps = None
        self.video_path = None

    def load_model(self) -> None:
        if self.model is not None:
            return

        print(f"Loading CLIP on {self.device}...")

        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _tensor(features):
        if isinstance(features, torch.Tensor):
            return features

        if hasattr(features, "pooler_output"):
            return features.pooler_output

        if hasattr(features, "last_hidden_state"):
            return features.last_hidden_state[:, 0]

        raise TypeError(f"Unsupported feature output: {type(features)}")

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.load_model()

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.inference_mode():
            features = self.model.get_image_features(
                pixel_values=pixel_values
            )

        features = self._tensor(features)
        features = torch.nn.functional.normalize(
            features.float(),
            dim=-1,
        )

        return features.cpu().numpy().astype("float32")

    def encode_text(self, text: str) -> np.ndarray:
        self.load_model()

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask"}
        }

        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)

        features = self._tensor(features)
        features = torch.nn.functional.normalize(
            features.float(),
            dim=-1,
        )

        return features.cpu().numpy().astype("float32")

    def build_index(
        self,
        video_path: str,
        sample_every: float = 1.0,
        batch_size: int = 16,
        output_dir: str = "outputs/retrieval",
    ) -> None:
        info = probe_video(video_path)

        self.video_path = info.path

        timestamps = np.arange(
            0.0,
            info.duration,
            sample_every,
            dtype=np.float32,
        )

        all_embeddings = []
        valid_timestamps = []

        for start in range(0, len(timestamps), batch_size):
            batch_times = timestamps[start:start + batch_size]

            images = []

            for timestamp in batch_times:
                frame = read_frame(
                    video_path,
                    float(timestamp),
                )

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                images.append(
                    Image.fromarray(rgb)
                )

            embeddings = self.encode_images(images)

            all_embeddings.append(embeddings)
            valid_timestamps.extend(
                float(t) for t in batch_times
            )

            print(
                f"Indexed {len(valid_timestamps)}/{len(timestamps)} frames"
            )

        matrix = np.concatenate(
            all_embeddings,
            axis=0,
        ).astype("float32")

        self.timestamps = np.asarray(
            valid_timestamps,
            dtype=np.float32,
        )

        self.index = faiss.IndexFlatIP(
            matrix.shape[1]
        )

        self.index.add(matrix)

        output = Path(output_dir)
        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(output / "index.faiss"),
        )

        np.save(
            output / "timestamps.npy",
            self.timestamps,
        )

        metadata = {
            "video": self.video_path,
            "model": self.model_id,
            "sample_every": sample_every,
            "frames_indexed": len(self.timestamps),
        }

        (output / "metadata.json").write_text(
            json.dumps(
                metadata,
                indent=2,
            )
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_gap: float = 2.0,
    ) -> list[dict]:
        if self.index is None:
            raise RuntimeError(
                "Index must be built before search."
            )

        query_embedding = self.encode_text(query)

        raw_k = min(
            self.index.ntotal,
            top_k * 10,
        )

        scores, indices = self.index.search(
            query_embedding,
            raw_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index < 0:
                continue

            timestamp = float(
                self.timestamps[index]
            )

            too_close = any(
                abs(
                    timestamp - result["timestamp"]
                ) < min_gap
                for result in results
            )

            if too_close:
                continue

            results.append(
                {
                    "timestamp": timestamp,
                    "score": float(score),
                }
            )

            if len(results) == top_k:
                break

        return results
