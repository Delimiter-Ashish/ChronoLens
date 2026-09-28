from __future__ import annotations

from pathlib import Path
import inspect

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from chronolens.video import read_frame


class EvidenceDetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading Grounding DINO on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        )

        self.model.to(self.device)
        self.model.eval()

    def detect(
        self,
        image: Image.Image,
        prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ) -> list[dict]:

        prompt = prompt.strip()

        if not prompt.endswith("."):
            prompt += "."

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        fn = self.processor.post_process_grounded_object_detection
        params = inspect.signature(fn).parameters

        kwargs = {
            "target_sizes": [(image.height, image.width)],
            "text_threshold": text_threshold,
        }

        if "threshold" in params:
            kwargs["threshold"] = box_threshold
        elif "box_threshold" in params:
            kwargs["box_threshold"] = box_threshold

        if "input_ids" in params:
            kwargs["input_ids"] = inputs["input_ids"]

        results = fn(
            outputs,
            **kwargs,
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]

        labels = results.get(
            "text_labels",
            results.get("labels", []),
        )

        detections = []

        for i in range(len(boxes)):
            box = boxes[i].detach().cpu().tolist()
            score = float(scores[i].detach().cpu())

            if i < len(labels):
                label = labels[i]

                if isinstance(label, torch.Tensor):
                    label = str(label.item())
                else:
                    label = str(label)
            else:
                label = prompt.rstrip(".")

            detections.append(
                {
                    "label": label,
                    "score": score,
                    "box": box,
                }
            )

        detections.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return detections

    def detect_video_frame(
        self,
        video_path: str,
        timestamp: float,
        prompt: str,
    ) -> tuple[Image.Image, list[dict]]:

        frame = read_frame(
            video_path,
            timestamp,
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(rgb)

        detections = self.detect(
            image,
            prompt,
        )

        return image, detections

    @staticmethod
    def draw_detections(
        image: Image.Image,
        detections: list[dict],
    ) -> Image.Image:

        output = image.copy()

        draw = ImageDraw.Draw(output)

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(0, 255, 180),
                width=5,
            )

            label = (
                f"{detection['label']} "
                f"{detection['score']:.2f}"
            )

            draw.text(
                (x1 + 5, max(5, y1 - 25)),
                label,
                fill=(0, 255, 180),
            )

        return output

    def save_evidence(
        self,
        video_path: str,
        timestamp: float,
        prompt: str,
        output_path: str,
    ) -> tuple[str, list[dict]]:

        image, detections = self.detect_video_frame(
            video_path,
            timestamp,
            prompt,
        )

        annotated = self.draw_detections(
            image,
            detections,
        )

        output = Path(output_path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        annotated.save(
            output,
            quality=95,
        )

        return str(output), detections
