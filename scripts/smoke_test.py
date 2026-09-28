from __future__ import annotations

import sys

import cv2
import faiss
import gradio
import imageio_ffmpeg
import numpy as np
import torch
import transformers

import chronolens


def main():
    print("ChronoLens smoke test")
    print(f"chronolens:   {chronolens.__version__}")
    print(f"python:       {sys.version.split()[0]}")
    print(f"torch:        {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"gradio:       {gradio.__version__}")
    print(f"opencv:       {cv2.__version__}")
    print(f"numpy:        {np.__version__}")
    print(f"faiss:        {getattr(faiss, '__version__', 'ok')}")
    print(f"ffmpeg:       {imageio_ffmpeg.get_ffmpeg_exe()}")
    print(f"cuda:         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu:          {torch.cuda.get_device_name(0)}")
        print(f"gpu_count:    {torch.cuda.device_count()}")
    else:
        raise SystemExit("CUDA is not available.")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
