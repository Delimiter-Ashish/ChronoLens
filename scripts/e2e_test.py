from pathlib import Path

from chronolens.engine import ChronoLensEngine


def main():
    video = Path("data/chronolens_demo.mp4")
    if not video.exists():
        raise SystemExit("Run: python scripts/make_demo_video.py")

    engine = ChronoLensEngine(output_root="outputs/e2e")
    meta = engine.index_video(str(video), sample_every_s=1.0, batch_size=16)
    print("indexed:", meta)

    result = engine.search(
        "black backpack",
        top_k=3,
        min_gap_s=3.0,
        evidence_prompt="black backpack",
    )
    if not result["hits"]:
        raise SystemExit("No search hits returned.")

    for h in result["hits"]:
        print(
            f"rank={h['rank']} timestamp={h['timestamp']:.2f}s "
            f"score={h['score']:.4f}"
        )

    if not Path(result["top_video"]).exists():
        raise SystemExit("Top result clip was not generated.")

    print("E2E TEST PASSED")


if __name__ == "__main__":
    main()
