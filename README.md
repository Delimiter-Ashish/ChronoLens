# ChronoLens

### Long-Video Retrieval, Temporal Localization & Visual Evidence Tracking

**ChronoLens** is a GPU-accelerated video investigation system for searching long videos with natural language, retrieving relevant moments, grounding the queried visual evidence, and tracking it through time.

Instead of sending an entire long video to a vision-language model for every query, ChronoLens builds a searchable visual index and uses a hierarchical retrieval pipeline to focus computation on relevant moments.

> **Search video in natural language → retrieve the moment → ground the object → track the evidence.**

---

## What ChronoLens Does

Given a video and a query such as:

```text
find the man walking with a backpack
