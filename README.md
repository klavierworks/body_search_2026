# BODY — pose search

Resurrection of the 2018 body-based image search engine. Stand in front of a
webcam, and the image whose figure most closely matches your pose comes up.

Two halves:

| | |
|---|---|
| [`preprocessing/`](preprocessing/) | Python CLI. Runs a pose detector on the Apple Neural Engine over a folder tree and builds a searchable index. |
| [`live/`](live/) | TypeScript + Vite. Webcam → MediaPipe → nearest neighbour → the matching image. |

## How the matching works

The matching is never image-to-image, and there is no learned embedding. Both
sides run a pose detector and compare the *configuration of the limbs*:

1. **Offline**, once — detect a pose in every source image. Keep the 17 COCO
   keypoints, each `(x, y, confidence)`. Images with no confident figure are
   dropped; that is what `-filtered-` meant on the original drive.
2. **Normalise** each pose into a comparable vector. Raw pixel coordinates are
   useless across images, since a figure can be anywhere in frame at any scale.
   So: take the bounding box of the confident joints, translate its centre to
   the origin, divide by its longer side, flatten to 34 floats, L2-normalise.
   What survives is only the pose.
3. **Live** — webcam → MediaPipe → *the same normalisation* → nearest
   neighbour → show the image.

Everything rests on step 2 producing the same vector on both sides. It is
implemented twice, in [`encoding.py`](preprocessing/bodypose/encoding.py) and
[`encoding.ts`](live/src/encoding.ts), and a golden-fixture test generated from
the Python side pins the TypeScript side to it — a silent drift between the two
would not degrade matching, it would destroy it, while looking like a bad model.

Distance is two-stage, following Move Mirror: cosine similarity for recall over
the whole corpus, then a confidence-weighted L2 over the shortlist, where each
joint counts in proportion to how sure *both* detectors were about it. That is
what stops an occluded ankle the detector was guessing at from dragging the
match somewhere silly.

### Two detectors, one vector space

The index is built with **Apple Vision** (Neural Engine, ~7× faster than the CPU
path) and queried with **MediaPipe BlazePose** in the browser. Different models —
but Vision's 19 joints and BlazePose's 33 landmarks both *contain* the 17 COCO
keypoints, so both reduce to the same vector.

That is an assumption worth testing rather than trusting, so it is testable:

```bash
bodypose doctor /path/to/images -n 60
```

runs both detectors over the same sample and reports how closely their vectors
agree, including whether mirroring one of them *improves* agreement — which is
how a left/right convention mismatch would show up, invisibly to any
single-detector test. If the answer is bad, `--backend mediapipe` builds the
index with the same detector the browser uses, trading speed for exactness.

## Quick start

```bash
# 1. index some images  (start small — the 43k output-images set takes ~5 min)
cd preprocessing
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m bodypose run /Volumes/SS2_OSX/output-images --thumbs

# 2. run the live app
cd ../live
npm install && npm run setup
BODY_ARTIFACTS=../preprocessing/out npm run dev
```

Then open http://127.0.0.1:5173 and hit **start camera**.

## Measured on this machine

M4 Pro, 14 cores, images on the internal SSD, 1024px decode:

| | |
|---|---|
| Pose detection, ANE | **7.46 ms/img** (134 img/s, one thread) |
| Pose detection, CPU | 53.58 ms/img (19 img/s) |
| End-to-end, 8 threads | **~138 img/s** — plateaus at 4 threads; the ANE is one shared block, not per-core |
| Search, 43k poses | well under 1 ms per frame in the browser |

So the full ~530k-image corpus is roughly an hour of detection, and the curated
43k `output-images` set about five minutes. Index building afterwards takes
seconds and can be re-run at different filter thresholds without touching the
images again — which is why detection and building are separate commands.

## What the original drive held

`DATASETS_BACKUP/` (614 GB, ~530k images across 23 folders) — Rijksmuseum
scrapes, MPII Human Pose, Flickr, Google image scrapes, primates, anatomy
plates, and a 78-image hand-curated `catalogue-filtered-ls78` that reads as the
aesthetic brief. `output-images/` (5.6 GB, 43,177 JPEGs at 1000px wide) is the
assembled working set: 19,211 Rijksmuseum images, 22,966 MPII frames, ~1,000
others, all written 15 Oct 2018.
