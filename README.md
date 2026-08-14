# BODY — pose search

Resurrection of the 2018 body-based image search engine. Stand in front of a
webcam, and the image whose figure most closely matches your pose comes up.

Two halves:

| | |
|---|---|
| [`INPUT/`](INPUT/) | Datasets, one folder per dataset. Usually symlinks to images living elsewhere. |
| [`OUTPUT/`](OUTPUT/) | The index built from each dataset. Derived data, gitignored. |
| [`preprocessing/`](preprocessing/) | Python CLI. Runs MediaPipe pose detection over a dataset and builds the index. Linux, macOS or Windows; GPU where there is one. |
| [`live/`](live/) | TypeScript + Vite. Webcam → MediaPipe → nearest neighbour → the matching image. |

A dataset is referred to by name, not by path. `bodypose run output-images`
reads `INPUT/output-images` and writes `OUTPUT/output-images`, and the live app
finds both without being told where anything is. The images themselves can sit
on any volume, because `INPUT/<dataset>` is normally a symlink — and since the
index records the dataset name rather than an absolute path, repointing that
symlink at a local copy does not invalidate it.

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

### One detector, both ends

Both halves run **MediaPipe BlazePose**: the Python CLI over the corpus, and
`@mediapipe/tasks-vision` in the browser over the webcam. Same model, same 33
landmarks, same left/right convention, so the offline vector and the live vector
come out of one detector and there is no cross-model agreement to worry about.

That is what the `universal` branch changed, and it buys two things. Matching
gets a guarantee rather than an assumption. And the pipeline stops being
macOS-only — MediaPipe runs on Linux, macOS and Windows, on GPU or CPU.

Apple Vision is still available on macOS as `--backend vision`, where the Neural
Engine is faster than anything MediaPipe reaches from Python. It reintroduces
the two-detector question, which is what `bodypose doctor` measures:

```bash
bodypose doctor output-images -n 60
```

It runs both over the same sample and reports how closely their vectors agree,
including whether mirroring one *improves* agreement — which is how a left/right
convention mismatch would show up, invisibly to any single-detector test.

## Quick start

```bash
# once
cd preprocessing && python3 -m venv .venv && .venv/bin/pip install -e . && cd ..
cd live && npm install && npm run setup && cd ..

# point a dataset at some images
ln -s /path/to/output-images INPUT/output-images

# see which delegate is faster on this machine (optional, ~2 min)
preprocessing/.venv/bin/bodypose bench output-images -n 200

# index it
preprocessing/.venv/bin/bodypose run output-images --thumbs

# run the live app — no configuration, it finds the index
cd live && npm run dev
```

The pose model downloads on first use, or is reused from `live/public/mediapipe/`
if `npm run setup` already fetched it, so both halves load the same file.

`bodypose datasets` lists what is in `INPUT/` and what has been indexed. If more
than one index exists, tell the live app which to use with
`BODY_DATASET=output-images npm run dev`.

Then open http://127.0.0.1:5173 and hit **start camera**.

## Performance

Detection throughput depends on the machine, the delegate and the image sizes,
so the pipeline measures rather than assumes:

```bash
bodypose bench output-images -n 200
```

That times GPU against CPU at several thread counts over a sample of the real
dataset and estimates the wall-clock for a 600k-image corpus. Roughly half the
cost is JPEG decoding, which is CPU work whichever delegate runs the model, so
GPU and CPU often land closer together than expected.

For reference, on an M4 Pro with the `--backend vision` path (Apple Vision on
the Neural Engine, macOS only): 7.46 ms/img, 134 img/s single-threaded, ~138
img/s end-to-end at 8 threads, plateauing at 4. That is the ceiling to beat, not
what MediaPipe delivers.

Search itself is well under 1 ms per frame in the browser at 43k poses.

Index building takes seconds and can be re-run at different filter thresholds
without touching the images again — which is why detection and building are
separate commands.

## What the original drive held

`DATASETS_BACKUP/` (614 GB, ~530k images across 23 folders) — Rijksmuseum
scrapes, MPII Human Pose, Flickr, Google image scrapes, primates, anatomy
plates, and a 78-image hand-curated `catalogue-filtered-ls78` that reads as the
aesthetic brief. `output-images/` (5.6 GB, 43,177 JPEGs at 1000px wide) is the
assembled working set: 19,211 Rijksmuseum images, 22,966 MPII frames, ~1,000
others, all written 15 Oct 2018.
