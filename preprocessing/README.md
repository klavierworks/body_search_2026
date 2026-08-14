# preprocessing

Runs a pose detector over a folder tree and builds the index the live app
searches.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Two dependencies do the work: `pyobjc-framework-Vision` for detection and
`pyobjc-framework-Quartz` for image decoding. Both are thin bindings over
frameworks already on the machine, so there is **no model to download** for the
default backend.

The optional MediaPipe backend needs `pip install -e '.[mediapipe]'`. Its wheels
lag the newest Python releases — 3.11 or 3.12 is the safe choice.

## Use

Datasets live in `INPUT/` and are named, not pathed:

```bash
ln -s /Volumes/SS2_OSX/output-images INPUT/output-images
bodypose run output-images --thumbs          # detect + build + thumbnails
```

That reads `INPUT/output-images` and writes `OUTPUT/output-images`. Or as
separate steps, which is usually what you want:

```bash
bodypose detect output-images                # slow: reads every image
bodypose build output-images --min-keypoints 10   # seconds: re-cut without re-reading
bodypose thumbs output-images                # web-sized copies of what survived
```

The dataset name can be omitted for `build`, `thumbs`, `stats` and `query` when
there is only one index in `OUTPUT/`. `-o` overrides the output directory, and
an absolute path works anywhere a dataset name does.

`BODY_ROOT`, `BODY_INPUT` and `BODY_OUTPUT` override the three directories if
you need the layout somewhere else.

| command | |
|---|---|
| `detect` | Walks the tree, runs pose detection, appends to `detections.jsonl`. Resumable. |
| `build` | Applies filters and writes the index. Cheap, re-run freely. |
| `thumbs` | Web-sized JPEGs for indexed images only, and points `index.json` at them. |
| `doctor` | Compares Vision against MediaPipe on a sample. See below. |
| `datasets` | Lists what is in `INPUT/` and what has been indexed into `OUTPUT/`. |
| `stats` | Summarises a built index. |
| `query` | Nearest indexed poses to a query image, from the terminal. |

### Why detect and build are separate

Detection is the expensive step and you want to do it exactly once. Filter
thresholds are the thing you actually want to iterate on, and rebuilding an
index from existing detections takes seconds. So `detect` keeps every pose it
finds, however marginal, and `build` decides what counts.

Detection streams results to an append-only JSONL as they land, so an
interrupted run — closed lid, Ctrl-C, a disconnected volume — resumes from where
it stopped. Just re-run the same command.

## The Neural Engine

The default backend is Apple Vision's `VNDetectHumanBodyPoseRequest`, explicitly
pinned to the ANE. Python is only the driver: pyobjc drops the GIL around the
Vision call, so the worker threads are genuinely concurrent and no arithmetic
happens in Python.

`detect` prints which device it actually got:

```
compute device: NeuralEngine[Main]
```

If it says `auto (...)` instead, pinning failed and you are running on whatever
Vision chose. Measured on an M4 Pro over 60 images at 1024px:

| | ms/img | img/s |
|---|---|---|
| pinned to Neural Engine | **7.46** | 134 |
| pinned to GPU | 7.46 | 134 |
| left unpinned | 8.67 | 115 |
| pinned to CPU | 53.58 | 19 |

GPU ties with the ANE on this model, but the ANE is the better pick anyway: it
leaves the GPU free and draws far less power over an hour-long run. Note that
pinning also beats letting Vision schedule for itself.

Threading plateaus at 4 workers (~135 img/s end-to-end including decode) and
stays flat to 10 — the ANE is one shared block, not a per-core unit. The default
of 8 sits in the flat region with enough headroom that the machine stays usable.

Decode is close to half the total cost, which is why images go through ImageIO's
thumbnail path (`--max-side`, default 1024) rather than being fully decoded — a
100-megapixel museum scan never materialises as a bitmap.

## Checking the two detectors agree

The index is built with Vision and queried with MediaPipe. Both reduce to the
same 17 COCO keypoints, but that is an assumption, and this tests it:

```bash
pip install -e '.[mediapipe]'
bodypose doctor output-images -n 60
```

It reports how closely the two agree, which joints disagree worst, and — the
important one — whether *mirroring* MediaPipe improves agreement. A left/right
convention mismatch is invisible to any single-detector test and would quietly
ruin every match, so it gets checked explicitly.

If agreement is poor, `bodypose detect --backend mediapipe` builds the index
with the browser's own detector. Slower, but exact.

## Filters

`build` decides what counts as a usable figure:

| flag | default | |
|---|---|---|
| `--min-keypoints` | 8 | joints that must clear the confidence floor |
| `--min-score` | 0.3 | detector's confidence in the person overall |
| `--min-bbox-frac` | 0.05 | rejects background extras in crowd scenes |
| `--no-torso` | off | by default 3 of 4 shoulder/hip joints are required |
| `--max-poses-per-image` | 2 | |
| `--include-mirror` | off | also index every pose reflected — doubles recall and index size |

Without a torso the bounding box is not a body box, so the normalisation that
depends on it is meaningless. That is why it is on by default.

## Encoding parameters

These change the vector space itself, so they are written into `index.json` and
read back by the browser at runtime rather than being duplicated as constants.

- `--origin center|corner` — `corner` is what Move Mirror did.
- `--scale uniform|independent` — `uniform` divides both axes by the longer
  bbox side and preserves aspect ratio. `independent` divides each axis by its
  own extent, the Move Mirror behaviour, which makes every pose fill a unit
  square and throws away the difference between a wide stance and a narrow one.
  Also blows up on a figure standing perfectly straight, whose bbox width is
  near zero. `uniform` is the better default; `independent` is there to
  reproduce the original.
- `--conf-floor` — below this a joint counts as absent.

### Regenerating the parity fixture

If you change `encoding.py`, regenerate the golden vectors the TypeScript tests
check themselves against, then re-run `npm test` in `live/`:

```bash
.venv/bin/python -c "
import json, random, os
from bodypose.encoding import EncodingParams, encode, mirror_keypoints
rng = random.Random(1234); cases = []
for origin in ('center','corner'):
    for scale in ('uniform','independent'):
        p = EncodingParams(origin=origin, scale=scale, conf_floor=0.1)
        for _ in range(3):
            kps = [[round(rng.random(),5) for _ in range(3)] for _ in range(17)]
            e, m = encode(kps, p), encode(mirror_keypoints(kps), p)
            cases.append({'params': {'version': p.version, 'origin': origin,
                'scale': scale, 'conf_floor': p.conf_floor}, 'keypoints': kps,
                'vector': [round(v,7) for v in e.vector],
                'weights': [round(v,7) for v in e.weights],
                'bbox': [round(v,7) for v in e.bbox], 'visible': e.visible,
                'mirrored_vector': [round(v,7) for v in m.vector]})
os.makedirs('../live/src/__fixtures__', exist_ok=True)
json.dump(cases, open('../live/src/__fixtures__/golden.json','w'), indent=1)
print('wrote', len(cases), 'cases')"
```

## Output

```
OUTPUT/<dataset>/
  detections.jsonl   append-only, one line per image — the resume log
  detect-meta.json   what was scanned, with what, on which device
  index.json         encoding parameters, filters, counts, dataset name
  vectors.bin        float32 LE, N x 34, unit norm per row
  weights.bin        float32 LE, N x 17, per-joint confidence
  meta.json          columnar: paths, bounding boxes, scores, mirror flags
  thumbs/            optional, mirrors the source tree
```

`index.json` records the dataset name as well as the absolute path the images
had at build time. The name is what gets used, so moving the repository or
repointing the symlink at a local copy leaves the index valid.

The browser fetches `vectors.bin` and `weights.bin` straight into typed arrays —
43k poses is about 5.9 MB and 2.9 MB respectively.
