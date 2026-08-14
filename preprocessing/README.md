# preprocessing

Runs a pose detector over a folder tree and builds the index the live app
searches.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Runs on Linux, macOS and Windows. Two dependencies do the work: `mediapipe` for
detection and `pillow` for decoding. The pose model is downloaded on first use —
or reused from `live/public/mediapipe/` if `npm run setup` already fetched it,
so both halves of the project load the same file.

MediaPipe does not publish a wheel for every Python version. 3.11 to 3.13 are
known to work.

Two optional extras:

```bash
.venv/bin/pip install -e '.[vision]'   # Apple Vision backend, macOS only
.venv/bin/pip install -e '.[heif]'     # HEIC/HEIF decoding
```

`[vision]` is inert off macOS — the pyobjc requirements carry a
`sys_platform == 'darwin'` marker, so pip skips rather than fails on them.

## Use

Datasets live in `INPUT/` and are named, not pathed:

```bash
ln -s /path/to/output-images INPUT/output-images
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
| `bench` | Times GPU against CPU on this machine. Run before a long job. |
| `doctor` | Compares Vision against MediaPipe on a sample, macOS only. See below. |
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

## Compute

Detection runs MediaPipe BlazePose — the same model the browser runs. TFLite
executes it through a *delegate*, which is the plug-in that decides what
hardware does the arithmetic:

| `--delegate` | |
|---|---|
| `auto` (default) | try GPU, fall back to CPU if it does not work here |
| `gpu` | OpenGL ES compute shaders on Linux, Metal on macOS |
| `cpu` | XNNPACK, the optimised CPU path |

`detect` prints what it got:

```
compute device: GPU (TFLite delegate)
```

### Which is faster is not predictable

The pose model is small. A many-core CPU running XNNPACK across eight threads is
a genuine competitor to a mid-range GPU, and about half the total cost is JPEG
decoding, which is CPU work either way. So measure instead of guessing:

```bash
bodypose bench output-images -n 200
```

That times both delegates at several thread counts over a random sample of the
real dataset and prints an estimated wall-clock for a 600k-image corpus. Pass
the winner to the real run as `--delegate` and `-j`.

### The GPU probe

MediaPipe is C++ underneath, and when it dislikes something it calls `CHECK`,
which aborts the process. An abort is not a Python exception — nothing catches
it. A half-working GL driver could otherwise kill an eight-hour run partway
through with no chance to fall back.

So `auto` decides by running one frame through the GPU delegate in a
*subprocess*. If that aborts, the child dies, the parent reads the exit code and
uses the CPU, and the run continues. It costs one interpreter start.

The GPU path also needs 4-channel RGBA frames where the CPU path takes 3-channel
RGB; handing the GPU an RGB frame is one of the aborts described above, so the
backend picks the layout to match the delegate.

### Decoding

Decode is close to half the total cost. Pillow's `draft()` handles it: JPEG
stores an image as blocks of frequency coefficients, and the decoder can
reconstruct them at 1/2, 1/4 or 1/8 scale for proportionally less work. Asking
for 1024px (`--max-side`) from an 8000px scan therefore decodes about a
sixty-fourth of the data, and a 100-megapixel museum scan never materialises as
a full bitmap. It applies to JPEG only; other formats decode fully and are
resized afterwards.

## Backends

| `--backend` | |
|---|---|
| `mediapipe` (default) | The detector the browser also runs. Works everywhere. |
| `vision` | Apple Vision on the Neural Engine. macOS only, `[vision]` extra. |

Vision is faster on Apple silicon — 134 img/s against the CPU path's 19, measured
on an M4 Pro over 60 images at 1024px, with threading plateauing at 4 workers.
But it puts a *second* detector in the loop: the index would be built by Vision
and queried by MediaPipe in the browser. Both reduce to the same 17 COCO
keypoints, which is an assumption rather than a guarantee, and `doctor` tests it:

```bash
bodypose doctor output-images -n 60
```

It reports how closely the two agree, which joints disagree worst, and — the
important one — whether *mirroring* MediaPipe improves agreement. A left/right
convention mismatch is invisible to any single-detector test and would quietly
ruin every match, so it gets checked explicitly.

On the default MediaPipe backend there is nothing for `doctor` to check, because
both ends already run the same model. That is the reason it is the default.

## Model sizes

`--model-size lite|full|heavy` picks the BlazePose variant. All three emit the
same 33 landmarks and therefore the same 17-joint vector, so this is accuracy
against speed and not a change of vector space:

| | size | |
|---|---|---|
| `lite` | ~5 MB | default, and what the live app loads |
| `full` | ~9 MB | noticeably better on partial and small figures |
| `heavy` | ~29 MB | best, several times slower |

Indexing with one size and querying with another works, but the small systematic
differences between them widen the gap between an offline vector and a live one
for no benefit. Match them, or change `live/src/main.ts` to load the same size.

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
