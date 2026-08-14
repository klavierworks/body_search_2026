# OUTPUT

One folder per dataset, holding the index built from it:

    OUTPUT/<dataset>/
      detections.jsonl   the raw detector output, append-only and resumable
      index.json         encoding parameters, filters, counts
      vectors.bin        float32, N x 34, unit norm per row
      weights.bin        float32, N x 17, per-joint confidence
      meta.json          paths, bounding boxes, scores
      thumbs/            web-sized copies, if `--thumbs` was passed

All of it is derived data, reproducible with `bodypose run <dataset>`, and all
of it is gitignored.

The live app reads from here. With one index present it needs no configuration;
with several, name one: `BODY_DATASET=<dataset> npm run dev`.
