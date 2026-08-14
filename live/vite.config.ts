/**
 * Vite is the whole server. Two middlewares are bolted on:
 *
 *   /api/*  the index artefacts, from OUTPUT/<dataset>/
 *   /img/*  the source images, from INPUT/<dataset>/
 *
 * Neither can be a static mount: INPUT/<dataset> is usually a symlink pointing
 * at another volume, and which dataset is in play is decided at run time. They
 * are registered on the preview server as well as the dev server so a
 * `vite build && vite preview` install behaves the same as development.
 *
 * Nothing needs configuring in the normal case. `npm run dev` finds the index
 * the preprocessor built and the images it was built from.
 */

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin, type ViteDevServer, type PreviewServer } from 'vite'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const REPO = path.resolve(HERE, '..')
const INPUT = path.resolve(process.env.BODY_INPUT ?? path.join(REPO, 'INPUT'))
const OUTPUT = path.resolve(process.env.BODY_OUTPUT ?? path.join(REPO, 'OUTPUT'))

const MIME: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.tif': 'image/tiff',
  '.tiff': 'image/tiff',
  '.bmp': 'image/bmp',
  '.heic': 'image/heic',
  '.heif': 'image/heif',
  '.json': 'application/json',
  '.bin': 'application/octet-stream',
}

/**
 * Which built index to serve.
 *
 * `BODY_ARTIFACTS` names a directory outright. Otherwise the dataset is
 * `BODY_DATASET`, or the only index in OUTPUT/ if there is just one — which is
 * the usual case and means no configuration at all.
 */
function artifactsDir(): string | null {
  if (process.env.BODY_ARTIFACTS) return path.resolve(process.env.BODY_ARTIFACTS)
  if (process.env.BODY_DATASET) return path.join(OUTPUT, process.env.BODY_DATASET)
  const built = listIndexes()
  if (built.length === 1) return path.join(OUTPUT, built[0])
  return null
}

function listIndexes(): string[] {
  const found: string[] = []
  const walk = (dir: string, prefix: string) => {
    let entries: fs.Dirent[]
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true })
    } catch {
      return
    }
    if (entries.some((e) => e.isFile() && e.name === 'index.json')) {
      found.push(prefix)
      return // an index directory has no nested indexes
    }
    for (const entry of entries) {
      if (entry.isDirectory() && !entry.name.startsWith('.')) {
        walk(path.join(dir, entry.name), prefix ? path.join(prefix, entry.name) : entry.name)
      }
    }
  }
  walk(OUTPUT, '')
  return found.filter(Boolean).sort()
}

interface Roots {
  images: string | null
  thumbs: string | null
}

/** Re-read on each request: rebuilding an index mid-session should just work. */
function readRoots(dir: string): Roots {
  try {
    const manifest = JSON.parse(fs.readFileSync(path.join(dir, 'index.json'), 'utf8'))
    // The dataset name wins over the absolute path baked in at build time, so
    // repointing INPUT/<dataset> at a local copy does not invalidate the index.
    const byDataset = manifest.dataset ? path.join(INPUT, manifest.dataset) : null
    return {
      images:
        process.env.BODY_IMAGES_ROOT ??
        (byDataset && fs.existsSync(byDataset) ? byDataset : manifest.images_root) ??
        null,
      thumbs: process.env.BODY_THUMBS_ROOT ?? manifest.thumbs_root ?? path.join(dir, 'thumbs'),
    }
  } catch {
    return { images: process.env.BODY_IMAGES_ROOT ?? null, thumbs: process.env.BODY_THUMBS_ROOT ?? null }
  }
}

/**
 * Resolve a request path under a root, refusing anything that escapes it.
 * These roots are whole photo libraries, so `../` has to be a hard no.
 *
 * Containment is checked after `realpath`, not before: `path.resolve` collapses
 * `..` textually but knows nothing about symlinks, so a link inside the images
 * root pointing anywhere on disk would otherwise be served happily. Resolving
 * both sides is also what makes INPUT/<dataset> work at all, since that is
 * itself normally a symlink to another volume.
 */
function safeJoin(root: string, relative: string): string | null {
  let realRoot: string
  let realFile: string
  try {
    realRoot = fs.realpathSync(root)
    realFile = fs.realpathSync(path.resolve(realRoot, relative))
  } catch {
    return null // missing file, dangling link, or an unreadable directory
  }
  return realFile === realRoot || realFile.startsWith(realRoot + path.sep) ? realFile : null
}

function send(res: import('node:http').ServerResponse, file: string, cache: string): boolean {
  let stat: fs.Stats
  try {
    stat = fs.statSync(file)
  } catch {
    return false
  }
  if (!stat.isFile()) return false

  res.setHeader('Content-Type', MIME[path.extname(file).toLowerCase()] ?? 'application/octet-stream')
  res.setHeader('Content-Length', stat.size)
  res.setHeader('Cache-Control', cache)
  fs.createReadStream(file).pipe(res)
  return true
}

function missing(res: import('node:http').ServerResponse, message: string): void {
  res.statusCode = 404
  res.setHeader('Content-Type', 'text/plain; charset=utf-8')
  res.end(message)
}

function noIndexMessage(): string {
  const built = listIndexes()
  if (built.length > 1) {
    return (
      `Several indexes in ${OUTPUT}. Name one:\n\n` +
      built.map((n) => `  BODY_DATASET=${n} npm run dev`).join('\n')
    )
  }
  return (
    `No index found in ${OUTPUT}\n\n` +
    'Build one:\n' +
    '  ln -s /path/to/images INPUT/my-dataset\n' +
    '  preprocessing/.venv/bin/bodypose run my-dataset --thumbs'
  )
}

function attach(server: ViteDevServer | PreviewServer): void {
  const dir = artifactsDir()
  const label = dir ? path.relative(REPO, dir) || dir : 'none'
  server.config.logger.info(`  ➜  index:  ${label}`)

  server.middlewares.use('/api', (req, res, next) => {
    const name = decodeURIComponent((req.url ?? '').split('?')[0].replace(/^\//, ''))
    if (!['index.json', 'meta.json', 'vectors.bin', 'weights.bin'].includes(name)) return next()
    if (!dir) return missing(res, noIndexMessage())
    // No caching: the artefacts are regenerated often and a stale vectors.bin
    // paired with a fresh meta.json is a confusing way to fail.
    if (!send(res, path.join(dir, name), 'no-store')) {
      missing(res, `${name} not found in ${dir}\n\n${noIndexMessage()}`)
    }
  })

  server.middlewares.use('/img', (req, res, next) => {
    const rel = decodeURIComponent((req.url ?? '').split('?')[0].replace(/^\//, ''))
    if (!rel || !dir) return next()
    const roots = readRoots(dir)

    // Thumbnails first — the originals can be 100MP museum scans, and fetching
    // one of those per match visibly stalls the display.
    if (roots.thumbs) {
      const file = safeJoin(roots.thumbs, rel + '.jpg')
      if (file && send(res, file, 'public, max-age=86400')) return
    }
    if (roots.images) {
      const file = safeJoin(roots.images, rel)
      if (file && send(res, file, 'public, max-age=86400')) return
    }
    missing(res, `image not found: ${rel}`)
  })
}

function artifactServer(): Plugin {
  return {
    name: 'body-artifact-server',
    configureServer: attach,
    configurePreviewServer: attach,
  }
}

export default defineConfig({
  plugins: [artifactServer()],
  server: { host: '127.0.0.1', port: 5173, open: false },
  preview: { host: '127.0.0.1', port: 4173 },
  optimizeDeps: { exclude: ['@mediapipe/tasks-vision'] },
})
