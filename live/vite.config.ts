/**
 * Vite is the whole server. Two middlewares are bolted on:
 *
 *   /api/*  the index artefacts, from OUTPUT/
 *   /img/*  the source images, from INPUT/
 *
 * Neither can be a static mount: INPUT/ normally contains symlinks pointing at
 * another volume, and paths are resolved per request. They are registered on
 * the preview server as well as the dev server so a `vite build && vite
 * preview` install behaves the same as development.
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
 * OUTPUT/ holds exactly one index, because INPUT/ is one dataset. So there is
 * nothing to choose between and no configuration in the normal case.
 * `BODY_ARTIFACTS` still points somewhere else outright if you need it.
 */
function artifactsDir(): string | null {
  if (process.env.BODY_ARTIFACTS) return path.resolve(process.env.BODY_ARTIFACTS)
  return fs.existsSync(path.join(OUTPUT, 'index.json')) ? OUTPUT : null
}

interface Roots {
  images: string | null
  thumbs: string | null
}

/** Re-read on each request: rebuilding an index mid-session should just work. */
function readRoots(dir: string): Roots {
  try {
    const manifest = JSON.parse(fs.readFileSync(path.join(dir, 'index.json'), 'utf8'))
    // Image paths in the index are relative to INPUT/, so INPUT/ wins over the
    // absolute path baked in at build time: repointing a symlink inside it at a
    // local copy does not invalidate the index.
    return {
      images:
        process.env.BODY_IMAGES_ROOT ??
        (fs.existsSync(INPUT) ? INPUT : manifest.images_root) ??
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
 * Two checks, because one is not enough on its own:
 *
 *  1. The relative path is rejected textually if any segment is `..`, empty or
 *     absolute. `path.resolve` would collapse `..` silently, so a request for
 *     `a/../../etc/passwd` has to be refused before it is resolved.
 *  2. The resolved file must sit inside the realpath of its *entry* — the
 *     top-level name under INPUT/ that the path starts with.
 *
 * The second check is what makes symlinked libraries work. INPUT/images is
 * normally a link to another volume, so the file's realpath does not sit under
 * INPUT/ at all and comparing against INPUT/ would refuse everything. Comparing
 * against the entry's own realpath still pins each file to the library it was
 * requested from, so a stray link inside that library pointing at /etc is
 * refused as before.
 */
function safeJoin(root: string, relative: string): string | null {
  const segments = relative.split(/[\\/]/)
  if (segments.some((s) => s === '..' || s === '' || path.isAbsolute(s))) return null

  // Anchor on the top-level entry when there is one, so a symlinked library is
  // its own containment boundary. A file directly in the root anchors on root.
  const anchor = segments.length > 1 ? path.join(root, segments[0]) : root
  let realAnchor: string
  let realFile: string
  try {
    realAnchor = fs.realpathSync(anchor)
    realFile = fs.realpathSync(path.resolve(root, relative))
  } catch {
    return null // missing file, dangling link, or an unreadable directory
  }
  return realFile === realAnchor || realFile.startsWith(realAnchor + path.sep) ? realFile : null
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
  return (
    `No index found in ${OUTPUT}\n\n` +
    'Build one:\n' +
    '  ln -s /path/to/images INPUT/images\n' +
    '  preprocessing/.venv/bin/bodypose run --thumbs'
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
  server: { host: '0.0.0.0', port: 5173 },
  preview: { host: '0.0.0.0', port: 4173 },
  optimizeDeps: { exclude: ['@mediapipe/tasks-vision'] },
})
