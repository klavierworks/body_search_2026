/**
 * Vite is the whole server. Two middlewares are bolted on:
 *
 *   /api/*  the index artefacts, read from wherever BODY_ARTIFACTS points
 *   /img/*  the source images themselves, which live outside the project
 *
 * Neither can be a static mount, because both directories are chosen at run
 * time and sit anywhere on disk. They are registered on the preview server as
 * well as the dev server so a `vite build && vite preview` install behaves the
 * same as development.
 */

import fs from 'node:fs'
import path from 'node:path'
import { defineConfig, type Plugin, type ViteDevServer, type PreviewServer } from 'vite'

const ARTIFACTS = path.resolve(process.env.BODY_ARTIFACTS ?? '../preprocessing/out')

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

interface Roots {
  images: string | null
  thumbs: string | null
}

/** Re-read on each request: rebuilding the index mid-session should just work. */
function readRoots(): Roots {
  try {
    const manifest = JSON.parse(fs.readFileSync(path.join(ARTIFACTS, 'index.json'), 'utf8'))
    return {
      images: process.env.BODY_IMAGES_ROOT ?? manifest.images_root ?? null,
      thumbs: process.env.BODY_THUMBS_ROOT ?? manifest.thumbs_root ?? null,
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
 * both sides also fixes a plain correctness bug on macOS, where `/tmp` is
 * itself a symlink to `/private/tmp` and a textual prefix test fails on paths
 * that are perfectly legitimate.
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

function attach(server: ViteDevServer | PreviewServer): void {
  server.middlewares.use('/api', (req, res, next) => {
    const name = decodeURIComponent((req.url ?? '').split('?')[0].replace(/^\//, ''))
    if (!['index.json', 'meta.json', 'vectors.bin', 'weights.bin'].includes(name)) return next()
    const file = path.join(ARTIFACTS, name)
    // No caching: the artefacts are regenerated often and a stale vectors.bin
    // paired with a fresh meta.json is a confusing way to fail.
    if (!send(res, file, 'no-store')) {
      missing(res, `${name} not found in ${ARTIFACTS}\nSet BODY_ARTIFACTS to your bodypose output directory.`)
    }
  })

  server.middlewares.use('/img', (req, res, next) => {
    const rel = decodeURIComponent((req.url ?? '').split('?')[0].replace(/^\//, ''))
    if (!rel) return next()
    const roots = readRoots()

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
