/**
 * Put MediaPipe's runtime and model under public/ so the app has no CDN
 * dependency — an installation on a gallery network with no outbound access
 * still starts.
 *
 * The wasm comes out of node_modules; the .task model has to be downloaded
 * once. Run `npm run setup` after `npm install`.
 */

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const target = path.join(root, 'public', 'mediapipe')

const MODELS = {
  lite: 'pose_landmarker_lite',
  full: 'pose_landmarker_full',
  heavy: 'pose_landmarker_heavy',
}

const wanted = process.argv.slice(2).filter((a) => a in MODELS)
const models = wanted.length ? wanted : ['lite']

fs.mkdirSync(target, { recursive: true })

// --- wasm runtime -----------------------------------------------------
const wasmSource = path.join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm')
if (!fs.existsSync(wasmSource)) {
  console.error('@mediapipe/tasks-vision is not installed. Run `npm install` first.')
  process.exit(1)
}
fs.cpSync(wasmSource, path.join(target, 'wasm'), { recursive: true })
console.log(`wasm    -> public/mediapipe/wasm (${fs.readdirSync(path.join(target, 'wasm')).length} files)`)

// --- models -----------------------------------------------------------
for (const key of models) {
  const name = MODELS[key]
  const dest = path.join(target, `${name}.task`)
  if (fs.existsSync(dest) && fs.statSync(dest).size > 0) {
    console.log(`${key.padEnd(7)} -> already present, skipping`)
    continue
  }
  const url =
    `https://storage.googleapis.com/mediapipe-models/pose_landmarker/${name}/float16/latest/${name}.task`
  process.stdout.write(`${key.padEnd(7)} -> downloading… `)
  const res = await fetch(url)
  if (!res.ok) {
    console.error(`\nfailed: ${url} -> ${res.status} ${res.statusText}`)
    process.exit(1)
  }
  const bytes = Buffer.from(await res.arrayBuffer())
  fs.writeFileSync(dest, bytes)
  console.log(`${(bytes.length / 1e6).toFixed(1)} MB`)
}

console.log('\nready. `npm run dev` next.')
console.log('The same .task file works for the Python side:')
console.log(`  export BODYPOSE_MP_MODEL=${path.join(target, 'pose_landmarker_lite.task')}`)
