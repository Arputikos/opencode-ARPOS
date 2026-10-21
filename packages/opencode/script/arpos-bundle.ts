#!/usr/bin/env bun
// ARPOS: one-directory bundle of the server the orchestrator runs (`opencode serve`) —
// loads one module graph instead of resolving and transpiling ~thousands of TS files
// on every session start. Built in the image only; dev runs the sources.
import path from "path"
import { createSolidTransformPlugin } from "@opentui/solid/bun-plugin"

const dir = path.resolve(import.meta.dirname, "..")
process.chdir(dir)

const result = await Bun.build({
  conditions: ["bun", "node", "browser"],
  tsconfig: "./tsconfig.json",
  plugins: [createSolidTransformPlugin()],
  // Native platform packages stay external: they resolve at runtime from node_modules.
  external: ["node-gyp", "@opentui/core-*", "@ff-labs/fff-bin-*", "@lydell/node-pty-*"],
  target: "bun",
  format: "esm",
  splitting: true,
  entrypoints: ["./src/index.ts", "./src/cli/tui/worker.ts"],
  outdir: "./dist/arpos-server",
})
if (!result.success) {
  for (const log of result.logs) console.error(log)
  process.exit(1)
}
console.log(`built ${result.outputs.length} files into dist/arpos-server`)
