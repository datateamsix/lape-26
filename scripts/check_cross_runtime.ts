import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { encodeText, type LapeMapping } from "../packages/core-ts/src/index.ts";

const mapping = JSON.parse(await readFile(resolve("mappings/lape-26-en-general-v0.1.json"), "utf8")) as LapeMapping;
const golden = JSON.parse(await readFile(resolve("tests/golden-vectors.json"), "utf8"));

for (const entry of golden.cases) {
  const tsMidi = encodeText(entry.input, mapping).map((event) => event.midi);
  const code = `import json; from lape26.core import encode_text; print(json.dumps([e['midi'] for e in encode_text(${JSON.stringify(entry.input)})]))`;
  const result = spawnSync("python3", ["-c", code], {
    encoding: "utf8",
    env: { ...process.env, PYTHONPATH: resolve("python") },
  });
  if (result.status !== 0) {
    console.error(result.stderr);
    process.exit(result.status ?? 1);
  }
  const pyMidi = JSON.parse(result.stdout);
  if (JSON.stringify(tsMidi) !== JSON.stringify(pyMidi)) {
    throw new Error(`Cross-runtime mismatch for ${entry.id}: TS=${tsMidi} Python=${pyMidi}`);
  }
}

console.log(`Cross-runtime parity passed for ${golden.cases.length} golden vectors.`);
