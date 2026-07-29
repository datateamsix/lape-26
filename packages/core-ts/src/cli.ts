import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { analyzeEvents, encodeText, type LapeMapping } from "./index.ts";

const here = dirname(fileURLToPath(import.meta.url));
const mappingPath = resolve(here, "../../../mappings/lape-26-en-general-v0.1.json");
const mapping = JSON.parse(await readFile(mappingPath, "utf8")) as LapeMapping;
const text = process.argv.slice(2).join(" ") || "HAMMER";
const events = encodeText(text, mapping);
console.log(JSON.stringify({ input: text, events, analysis: analyzeEvents(events) }, null, 2));
