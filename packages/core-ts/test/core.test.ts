import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { analyzeEvents, encodeText, midiToFrequency, normalizeText, type LapeMapping } from "../src/index.ts";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../../..");
const mapping = JSON.parse(await readFile(resolve(root, "mappings/lape-26-en-general-v0.1.json"), "utf8")) as LapeMapping;
const golden = JSON.parse(await readFile(resolve(root, "tests/golden-vectors.json"), "utf8"));

test("A4 equals 440 Hz", () => {
  assert.equal(midiToFrequency(69), 440);
  assert.ok(Math.abs(midiToFrequency(60) - 261.625565) < 0.00001);
});

test("normalization follows profile v0.1", () => {
  assert.equal(normalizeText("Café, don't! 26"), "CAFEDONT");
});

test("HAMMER matches the initial vector", () => {
  const events = encodeText("HAMMER", mapping);
  assert.deepEqual(events.map((event) => event.midi), [69, 60, 63, 63, 64, 57]);
  assert.deepEqual(analyzeEvents(events).intervals, [-9, 3, 0, 1, -7]);
});

test("shared golden vectors match", () => {
  for (const entry of golden.cases) {
    const events = encodeText(entry.input, mapping);
    assert.equal(events.map((event) => event.normalizedCharacter).join(""), entry.normalized, entry.id);
    assert.deepEqual(events.map((event) => event.midi), entry.midi, entry.id);
    assert.deepEqual(analyzeEvents(events).intervals, entry.intervals, entry.id);
  }
});
