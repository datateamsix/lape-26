export type LetterMapping = {
  pitch: string;
  midi: number;
  frequencyHz: number;
};

export type LapeMapping = {
  mappingId: string;
  normalizationProfile: string;
  letters: Record<string, LetterMapping>;
};

export type LetterEvent = {
  sourceCharacter: string;
  normalizedCharacter: string;
  sourceIndex: number;
  normalizedIndex: number;
  pitch: string;
  midi: number;
  frequencyHz: number;
  mappingVersion: string;
  normalizationProfile: string;
};

export function midiToFrequency(midi: number, referenceHz = 440): number {
  if (!Number.isInteger(midi) || midi < 0 || midi > 127) {
    throw new RangeError("MIDI pitch must be an integer between 0 and 127");
  }
  return referenceHz * 2 ** ((midi - 69) / 12);
}

export function normalizeText(text: string): string {
  return text
    .normalize("NFKD")
    .toUpperCase()
    .replace(/[^A-Z]/g, "");
}

export function validateMapping(mapping: LapeMapping): void {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const keys = Object.keys(mapping.letters).sort().join("");
  if (keys !== alphabet) throw new Error("Mapping must define exactly A-Z");

  const midi = Object.values(mapping.letters).map((entry) => entry.midi);
  if (new Set(midi).size !== 26) throw new Error("Mapping MIDI assignments must be unique");
}

export function encodeText(text: string, mapping: LapeMapping): LetterEvent[] {
  validateMapping(mapping);
  const events: LetterEvent[] = [];
  let normalizedIndex = 0;

  for (const [sourceIndex, sourceCharacter] of Array.from(text.normalize("NFKD")).entries()) {
    const normalizedCharacter = sourceCharacter.toUpperCase();
    if (!/^[A-Z]$/.test(normalizedCharacter)) continue;
    const entry = mapping.letters[normalizedCharacter];
    events.push({
      sourceCharacter,
      normalizedCharacter,
      sourceIndex,
      normalizedIndex,
      pitch: entry.pitch,
      midi: entry.midi,
      frequencyHz: midiToFrequency(entry.midi),
      mappingVersion: mapping.mappingId,
      normalizationProfile: mapping.normalizationProfile,
    });
    normalizedIndex += 1;
  }

  if (events.map((event) => event.normalizedCharacter).join("") !== normalizeText(text)) {
    throw new Error("Normalization and encoding paths diverged");
  }
  return events;
}

export function intervals(midi: number[]): number[] {
  return midi.slice(1).map((value, index) => value - midi[index]);
}

export function analyzeEvents(events: LetterEvent[]) {
  const midi = events.map((event) => event.midi);
  const movement = intervals(midi);
  const motion = movement.reduce((sum, value) => sum + Math.abs(value), 0);
  return {
    noteCount: midi.length,
    midi,
    intervals: movement,
    registerCenterMidi: midi.length ? midi.reduce((a, b) => a + b, 0) / midi.length : null,
    pitchSpanSemitones: midi.length ? Math.max(...midi) - Math.min(...midi) : 0,
    directionalBalance: motion ? movement.reduce((a, b) => a + b, 0) / motion : 0,
    repetitionIndex: movement.length ? movement.filter((value) => value === 0).length / movement.length : 0,
  };
}
