import type { LetterEvent } from "../../core-ts/src/index.ts";

export type PlaybackMode = "melody" | "chord";

export type PlaybackPlan = {
  mode: PlaybackMode;
  frequenciesHz: number[];
};

export function createPlaybackPlan(events: LetterEvent[], mode: PlaybackMode): PlaybackPlan {
  if (mode === "melody") {
    return { mode, frequenciesHz: events.map((event) => event.frequencyHz) };
  }
  return { mode, frequenciesHz: [...new Set(events.map((event) => event.frequencyHz))] };
}
