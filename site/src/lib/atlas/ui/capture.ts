// The props every adapter's Capture.svelte receives (kit adapters in atlas/adapters/<id>/, package adapters in
// explainers/<slug>/adapters/<id>/). A Capture edits the learner's raw input and reports it; the runtime preprocesses it.
//   <Capture {rt} raw={rt.input} presets={rt.presets} onchange={(raw, o) => …} compact />
// `onchange(raw, { live: true })` is a change in progress (a stroke being drawn): the page coalesces those to one per frame.
// `compact` is the small copy in the lesson's side panel.
import type { I18n } from '../types/core.ts';
import type { AtlasRuntime } from '../runtime/runtime.ts';

export interface CaptureProps {
	rt: AtlasRuntime;
	raw: unknown;
	presets: { id: string; label: I18n; raw: unknown }[];
	onchange: (raw: unknown, o?: { live?: boolean }) => void;
	compact?: boolean;
}
