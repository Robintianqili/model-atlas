/**
 * The one icon set (redesign spec §4.2): UI icons on a 24 grid, drawn with a 1.75 stroke and round caps
 * (Lucide-compatible shapes), and monochrome area and section glyphs on a 20 grid at 1.6 stroke, drawn from the data
 * grammar. Areas get glyphs, not colours (eleven hues would compete with the data).
 *
 * Every icon is a named export holding its SVG inner markup, so a page bundles only the icons it imports:
 *
 *   import Icon from '$lib/kit/Icon.svelte';
 *   import { search, glyphVision } from '$lib/kit/icons';
 *   <Icon d={search} />                 a 24-grid UI icon, aria-hidden (put a word beside it, or a label on the button)
 *   <Icon d={glyphVision} glyph />      a 20-grid glyph
 *   <Icon d={info} label="About" />     an icon that is the only content: role="img" with that label
 *
 * Data-driven code (an area's glyph by its slug) uses glyphFor(slug) / iconFor(name), which bundle the whole table.
 */

// ------------------------------------------------------------------ UI icons (24 grid)
export const x = '<path d="M6 6l12 12M18 6L6 18"/>';
export const left = '<path d="M15 5l-7 7 7 7"/>';
export const right = '<path d="M9 5l7 7-7 7"/>';
export const arrow = '<path d="M5 12h14M13 6l6 6-6 6"/>';
export const play = '<path d="M8 5.5v13l11-6.5z" fill="currentColor" stroke="none"/>';
export const pause = '<path d="M8 5v14M16 5v14"/>';
export const search = '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>';
export const moon = '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>';
export const sun =
	'<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>';
/** the theme follows the system: a half-filled circle */
export const contrast = '<circle cx="12" cy="12" r="8.5"/><path d="M12 3.5a8.5 8.5 0 0 1 0 17z" fill="currentColor"/>';
export const list =
	'<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4.5" cy="6" r="1" fill="currentColor"/><circle cx="4.5" cy="12" r="1" fill="currentColor"/><circle cx="4.5" cy="18" r="1" fill="currentColor"/>';
export const more =
	'<circle cx="5" cy="12" r="1.4" fill="currentColor"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/><circle cx="19" cy="12" r="1.4" fill="currentColor"/>';
export const menu = '<path d="M4 7h16M4 12h16M4 17h16"/>';
export const pencil = '<path d="M4 20l4-1 11-11-3-3L5 16z"/><path d="M14 7l3 3"/>';
export const check = '<path d="M5 12.5l4.5 4.5L19 7.5"/>';
export const eye = '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>';
export const book = '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z"/><path d="M4 20.5A2.5 2.5 0 0 0 6.5 23H20v-5"/>';
export const code = '<path d="M8.5 7L3.5 12l5 5M15.5 7l5 5-5 5"/>';
export const chart = '<path d="M4 4v16h16"/><path d="M7.5 15l3.5-4 3 2.5L19.5 7"/>';
export const compass = '<circle cx="12" cy="12" r="9"/><path d="M15.5 8.5l-2 5-5 2 2-5z"/>';
export const lesson = '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M10 9l5 3-5 3z" fill="currentColor"/>';
export const help = '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6v.6M12 17h.01"/>';
export const info = '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5h.01"/>';
export const restart = '<path d="M4 12a8 8 0 1 0 2.3-5.7L4 8.5"/><path d="M4 4v4.5h4.5"/>';
export const clock = '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>';
export const grid =
	'<rect x="4" y="4" width="7" height="7" rx="1"/><rect x="13" y="4" width="7" height="7" rx="1"/><rect x="4" y="13" width="7" height="7" rx="1"/><rect x="13" y="13" width="7" height="7" rx="1"/>';
export const rows = '<path d="M4 6h16M4 12h16M4 18h16"/>';
export const sources = '<path d="M7 4h10v16l-5-3.5L7 20z"/>';
export const flask = '<path d="M9 3h6M10 3v6l-5 9.5A1.5 1.5 0 0 0 6.3 21h11.4a1.5 1.5 0 0 0 1.3-2.5L14 9V3"/>';
export const quiz = '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M8 12.5l2.5 2.5L16 9.5"/>';
export const lock = '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>';
export const external = '<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>';
// input kinds
export const draw = '<path d="M4 20l4-1 11-11-3-3L5 16z"/>';
export const type = '<path d="M5 6V4.5h14V6M12 4.5v15M9 19.5h6"/>';
export const points =
	'<circle cx="6" cy="16" r="1.4" fill="currentColor"/><circle cx="10" cy="9" r="1.4" fill="currentColor"/><circle cx="14" cy="13" r="1.4" fill="currentColor"/><circle cx="18" cy="6" r="1.4" fill="currentColor"/><path d="M3 20h18" opacity=".6"/>';
export const game = '<rect x="3" y="7" width="18" height="10" rx="5"/><path d="M8 10v4M6 12h4M15.5 11h.01M17.5 13h.01"/>';
export const graph = '<circle cx="6" cy="6" r="2.2"/><circle cx="18" cy="8" r="2.2"/><circle cx="9" cy="18" r="2.2"/><path d="M8 7l8 1M7 8l1.5 8"/>';

// ------------------------------------------------------------------ area and section glyphs (20 grid)
export const glyphVision =
	'<rect x="3.5" y="3.5" width="13" height="13" rx="1.5"/><path d="M7.8 3.5v13M12.2 3.5v13M3.5 7.8h13M3.5 12.2h13"/><rect x="7.8" y="7.8" width="4.4" height="4.4" fill="currentColor" stroke="none"/>';
export const glyphLanguage =
	'<rect x="3" y="5" width="5" height="4" rx="1"/><rect x="9.5" y="5" width="7.5" height="4" rx="1"/><rect x="3" y="11" width="8" height="4" rx="1"/><rect x="12.5" y="11" width="4.5" height="4" rx="1" fill="currentColor"/>';
export const glyphAudio = '<path d="M3 10h1.5M6 7v6M8.5 4.5v11M11 7.5v5M13.5 5.5v9M16 8.5v3"/>';
export const glyphTimeSeries = '<path d="M3 14l3-3 3 2 4-6 4 3"/><path d="M3 16.5h14" stroke-opacity=".5"/>';
export const glyphRecommendation =
	'<rect x="3.5" y="3.5" width="13" height="13" rx="1.5"/><path d="M3.5 10h13M10 3.5v13"/><path d="M3.5 3.5h6.5v6.5H3.5zM10 10h6.5v6.5H10z" fill="currentColor" stroke="none" opacity=".85"/>';
export const glyphTabular = '<rect x="3.5" y="4" width="13" height="12" rx="1.5"/><path d="M3.5 8h13M3.5 12h13M8.5 4v12"/>';
export const glyphRl = '<rect x="3.5" y="3.5" width="13" height="13" rx="1.5"/><path d="M6.5 13.5h3.5v-7h3.5"/><path d="M11.8 5l1.7 1.5-1.7 1.5"/>';
export const glyphGraphs =
	'<circle cx="5" cy="6" r="2"/><circle cx="15" cy="5" r="2"/><circle cx="10" cy="15" r="2"/><path d="M6.8 7l2.4 6.2M13.6 6.6l-2.8 6.6M7 6l6-.8"/>';
export const glyphGenerative =
	'<circle cx="6" cy="12" r="1.2" fill="currentColor"/><circle cx="9" cy="8" r="1.2" fill="currentColor"/><circle cx="12" cy="11" r="1.2" fill="currentColor"/><circle cx="14.5" cy="6.5" r="1.2" fill="currentColor"/><path d="M3.5 16c4-1 9-6 13-12" stroke-dasharray="1.5 2"/>';
export const glyphMultimodal = '<rect x="3" y="4" width="6" height="6" rx="1"/><path d="M12 5h5M12 8h3.5M12 13h5M12 16h3.5M3 14h6"/><path d="M9 7h3M9 14h3"/>';
export const glyphScience = '<path d="M3 10c2-6 4-6 6 0s4 6 6 0"/><circle cx="16.5" cy="10" r="1.3" fill="currentColor"/>';
export const glyphFoundations = '<path d="M3 15l4-6 3 3 3-5 4 8z"/>';
export const glyphPretraining =
	'<rect x="3" y="7" width="3.5" height="6" rx="1"/><rect x="8.25" y="7" width="3.5" height="6" rx="1"/><rect x="13.5" y="7" width="3.5" height="6" rx="1" stroke-dasharray="1.5 1.5"/>';
export const glyphFinetuning = '<path d="M4 15V5M4 10h6l3-4M10 10l3 4h3"/>';
export const glyphPeft = '<rect x="3" y="3.5" width="9" height="13" rx="1"/><rect x="13.5" y="3.5" width="3.5" height="13" rx="1" fill="currentColor" stroke="none"/>';
export const glyphAlignment = '<circle cx="10" cy="10" r="6.5"/><circle cx="10" cy="10" r="2.6" fill="currentColor" stroke="none"/>';
export const glyphCompression = '<path d="M4 4l4 4M16 4l-4 4M4 16l4-4M16 16l-4-4"/><rect x="8" y="8" width="4" height="4" rx=".8"/>';
export const glyphInterpretability = '<circle cx="8.5" cy="8.5" r="5"/><path d="M12.2 12.2l4.3 4.3M6.5 8.5h4M8.5 6.5v4"/>';
export const glyphEvaluation = '<path d="M3.5 6.2l1.6 1.6L8 5M3.5 12.7l1.6 1.6L8 11.5M10.5 6.5h6M10.5 13h6"/>';
export const glyphInference = '<path d="M3 10h9M9 6l4 4-4 4"/><path d="M15.5 5v10"/>';
export const glyphRobustness = '<path d="M10 3l6 2.5V10c0 3.4-2.6 5.9-6 7-3.4-1.1-6-3.6-6-7V5.5z"/>';

// ------------------------------------------------------------------ lookups for data-driven code (bundle the tables)
const UI_ICONS = {
	x, left, right, arrow, play, pause, search, moon, sun, contrast, list, more, menu, pencil, check, eye, book, code,
	chart, compass, lesson, help, info, restart, clock, grid, rows, sources, flask, quiz, lock, external,
	draw, type, points, game, graph
} as const;
export type IconName = keyof typeof UI_ICONS;

/** Area slugs (catalog/areas.ts) and training-method sections (catalog concept sections) to their glyphs. */
const GLYPHS: Record<string, string> = {
	vision: glyphVision,
	language: glyphLanguage,
	audio: glyphAudio,
	'time-series': glyphTimeSeries,
	recommendation: glyphRecommendation,
	tabular: glyphTabular,
	rl: glyphRl,
	graphs: glyphGraphs,
	generative: glyphGenerative,
	multimodal: glyphMultimodal,
	science: glyphScience,
	foundations: glyphFoundations,
	pretraining: glyphPretraining,
	finetuning: glyphFinetuning,
	peft: glyphPeft,
	alignment: glyphAlignment,
	compression: glyphCompression,
	interpretability: glyphInterpretability,
	evaluation: glyphEvaluation,
	inference: glyphInference,
	robustness: glyphRobustness
};

/** A UI icon's markup by name. */
export function iconFor(name: IconName): string {
	return UI_ICONS[name];
}

/** An area's or section's glyph by its slug; an unknown slug gets the foundations glyph. */
export function glyphFor(slug: string): string {
	return GLYPHS[slug] ?? glyphFoundations;
}

/** The slugs that have their own glyph (for tests and the component gallery). */
export function glyphSlugs(): string[] {
	return Object.keys(GLYPHS);
}
