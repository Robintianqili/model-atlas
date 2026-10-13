/**
 * The icons the catalog pages use beyond the header's, and the area and section glyphs, copied from kit/icons.ts (same
 * markup, same grids). They live here because kit/icons.ts sits in a chunk every page loads (the header imports it) and
 * Rollup keeps a module whole: using glyphFor() from there put the 21 glyphs (about 1.4 KB gzip) into every shell,
 * planned pages and lessons included. Only the catalog pages import this module.
 *   <Icon d={glyphFor('vision')} glyph />   <Icon d={arrow} />
 */

// ------------------------------------------------------------------ UI icons (24 grid)
export const arrow = '<path d="M5 12h14M13 6l6 6-6 6"/>';
export const right = '<path d="M9 5l7 7-7 7"/>';
export const left = '<path d="M15 5l-7 7 7 7"/>';
export const draw = '<path d="M4 20l4-1 11-11-3-3L5 16z"/>';
export const type = '<path d="M5 6V4.5h14V6M12 4.5v15M9 19.5h6"/>';
export const points = '<circle cx="6" cy="16" r="1.4" fill="currentColor"/><circle cx="10" cy="9" r="1.4" fill="currentColor"/><circle cx="14" cy="13" r="1.4" fill="currentColor"/><circle cx="18" cy="6" r="1.4" fill="currentColor"/><path d="M3 20h18" opacity=".6"/>';
export const game = '<rect x="3" y="7" width="18" height="10" rx="5"/><path d="M8 10v4M6 12h4M15.5 11h.01M17.5 13h.01"/>';
export const graph = '<circle cx="6" cy="6" r="2.2"/><circle cx="18" cy="8" r="2.2"/><circle cx="9" cy="18" r="2.2"/><path d="M8 7l8 1M7 8l1.5 8"/>';
export const rows = '<path d="M4 6h16M4 12h16M4 18h16"/>';
export const grid = '<rect x="4" y="4" width="7" height="7" rx="1"/><rect x="13" y="4" width="7" height="7" rx="1"/><rect x="4" y="13" width="7" height="7" rx="1"/><rect x="13" y="13" width="7" height="7" rx="1"/>';

// ------------------------------------------------------------------ area and section glyphs (20 grid)
const GLYPHS: Record<string, string> = {
	vision: '<rect x="3.5" y="3.5" width="13" height="13" rx="1.5"/><path d="M7.8 3.5v13M12.2 3.5v13M3.5 7.8h13M3.5 12.2h13"/><rect x="7.8" y="7.8" width="4.4" height="4.4" fill="currentColor" stroke="none"/>',
	language: '<rect x="3" y="5" width="5" height="4" rx="1"/><rect x="9.5" y="5" width="7.5" height="4" rx="1"/><rect x="3" y="11" width="8" height="4" rx="1"/><rect x="12.5" y="11" width="4.5" height="4" rx="1" fill="currentColor"/>',
	audio: '<path d="M3 10h1.5M6 7v6M8.5 4.5v11M11 7.5v5M13.5 5.5v9M16 8.5v3"/>',
	'time-series': '<path d="M3 14l3-3 3 2 4-6 4 3"/><path d="M3 16.5h14" stroke-opacity=".5"/>',
	recommendation: '<rect x="3.5" y="3.5" width="13" height="13" rx="1.5"/><path d="M3.5 10h13M10 3.5v13"/><path d="M3.5 3.5h6.5v6.5H3.5zM10 10h6.5v6.5H10z" fill="currentColor" stroke="none" opacity=".85"/>',
	tabular: '<rect x="3.5" y="4" width="13" height="12" rx="1.5"/><path d="M3.5 8h13M3.5 12h13M8.5 4v12"/>',
	rl: '<rect x="3.5" y="3.5" width="13" height="13" rx="1.5"/><path d="M6.5 13.5h3.5v-7h3.5"/><path d="M11.8 5l1.7 1.5-1.7 1.5"/>',
	graphs: '<circle cx="5" cy="6" r="2"/><circle cx="15" cy="5" r="2"/><circle cx="10" cy="15" r="2"/><path d="M6.8 7l2.4 6.2M13.6 6.6l-2.8 6.6M7 6l6-.8"/>',
	generative: '<circle cx="6" cy="12" r="1.2" fill="currentColor"/><circle cx="9" cy="8" r="1.2" fill="currentColor"/><circle cx="12" cy="11" r="1.2" fill="currentColor"/><circle cx="14.5" cy="6.5" r="1.2" fill="currentColor"/><path d="M3.5 16c4-1 9-6 13-12" stroke-dasharray="1.5 2"/>',
	multimodal: '<rect x="3" y="4" width="6" height="6" rx="1"/><path d="M12 5h5M12 8h3.5M12 13h5M12 16h3.5M3 14h6"/><path d="M9 7h3M9 14h3"/>',
	science: '<path d="M3 10c2-6 4-6 6 0s4 6 6 0"/><circle cx="16.5" cy="10" r="1.3" fill="currentColor"/>',
	foundations: '<path d="M3 15l4-6 3 3 3-5 4 8z"/>',
	pretraining: '<rect x="3" y="7" width="3.5" height="6" rx="1"/><rect x="8.25" y="7" width="3.5" height="6" rx="1"/><rect x="13.5" y="7" width="3.5" height="6" rx="1" stroke-dasharray="1.5 1.5"/>',
	finetuning: '<path d="M4 15V5M4 10h6l3-4M10 10l3 4h3"/>',
	peft: '<rect x="3" y="3.5" width="9" height="13" rx="1"/><rect x="13.5" y="3.5" width="3.5" height="13" rx="1" fill="currentColor" stroke="none"/>',
	alignment: '<circle cx="10" cy="10" r="6.5"/><circle cx="10" cy="10" r="2.6" fill="currentColor" stroke="none"/>',
	compression: '<path d="M4 4l4 4M16 4l-4 4M4 16l4-4M16 16l-4-4"/><rect x="8" y="8" width="4" height="4" rx=".8"/>',
	interpretability: '<circle cx="8.5" cy="8.5" r="5"/><path d="M12.2 12.2l4.3 4.3M6.5 8.5h4M8.5 6.5v4"/>',
	evaluation: '<path d="M3.5 6.2l1.6 1.6L8 5M3.5 12.7l1.6 1.6L8 11.5M10.5 6.5h6M10.5 13h6"/>',
	inference: '<path d="M3 10h9M9 6l4 4-4 4"/><path d="M15.5 5v10"/>',
	robustness: '<path d="M10 3l6 2.5V10c0 3.4-2.6 5.9-6 7-3.4-1.1-6-3.6-6-7V5.5z"/>',
};

/** An area's or section's glyph by its slug; an unknown slug gets the foundations glyph. */
export function glyphFor(slug: string): string {
	return GLYPHS[slug] ?? GLYPHS.foundations;
}
