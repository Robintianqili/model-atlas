// e2e passes (K2 speed pack): which browser passes `atlas e2e` and `atlas check` run.
//
// A pass is named <width>-<lang>-<scheme>[-reduced]:
//   width   desktop (1280 x 900), phone (390 x 844), narrow (360 x 800), a number of px (280-1920), or an exact window
//           <width>x<height> (1280x680, 360x640; redesign F)
//   lang    en | zh
//   scheme  light | dark
//   reduced (optional) prefers-reduced-motion: one pass over the step list, nothing may autoplay
// Presets: default (2 passes: desktop-en-light, phone-zh-dark; every gate runs these, check --all too since the lean
// gates of 2026-10-13), all (5: the default two plus narrow-en-dark, the 360 px scroll gate, desktop-zh-light and
// desktop-en-light-reduced; on request) and full (all + desktop-en-dark, narrow-zh-light and phone-en-light). layout
// (redesign F): the lesson app's windows 1280x800, 1280x680, 1024x768 (app mode), 390 x 844 zh dark and 360x640 (flow).
// Lists mix names and presets: --passes default,narrow-en-dark.
// Every pass at a width <= 430 checks horizontal scroll (AK602); a pass at 361-430 px also probes the page at 360 px
// after each step (the AK602 gate width), so the default two passes keep the 360 px gate.
import { AtlasError } from '../errors.ts';

export interface Pass {
	name: string;
	width: number;
	height: number;
	scheme: 'light' | 'dark';
	lang: 'en' | 'zh';
	reduced: boolean;
	/** a screenshot of every step (and a contact sheet); off for the reduced-motion pass */
	shots: boolean;
	/** the stage must show a provenance badge on every step (AK502): the first desktop pass in English */
	badge: boolean;
	/** horizontal scroll check at the pass width (AK602) */
	scroll: boolean;
	/** also probe horizontal scroll at 360 px after each step (a 361-430 px pass) */
	probe360: boolean;
}

export const WIDTHS: Record<string, { width: number; height: number }> = {
	desktop: { width: 1280, height: 900 },
	phone: { width: 390, height: 844 },
	narrow: { width: 360, height: 800 }
};

export const PRESETS: Record<string, string[]> = {
	default: ['desktop-en-light', 'phone-zh-dark'],
	// redesign F: the lesson app's windows (app mode at 1280 x 800, 1280 x 680 and 1024 x 768; flow on phones)
	layout: ['1280x800-en-light', '1280x680-en-light', '1024x768-en-dark', 'phone-zh-dark', '360x640-en-light'],
	all: ['desktop-en-light', 'phone-zh-dark', 'narrow-en-dark', 'desktop-zh-light', 'desktop-en-light-reduced'],
	full: ['desktop-en-light', 'phone-zh-dark', 'narrow-en-dark', 'desktop-zh-light', 'desktop-en-light-reduced', 'desktop-en-dark', 'narrow-zh-light', 'phone-en-light']
};

const NAME_RE = /^(desktop|phone|narrow|\d{3,4}(?:x\d{3,4})?)-(en|zh)-(light|dark)(-reduced)?$/;

/** One pass from its name (AK301 for a name that is not <width>-<lang>-<scheme>[-reduced]). */
export function parsePass(name: string): Pass {
	const m = NAME_RE.exec(name.trim());
	if (!m) throw new AtlasError('AK301', `unknown e2e pass '${name}'`, { fix: `a pass is <desktop|phone|narrow|px>-<en|zh>-<light|dark>[-reduced], e.g. phone-zh-dark or 414-en-light; presets: ${Object.keys(PRESETS).join(', ')}` });
	const [, w, lang, scheme, red] = m;
	let size = WIDTHS[w];
	if (!size && w.includes('x')) {
		// redesign F: an exact window, <width>x<height> (1280x680, 360x640)
		const [pw, ph] = w.split('x').map(Number);
		if (pw < 280 || pw > 1920 || ph < 400 || ph > 1440) throw new AtlasError('AK301', `e2e pass '${name}': ${pw} x ${ph} is outside 280-1920 x 400-1440`);
		size = { width: pw, height: ph };
	}
	if (!size) {
		const px = Number(w);
		if (px < 280 || px > 1920) throw new AtlasError('AK301', `e2e pass '${name}': width ${px} px is outside 280-1920`);
		size = { width: px, height: px <= 700 ? 844 : 900 };
	}
	const reduced = !!red;
	return {
		name: m[0],
		width: size.width,
		height: size.height,
		scheme: scheme as Pass['scheme'],
		lang: lang as Pass['lang'],
		reduced,
		shots: !reduced,
		badge: false,
		scroll: size.width <= 430,
		probe360: size.width > 360 && size.width <= 430
	};
}

/** The passes of a --passes list (names and presets, comma-separated; default 'default'), in order, without repeats.
 *  The first non-reduced desktop pass in English checks the provenance badges (AK502); without one, the first English
 *  pass does. */
export function resolvePasses(spec?: string | string[]): Pass[] {
	const parts = (Array.isArray(spec) ? spec : (spec ?? 'default').split(','))
		.map((s) => s.trim())
		.filter(Boolean);
	const names: string[] = [];
	for (const p of parts.length ? parts : ['default']) for (const n of PRESETS[p] ?? [p]) if (!names.includes(n)) names.push(n);
	const passes = names.map(parsePass);
	const badge = passes.find((p) => !p.reduced && p.lang === 'en' && p.width >= 1000) ?? passes.find((p) => !p.reduced && p.lang === 'en');
	if (badge) badge.badge = true;
	return passes;
}

/** The pass list of the old flags: --full (the full preset) and --phone <px> (adds <px>-en-light). */
export function legacyPasses(o: { full?: boolean; phone?: number; passes?: string }): string {
	const list = [o.passes ?? (o.full ? 'full' : 'default')];
	if (o.phone && o.phone !== 360) list.push(`${o.phone}-en-light`);
	return list.join(',');
}
