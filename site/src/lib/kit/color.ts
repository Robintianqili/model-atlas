/**
 * Model Atlas color scales — the one visual grammar for values (brief §4.7).
 *
 *  - diverging : signed values (activations, weights, gradients, diffs).
 *                blue = negative, near-neutral gray = 0, orange = positive.
 *  - sequential: non-negative values (probabilities, attention, magnitudes). One purple ramp.
 *  - categorical: 10 fixed class colors; class k has the same color everywhere.
 *
 * Ramps are built in OKLCH (perceptual) and converted to sRGB once into 256-entry lookup tables,
 * so `scale(v)` is a couple of arithmetic ops plus an array read: fast enough for canvas loops.
 * Light and dark themes have their own ramps (the zero / low end sits near that theme's surface).
 *
 * Pure TS: no DOM access at import time, safe in workers and in Node.
 */

export type Theme = 'light' | 'dark';
export type ScaleKind = 'diverging' | 'sequential';
/** [lo, hi]. For diverging scales the domain is made symmetric around 0 (max |lo|,|hi|). */
export type Domain = [number, number];
/** How a color domain is chosen: this tensor/channel only, the whole layer, or all layers. */
export type ColorScope = 'unit' | 'layer' | 'global';
export type RGB = [number, number, number];

// ---------------------------------------------------------------- OKLCH -> sRGB

function lin2srgb(x: number): number {
	const v = x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
	return v;
}

/** Convert OKLCH (L 0..1, C, h degrees) to linear-light sRGB components (may be out of gamut). */
function oklchToLinear(L: number, C: number, h: number): [number, number, number] {
	const hr = (h * Math.PI) / 180;
	const a = C * Math.cos(hr);
	const b = C * Math.sin(hr);
	const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
	const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
	const s_ = L - 0.0894841775 * a - 1.291485548 * b;
	const l = l_ * l_ * l_;
	const m = m_ * m_ * m_;
	const s = s_ * s_ * s_;
	return [
		4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
		-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
		-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s
	];
}

function inGamut([r, g, b]: [number, number, number]): boolean {
	const e = 1e-4;
	return r >= -e && r <= 1 + e && g >= -e && g <= 1 + e && b >= -e && b <= 1 + e;
}

/** OKLCH to 8-bit sRGB, reducing chroma (keeping L and h) until the color is in gamut. */
export function oklch(L: number, C: number, h: number): RGB {
	let lin = oklchToLinear(L, C, h);
	if (!inGamut(lin)) {
		let lo = 0;
		let hi = C;
		for (let i = 0; i < 24; i++) {
			const mid = (lo + hi) / 2;
			if (inGamut(oklchToLinear(L, mid, h))) lo = mid;
			else hi = mid;
		}
		lin = oklchToLinear(L, lo, h);
	}
	return lin.map((x) => Math.round(Math.min(1, Math.max(0, lin2srgb(x))) * 255)) as RGB;
}

// ---------------------------------------------------------------- ramp anchors

type Anchor = { t: number; L: number; C: number; h: number };

/** Piecewise-linear interpolation of L and C between anchors (hue taken from the anchor pair). */
function rampAt(anchors: Anchor[], t: number): RGB {
	if (t <= anchors[0].t) return oklch(anchors[0].L, anchors[0].C, anchors[0].h);
	for (let i = 1; i < anchors.length; i++) {
		const a = anchors[i - 1];
		const b = anchors[i];
		if (t <= b.t) {
			const u = (t - a.t) / (b.t - a.t);
			// hue: near-gray anchors (C < 0.02) borrow the other anchor's hue
			const h = a.C < 0.02 ? b.h : b.C < 0.02 ? a.h : a.h + (b.h - a.h) * u;
			return oklch(a.L + (b.L - a.L) * u, a.C + (b.C - a.C) * u, h);
		}
	}
	const z = anchors[anchors.length - 1];
	return oklch(z.L, z.C, z.h);
}

/**
 * Diverging anchors over t in [-1, 1] (stored as t in [0,1] with 0.5 = zero).
 * Light: deep blue <- warm gray -> deep orange. Dark: light blue <- slate -> light orange.
 */
const DIVERGING: Record<Theme, Anchor[]> = {
	light: [
		{ t: 0, L: 0.46, C: 0.15, h: 257 },
		{ t: 0.25, L: 0.7, C: 0.1, h: 252 },
		{ t: 0.5, L: 0.955, C: 0.004, h: 90 },
		{ t: 0.75, L: 0.78, C: 0.12, h: 62 },
		{ t: 1, L: 0.56, C: 0.16, h: 48 }
	],
	dark: [
		{ t: 0, L: 0.78, C: 0.12, h: 250 },
		{ t: 0.25, L: 0.55, C: 0.12, h: 254 },
		{ t: 0.5, L: 0.285, C: 0.006, h: 270 },
		{ t: 0.75, L: 0.58, C: 0.13, h: 55 },
		{ t: 1, L: 0.82, C: 0.14, h: 65 }
	]
};

/** Sequential purple ramp over t in [0, 1]: low values recede toward the surface. */
const SEQUENTIAL: Record<Theme, Anchor[]> = {
	light: [
		{ t: 0, L: 0.975, C: 0.01, h: 305 },
		{ t: 0.45, L: 0.73, C: 0.12, h: 303 },
		{ t: 1, L: 0.37, C: 0.17, h: 298 }
	],
	dark: [
		{ t: 0, L: 0.25, C: 0.02, h: 300 },
		{ t: 0.55, L: 0.58, C: 0.17, h: 302 },
		{ t: 1, L: 0.88, C: 0.09, h: 306 }
	]
};

const LUT_SIZE = 256;
const luts: Partial<Record<`${ScaleKind}-${Theme}`, Uint8ClampedArray>> = {};

function lut(kind: ScaleKind, theme: Theme): Uint8ClampedArray {
	const key = `${kind}-${theme}` as const;
	let table = luts[key];
	if (!table) {
		table = new Uint8ClampedArray(LUT_SIZE * 3);
		const anchors = kind === 'diverging' ? DIVERGING[theme] : SEQUENTIAL[theme];
		for (let i = 0; i < LUT_SIZE; i++) {
			const [r, g, b] = rampAt(anchors, i / (LUT_SIZE - 1));
			table[i * 3] = r;
			table[i * 3 + 1] = g;
			table[i * 3 + 2] = b;
		}
		luts[key] = table;
	}
	return table;
}

// ---------------------------------------------------------------- theme

/**
 * Resolved theme of the page right now. Reads `data-theme` on <html> or the OS preference.
 * Returns 'light' outside the browser. Components that draw on canvas should read the
 * reactive `themeState.resolved` (theme.svelte.ts) so they redraw when the theme changes.
 */
export function currentTheme(): Theme {
	if (typeof document === 'undefined') return 'light';
	const attr = document.documentElement.getAttribute('data-theme');
	if (attr === 'light' || attr === 'dark') return attr;
	try {
		return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
	} catch {
		return 'light';
	}
}

// ---------------------------------------------------------------- scales

export interface ColorScale {
	kind: ScaleKind;
	theme: Theme;
	/** The effective domain (diverging: symmetric [-m, m]). */
	domain: Domain;
	/** Value -> [r, g, b] (0..255). Out-of-domain values clamp. NaN -> mid gray. */
	rgb(v: number): RGB;
	/** Value -> CSS color string `rgb(r g b)`. */
	css(v: number): string;
	/** Write the color of `v` into `out[offset..offset+3]` (RGBA, alpha = 255). For ImageData loops. */
	write(v: number, out: Uint8ClampedArray, offset: number): void;
	/** Normalized position 0..1 of v on the ramp (0.5 = zero for diverging). */
	position(v: number): number;
	/** Text color (ink) that stays readable on top of the color for v. */
	ink(v: number): string;
}

/**
 * Make a color scale.
 *   makeScale('diverging', [-2, 3])      // symmetric: [-3, 3]
 *   makeScale('sequential', [0, 1], 'dark')
 * The theme defaults to the current page theme.
 */
export function makeScale(kind: ScaleKind, domain: Domain, theme: Theme = currentTheme()): ColorScale {
	const table = lut(kind, theme);
	let lo: number;
	let hi: number;
	if (kind === 'diverging') {
		const m = Math.max(Math.abs(domain[0]), Math.abs(domain[1])) || 1;
		lo = -m;
		hi = m;
	} else {
		lo = domain[0];
		hi = domain[1];
		if (!(hi > lo)) hi = lo + 1;
	}
	const span = hi - lo;
	const position = (v: number): number => {
		if (Number.isNaN(v)) return kind === 'diverging' ? 0.5 : 0;
		const p = (v - lo) / span;
		return p < 0 ? 0 : p > 1 ? 1 : p;
	};
	const index = (v: number): number => Math.round(position(v) * (LUT_SIZE - 1)) * 3;
	const rgb = (v: number): RGB => {
		const i = index(v);
		return [table[i], table[i + 1], table[i + 2]];
	};
	return {
		kind,
		theme,
		domain: [lo, hi],
		rgb,
		css(v) {
			const [r, g, b] = rgb(v);
			return `rgb(${r} ${g} ${b})`;
		},
		write(v, out, offset) {
			const i = index(v);
			out[offset] = table[i];
			out[offset + 1] = table[i + 1];
			out[offset + 2] = table[i + 2];
			out[offset + 3] = 255;
		},
		position,
		ink(v) {
			return inkOn(rgb(v));
		}
	};
}

// ---------------------------------------------------------------- ink on a color

const INK_DARK = '#16150f';
const INK_LIGHT = '#fbfaf6';
const lin8 = (c: number): number => {
	const x = c / 255;
	return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
};

/** WCAG 2 relative luminance of an 8-bit sRGB color (0 = black, 1 = white). */
export function luminance([r, g, b]: RGB): number {
	return 0.2126 * lin8(r) + 0.7152 * lin8(g) + 0.0722 * lin8(b);
}

/** WCAG 2 contrast ratio of two colors (1 to 21). */
export function contrast(a: RGB, b: RGB): number {
	const x = luminance(a);
	const y = luminance(b);
	return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}

// hexToRgb is a function declaration (hoisted), so these can be computed here
const L_INK_DARK = luminance(hexToRgb(INK_DARK));
const L_INK_LIGHT = luminance(hexToRgb(INK_LIGHT));

/**
 * The text color (ink) to write on a background color: the dark or the light ink, whichever contrasts more. On the
 * few mid-tone colors where neither reaches 4.5:1 (WCAG AA for small text) it is pure black or white (always >= 4.58:1).
 */
export function inkOn(bg: RGB): string {
	const L = luminance(bg);
	const dark = (L + 0.05) / (L_INK_DARK + 0.05);
	const light = (L_INK_LIGHT + 0.05) / (L + 0.05);
	if (Math.max(dark, light) >= 4.5) return dark >= light ? INK_DARK : INK_LIGHT;
	return (L + 0.05) / 0.05 >= 1.05 / (L + 0.05) ? '#000000' : '#ffffff';
}

/** Diverging color for one value: blue (<0), gray (0), orange (>0). `maxAbs` sets the saturation point. */
export function divergingRgb(v: number, maxAbs: number, theme: Theme = currentTheme()): RGB {
	return makeScale('diverging', [-maxAbs, maxAbs], theme).rgb(v);
}

/** Sequential purple for one value in [lo, hi]. */
export function sequentialRgb(v: number, domain: Domain = [0, 1], theme: Theme = currentTheme()): RGB {
	return makeScale('sequential', domain, theme).rgb(v);
}

/** CSS linear-gradient string for legends (left = low / negative). */
export function gradientCss(kind: ScaleKind, theme: Theme = currentTheme(), steps = 9): string {
	const table = lut(kind, theme);
	const stops: string[] = [];
	for (let i = 0; i < steps; i++) {
		const t = i / (steps - 1);
		const k = Math.round(t * (LUT_SIZE - 1)) * 3;
		stops.push(`rgb(${table[k]} ${table[k + 1]} ${table[k + 2]}) ${(t * 100).toFixed(1)}%`);
	}
	return `linear-gradient(90deg, ${stops.join(', ')})`;
}

// ---------------------------------------------------------------- domains and scope

/**
 * Domain of one or more arrays for a scale kind.
 *   diverging  -> [-max|v|, max|v|]
 *   sequential -> [min(0, min v), max v]  (probabilities start at 0)
 * Use it to build `layer` or `global` scope domains: pass every tensor in that scope.
 */
export function domainOf(kind: ScaleKind, ...arrays: ArrayLike<number>[]): Domain {
	let lo = Infinity;
	let hi = -Infinity;
	for (const arr of arrays) {
		for (let i = 0; i < arr.length; i++) {
			const v = arr[i];
			if (Number.isNaN(v)) continue;
			if (v < lo) lo = v;
			if (v > hi) hi = v;
		}
	}
	if (lo === Infinity) return kind === 'diverging' ? [-1, 1] : [0, 1];
	if (kind === 'diverging') {
		const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
		return [-m, m];
	}
	return [Math.min(0, lo), hi > Math.min(0, lo) ? hi : Math.min(0, lo) + 1];
}

/**
 * Pick the domain for a color scope. `unit` uses `own` (this tensor/channel); `layer` and
 * `global` use the domains the page passes in (fall back to `own` if missing).
 */
export function scopeDomain(
	kind: ScaleKind,
	scope: ColorScope,
	own: ArrayLike<number>,
	scopes: { layer?: Domain; global?: Domain } = {}
): Domain {
	if (scope === 'layer' && scopes.layer) return scopes.layer;
	if (scope === 'global' && scopes.global) return scopes.global;
	return domainOf(kind, own);
}

// ---------------------------------------------------------------- categorical

/**
 * Fixed 10-color class palette. Slots 1–8 are a validated colorblind-aware order
 * (adjacent-pair CVD ΔE ≥ 8.4 in both themes, checked with the dataviz validator, OKLab×100);
 * slots 9–10 (cyan, brown) were added and the full 10 re-validated on adjacent pairs.
 * Not every pair is distinct under CVD, so class colors always appear with a text label.
 */
export const CATEGORICAL: Record<Theme, readonly string[]> = {
	light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948', '#0aa2c0', '#9b5a14'],
	dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767', '#1597ad', '#b06a2a']
};

/** Color of class k (wraps after 10; pages with > 10 classes should group into "other"). */
export function categorical(k: number, theme: Theme = currentTheme()): string {
	const p = CATEGORICAL[theme];
	return p[((k % p.length) + p.length) % p.length];
}

/** Hex '#rrggbb' -> RGB. */
export function hexToRgb(hex: string): RGB {
	const h = hex.replace('#', '');
	return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/** Fixed semantic colors that are not ramps (match the CSS tokens in tokens.css). */
export const SEMANTIC: Record<Theme, { real: string; generated: string; q: string; k: string; v: string; accent: string }> = {
	light: { real: '#1a8a4a', generated: '#c2338a', q: '#2a78d6', k: '#d93b3b', v: '#1a8a4a', accent: '#0b6b70' },
	dark: { real: '#3cb371', generated: '#e05aa8', q: '#5b9cf0', k: '#f07070', v: '#3cb371', accent: '#4fc4c0' }
};
