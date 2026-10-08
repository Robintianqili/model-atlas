/** Tiny chart helpers for the page's SVG charts (linear scales, nice ticks). Pure TS. */

export interface Scale {
	(v: number): number;
	domain: [number, number];
	range: [number, number];
	invert(px: number): number;
}

export function linear(domain: [number, number], range: [number, number]): Scale {
	const [d0, d1] = domain;
	const [r0, r1] = range;
	const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
	const f = ((v: number) => r0 + (v - d0) * k) as Scale;
	f.domain = domain;
	f.range = range;
	f.invert = (px: number) => (k === 0 ? d0 : d0 + (px - r0) / k);
	return f;
}

/** About `count` round tick values covering [lo, hi]. */
export function ticks(lo: number, hi: number, count = 5): number[] {
	if (!(hi > lo)) return [lo];
	const raw = (hi - lo) / Math.max(1, count);
	const mag = Math.pow(10, Math.floor(Math.log10(raw)));
	const norm = raw / mag;
	const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
	const out: number[] = [];
	const start = Math.ceil(lo / step - 1e-9) * step;
	for (let v = start; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
	return out;
}

export function clamp(v: number, lo: number, hi: number): number {
	return v < lo ? lo : v > hi ? hi : v;
}

/** SVG path for a polyline through points (skips NaN). */
export function linePath(pts: [number, number][]): string {
	let d = '';
	let pen = false;
	for (const [x, y] of pts) {
		if (!Number.isFinite(x) || !Number.isFinite(y)) {
			pen = false;
			continue;
		}
		d += (pen ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1);
		pen = true;
	}
	return d;
}

/** SVG path for a band between lower and upper polylines (same x). */
export function bandPath(xs: number[], lo: number[], hi: number[]): string {
	if (!xs.length) return '';
	let d = `M${xs[0].toFixed(1)} ${hi[0].toFixed(1)}`;
	for (let i = 1; i < xs.length; i++) d += `L${xs[i].toFixed(1)} ${hi[i].toFixed(1)}`;
	for (let i = xs.length - 1; i >= 0; i--) d += `L${xs[i].toFixed(1)} ${lo[i].toFixed(1)}`;
	return d + 'Z';
}

/** Marker shapes (secondary encoding next to color). Path centered at (0,0), size ≈ 2r. */
export function markerPath(shape: 'circle' | 'square' | 'triangle' | 'diamond', r: number): string {
	switch (shape) {
		case 'square':
			return `M${-r} ${-r}h${2 * r}v${2 * r}h${-2 * r}Z`;
		case 'triangle': {
			const h = r * 1.15;
			return `M0 ${-h}L${h} ${h * 0.85}L${-h} ${h * 0.85}Z`;
		}
		case 'diamond':
			return `M0 ${-r * 1.25}L${r * 1.25} 0L0 ${r * 1.25}L${-r * 1.25} 0Z`;
		default:
			return `M${-r} 0a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
	}
}

/** Format an accuracy fraction (0..1) as a percent with `d` decimals. */
export function pct(v: number, d = 1): string {
	return (v * 100).toFixed(d) + '%';
}
