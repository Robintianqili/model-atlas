// Accessibility guards for the kit's colors (the axe color-contrast rule that `atlas e2e` runs, AK604), in Node:
// the ink written on every color of the value ramps, and the text tokens of tokens.css in both themes.
// Run from site/: node --test src/lib/atlas/ui/a11y.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CATEGORICAL, contrast, hexToRgb, inkOn, makeScale } from '../../kit/color.ts';

const AA = 4.5;
const ratio = (a: string, b: string) => contrast(hexToRgb(a), hexToRgb(b));

test('kit color: scale.ink(v) keeps 4.5:1 on every ramp color, both themes', () => {
	for (const theme of ['light', 'dark'] as const)
		for (const kind of ['diverging', 'sequential'] as const) {
			const sc = makeScale(kind, [-1, 1], theme);
			for (let i = 0; i <= 255; i++) {
				const v = -1 + (2 * i) / 255;
				const c = contrast(sc.rgb(v), hexToRgb(sc.ink(v)));
				assert.ok(c >= AA, `${theme} ${kind} at ${v.toFixed(3)}: ${c.toFixed(2)}:1`);
			}
		}
});

test('kit color: inkOn on the class colors and on the worst mid grays', () => {
	for (const hex of [...CATEGORICAL.light, ...CATEGORICAL.dark, '#767676', '#777777', '#7a7a7a', '#808080'])
		assert.ok(contrast(hexToRgb(hex), hexToRgb(inkOn(hexToRgb(hex)))) >= AA, hex);
});

/** The custom properties of one rule block of tokens.css (`:root {` or `:root[data-theme='dark'] {`): hex colors and
 *  var() aliases (the redesign's legacy names, e.g. --bg: var(--paper)). */
function block(css: string, head: string): Record<string, string> {
	const at = css.indexOf(head);
	assert.ok(at >= 0, `tokens.css has no ${head}`);
	const body = css.slice(at + head.length, css.indexOf('}', at));
	return Object.fromEntries([...body.matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6}|var\(--[\w-]+\))\s*;/g)].map((m) => [m[1], m[2]]));
}

/** Resolve var(--x) aliases to their hex value in the same theme. */
function resolved(tk: Record<string, string>): Record<string, string> {
	const out: Record<string, string> = {};
	for (const k of Object.keys(tk)) {
		let v = tk[k];
		for (let i = 0; i < 4 && v?.startsWith('var('); i++) v = tk[v.slice(6, -1)];
		if (v) out[k] = v;
	}
	return out;
}

test('tokens.css: text tokens keep 4.5:1 on the page surfaces, both themes', () => {
	const css = readFileSync(join(fileURLToPath(new URL('.', import.meta.url)), '../../styles/tokens.css'), 'utf8');
	const light = block(css, ':root {');
	const darkOwn = block(css, ":root[data-theme='dark'] {");
	// the aliases live on :root once and follow the dark values
	const aliases = Object.fromEntries(Object.entries(light).filter(([, v]) => v.startsWith('var(')));
	const themes = { light: resolved(light), dark: resolved({ ...light, ...darkOwn, ...aliases }) };
	const texts = ['ink', 'ink-2', 'muted', 'accent', 'div-neg-text', 'div-pos-text', 'q-text', 'k-text', 'v-text', 'real-text', 'generated-text'];
	for (const [theme, tk] of Object.entries(themes)) {
		for (const t of texts)
			for (const bg of ['bg', 'surface', 'surface-2']) {
				assert.ok(tk[t] && tk[bg], `${theme}: --${t} and --${bg} are defined`);
				assert.ok(ratio(tk[t], tk[bg]) >= AA, `${theme}: --${t} on --${bg} is ${ratio(tk[t], tk[bg]).toFixed(2)}:1`);
			}
		// the current row, the highlighted code line and the pressed chip use --accent-soft behind muted text
		assert.ok(ratio(tk.muted, tk['accent-soft']) >= AA, `${theme}: --muted on --accent-soft is ${ratio(tk.muted, tk['accent-soft']).toFixed(2)}:1`);
		for (const [fg, bg] of [['live', 'live-soft'], ['pre', 'pre-soft'], ['illus', 'illus-soft']]) assert.ok(ratio(tk[fg], tk[bg]) >= AA, `${theme}: provenance badge --${fg} on --${bg}`);
	}
	// the media-query copy of the dark theme says the same as [data-theme='dark']
	const media = css.slice(css.indexOf('@media (prefers-color-scheme: dark)'));
	const m = block(media, ":root:not([data-theme='light']) {");
	for (const [k, v] of Object.entries(darkOwn)) assert.equal(m[k], v, `--${k}: the two dark blocks differ`);
});
