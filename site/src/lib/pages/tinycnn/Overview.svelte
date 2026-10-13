<!--
  Overview: every layer of TinyCNN as a column of feature-map tiles (CNN Explainer style), with
  edges showing what feeds what. Left to right on wide screens, top to bottom on narrow ones.
  Clicking (or Enter on) a tile selects that unit for the zoom view below; hovering a tile
  highlights the tiles that feed it.

  Keyboard: one tab stop. Arrow keys move between tiles (selection follows), Enter jumps to the
  zoom view.

  Fit: when the horizontal layout is wider than its box, it is scaled down to fit (down to 80 %), so
  Flatten, Dense and Softmax stay in view; below that it switches to the vertical layout. If it
  still overflows, a fade and "scroll →" cue mark the hidden part.
-->
<script lang="ts">
	import { tick } from 'svelte';
	import MapCanvas from './MapCanvas.svelte';
	import ScopeSwitch from '$lib/kit/ScopeSwitch.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import { categorical, makeScale } from '$lib/kit/color';
	import { themeState, motion } from '$lib/kit/theme.svelte';
	import { fmt3, fmtPct, fmtShape } from '$lib/kit/format';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import { channelDomain, isSpatial, unitsOf, type LayerSpec } from './data';
	import type { CnnStore } from './store.svelte';

	let { S, onzoom }: { S: CnnStore; onzoom?: () => void } = $props();

	const layers = $derived(S.spec.layers);
	const classes = $derived(S.spec.classes);

	let wrap: HTMLDivElement | undefined = $state();
	let track: HTMLDivElement | undefined = $state();
	let width = $state(0);
	/** the flowing-dash animation runs only while the overview is on screen (it repaints every frame) */
	let onscreen = $state(false);
	/** natural (unscaled) size of the horizontal layout, measured */
	let natW = $state(0);
	let natH = $state(0);
	let overflowing = $state(false);
	const MIN_FIT = 0.8;
	const vertical = $derived(width > 0 && width < Math.max(700, natW * MIN_FIT));
	/** scale of the horizontal layout so that it fits its box */
	const fit = $derived(!vertical && natW > 0 && width > 0 ? Math.max(MIN_FIT, Math.min(1, (width - 4) / natW)) : 1);

	// ------------------------------------------------------------ sizes
	const AREA_H = 352; // height of the tile area of every column (horizontal layout)
	function tilePx(l: LayerSpec): number {
		const H = l.outShape[1] ?? 1;
		if (vertical) {
			const per = Math.min(8, l.outShape[0]);
			const w = Math.floor((width - 2 - (per - 1) * 4) / per);
			if (l.op === 'input') return Math.min(112, Math.max(72, Math.floor(width / 3)));
			return Math.max(26, Math.min(H >= 12 ? 48 : 40, w));
		}
		return H >= 28 ? 52 : H >= 24 ? 40 : H >= 12 ? 34 : H >= 8 ? 28 : 22;
	}
	const units = (l: LayerSpec) => Array.from({ length: unitsOf(l) }, (_, i) => i);

	// ------------------------------------------------------------ element registry and positions
	type Rect = { l: number; r: number; t: number; b: number };
	const els = new Map<string, HTMLElement>();
	let pos = $state.raw<Record<string, Rect>>({});
	const hasTrace = $derived(!!S.trace);
	let measureTick = $state(0);

	function reg(node: HTMLElement, key: string) {
		els.set(key, node);
		return {
			update(k: string) {
				els.delete(key);
				key = k;
				els.set(key, node);
			},
			destroy() {
				if (els.get(key) === node) els.delete(key);
			}
		};
	}

	function measure() {
		if (!track) return;
		if (!vertical) {
			// layout size (transforms do not change offsetWidth)
			natW = track.offsetWidth;
			natH = track.offsetHeight;
		}
		// positions in the track's own (unscaled) coordinates, where the SVG edges are drawn
		const f = fit;
		const tr = track.getBoundingClientRect();
		const out: Record<string, Rect> = {};
		for (const [k, el] of els) {
			const r = el.getBoundingClientRect();
			out[k] = { l: (r.left - tr.left) / f, r: (r.right - tr.left) / f, t: (r.top - tr.top) / f, b: (r.bottom - tr.top) / f };
		}
		pos = out;
		const sc = wrap?.querySelector('.scroller');
		overflowing = !!sc && sc.scrollWidth > sc.clientWidth + 2;
	}

	$effect(() => {
		if (!wrap) return;
		const ro = new ResizeObserver((e) => {
			width = Math.floor(e[0].contentRect.width);
			measureTick++;
		});
		ro.observe(wrap);
		return () => ro.disconnect();
	});
	$effect(() => {
		if (!wrap) return;
		const io = new IntersectionObserver((e) => (onscreen = e[e.length - 1].isIntersecting));
		io.observe(wrap);
		return () => io.disconnect();
	});
	$effect(() => {
		if (!track) return;
		const ro = new ResizeObserver(() => measureTick++);
		ro.observe(track);
		return () => ro.disconnect();
	});
	$effect(() => {
		void measureTick;
		void vertical;
		void fit;
		void i18n.lang;
		void hasTrace; // the first trace swaps placeholders for tiles
		tick().then(measure);
	});

	// ------------------------------------------------------------ edges
	type Seg = [number, number, number, number];
	const keyOf = (ci: number, u: number) => `${ci}:${u}`;
	/** anchor points of the flatten strip: one per pool2 channel (16 segments) */
	function flatAnchor(ci: number, k: number, n: number, side: 'in' | 'out'): [number, number] | null {
		const r = pos[keyOf(ci, 0)];
		if (!r) return null;
		if (vertical) {
			const x = r.l + ((k + 0.5) / n) * (r.r - r.l);
			return [x, side === 'in' ? r.t : r.b];
		}
		const y = r.t + ((k + 0.5) / n) * (r.b - r.t);
		return [side === 'in' ? r.l : r.r, y];
	}
	function outPt(ci: number, u: number): [number, number] | null {
		const l = layers[ci];
		if (l.op === 'flatten') return null;
		const r = pos[keyOf(ci, u)];
		if (!r) return null;
		return vertical ? [(r.l + r.r) / 2, r.b] : [r.r, (r.t + r.b) / 2];
	}
	function inPt(ci: number, u: number): [number, number] | null {
		const r = pos[keyOf(ci, u)];
		if (!r) return null;
		return vertical ? [(r.l + r.r) / 2, r.t] : [r.l, (r.t + r.b) / 2];
	}
	/** incoming edges of unit u of column ci, as [x1, y1, x2, y2] */
	function incoming(ci: number, u: number): Seg[] {
		if (ci === 0) return [];
		const l = layers[ci];
		const prev = layers[ci - 1];
		const segs: Seg[] = [];
		const push = (a: [number, number] | null, b: [number, number] | null) => {
			if (a && b) segs.push([a[0], a[1], b[0], b[1]]);
		};
		const nPrev = unitsOf(prev);
		if (l.op === 'conv2d') for (let v = 0; v < nPrev; v++) push(outPt(ci - 1, v), inPt(ci, u));
		else if (l.op === 'relu' || l.op === 'maxpool2d') push(outPt(ci - 1, u), inPt(ci, u));
		else if (l.op === 'flatten') for (let v = 0; v < nPrev; v++) push(outPt(ci - 1, v), flatAnchor(ci, v, nPrev, 'in'));
		else if (l.op === 'dense') {
			const nC = layers[ci - 2]?.outShape[0] ?? 16;
			for (let v = 0; v < nC; v++) push(flatAnchor(ci - 1, v, nC, 'out'), inPt(ci, u));
		} else if (l.op === 'softmax') push(outPt(ci - 1, u), inPt(ci, u));
		return segs;
	}
	function pathOf(segs: Seg[]): string {
		let d = '';
		for (const [x1, y1, x2, y2] of segs) {
			if (vertical) {
				const m = (y2 - y1) / 2;
				d += `M${x1.toFixed(1)} ${y1.toFixed(1)}C${x1.toFixed(1)} ${(y1 + m).toFixed(1)} ${x2.toFixed(1)} ${(y2 - m).toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
			} else {
				const m = (x2 - x1) / 2;
				d += `M${x1.toFixed(1)} ${y1.toFixed(1)}C${(x1 + m).toFixed(1)} ${y1.toFixed(1)} ${(x2 - m).toFixed(1)} ${y2.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
			}
		}
		return d;
	}
	const basePath = $derived.by(() => {
		void pos;
		if (vertical) return '';
		const segs: Seg[] = [];
		for (let ci = 1; ci < layers.length; ci++) for (const u of units(layers[ci])) segs.push(...incoming(ci, u));
		return pathOf(segs);
	});

	// ------------------------------------------------------------ hover / selection
	let hovered = $state<{ ci: number; u: number } | null>(null);
	const selCi = $derived(layers.findIndex((l) => l.id === S.sel.layer));
	const active = $derived(hovered ?? (selCi >= 0 ? { ci: selCi, u: unitsOf(layers[selCi]) > 1 ? S.sel.ch : 0 } : null));
	const hiPath = $derived.by(() => {
		void pos;
		if (!active) return '';
		const l = layers[active.ci];
		if (l.op === 'softmax') {
			// softmax normalizes over every logit
			const segs: Seg[] = [];
			for (let v = 0; v < unitsOf(l); v++) segs.push(...incoming(active.ci, v));
			return pathOf(segs);
		}
		return pathOf(incoming(active.ci, active.u));
	});
	/** source tiles of the active unit (outlined) */
	const sources = $derived.by(() => {
		const s = new Set<string>();
		if (!active || active.ci === 0) return s;
		const l = layers[active.ci];
		const p = active.ci - 1;
		if (l.op === 'conv2d' || l.op === 'flatten' || l.op === 'softmax') for (const v of units(layers[p])) s.add(keyOf(p, v));
		else if (l.op === 'relu' || l.op === 'maxpool2d') s.add(keyOf(p, active.u));
		else if (l.op === 'dense') s.add(keyOf(p, 0));
		return s;
	});

	function choose(ci: number, u: number) {
		const l = layers[ci];
		S.select(l.id, u);
	}

	async function onkeydown(e: KeyboardEvent) {
		const ci = selCi < 0 ? 0 : selCi;
		const l = layers[ci];
		const n = unitsOf(l);
		const u = n > 1 ? S.sel.ch : 0;
		let nci = ci;
		let nu = u;
		const nextLayer = vertical ? 'ArrowDown' : 'ArrowRight';
		const prevLayer = vertical ? 'ArrowUp' : 'ArrowLeft';
		const nextUnit = vertical ? 'ArrowRight' : 'ArrowDown';
		const prevUnit = vertical ? 'ArrowLeft' : 'ArrowUp';
		if (e.key === nextLayer) nci = Math.min(layers.length - 1, ci + 1);
		else if (e.key === prevLayer) nci = Math.max(0, ci - 1);
		else if (e.key === nextUnit) nu = Math.min(n - 1, u + 1);
		else if (e.key === prevUnit) nu = Math.max(0, u - 1);
		else if (e.key === 'Home') nu = 0;
		else if (e.key === 'End') nu = n - 1;
		else if (e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			onzoom?.();
			return;
		} else return;
		e.preventDefault();
		if (nci !== ci) nu = Math.min(unitsOf(layers[nci]) - 1, u);
		choose(nci, nu);
		await tick();
		els.get(keyOf(nci, nu))?.focus();
	}

	// ------------------------------------------------------------ colors and text
	const tr = $derived(S.trace);
	function tileData(l: LayerSpec, u: number): Float32Array | null {
		const d = tr?.t[l.id]?.data;
		if (!d) return null;
		const hw = (l.outShape[1] ?? 1) * (l.outShape[2] ?? 1);
		return d.subarray(u * hw, (u + 1) * hw);
	}
	function tileDomain(l: LayerSpec, u: number): [number, number] {
		const d = tr?.t[l.id]?.data;
		const hw = (l.outShape[1] ?? 1) * (l.outShape[2] ?? 1);
		return S.domainFor(l.id, () => (d ? channelDomain(d, u, hw) : [-1, 1]));
	}
	const denseScale = $derived(makeScale('diverging', S.domains?.layer.dense ?? [-1, 1], themeState.resolved));
	const unitName = (l: LayerSpec, u: number) => (l.op === 'dense' || l.op === 'softmax' ? t(classes[u]) : `${u}`);

	const status = $derived.by(() => {
		if (!active || !tr) return '';
		const l = layers[active.ci];
		const name = t(l.label);
		const u = active.u;
		if (isSpatial(l)) {
			const d = tileData(l, u)!;
			let lo = Infinity;
			let hi = -Infinity;
			for (let i = 0; i < d.length; i++) {
				if (d[i] < lo) lo = d[i];
				if (d[i] > hi) hi = d[i];
			}
			const ch = l.op === 'input' ? '' : ` · ${t({ en: 'channel', zh: '通道' })} ${u}`;
			return `${name}${ch} · ${fmtShape(l.outShape.slice(1))} · ${t({ en: 'values', zh: '数值' })} ${fmt3(lo)} … ${fmt3(hi)}`;
		}
		if (l.op === 'flatten') return `${name} · ${t({ en: '256 numbers in one list', zh: '一列 256 个数' })}`;
		if (l.op === 'dense') return `${name} · ${t(classes[u])}: logit ${fmt3(tr.t.dense.data[u])}`;
		return `${name} · ${t(classes[u])}: ${fmtPct(tr.t.softmax.data[u])}`;
	});

	const legendLayer = $derived(selCi >= 0 && isSpatial(layers[selCi]) && layers[selCi].op !== 'input' ? layers[selCi] : layers.find((l) => l.id === 'conv1_1')!);
	const legendDomain = $derived(tileDomain(legendLayer, legendLayer.id === S.sel.layer ? S.sel.ch : 0));
	const scopeText = $derived(
		S.scope === 'unit'
			? t({ en: 'each map uses its own range', zh: '每张图使用自己的范围' })
			: S.scope === 'layer'
				? t({ en: 'one range per layer', zh: '每层一个范围' })
				: t({ en: 'one range for all layers', zh: '所有层共用一个范围' })
	);
</script>

<div class="ov-top">
	<ScopeSwitch bind:value={S.scope} />
	<div class="legends">
		<ColorLegend kind="sequential" domain={[0, 1]} label={t({ en: 'Input ink', zh: '输入墨迹' })} width={120} />
		<ColorLegend
			kind="diverging"
			domain={legendDomain}
			label={t({ en: `Activations (${t(legendLayer.label)}; ${scopeText})`, zh: `激活值（${t(legendLayer.label)}；${scopeText}）` })}
			width={190}
		/>
	</div>
</div>

<div class="ov" bind:this={wrap} class:vertical class:overflowing>
	<div class="scroller">
		<div class="fitbox" style={fit < 1 ? `width: ${Math.ceil(natW * fit)}px; height: ${Math.ceil(natH * fit)}px` : undefined}>
		<!-- svelte-ignore a11y_no_noninteractive_element_interactions (composite widget: arrow keys move between tiles) -->
		<div
			class="track"
			style={fit < 1 ? `transform: scale(${fit}); transform-origin: 0 0` : undefined}
			bind:this={track}
			role="group"
			aria-label={t({ en: 'TinyCNN layers. Arrow keys move between maps; Enter zooms in.', zh: 'TinyCNN 各层。方向键在特征图之间移动；回车放大查看。' })}
			{onkeydown}
		>
			<svg class="edges" aria-hidden="true">
				{#if basePath}<path class="base" d={basePath} />{/if}
				{#if hiPath}<path class="hi" class:flow={!motion.reduced && onscreen} d={hiPath} />{/if}
			</svg>
			{#each layers as l, ci (l.id)}
				{@const tp = tilePx(l)}
				<div class="col op-{l.op}" class:two={!vertical && unitsOf(l) > 8 && isSpatial(l)} class:cur={selCi === ci}>
					<div class="head">
						<span class="name">{t(l.label)}</span>
						<span class="shape num">{fmtShape(l.outShape)}</span>
					</div>
					<div class="area" style={vertical ? '' : `height: ${AREA_H}px`}>
						{#if isSpatial(l)}
							<div class="tiles" style="--tp: {tp}px">
								{#each units(l) as u (u)}
									{@const data = tileData(l, u)}
									{@const k = keyOf(ci, u)}
									<button
										type="button"
										class="tile"
										class:sel={S.sel.layer === l.id && S.sel.ch === u}
										class:src={sources.has(k)}
										tabindex={S.sel.layer === l.id && S.sel.ch === u ? 0 : -1}
										aria-label="{t(l.label)}{l.op === 'input' ? '' : `, ${t({ en: 'channel', zh: '通道' })} ${u}`}"
										aria-pressed={S.sel.layer === l.id && S.sel.ch === u}
										use:reg={k}
										onclick={() => choose(ci, u)}
										onpointerenter={() => (hovered = { ci, u })}
										onpointerleave={() => (hovered = null)}
									>
										{#if data}
											<MapCanvas
												{data}
												rows={l.outShape[1]}
												cols={l.outShape[2]}
												kind={l.op === 'input' ? 'sequential' : 'diverging'}
												domain={tileDomain(l, u)}
												size={tp}
											/>
										{:else}
											<span class="ph" style="width: {tp}px; height: {tp}px"></span>
										{/if}
									</button>
								{/each}
							</div>
						{:else if l.op === 'flatten'}
							{@const d = tr?.t.flatten.data}
							<button
								type="button"
								class="tile flat"
								class:sel={S.sel.layer === l.id}
								class:src={sources.has(keyOf(ci, 0))}
								tabindex={S.sel.layer === l.id ? 0 : -1}
								aria-label={t({ en: 'Flatten: 256 numbers', zh: '展平：256 个数' })}
								aria-pressed={S.sel.layer === l.id}
								use:reg={keyOf(ci, 0)}
								onclick={() => choose(ci, 0)}
								onpointerenter={() => (hovered = { ci, u: 0 })}
								onpointerleave={() => (hovered = null)}
							>
								{#if d}
									{#if vertical}
										<MapCanvas data={d} rows={1} cols={256} domain={S.domainFor('flatten', () => [-1, 1])} size={Math.max(120, width - 6)} height={16} />
									{:else}
										<MapCanvas data={d} rows={256} cols={1} domain={S.domainFor('flatten', () => [-1, 1])} size={12} height={AREA_H - 16} />
									{/if}
								{/if}
							</button>
						{:else if l.op === 'dense'}
							<div class="rows dense">
								{#each units(l) as u (u)}
									{@const v = tr?.t.dense.data[u] ?? 0}
									{@const k = keyOf(ci, u)}
									<button
										type="button"
										class="row-btn dcell"
										class:sel={S.sel.layer === l.id && S.sel.ch === u}
										class:src={sources.has(k)}
										tabindex={S.sel.layer === l.id && S.sel.ch === u ? 0 : -1}
										aria-label="{t(l.label)}, {t(classes[u])}: {fmt3(v)}"
										aria-pressed={S.sel.layer === l.id && S.sel.ch === u}
										use:reg={k}
										onclick={() => choose(ci, u)}
										onpointerenter={() => (hovered = { ci, u })}
										onpointerleave={() => (hovered = null)}
									>
										<span class="sq num" style="background: {tr ? denseScale.css(v) : 'var(--surface-2)'}; color: {denseScale.ink(v)}">{tr ? fmt3(v) : ''}</span>
									</button>
								{/each}
							</div>
						{:else if l.op === 'softmax'}
							<div class="rows soft">
								{#each units(l) as u (u)}
									{@const p = tr?.t.softmax.data[u] ?? 0}
									{@const k = keyOf(ci, u)}
									<button
										type="button"
										class="row-btn prow"
										class:sel={S.sel.layer === l.id && S.sel.ch === u}
										class:top={tr && S.top === u}
										tabindex={S.sel.layer === l.id && S.sel.ch === u ? 0 : -1}
										aria-label="{t(classes[u])}: {fmtPct(p)}"
										aria-pressed={S.sel.layer === l.id && S.sel.ch === u}
										use:reg={k}
										onclick={() => choose(ci, u)}
										onpointerenter={() => (hovered = { ci, u })}
										onpointerleave={() => (hovered = null)}
									>
										<span class="sw" style="background: {categorical(u, themeState.resolved)}"></span>
										<span class="cn">{unitName(l, u)}</span>
										<span class="bar"><span style="width: {Math.max(1, p * 100)}%"></span></span>
										<span class="pv num">{tr ? fmtPct(p, 0) : ''}</span>
									</button>
								{/each}
							</div>
						{/if}
					</div>
				</div>
			{/each}
		</div>
		</div>
	</div>
	{#if overflowing}<span class="cue" aria-hidden="true">{t({ en: 'scroll', zh: '滑动' })} →</span>{/if}
	<p class="status num" aria-live="polite">{status || ' '}</p>
</div>

<style>
	.ov-top {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 24px;
		align-items: flex-end;
		justify-content: space-between;
		margin-bottom: 10px;
	}
	.legends {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 18px;
	}
	.ov {
		position: relative;
		min-width: 0;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		padding: 10px 10px 4px;
	}
	.scroller {
		overflow-x: auto;
		overscroll-behavior-x: contain;
	}
	.fitbox {
		margin-inline: auto;
	}
	.overflowing .scroller {
		mask-image: linear-gradient(90deg, #000 calc(100% - 48px), transparent);
	}
	.cue {
		position: absolute;
		right: 12px;
		top: 12px;
		padding: 1px 8px;
		border-radius: var(--r-pill);
		background: var(--surface-2);
		border: 1px solid var(--rule-strong);
		font-size: var(--fs-xs);
		color: var(--ink-2);
		pointer-events: none;
	}
	.track {
		position: relative;
		display: flex;
		gap: 15px;
		width: max-content;
		margin-inline: auto;
		padding: 2px 4px 6px;
	}
	.vertical .track {
		flex-direction: column;
		gap: 26px;
		width: 100%;
		padding: 0;
	}
	.edges {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		overflow: visible;
		pointer-events: none;
	}
	.edges path {
		fill: none;
	}
	.base {
		stroke: var(--rule-strong);
		stroke-width: 0.8;
		opacity: 0.45;
	}
	.hi {
		stroke: var(--accent);
		stroke-width: 1.6;
		opacity: 0.9;
		stroke-dasharray: 6 4;
	}
	.hi.flow {
		animation: dash 0.8s linear infinite;
	}
	@keyframes dash {
		to {
			stroke-dashoffset: -10;
		}
	}
	.col {
		position: relative;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 6px;
	}
	.vertical .col {
		align-items: stretch;
	}
	.head {
		display: grid;
		justify-items: center;
		line-height: 1.15;
		min-height: 2.4em;
	}
	.vertical .head {
		display: flex;
		gap: 10px;
		align-items: baseline;
		min-height: 0;
	}
	.name {
		font-size: 12px;
		font-weight: 650;
		white-space: nowrap;
	}
	.cur .name {
		color: var(--accent);
	}
	.shape {
		font-size: var(--t-xs);
		color: var(--muted);
		white-space: nowrap;
	}
	.area {
		display: flex;
		flex-direction: column;
		justify-content: center;
	}
	.tiles {
		display: grid;
		grid-auto-flow: row;
		gap: 4px;
		justify-content: center;
	}
	.two .tiles {
		grid-template-rows: repeat(8, auto);
		grid-auto-flow: column;
	}
	.vertical .tiles {
		grid-template-columns: repeat(auto-fill, var(--tp));
		justify-content: start;
	}
	.vertical .op-input .tiles {
		grid-template-columns: var(--tp);
	}
	.tile {
		display: block;
		padding: 0;
		border: 0;
		background: none;
		border-radius: 3px;
		outline: 1px solid var(--rule);
		position: relative;
		z-index: 1;
		cursor: pointer;
	}
	.tile:hover {
		outline: 2px solid var(--ink-2);
	}
	.tile.src {
		outline: 2px dashed var(--accent);
		outline-offset: 1px;
	}
	.tile.sel {
		outline: 3px solid var(--accent);
		outline-offset: 1px;
	}
	.tile:focus-visible {
		outline: 3px solid var(--accent);
		outline-offset: 3px;
	}
	.ph {
		display: block;
		background: var(--surface-2);
		border-radius: 2px;
	}
	.flat {
		padding: 0;
	}
	.rows {
		display: grid;
		gap: 4px;
	}
	.vertical .rows.dense {
		grid-template-columns: repeat(5, minmax(0, 1fr));
	}
	.row-btn {
		position: relative;
		z-index: 1;
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 0 4px;
		min-height: 32px;
		border: 1px solid transparent;
		border-radius: 4px;
		background: var(--surface);
		font-size: 12px;
		text-align: left;
		cursor: pointer;
	}
	.row-btn:hover {
		border-color: var(--rule-strong);
	}
	.row-btn.sel {
		border-color: var(--accent);
		box-shadow: 0 0 0 1px var(--accent);
	}
	.row-btn.src {
		border: 1.5px dashed var(--accent);
	}
	.sq {
		display: grid;
		place-items: center;
		min-width: 42px;
		height: 24px;
		border-radius: 3px;
		font-size: var(--t-xs);
	}
	.vertical .sq {
		min-width: 0;
		width: 100%;
	}
	.vertical .dcell {
		padding: 2px;
	}
	.prow {
		width: 150px;
	}
	.vertical .prow {
		width: 100%;
	}
	.prow.top .cn {
		font-weight: 700;
	}
	.sw {
		width: 9px;
		height: 9px;
		border-radius: 2px;
		flex: 0 0 auto;
	}
	.cn {
		flex: 0 0 60px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.vertical .cn {
		flex-basis: 84px;
	}
	.bar {
		flex: 1 1 auto;
		height: 9px;
		background: var(--surface-2);
		border-radius: 0 3px 3px 0;
		overflow: hidden;
	}
	.bar span {
		display: block;
		height: 100%;
		background: var(--seq-mid);
	}
	.prow.top .bar span {
		background: var(--seq-hi);
	}
	.pv {
		flex: 0 0 34px;
		text-align: right;
		font-size: var(--t-xs);
	}
	.status {
		min-height: 1.5em;
		margin: 4px 2px 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
</style>
