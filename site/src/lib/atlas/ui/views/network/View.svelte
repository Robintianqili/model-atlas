<!--
  network: an MLP as nodes and lines on the recorded weights (or their gradients, or this update's change). Canvas.
  Keyboard: focus the diagram, ←/→ move between layers, ↑/↓ between nodes; the readout names the node, its bias or
  value, and its strongest incoming line. Wide layers are elided ("showing 12 of 64").
-->
<script lang="ts">
	import type { StageProps } from '../../../types/view.ts';
	import type { I18n } from '../../../types/core.ts';
	import { colon, t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '../../../core/format.ts';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { isLive, tapeOf } from '../../train/refs.ts';
	import meta, { netData } from './meta.ts';

	// props one by one (no rest/spread props: they would grow the Svelte runtime chunk every page preloads)
	let { rt, source, at, focus, sel: selection, ctx, provenance, autoplay, onend, props }: StageProps = $props();
	const p: StageProps = $derived({ rt, source, at, focus, sel: selection, ctx, provenance, autoplay, onend, props });

	let version = $state(0);
	$effect(() => {
		const tp = tapeOf(p.rt, p.source);
		if (isLive(tp)) return tp.subscribe(() => version++);
	});
	const d = $derived.by(() => {
		void version;
		return netData(p);
	});
	const theme = $derived(themeState.resolved);
	const maxNodes = $derived(Math.max(4, Math.floor(Number(p.props?.maxNodes ?? 12)) || 12));

	let W = $state(520);
	const width = $derived(Math.max(260, Math.min(W, 640)));
	const height = $derived(Math.min(300, Math.max(200, (Math.min(maxNodes, Math.max(...(d.sizes ?? [4]))) + 1) * 24)));

	/** shown node indices of a layer of size n: all, or the first half and the last half with a gap */
	function shown(n: number): number[] {
		if (n <= maxNodes) return Array.from({ length: n }, (_, i) => i);
		const a = Math.ceil(maxNodes / 2),
			b = maxNodes - a;
		return [...Array.from({ length: a }, (_, i) => i), ...Array.from({ length: b }, (_, i) => n - b + i)];
	}
	const layout = $derived.by(() => {
		const sizes = d.sizes ?? [];
		const L = sizes.length;
		const padX = 34;
		return sizes.map((n, li) => {
			const idx = shown(n);
			const x = L > 1 ? padX + (li / (L - 1)) * (width - 2 * padX) : width / 2;
			const gap = height / (idx.length + 1);
			return { n, idx, x, y: (k: number) => gap * (k + 1) + (idx.length < n && k >= Math.ceil(maxNodes / 2) ? 0 : 0) };
		});
	});
	const pos = (li: number, node: number) => {
		const l = layout[li];
		const k = l.idx.indexOf(node);
		return k < 0 ? null : { x: l.x, y: l.y(k) };
	};

	let canvas = $state<HTMLCanvasElement | null>(null);
	let sel = $state<{ layer: number; node: number } | null>(null);
	let colors: { theme: string; surface: string; ink2: string; rule: string; accent: string } | null = null;

	function draw() {
		const c = canvas;
		if (!c || !d.sizes) return;
		const dpr = typeof devicePixelRatio === 'number' ? Math.min(3, devicePixelRatio) : 1;
		c.width = Math.round(width * dpr);
		c.height = Math.round(height * dpr);
		const g = c.getContext('2d');
		if (!g) return;
		g.setTransform(dpr, 0, 0, dpr, 0, 0);
		if (!colors || colors.theme !== theme) {
			const css = getComputedStyle(c);
			colors = { theme, surface: css.getPropertyValue('--surface').trim() || '#fff', ink2: css.getPropertyValue('--ink-2').trim() || '#555', rule: css.getPropertyValue('--rule').trim() || '#ccc', accent: css.getPropertyValue('--accent').trim() || '#0b6b70' };
		}
		g.clearRect(0, 0, width, height);
		const sc = makeScale('diverging', [-(d.maxAbs || 1), d.maxAbs || 1], theme);
		// edges
		d.layers.forEach((l, li) => {
			const I = l.weight.shape[1];
			for (const o of layout[li + 1].idx)
				for (const i of layout[li].idx) {
					const v = l.edge[o * I + i];
					const u = d.maxAbs ? Math.abs(v) / d.maxAbs : 0;
					const a = pos(li, i)!,
						b = pos(li + 1, o)!;
					g.strokeStyle = sc.css(v);
					g.globalAlpha = 0.2 + 0.8 * u;
					g.lineWidth = 0.5 + 3.5 * u;
					g.beginPath();
					g.moveTo(a.x, a.y);
					g.lineTo(b.x, b.y);
					g.stroke();
				}
		});
		g.globalAlpha = 1;
		// the focused edge
		if (d.focus && d.layers[d.focus.layer]) {
			const a = pos(d.focus.layer, d.focus.i),
				b = pos(d.focus.layer + 1, d.focus.o);
			if (a && b) {
				g.strokeStyle = colors!.accent;
				g.lineWidth = 4;
				g.beginPath();
				g.moveTo(a.x, a.y);
				g.lineTo(b.x, b.y);
				g.stroke();
			}
		}
		// nodes
		const r = Math.max(5, Math.min(10, height / (maxNodes * 2.6)));
		let vmax = 0;
		for (const vals of d.values ?? []) if (vals) for (const v of vals) vmax = Math.max(vmax, Math.abs(v));
		let bmax = 0;
		for (const l of d.layers) if (l.bias) for (const v of l.bias.data) bmax = Math.max(bmax, Math.abs(v));
		const vs = makeScale('diverging', [-(vmax || 1), vmax || 1], theme);
		const bs = makeScale('diverging', [-(bmax || 1), bmax || 1], theme);
		layout.forEach((l, li) => {
			l.idx.forEach((node, k) => {
				const y = l.y(k);
				const vals = d.values?.[li];
				let fill = colors!.surface;
				if (vals) fill = vs.css(vals[node]);
				else if (li > 0 && d.layers[li - 1].bias) fill = bs.css(d.layers[li - 1].bias!.data[node]);
				g.beginPath();
				g.arc(l.x, y, r, 0, Math.PI * 2);
				g.fillStyle = fill;
				g.fill();
				g.lineWidth = sel && sel.layer === li && sel.node === node ? 3 : 1.2;
				g.strokeStyle = sel && sel.layer === li && sel.node === node ? colors!.accent : colors!.ink2;
				g.stroke();
			});
			if (l.idx.length < l.n) {
				const k = Math.ceil(maxNodes / 2);
				const y = (l.y(k - 1) + l.y(k)) / 2;
				g.fillStyle = colors!.ink2;
				g.font = '12px sans-serif';
				g.textAlign = 'center';
				g.fillText('⋮', l.x, y + 4);
			}
		});
	}
	$effect(() => {
		void d;
		void width;
		void height;
		void theme;
		void sel;
		draw();
	});

	const label = (v: I18n | string | undefined, fallback: string) => (v === undefined ? fallback : typeof v === 'string' ? v : t(v));
	const inputs = $derived((p.props?.inputs as (I18n | string)[] | undefined) ?? []);
	const outputs = $derived((p.props?.outputs as (I18n | string)[] | undefined) ?? []);
	const nodeName = (li: number, node: number) => {
		const L = (d.sizes?.length ?? 0) - 1;
		if (li === 0) return label(inputs[node], d.sizes?.[0] === 2 ? ['x', 'y'][node] : d.sizes?.[0] === 3 ? ['x', 'y', 't'][node] : `x${node}`);
		if (li === L) return label(outputs[node], t({ en: `output ${node}`, zh: `输出 ${node}` }));
		return t({ en: `hidden layer ${li}, node ${node} (counting from 0)`, zh: `隐藏层 ${li} 的节点 ${node}（从 0 数起）` });
	};
	const selInfo = $derived.by(() => {
		if (!sel || !d.sizes) return null;
		const { layer: li, node } = sel;
		const vals = d.values?.[li];
		const l = li > 0 ? d.layers[li - 1] : null;
		let strong: { i: number; v: number } | null = null;
		if (l) {
			const I = l.weight.shape[1];
			for (let i = 0; i < I; i++) {
				const v = l.edge[node * I + i];
				if (!strong || Math.abs(v) > Math.abs(strong.v)) strong = { i, v };
			}
		}
		return { name: nodeName(li, node), value: vals ? vals[node] : NaN, bias: l?.bias ? l.bias.data[node] : NaN, strong: strong ? { from: nodeName(li - 1, strong.i), v: strong.v } : null };
	});
	function key(e: KeyboardEvent) {
		if (!d.sizes) return;
		const cur = sel ?? { layer: 0, node: 0 };
		let { layer, node } = cur;
		if (e.key === 'ArrowRight') layer = Math.min(d.sizes.length - 1, layer + 1);
		else if (e.key === 'ArrowLeft') layer = Math.max(0, layer - 1);
		else if (e.key === 'ArrowDown') node = node + 1;
		else if (e.key === 'ArrowUp') node = node - 1;
		else if (e.key === 'Escape') {
			sel = null;
			return;
		} else return;
		e.preventDefault();
		const n = d.sizes[layer];
		sel = { layer, node: Math.max(0, Math.min(n - 1, node)) };
	}
	function pick(e: PointerEvent) {
		const r = (canvas ?? (e.currentTarget as HTMLElement)).getBoundingClientRect();
		const x = ((e.clientX - r.left) / r.width) * width,
			y = ((e.clientY - r.top) / r.height) * height;
		let best: { layer: number; node: number } | null = null,
			bd = 18 * 18;
		layout.forEach((l, li) =>
			l.idx.forEach((node, k) => {
				const dd = (l.x - x) ** 2 + (l.y(k) - y) ** 2;
				if (dd < bd) {
					bd = dd;
					best = { layer: li, node };
				}
			})
		);
		sel = best;
	}
	const flatSel = $derived(sel && d.sizes ? d.sizes.slice(0, sel.layer).reduce((a, b) => a + b, 0) + sel.node : 0);
	const what = $derived(d.color === 'grad' ? t({ en: 'gradient', zh: '梯度' }) : d.color === 'update' ? t({ en: 'change in this update', zh: '这次更新的变化量' }) : t({ en: 'weight', zh: '权重' }));
	const description = $derived.by(() => {
		void version;
		return t(meta.describe(p));
	});
</script>

<figure class="nw" bind:clientWidth={W} style:min-height="{meta.minHeight}px">
	{#if d.problem}
		<p class="muted">{t(d.problem)}</p>
	{:else}
		<div
			class="pad"
			tabindex="0"
			role="slider"
			aria-roledescription={t({ en: 'network diagram: arrow keys move between nodes', zh: '网络图：用方向键在节点之间移动' })}
			aria-label={description}
			aria-valuemin={0}
			aria-valuemax={Math.max(0, (d.sizes ?? [1]).reduce((a, b) => a + b, 0) - 1)}
			aria-valuenow={flatSel}
			aria-valuetext={selInfo ? selInfo.name : t({ en: 'no node selected', zh: '未选中节点' })}
			onkeydown={key}
			onpointerdown={pick}
			onblur={() => (sel = null)}
		>
			<canvas bind:this={canvas} style:width="{width}px" style:height="{height}px" aria-hidden="true"></canvas>
		</div>
		<figcaption class="small">
			<span class="muted">{t({ en: 'Lines', zh: '连线' })}{colon()}{t({ en: ' ', zh: '' })}{what} ({t({ en: 'blue below zero, orange above; thicker = larger', zh: '小于零为蓝，大于零为橙；越粗越大' })})</span>
			<span class="muted">{d.values ? t({ en: 'Circles: the values for the first example of the batch', zh: '圆圈：这一批中第一个样例的数值' }) : t({ en: 'Circles: each node’s bias', zh: '圆圈：每个节点的偏置' })}</span>
			{#each layout as l, li (li)}{#if l.idx.length < l.n}<span class="muted">{t({ en: `layer ${li}: showing ${l.idx.length} of ${l.n} nodes`, zh: `第 ${li} 层：显示 ${l.n} 个节点中的 ${l.idx.length} 个` })}</span>{/if}{/each}
			{#if selInfo}
				<span class="readout">
					<strong>{selInfo.name}</strong>
					{#if Number.isFinite(selInfo.value)} · {t({ en: 'value', zh: '数值' })} <span class="num">{fmt3(selInfo.value)}</span>{/if}
					{#if Number.isFinite(selInfo.bias)} · {t({ en: 'bias', zh: '偏置' })} <span class="num">{fmt3(selInfo.bias)}</span>{/if}
					{#if selInfo.strong} · {t({ en: `strongest incoming ${what}`, zh: `最强的输入${what}` })}{colon()}{t({ en: ' ', zh: '' })}<span class="num">{fmt3(selInfo.strong.v)}</span> {t({ en: 'from', zh: '来自' })} {selInfo.strong.from}{/if}
				</span>
			{/if}
		</figcaption>
	{/if}
</figure>

<style>
	.nw {
		margin: 0;
		display: grid;
		gap: 6px;
		min-width: 0;
		align-content: start;
	}
	canvas {
		display: block;
		max-width: 100%;
		touch-action: manipulation;
	}
	.pad {
		width: fit-content;
		max-width: 100%;
		border-radius: var(--r1);
	}
	.pad:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		color: var(--ink-2);
	}
</style>
