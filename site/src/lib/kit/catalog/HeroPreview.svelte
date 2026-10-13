<!--
  The home page's answer-first preview (redesign spec §4.5, §6.3; C's graft): "TinyCNN reads a drawing of a house". The
  input, the 8 maps of Conv 1-1, the 8 of Pool 1, the 16 of Pool 2 and the top 5 of the softmax, drawn on canvases from
  build-time data (HeroData: the shipped weights on the lesson's drawing; no model code on the home page) with the site's
  colour scales. Labelled Precomputed.
    <HeroPreview data={data.hero} />
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Icon from '../Icon.svelte';
	import ProvenanceBadge from '../ProvenanceBadge.svelte';
	import { arrow } from './glyphs';
	import { makeScale } from '../color';
	import { themeState } from '../theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { ct } from './text';
	import type { HeroData } from './types';

	let { data }: { data: HeroData } = $props();

	const bytes = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
	type Maps = { n: number; count: number; px: Uint8Array; kind: 'sequential' | 'diverging' };
	const cols = $derived<{ id: string; label: { en: string; zh: string }; shape: string; m: Maps; per: number }[]>([
		{ id: 'input', label: { en: 'Input', zh: '输入' }, shape: '28×28', m: { n: 28, count: 1, px: bytes(data.input), kind: 'sequential' }, per: 1 },
		{ id: 'conv', label: { en: 'Conv 1-1', zh: '卷积 1-1' }, shape: '8×26²', m: { n: 26, count: 8, px: bytes(data.conv), kind: 'diverging' }, per: 2 },
		{ id: 'pool1', label: { en: 'Pool 1', zh: '池化 1' }, shape: '8×12²', m: { n: 12, count: 8, px: bytes(data.pool1), kind: 'diverging' }, per: 2 },
		{ id: 'pool2', label: { en: 'Pool 2', zh: '池化 2' }, shape: '16×4²', m: { n: 4, count: 16, px: bytes(data.pool2), kind: 'diverging' }, per: 4 }
	]);

	/** draw map k of a layer into its canvas: one pixel per cell, scaled up by CSS (pixelated) */
	function tile(canvas: HTMLCanvasElement, a: { m: Maps; k: number }) {
		const draw = () => {
			const { m, k } = a;
			const ctx = canvas.getContext('2d');
			if (!ctx) return;
			const scale = makeScale(m.kind, m.kind === 'sequential' ? [0, 1] : [-1, 1], themeState.resolved);
			const img = ctx.createImageData(m.n, m.n);
			const off = k * m.n * m.n;
			for (let i = 0; i < m.n * m.n; i++) {
				const b = m.px[off + i] / 255;
				scale.write(m.kind === 'sequential' ? b : b * 2 - 1, img.data, i * 4);
			}
			ctx.putImageData(img, 0, 0);
		};
		$effect(() => {
			void themeState.resolved;
			draw();
		});
	}

	const pct = (p: number) => (p > 0.999 ? '>99.9%' : p < 0.001 ? '<0.1%' : `${(p * 100).toFixed(1)}%`);
	const top = $derived(data.top[0]);
</script>

<figure class="pv" aria-labelledby="pv-title">
	<div class="pv-h">
		<div>
			<p class="pv-t" id="pv-title">{t({ en: 'TinyCNN reads a drawing of a house', zh: 'TinyCNN 读一幅房子的画' })}</p>
			<p class="pv-s">{t({ en: `a real network with ${data.params.toLocaleString('en')} learned numbers`, zh: `一个真实的网络，有 ${data.params.toLocaleString('en')} 个学来的数` })}</p>
		</div>
		<span class="prov"><ProvenanceBadge mode="precomputed" note={{ en: 'Computed when the site was built: the shipped TinyCNN weights on the lesson’s drawing of a house.', zh: '在构建网站时算出：发布的 TinyCNN 权重作用在课程里那幅房子的画上。' }} /><span class="pl">· {t({ en: 'a real run', zh: '真实运行' })}</span></span>
	</div>
	<div class="pv-b">
		{#each cols as c (c.id)}
			<div class="col {c.id}">
				<p class="lbl">{t(c.label)}<span class="num">{c.shape}</span></p>
				<div class="tiles" style:--per={c.per}>
					{#each { length: c.m.count } as _, k (k)}
						<canvas width={c.m.n} height={c.m.n} use:tile={{ m: c.m, k }} aria-hidden="true"></canvas>
					{/each}
				</div>
			</div>
		{/each}
		<div class="col out">
			<p class="lbl">{t({ en: 'Softmax, top 5', zh: 'Softmax 前 5' })}</p>
			<dl class="probs">
				{#each data.top as s, i (s.en)}
					<div class:top={i === 0}>
						<dt>{t(s)}</dt>
						<dd><span class="bar" aria-hidden="true"><i style:width="{Math.max(0.6, s.p * 100)}%"></i></span><span class="num">{pct(s.p)}</span></dd>
					</div>
				{/each}
			</dl>
		</div>
		<p class="visually-hidden">
			{t({
				en: `The drawing of a house, 28 by 28 pixels, then the 8 maps of the first convolution, the 8 maps of the first pooling layer and the 16 maps of the second. The network's answer: ${top.en}, ${pct(top.p)}.`,
				zh: `一幅 28×28 像素的房子画，接着是第一个卷积层的 8 张图、第一个池化层的 8 张图和第二个池化层的 16 张图。网络的回答：${top.zh}，${pct(top.p)}。`
			})}
		</p>
	</div>
	<figcaption class="pv-f">
		<span>{t({ en: 'Every tile is computed from this drawing.', zh: '每个小图都由这幅画算出。' })}</span>
		<a class="btn" href="{base}/m/tinycnn/">{ct('followIt')}<Icon d={arrow} size={16} /></a>
	</figcaption>
</figure>

<style>
	.pv {
		margin: 0;
		min-width: 0;
		border: 1px solid var(--rule);
		border-radius: var(--r5);
		background: var(--panel);
		box-shadow: var(--shadow-2);
		overflow: hidden;
	}
	.pv-h {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 12px;
		padding: 16px 18px 12px;
		border-bottom: 1px solid var(--rule);
	}
	.pv-h p {
		margin: 0;
	}
	.pv-t {
		font-size: var(--t-md);
		line-height: var(--lh-md);
		font-weight: var(--w-title);
	}
	.pv-s {
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	.prov {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		flex: none;
	}
	.pl {
		font-size: var(--t-sm);
		color: var(--muted);
	}
	.pv-b {
		position: relative;
		display: grid;
		grid-template-columns: 1.15fr 0.9fr 0.9fr 0.9fr 1.5fr;
		gap: 14px;
		padding: 16px 18px;
		background: var(--stage);
	}
	.col {
		min-width: 0;
	}
	.col + .col {
		padding-left: 12px;
		border-left: 1px dashed var(--rule-strong);
	}
	.lbl {
		display: flex;
		flex-direction: column;
		margin: 0 0 8px;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		font-weight: var(--w-title);
		color: var(--ink-2);
	}
	.lbl .num {
		font-weight: var(--w-body);
		color: var(--muted);
	}
	.tiles {
		display: grid;
		grid-template-columns: repeat(var(--per), minmax(0, 1fr));
		gap: 3px;
	}
	canvas {
		display: block;
		width: 100%;
		aspect-ratio: 1;
		image-rendering: pixelated;
		border-radius: 1px;
		background: var(--well);
	}
	.input canvas {
		outline: 1px solid var(--rule-strong);
	}
	.probs {
		display: flex;
		flex-direction: column;
		gap: 7px;
		margin: 0;
	}
	.probs div {
		display: grid;
		gap: 2px;
	}
	dt {
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--ink-2);
	}
	.top dt {
		font-weight: var(--w-title);
		color: var(--ink);
	}
	dd {
		display: flex;
		align-items: center;
		gap: 8px;
		margin: 0;
	}
	.bar {
		flex: 1;
		height: 6px;
		border-radius: var(--r-pill);
		background: var(--rule);
	}
	.bar i {
		display: block;
		height: 100%;
		border-radius: var(--r-pill);
		background: var(--seq-hi);
	}
	dd .num {
		min-width: 52px;
		text-align: right;
		font-size: var(--t-xs);
		color: var(--muted);
	}
	.top dd .num {
		color: var(--ink);
		font-weight: var(--w-ui);
	}
	.pv-f {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
		padding: 12px 18px;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
	}
	.pv-f .btn {
		min-height: 40px;
		font-size: var(--t-sm);
	}
	@media (max-width: 759px) {
		.pv-h {
			padding: 14px 14px 10px;
		}
		.pl {
			display: none;
		}
		.pv-b {
			grid-template-columns: 1fr 1fr 1.5fr;
			gap: 10px;
			padding: 12px;
		}
		.pool1,
		.pool2 {
			display: none;
		}
		.pv-f {
			flex-direction: column;
			align-items: stretch;
			padding: 12px 14px;
		}
	}
</style>
