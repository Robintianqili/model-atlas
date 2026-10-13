<!--
  'regularization/field' (meta.ts has the props): the network's answer over the plane (a canvas: each class's color,
  stronger where the network is surer, as plane2d draws it), its border where the share is one half (marching squares,
  SVG), the training points (filled), the test points (rings) and a second ring around each test point it gets wrong.
  Below: the frame's numbers and the key.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtInt, fmtPct } from '$lib/kit/format';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { BOUNDS, RES_X, RES_Y, fieldOf } from '../../net.ts';
	import { CLASS_NAMES, fieldPic } from './meta.ts';

	let { at, props }: StageProps = $props();
	const pic = $derived(fieldPic({ at }));
	const theme = $derived(themeState.resolved);
	const showField = $derived(props?.field !== false);
	const field = $derived(pic?.params && showField ? fieldOf(pic.params) : null);

	const W = 480;
	const H = Math.round((W * (BOUNDS.y1 - BOUNDS.y0)) / (BOUNDS.x1 - BOUNDS.x0));
	const sx = (x: number) => ((x - BOUNDS.x0) / (BOUNDS.x1 - BOUNDS.x0)) * W;
	const sy = (y: number) => ((BOUNDS.y1 - y) / (BOUNDS.y1 - BOUNDS.y0)) * H;
	const inside = (x: number, y: number) => x >= BOUNDS.x0 && x <= BOUNDS.x1 && y >= BOUNDS.y0 && y <= BOUNDS.y1;

	const rgbOf = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
	let canvas = $state<HTMLCanvasElement | null>(null);
	$effect(() => {
		const c = canvas;
		const f = field;
		const g = c?.getContext('2d');
		if (!c || !g) return;
		const surface = getComputedStyle(c).getPropertyValue('--surface').trim();
		const base = /^#[0-9a-f]{6}$/i.test(surface) ? rgbOf(surface) : theme === 'dark' ? [26, 29, 33] : [255, 253, 248];
		g.fillStyle = `rgb(${base.join(',')})`;
		g.fillRect(0, 0, W, H);
		if (!f) return;
		const c0 = rgbOf(categorical(0, theme));
		const c1 = rgbOf(categorical(1, theme));
		const img = new ImageData(RES_X, RES_Y);
		for (let i = 0; i < f.length; i++) {
			const u = 0.12 + 0.43 * Math.min(1, Math.abs(f[i] - 0.5) * 2);
			const to = f[i] >= 0.5 ? c1 : c0;
			for (let ch = 0; ch < 3; ch++) img.data[i * 4 + ch] = Math.round(base[ch] + (to[ch] - base[ch]) * u);
			img.data[i * 4 + 3] = 255;
		}
		const off = document.createElement('canvas');
		off.width = RES_X;
		off.height = RES_Y;
		off.getContext('2d')!.putImageData(img, 0, 0);
		g.imageSmoothingEnabled = true;
		g.drawImage(off, 0, 0, W, H);
	});

	/** the border P(class B) = 0.5: marching squares between cell centres, linear along the edges */
	const border = $derived.by(() => {
		const f = field;
		if (!f) return '';
		const cw = W / RES_X;
		const ch = H / RES_Y;
		const v = (r: number, c: number) => f[r * RES_X + c] - 0.5;
		let d = '';
		for (let r = 0; r < RES_Y - 1; r++)
			for (let c = 0; c < RES_X - 1; c++) {
				const k: [number, number, number][] = [
					[c, r, v(r, c)],
					[c + 1, r, v(r, c + 1)],
					[c + 1, r + 1, v(r + 1, c + 1)],
					[c, r + 1, v(r + 1, c)]
				];
				const pts: string[] = [];
				for (let e = 0; e < 4; e++) {
					const [x0, y0, a] = k[e];
					const [x1, y1, b] = k[(e + 1) % 4];
					if (a < 0 !== b < 0) {
						const s = a / (a - b);
						pts.push(`${((x0 + s * (x1 - x0) + 0.5) * cw).toFixed(1)},${((y0 + s * (y1 - y0) + 0.5) * ch).toFixed(1)}`);
					}
				}
				for (let q = 0; q + 1 < pts.length; q += 2) d += `M${pts[q]}L${pts[q + 1]}`;
			}
		return d;
	});

	const L = {
		plot: { en: 'Training points, test points and the network’s border', zh: '训练点、测试点和网络的分界线' },
		trainDot: { en: 'training point', zh: '训练点' },
		testDot: { en: 'test point', zh: '测试点' },
		wrong: { en: 'test point it gets wrong', zh: '被分错的测试点' }
	};
	const nWrong = $derived(pic ? pic.wrong.filter(Boolean).length : 0);
</script>

{#if pic}
	<figure class="field">
		<div class="plot" style:aspect-ratio="{W} / {H}">
			<canvas bind:this={canvas} width={W} height={H} aria-hidden="true"></canvas>
			<svg viewBox="0 0 {W} {H}" role="img" aria-label={t(L.plot)}>
				<line class="axis" x1={sx(0)} x2={sx(0)} y1="0" y2={H} />
				<line class="axis" x1="0" x2={W} y1={sy(0)} y2={sy(0)} />
				{#if border}<path class="border" d={border} />{/if}
				{#each { length: pic.test.n } as _, i (i)}
					{@const x = pic.test.xy[2 * i]}
					{@const y = pic.test.xy[2 * i + 1]}
					{#if inside(x, y)}
						<circle class="test" cx={sx(x)} cy={sy(y)} r="4.2" style:stroke={categorical(pic.test.label[i], theme)} />
						{#if showField && pic.wrong[i]}<circle class="wrong" cx={sx(x)} cy={sy(y)} r="7.6" />{/if}
					{/if}
				{/each}
				{#each { length: pic.train.n } as _, i (i)}
					{@const x = pic.train.xy[2 * i]}
					{@const y = pic.train.xy[2 * i + 1]}
					{#if inside(x, y)}<circle class="train" cx={sx(x)} cy={sy(y)} r="4.6" style:fill={categorical(pic.train.label[i], theme)} />{/if}
				{/each}
			</svg>
		</div>
		<p class="nums">
			{t({ en: 'after', zh: '更新' })} <strong class="num">{fmtInt(pic.m.step)}</strong> {t({ en: 'updates', zh: '次之后' })} ·
			{t({ en: 'training loss', zh: '训练损失' })} <strong class="num">{fmt3(pic.m.trainLoss)}</strong> ·
			{t({ en: 'test loss', zh: '测试损失' })} <strong class="num">{fmt3(pic.m.testLoss)}</strong> ·
			{t({ en: 'accuracy', zh: '准确率' })} <span class="num">{fmtPct(pic.m.trainAcc)}</span> / <span class="num">{fmtPct(pic.m.testAcc)}</span>
			<span class="muted">{t({ en: '(training / test)', zh: '（训练 / 测试）' })}</span>
		</p>
		<figcaption>
			<span class="key"><i class="dot"></i>{t(L.trainDot)}</span>
			<span class="key"><i class="ring"></i>{t(L.testDot)}</span>
			{#if showField}<span class="key"><i class="ring2"></i>{t(L.wrong)} <span class="num">({fmtInt(nWrong)})</span></span>{/if}
			{#each CLASS_NAMES as c, k (k)}<span class="key"><i class="swatch" style:background={categorical(k, theme)}></i>{t(c)}</span>{/each}
		</figcaption>
	</figure>
{/if}

<style>
	.field {
		margin: 0;
		display: grid;
		gap: 6px;
		min-width: 0;
		max-width: 560px;
	}
	.plot {
		position: relative;
		width: 100%;
		border: 1px solid var(--rule);
		border-radius: 6px;
		overflow: hidden;
	}
	canvas,
	svg {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		display: block;
	}
	.axis {
		stroke: var(--rule-strong);
		stroke-width: 1;
		stroke-dasharray: 2 3;
	}
	.border {
		fill: none;
		stroke: var(--ink-2);
		stroke-width: 1.8;
	}
	.train {
		stroke: var(--surface);
		stroke-width: 1.2;
	}
	.test {
		fill: var(--surface);
		stroke-width: 2;
	}
	.wrong {
		fill: none;
		stroke: var(--highlight);
		stroke-width: 2.2;
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 12px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.key {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}
	.key i {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 50%;
		box-sizing: border-box;
	}
	.dot {
		background: var(--ink-2);
	}
	.ring {
		border: 2px solid var(--ink-2);
	}
	.key i.ring2 {
		border: 2px solid var(--highlight);
		width: 13px;
		height: 13px;
	}
	.key i.swatch {
		border-radius: 2px;
	}
	.nums {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.muted {
		color: var(--muted);
	}
</style>
