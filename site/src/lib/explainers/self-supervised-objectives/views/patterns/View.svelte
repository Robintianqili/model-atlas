<!-- self-supervised-objectives 'patterns' (meta.ts): each objective as a mask pattern over the learner's tokens or the
     drawing's 4 x 4 patches. Tokens: plain = seen, dashed and muted = hidden, accent outline = to predict, italic = a
     marker ([CLS], a sentinel, a label). Patches: hatched = hidden; an outlined patch is one to predict. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import type { Picture } from '../../objectives.ts';
	import { GRID, noisy, pixels } from '../../patterns.ts';
	import { DRAWING } from '../../source_data.ts';
	import { patternsOf } from './meta.ts';

	let P: StageProps = $props();
	const D = $derived(patternsOf(P));
	let chosen = $state('clm');
	const list = $derived(D.pick ? D.all.filter((o) => o.id === chosen) : D.list);
	const PX = pixels();
	const NOISE = noisy(0.5);

	/** Draw one picture of the drawing: ink on the surface colour, the patch grid, hidden patches hatched (redrawn when the
	 *  theme changes: the theme is part of the action's argument). */
	function paint(cv: HTMLCanvasElement, arg: { pic: Picture; theme: string }) {
		const draw = (p: Picture) => {
			const css = getComputedStyle(cv);
			const col = (v: string) => css.getPropertyValue(v).trim();
			const S = 5;
			const dpr = window.devicePixelRatio || 1;
			const crop = p.crop;
			const side = crop ? crop.size : GRID.side;
			const W = side * S;
			cv.width = W * dpr;
			cv.height = W * dpr;
			cv.style.width = `${W}px`;
			cv.style.height = `${W}px`;
			const g = cv.getContext('2d');
			if (!g) return;
			g.setTransform(dpr, 0, 0, dpr, 0, 0);
			g.fillStyle = col('--surface');
			g.fillRect(0, 0, W, W);
			g.fillStyle = col('--ink');
			const src = p.mode === 'noise' ? NOISE : PX;
			for (let y = 0; y < side; y++)
				for (let x = 0; x < side; x++) {
					const sx = crop ? crop.x + (crop.flip ? side - 1 - x : x) : x;
					const sy = crop ? crop.y + y : y;
					g.globalAlpha = Math.min(1, Math.max(0, src[sy * GRID.side + sx]));
					g.fillRect(x * S, y * S, S, S);
				}
			g.globalAlpha = 1;
			if (crop) return;
			const pw = GRID.patch * S;
			for (let i = 0; i < GRID.n * GRID.n; i++) {
				const x = (i % GRID.n) * pw;
				const y = Math.floor(i / GRID.n) * pw;
				if (p.hidden?.[i]) {
					g.fillStyle = col('--surface-3');
					g.fillRect(x, y, pw, pw);
					g.strokeStyle = col('--muted');
					g.lineWidth = 1;
					g.beginPath();
					for (let d = -pw; d < pw; d += 7) (g.moveTo(x + Math.max(0, d), y + Math.max(0, -d)), g.lineTo(x + Math.min(pw, d + pw), y + Math.min(pw, pw - d)));
					g.stroke();
				} else if (p.target) {
					g.strokeStyle = col('--accent');
					g.lineWidth = 2;
					g.strokeRect(x + 1, y + 1, pw - 2, pw - 2);
				}
			}
			g.strokeStyle = col('--rule-strong');
			g.lineWidth = 1;
			for (let i = 1; i < GRID.n; i++) {
				g.beginPath();
				g.moveTo(i * pw + 0.5, 0);
				g.lineTo(i * pw + 0.5, W);
				g.moveTo(0, i * pw + 0.5);
				g.lineTo(W, i * pw + 0.5);
				g.stroke();
			}
		};
		draw(arg.pic);
		return { update: (a: { pic: Picture; theme: string }) => draw(a.pic) };
	}
	const altOf = (p: Picture) => {
		const k = p.hidden?.filter(Boolean).length ?? 0;
		const what = t({ en: `a drawing of a ${DRAWING.en}`, zh: `一幅${DRAWING.zh}的简笔画` });
		if (p.mode === 'crop') return t({ en: `Part of ${what}${p.crop?.flip ? ', mirrored' : ''}.`, zh: `${what}的一部分${p.crop?.flip ? '，左右翻转' : ''}。` });
		if (p.mode === 'noise') return t({ en: `${what} with noise on every pixel.`, zh: `每个像素都加了噪声的${what}。` });
		return t({ en: `${what}, cut into 16 patches; ${k} are hatched (hidden)${p.target ? ', the others are outlined: these are the target' : ''}.`, zh: `${what}，切成 16 个图块；${k} 个打了斜线（藏起）${p.target ? '，其余加框的是目标' : ''}。` });
	};
</script>

<div class="pt">
	{#if D.pick}
		<div class="pick" role="group" aria-label={t({ en: 'Objectives', zh: '训练目标' })}>
			{#each D.all as o (o.id)}<button type="button" class="btn" aria-pressed={o.id === chosen} onclick={() => (chosen = o.id)}>{t(o.name)}</button>{/each}
		</div>
	{/if}
	{#each list as o (o.id)}
		<section class="ob">
			<h4>{t(o.name)} <span class="ex">{t(o.example)}</span> <span class="badge" class:live={o.live}>{t(o.live ? { en: 'live model', zh: '实时模型' } : { en: 'illustrative', zh: '示意' })}</span></h4>
			{#if !D.brief}<p class="small"><b>{t({ en: 'Sees', zh: '看到' })}:</b> {t(o.sees)}. <b>{t({ en: 'Predicts', zh: '预测' })}:</b> {t(o.predicts)}.</p>{/if}
			{#each o.lines as l, li (li)}
				<div class="line">
					<span class="lab">{t(l.label)}</span>
					<span class="toks">{#each l.chips as c, ci (ci)}<span class="tk {c.role}">{c.text}</span>{/each}</span>
				</div>
			{/each}
			{#if o.pictures.length}
				<div class="pics">
					{#each o.pictures as pic, pi (pi)}
						<figure><span class="cv" role="img" aria-label={altOf(pic)}><canvas use:paint={{ pic, theme: themeState.resolved }} aria-hidden="true"></canvas></span><figcaption>{t(pic.label)}</figcaption></figure>
					{/each}
				</div>
			{/if}
			{#if o.href}<a class="small" href={`${base}${o.href}`}>{t({ en: 'See it on a live model', zh: '在实时模型上看看' })} →</a>{/if}
		</section>
	{/each}
	<p class="legend small">
		<span class="tk seen">{t({ en: 'seen', zh: '看得见' })}</span>
		<span class="tk gone">{t({ en: 'hidden', zh: '藏起' })}</span>
		<span class="tk target">{t({ en: 'to predict', zh: '要预测' })}</span>
		<span class="tk mark">{t({ en: 'marker', zh: '标记' })}</span>
		{t({ en: '␣ is a space.', zh: '␣ 表示空格。' })}{#if list.some((o) => o.pictures.length)}{t({ en: ' Hatched patches are hidden.', zh: '打斜线的图块被藏起。' })}{/if}
	</p>
</div>

<style>
	.pt { display: grid; gap: 12px; min-width: 0; }
	.pick { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.ob { display: grid; gap: 6px; min-width: 0; padding: 8px 10px; border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); }
	h4 { margin: 0; font-size: var(--fs-sm); }
	.ex { font-weight: normal; color: var(--ink-2); }
	.badge { font-weight: normal; font-size: var(--fs-xs); padding: 0 6px; border-radius: 999px; background: var(--illus-soft); color: var(--ink); }
	.badge.live { background: var(--live-soft); }
	.line { display: grid; grid-template-columns: minmax(0, 1fr); gap: 2px; min-width: 0; }
	.lab { font-size: var(--fs-xs); color: var(--ink-2); }
	.toks { display: flex; flex-wrap: wrap; gap: 3px; min-width: 0; }
	.tk { display: inline-block; padding: 1px 5px; border-radius: 4px; border: 1px solid var(--rule-strong); font-family: var(--font-mono); font-size: var(--fs-xs); white-space: pre; color: var(--ink); background: var(--surface-2); }
	.tk.gone { border-style: dashed; color: var(--muted); background: var(--surface); }
	.tk.target { border: 2px solid var(--accent); background: var(--accent-soft); }
	.tk.mark { font-style: italic; border-style: dashed; background: var(--surface); }
	.pics { display: flex; flex-wrap: wrap; gap: 12px; }
	figure { margin: 0; display: grid; gap: 2px; justify-items: start; }
	.cv { display: block; line-height: 0; }
	figcaption { font-size: var(--fs-xs); color: var(--ink-2); }
	.small { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
	.legend { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
</style>
