<!-- vq-vae/grid (see meta.ts): the drawing, its G × G code indices, the redrawing; a patch picker and a code swap. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import Num from '../../../../atlas/ui/views/kit.parts/Num.svelte';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import Pic from '../../parts/Pic.svelte';
	import { T, blankOf, cellOf, codesOf, distOf, extraOf, rankOf, swapOf, tilesOf, withExtra } from '../../vq.ts';

	let P: StageProps = $props();
	const mode = $derived(String(P.props.mode ?? 'redraw'));
	const x = $derived(T(P.rt, 'input')?.data);
	const y = $derived(T(P.rt, 'out')?.data);
	const codes = $derived(codesOf(P.rt));
	const G = $derived(Math.round(Math.sqrt(codes.length)) || 4);
	const cell = $derived(cellOf(P.rt));
	const model = $derived(P.rt.model);
	const blank = $derived(model ? blankOf(model) : -1);
	const tiles = $derived(model && mode === 'swap' ? tilesOf(model) : []);
	const sw = $derived(mode === 'swap' ? swapOf(P.rt) : null);
	/** the default swap target: the first blank patch next to an inked one (else the first blank one, else 0) */
	const firstBlank = $derived.by(() => {
		const near = (c: number) => [c - G, c + G, c % G ? c - 1 : -1, (c + 1) % G ? c + 1 : -1].some((n) => n >= 0 && n < codes.length && codes[n] !== blank);
		return codes.findIndex((k, c) => k === blank && near(c));
	});
	let picked = $state<number | null>(null);
	const swapCell = $derived(picked ?? sw?.cell ?? (firstBlank >= 0 ? firstBlank : Math.max(0, codes.indexOf(blank))));
	const shown = $derived(mode === 'swap' ? swapCell : cell);
	const err = $derived.by(() => {
		const s = T(P.rt, 'img')?.data;
		if (mode !== 'limits' || !s || !x) return null;
		const e = Float32Array.from(s, (v, i) => Math.max(v, 0) + Math.log1p(Math.exp(-Math.abs(v))) - v * x[i]);
		return { e, max: Math.max(...e) };
	});
	const change = $derived(sw ? Float32Array.from(sw.after, (v, i) => v - sw.before[i]) : null);
	const where = (c: number) => t({ en: `row ${Math.floor(c / G)}, column ${c % G}`, zh: `第 ${Math.floor(c / G)} 行第 ${c % G} 列` });
	/** pixels above half ink (step 'decode' narrates the same counts); the outlined patch's two nearest codes (step 'indices') */
	const inkN = (a: ArrayLike<number> | undefined) => (a ? Array.from(a).filter((v) => v > 0.5).length : 0);
	const near = $derived(mode === 'codes' && codes.length ? rankOf(distOf(P.rt, cell)) : []);

	function pickCell(c: number) {
		if (mode === 'swap') picked = c;
		else P.setInput?.(withExtra(P.rt.input, { cell: c }));
	}
	function setCode(k: number) {
		P.setInput?.(withExtra(P.rt.input, { swap: { cell: swapCell, code: k } }));
	}
	function undo() {
		const { swap: _s, ...rest } = withExtra(P.rt.input, {}) as Record<string, unknown>;
		P.setInput?.(rest);
	}
	$effect(() => {
		if (P.autoplay && x) P.onend();
	});
</script>

{#snippet cells(labels: boolean, pick: boolean)}
	<div class="cells" role="group" aria-label={t({ en: 'The patches', zh: '图块' })} style:grid-template-columns="repeat({G}, 1fr)">
		{#each codes as k, c (c)}
			<svelte:element this={pick ? 'button' : 'span'} type={pick ? 'button' : undefined} role={pick ? undefined : 'presentation'} class="cell" class:on={c === shown} aria-pressed={pick ? c === shown : undefined} aria-label={pick ? t({ en: `Patch ${where(c)}, code ${k}`, zh: `图块 ${where(c)}，码字 ${k}` }) : undefined} onclick={pick ? () => pickCell(c) : undefined}>{#if labels}<span class="chip">{k}</span>{/if}</svelte:element>
		{/each}
	</div>
{/snippet}

{#if !x || !y || !codes.length}
	<Notice text={{ en: 'Draw something first: the codes come from your drawing.', zh: '先画点什么：码字来自你的画。' }} />
{:else}
	<div class="grid-view">
		<div class="row">
			<figure>
				<Pic data={x} size={196} grid={G} frame="real" label={t({ en: 'Your drawing in patches', zh: '切成图块的你的画' })}>
					{@render cells(mode === 'codes' || mode === 'swap', mode === 'patches' || mode === 'codes' || mode === 'swap')}
				</Pic>
				<figcaption>{t(mode === 'patches' ? { en: `your drawing: ${G}×${G} patches, pick one to outline it`, zh: `你的画：${G}×${G} 个图块，点一个把它框出来` } : mode === 'redraw' ? { en: `your drawing: ${inkN(x)} pixels above half ink`, zh: `你的画：墨量超过一半的像素 ${inkN(x)} 个` } : { en: 'your drawing and the code of each patch', zh: '你的画和每个图块的码字' })}</figcaption>
			</figure>
			{#if mode === 'codes' || mode === 'redraw' || mode === 'limits'}
				<table class="idx" aria-label={t({ en: 'Code indices, row by row', zh: '码字编号，逐行' })}>
					<tbody>
						{#each Array.from({ length: G }, (_, r) => r) as r (r)}
							<tr>{#each codes.slice(r * G, r * G + G) as k, j (j)}<td class:on={r * G + j === cell} class:blank={k === blank}>{k}</td>{/each}</tr>
						{/each}
					</tbody>
				</table>
			{/if}
			{#if mode !== 'patches' && mode !== 'codes'}
				<figure>
					<Pic data={sw ? sw.after : y} size={196} grid={mode === 'limits' || mode === 'swap' ? G : 0} frame="generated" label={t({ en: 'The redrawing', zh: '重画' })} />
					<figcaption>{t(sw ? { en: 'the redrawing after your swap', zh: '换码字后的重画' } : mode === 'redraw' ? { en: `the redrawing from the codes: ${inkN(y)} pixels above half ink`, zh: `从码字重画出的画：墨量超过一半的像素 ${inkN(y)} 个` } : { en: 'the redrawing from the codes', zh: '从码字重画出的画' })}</figcaption>
				</figure>
			{/if}
			{#if sw && change}
				<figure>
					<Pic data={change} size={196} kind="change" grid={G} label={t({ en: 'What the swap changed', zh: '替换改变了什么' })} />
					<figcaption>{t({ en: 'change: orange more ink, blue less', zh: '变化：橙色墨多了，蓝色墨少了' })}</figcaption>
				</figure>
			{/if}
			{#if err}
				<figure>
					<Pic data={err.e} size={196} kind="heat" max={err.max} grid={G} label={t({ en: 'Redrawing error of each pixel', zh: '每个像素的重画误差' })} />
					<figcaption>{t({ en: 'error of each pixel (stronger colour: larger)', zh: '每个像素的误差（颜色越浓越大）' })}</figcaption>
				</figure>
			{/if}
		</div>
		{#if near.length > 1}
			<p class="small">{t({ en: `Outlined patch: code ${near[0].k}, squared distance`, zh: `框出的图块：码字 ${near[0].k}，距离平方` })} <Num value={near[0].d} />{t({ en: `; runner-up: code ${near[1].k},`, zh: `；第二近：码字 ${near[1].k}，距离平方` })} <Num value={near[1].d} />.</p>
		{/if}
		{#if mode === 'swap'}
			<p class="small">
				{t({ en: `Patch ${where(swapCell)} (counting from 0) has code ${codes[swapCell]}. Pick a code to put there:`, zh: `图块 ${where(swapCell)}（从 0 数起）的码字是 ${codes[swapCell]}。挑一个码字放进去：` })}
				{#if sw}<button type="button" class="undo" onclick={undo}>{t({ en: 'Undo the swap', zh: '撤销替换' })}</button>{/if}
			</p>
			<div class="palette" role="group" aria-label={t({ en: 'The codebook', zh: '码本' })} use:keyScroll>
				{#each tiles as tile, k (k)}
					<button type="button" class="code" class:on={sw?.to === k} aria-pressed={sw?.to === k} onclick={() => setCode(k)}>
						<Pic data={tile} size={36} label={t({ en: `Code ${k}`, zh: `码字 ${k}` })} /><span>{k}</span>
					</button>
				{/each}
			</div>
			{#if sw}
				<p class="small">{t({ en: 'Pixels above half ink in that block: from', zh: '这一块里墨量超过一半的像素：从' })} <Num value={sw.ink0} /> {t({ en: 'to', zh: '个变成' })} <Num value={sw.ink1} />{t({ en: '. Ink change added up over the pixels: inside the block', zh: ' 个。各像素墨量变化之和：块内' })} <Num value={sw.inside} />, {t({ en: 'outside it', zh: '块外' })} <Num value={sw.outside} />.</p>
			{/if}
		{/if}
	</div>
{/if}

<style>
	.grid-view { display: grid; gap: 10px; min-width: 0; }
	.row { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; }
	figure { margin: 0; display: grid; gap: 4px; justify-items: center; }
	figcaption, .small { font-size: var(--fs-sm); }
	figcaption { color: var(--muted); max-width: 200px; text-align: center; }
	p { margin: 0; }
	.cells { position: absolute; inset: 2px; display: grid; }
	.cell { all: unset; box-sizing: border-box; display: grid; place-items: start; cursor: default; }
	button.cell { cursor: pointer; }
	button.cell:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
	.cell.on { box-shadow: inset 0 0 0 3px var(--accent); }
	.chip { font-size: 11px; line-height: 1; padding: 2px 3px; margin: 2px; border-radius: 3px; background: var(--surface); color: var(--ink); }
	.idx { border-collapse: collapse; font-variant-numeric: tabular-nums; }
	.idx td { border: 1px solid var(--rule); padding: 4px 7px; text-align: center; }
	.idx td.blank { color: var(--muted); border-style: dashed; }
	.idx td.on { box-shadow: inset 0 0 0 2px var(--accent); font-weight: 700; }
	.palette { display: grid; grid-template-columns: repeat(auto-fill, minmax(46px, 1fr)); gap: 4px; max-height: 210px; overflow: auto; min-width: 0; }
	.code { display: grid; justify-items: center; gap: 1px; padding: 3px; border: 1px solid var(--rule); border-radius: 4px; background: var(--surface); color: var(--ink); font: inherit; font-size: 11px; cursor: pointer; }
	.code.on { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
	.undo { margin-left: 8px; padding: 2px 8px; border: 1px solid var(--rule); border-radius: 4px; background: var(--surface); color: var(--ink); font: inherit; cursor: pointer; }
</style>
