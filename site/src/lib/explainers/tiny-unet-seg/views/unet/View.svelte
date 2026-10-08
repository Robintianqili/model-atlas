<!-- tiny-unet-seg/unet (see meta.ts): the U with one live map per level, the arrows between levels (the skip connections
     across), and the maps of one level as tiles below. Tap a level to open its maps. Nothing animates. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import TileStack from '$lib/kit/TileStack.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import Seg from '../../parts/Seg.svelte';
	import { T, classMap } from '../../seg.ts';
	import { LEVELS, NAMES, openOf } from './meta.ts';

	let P: StageProps = $props();
	const part = $derived(String(P.props.part ?? 'all'));
	let picked = $state<string | null>(null);
	const open = $derived(picked ?? openOf(P));
	const ot = $derived(T(P.rt, open) ?? T(P.rt, openOf(P)));
	const cm = $derived(classMap(P.rt));
	const img = $derived(T(P.rt, 'input')?.data);
	const lit = (id: string) => part === 'all' || LEVELS.find((l) => l.id === id)?.part === part || (part === 'skip' && id !== 'bott');
	/** the strongest map of a level (largest sum of |values|) */
	function strongest(id: string) {
		const x = T(P.rt, id);
		if (!x) return null;
		const [C, H, W] = x.shape;
		let b = 0, bs = -1;
		for (let c = 0; c < C; c++) {
			let s = 0;
			for (let i = 0; i < H * W; i++) s += Math.abs(x.data[c * H * W + i]);
			if (s > bs) (bs = s), (b = c);
		}
		return { data: x.data.subarray(b * H * W, (b + 1) * H * W), H, W, C, ch: b };
	}
	const sees = (id: string) => {
		const H = T(P.rt, id)?.shape[1] ?? 0, m = P.rt.sources.fwd?.model ?? P.rt.model;
		const b = m?.receptiveField(id, { y0: H / 2, x0: H / 2, y1: H / 2, x1: H / 2 });
		return b ? b.x1 - b.x0 + 1 : null;
	};
	const nm = $derived(NAMES[open] ?? { en: open, zh: open });
	const skipC = $derived(open.startsWith('cat') ? (T(P.rt, open === 'cat1' ? 'enc1' : 'enc2')?.shape[0] ?? 0) : 0);
	const tileLabels = $derived(ot && skipC ? Array.from({ length: ot.shape[0] }, (_, c) => (c < skipC ? `⇢${c}` : `↑${c - skipC}`)) : undefined);
	$effect(() => {
		if (P.autoplay && ot) P.onend();
	});
</script>

{#snippet level(id: string, size: number)}
	{@const m = strongest(id)}
	{@const L = LEVELS.find((l) => l.id === id)!}
	{@const r = L.part === 'down' || id === 'bott' ? sees(id) : null}
	<button type="button" class="node" class:lit={lit(id)} aria-pressed={open === id} onclick={() => (picked = id)}>
		{#if m}<span class="thumb" style:max-width="{size}px"><TensorGrid data={m.data} rows={m.H} cols={m.W} interactive={false} readout={false} gridLines={false} labels={false} minCell={1} maxCell={8} label={t({ en: `${L.en}: map ${m.ch}`, zh: `${L.zh}：第 ${m.ch} 张图` })} /></span>{/if}
		<span class="lab">{t(L)}</span>
		{#if m}<span class="num small">{m.C} × {m.H}×{m.W}</span>{/if}
		{#if r}<span class="muted small">{t({ en: `one cell sees ${r}×${r}`, zh: `一个格子看到 ${r}×${r}` })}</span>{/if}
	</button>
{/snippet}

{#if !T(P.rt, 'enc1') || !img}
	<Notice />
{:else}
	<div class="u">
		<div class="grid">
			<div class="c1">{@render level('enc1', 96)}</div>
			<div class="ar across" class:on={part === 'skip' || part === 'all'}>{t({ en: 'skip 1: copy', zh: '跳跃 1：复制' })} →</div>
			<div class="c3">{@render level('dec1', 96)}</div>
			<div class="out"><span class="ar">→</span>{#if cm}<Seg {img} cls={cm.k} size={72} label={t({ en: 'The class map', zh: '类别图' })} />{/if}<span class="small">{t({ en: 'classes', zh: '类别' })}</span></div>
			<div class="c1 ar" class:on={part === 'down' || part === 'all'}>↓ {t({ en: 'max pool', zh: '最大池化' })}</div>
			<div class="c3 ar" class:on={part === 'up' || part === 'all'}>↑ {t({ en: 'upsample', zh: '上采样' })}</div>
			<div class="c1">{@render level('enc2', 72)}</div>
			<div class="ar across" class:on={part === 'skip' || part === 'all'}>{t({ en: 'skip 2: copy', zh: '跳跃 2：复制' })} →</div>
			<div class="c3">{@render level('dec2', 72)}</div>
			<div class="c1 ar" class:on={part === 'down' || part === 'all'}>↘ {t({ en: 'max pool', zh: '最大池化' })}</div>
			<div class="c3 ar" class:on={part === 'up' || part === 'all'}>↗ {t({ en: 'upsample', zh: '上采样' })}</div>
			<div class="bott">{@render level('bott', 56)}</div>
		</div>
		{#if ot}
			<div class="tiles" use:keyScroll>
				<p class="small">
					{#if skipC}{t({ en: `${nm.en}: the ${skipC} maps marked ⇢ are copied across the skip connection; the ${ot.shape[0] - skipC} marked ↑ came up from below.`, zh: `${nm.zh}：标 ⇢ 的 ${skipC} 张图是经跳跃连接复制过来的；标 ↑ 的 ${ot.shape[0] - skipC} 张是从下面上来的。` })}
					{:else}{t({ en: `All ${ot.shape[0]} maps of ${nm.en} (${ot.shape[1]}×${ot.shape[2]} each).`, zh: `${nm.zh} 的全部 ${ot.shape[0]} 张图（每张 ${ot.shape[1]}×${ot.shape[2]}）。` })}{/if}
				</p>
				<TileStack scope="unit" data={ot.data} channels={ot.shape[0]} height={ot.shape[1]} width={ot.shape[2]} tile={ot.shape[1] > 8 ? 64 : 48} channelLabels={tileLabels} label={t({ en: `Maps of ${nm.en}`, zh: `${nm.zh} 的图` })} />
			</div>
		{/if}
	</div>
{/if}

<style>
	.u { display: grid; gap: 12px; min-width: 0; }
	.grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 0.7fr) minmax(0, 1fr) auto; gap: 4px 6px; align-items: center; justify-items: center; }
	.c1 { grid-column: 1; } .c3 { grid-column: 3; }
	.bott { grid-column: 1 / 4; }
	.out { display: grid; justify-items: center; gap: 2px; width: 76px; }
	/* a level outside the step's part: its map faded, its text --muted with a dashed border (no opacity on text: AK604) */
	.node { display: grid; justify-items: center; gap: 2px; background: none; border: 1px dashed var(--rule-strong); border-radius: 6px; padding: 4px; color: var(--muted); cursor: pointer; width: 100%; min-width: 0; }
	.node .thumb { opacity: 0.45; }
	.node.lit { border: 1px solid var(--rule); color: var(--ink); }
	.node.lit .thumb { opacity: 1; }
	.node[aria-pressed='true'] { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
	.thumb { display: block; width: 100%; }
	.lab { font-size: var(--fs-sm); font-weight: 600; }
	.small { font-size: var(--fs-xs); }
	.ar { font-size: var(--fs-xs); color: var(--muted); text-align: center; }
	.ar.on { color: var(--accent); font-weight: 700; }
	.across { grid-column: 2; }
	.tiles { min-width: 0; overflow-x: auto; }
	.tiles p { margin: 0 0 6px; }
</style>
