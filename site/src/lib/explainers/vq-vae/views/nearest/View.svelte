<!-- vq-vae/nearest (see meta.ts): one patch, its numbers z_e, its distance to every code, and (mode 'train') the
     gradient the straight-through estimator copies from z_q to z_e. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import Num from '../../../../atlas/ui/views/kit.parts/Num.svelte';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import Pic from '../../parts/Pic.svelte';
	import { T, lossOf, tilesOf } from '../../vq.ts';
	import { nearestOf } from './meta.ts';

	let P: StageProps = $props();
	const m = $derived(nearestOf(P));
	const train = $derived(P.props.mode === 'train');
	const tiles = $derived(P.rt.model ? tilesOf(P.rt.model) : []);
	/** the outlined patch's block of a 28 × 28 picture: the drawing, or (snap) the redrawing, where the code sits among the drawing's other codes */
	const blockOf = (x: ArrayLike<number> | undefined) => {
		if (!x || !m) return null;
		const W = Math.round(Math.sqrt(x.length)), S = W / m.G;
		return Float32Array.from({ length: S * S }, (_, i) => x[(m.row * S + Math.floor(i / S)) * W + m.col * S + (i % S)]);
	};
	const patch = $derived(blockOf(T(P.rt, 'input')?.data));
	const inPlace = $derived(train ? null : blockOf(T(P.rt, 'out')?.data));
	const best = $derived(m?.rank[0]);
	const max = $derived(m ? Math.max(...m.rank.map((r) => r.d)) : 1);
	const sc = $derived(makeScale('sequential', [0, max || 1], themeState.resolved));
	const terms = $derived(m && best ? m.ze.map((z, d) => ({ d, z, e: m.zq[d], sq: (z - m.zq[d]) ** 2 })) : []);
	const L = $derived(train ? lossOf(P.rt) : null);
	$effect(() => {
		if (P.autoplay && m) P.onend();
	});
</script>

{#if !m || !best}
	<Notice text={{ en: 'Draw something first: the patch comes from your drawing.', zh: '先画点什么：图块来自你的画。' }} />
{:else}
	<div class="near">
		<div class="row">
			<figure>
				<Pic data={patch} size={112} frame="real" label={t({ en: 'The outlined patch', zh: '框出的图块' })} />
				<figcaption>{t({ en: `patch at row ${m.row}, column ${m.col} (counting from 0)`, zh: `第 ${m.row} 行第 ${m.col} 列的图块（从 0 数起）` })}</figcaption>
			</figure>
			{#if tiles[best.k]}
				<figure>
					<Pic data={tiles[best.k]} size={112} frame="generated" label={t({ en: `What code ${best.k} draws`, zh: `码字 ${best.k} 画出的样子` })} />
					<figcaption>{t({ en: `code ${best.k}, drawn among empty patches`, zh: `码字 ${best.k}，放在空白图块中间画出的样子` })}</figcaption>
				</figure>
			{/if}
			{#if inPlace}
				<figure>
					<Pic data={inPlace} size={112} frame="generated" label={t({ en: `Code ${best.k} redrawn in place`, zh: `码字 ${best.k} 在原位重画的样子` })} />
					<figcaption>{t({ en: `code ${best.k} in place, among your drawing’s other codes`, zh: `码字 ${best.k} 在原位、和你的画的其他码字一起画出的样子` })}</figcaption>
				</figure>
			{/if}
		</div>
		<div class="tbl" use:keyScroll><table class="terms">
			<caption>{t(train ? { en: 'Training, number by number: forward the decoder sees z_q = e_k; backward the gradient at z_q is copied to z_e', zh: '逐个数看训练：前向时解码器看到 z_q = e_k；反向时 z_q 处的梯度被复制给 z_e' } : { en: `The squared distance to the nearest code, ${best.k}, number by number`, zh: `到最近码字 ${best.k} 的距离平方，逐个数` })}</caption>
			<thead>
				<tr><th scope="col">{t({ en: 'no.', zh: '序号' })}</th><th scope="col">z_e</th><th scope="col">e_{best.k}</th>{#if train}<th scope="col">∂L/∂z_q</th><th scope="col">∂L/∂z_e</th>{:else}<th scope="col">(z_e − e)²</th>{/if}</tr>
			</thead>
			<tbody>
				{#each terms as r (r.d)}
					<tr><th scope="row">{r.d}</th><td><Num value={r.z} /></td><td><Num value={r.e} /></td>{#if train}<td><Num value={m.grad[r.d]} /></td><td><Num value={m.grad[r.d]} /></td>{:else}<td><Num value={r.sq} /></td>{/if}</tr>
				{/each}
				{#if !train}<tr class="sum"><th scope="row">{t({ en: 'sum', zh: '和' })}</th><td></td><td></td><td><Num value={best.d} /></td></tr>{/if}
			</tbody>
		</table></div>
		{#if train && L}
			<p class="small">{t({ en: 'This drawing: redrawing error (BCE averaged over the pixels)', zh: '这幅画：重画误差（像素平均的 BCE）' })} <Num value={L.rec} />; {t({ en: 'commitment term, (z_e − z_q)² averaged over all patches and numbers,', zh: '承诺项，即 (z_e − z_q)² 在所有图块和所有数上的平均，' })} <Num value={L.commit} />.</p>
		{:else if !train}
			<div class="dists" use:keyScroll>
				<p class="small head">{t({ en: 'Squared distance to every code, nearest first (bar length = distance)', zh: '到每个码字的距离平方，从近到远（条长 = 距离）' })}</p>
				<ol>
					{#each m.rank as r, i (r.k)}
						<li class:win={i === 0}>
							{#if tiles[r.k]}<Pic data={tiles[r.k]} size={22} label={t({ en: `Code ${r.k}`, zh: `码字 ${r.k}` })} />{:else}<span></span>{/if}
							<span class="k">{t({ en: `code ${r.k}`, zh: `码字 ${r.k}` })}</span>
							<span class="bar"><span style:width="{(100 * r.d) / (max || 1)}%" style:background={sc.css(r.d)}></span></span>
							<span class="v"><Num value={r.d} /></span>
						</li>
					{/each}
				</ol>
			</div>
		{/if}
	</div>
{/if}

<style>
	.near { display: grid; gap: 10px; min-width: 0; }
	.row { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; }
	figure { margin: 0; display: grid; gap: 4px; justify-items: center; }
	figcaption, .small, caption { font-size: var(--fs-sm); }
	figcaption { color: var(--muted); max-width: 170px; text-align: center; }
	caption { text-align: left; color: var(--muted); padding-bottom: 4px; }
	p { margin: 0; }
	table { border-collapse: collapse; font-variant-numeric: tabular-nums; max-width: 100%; }
	th, td { padding: 2px 10px 2px 0; text-align: right; font-weight: normal; white-space: nowrap; }
	thead th { border-bottom: 1px solid var(--rule); }
	tr.sum th, tr.sum td { font-weight: 700; border-top: 1px solid var(--rule); }
	.tbl { overflow-x: auto; min-width: 0; }
	.dists { max-height: 230px; overflow: auto; min-width: 0; }
	.head { color: var(--muted); margin-bottom: 4px; }
	ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
	li { display: grid; grid-template-columns: 24px 72px minmax(40px, 1fr) 56px; align-items: center; gap: 6px; font-size: var(--fs-sm); }
	li.win { font-weight: 700; }
	.bar { height: 10px; background: var(--surface-2); border-radius: 2px; overflow: hidden; }
	.bar span { display: block; height: 100%; }
	.v { text-align: right; }
</style>
