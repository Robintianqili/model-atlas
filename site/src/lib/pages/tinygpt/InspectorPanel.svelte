<!--
  The kit Inspector over all 66 traced tensors, with TinyGPT-specific tabs:
  Values (attention per head, masked cells hatched; logits/probs with the top tokens),
  Stats (masked −∞ cells left out), Math (live arithmetic), Code (PyTorch reference + the JS line).
-->
<script lang="ts">
	import Inspector from '$lib/kit/Inspector.svelte';
	import TensorView from '$lib/kit/TensorView.svelte';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import TensorStrip from '$lib/kit/TensorStrip.svelte';
	import TensorStats from '$lib/kit/TensorStats.svelte';
	import ScopeSwitch from '$lib/kit/ScopeSwitch.svelte';
	import AttentionMatrix from '$lib/kit/AttentionMatrix.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import CodeBlock from '$lib/kit/CodeBlock.svelte';
	import InspectorMath from './InspectorMath.svelte';
	import { specToLayers, type ModelSpec } from '$lib/kit/spec';
	import { domainOf, type ColorScope, type Domain } from '$lib/kit/color';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtShape, fmtPct } from '$lib/kit/format';
	import type { TGState } from './state.svelte';
	import { H, NL, V } from './state.svelte';
	import { row, topIndices } from './analysis';
	import { TRACE_CODE, codeLines, jsLine } from './code';

	let { S, spec }: { S: TGState; spec: ModelSpec } = $props();

	const layers = $derived(specToLayers(spec));
	const f = $derived(S.run?.forward);
	const T = $derived(S.T);
	let scope = $state<ColorScope>('layer');

	const isAttn = (id: string) => /attn\.(scores|masked|weights)$/.test(id);
	const kindOf = (id: string) => (/attn\.weights$/.test(id) || id === 'probs' ? 'sequential' : 'diverging');

	// 'global' scope: one range for the same tensor in all 4 blocks (e.g. every resid2)
	const globalCache = new Map<string, Domain>();
	let cacheFor: unknown = null;
	function globalDomain(id: string): Domain | undefined {
		if (!f) return undefined;
		if (cacheFor !== f) {
			globalCache.clear();
			cacheFor = f;
		}
		const m = /^L\d\.(.+)$/.exec(id);
		if (!m) return undefined;
		if (!globalCache.has(m[1])) {
			const arrs = Array.from({ length: NL }, (_, l) => f.t[`L${l}.${m[1]}`].data);
			globalCache.set(m[1], domainOf(kindOf(id), ...arrs));
		}
		return globalCache.get(m[1]);
	}
	function finite(data: Float32Array): Float32Array {
		return data.filter((v) => Number.isFinite(v));
	}
	function finiteDomain(data: ArrayLike<number>): Domain {
		const fin: number[] = [];
		for (let i = 0; i < data.length; i++) if (Number.isFinite(data[i])) fin.push(data[i]);
		return domainOf('diverging', fin);
	}
	function headOf(id: string) {
		return f!.t[id].data.subarray(S.head * T * T, (S.head + 1) * T * T);
	}
	const heads = Array.from({ length: H }, (_, i) => i);
</script>

<Inspector
	{layers}
	bind:current={S.layerId}
	provenance="live"
	precision="fp32"
	tolerance={1e-4}
	title={{ en: 'Inspector · every traced tensor', zh: '检查器 · 每一个被追踪的张量' }}
>
	{#snippet header(id)}
		{#if f?.t[id]}
			{@const sh = f.t[id].shape}
			<p class="hdr num">
				{t({ en: 'this run:', zh: '本次运行：' })} {fmtShape(sh)}
				· {t({ en: 'token', zh: '词元' })} <span class="tok">{S.label(S.pos)}</span> ({t({ en: 'position', zh: '位置' })} {S.pos})
				{#if isAttn(id)}· {t({ en: 'head', zh: '头' })} {S.head + 1}{/if}
			</p>
			{#if sh.length === 2}
				<TensorStrip
					data={row(f.t[id].data, S.pos, sh[1])}
					kind={kindOf(id)}
					domain={kindOf(id) === 'sequential' ? [0, Math.max(1e-6, ...Array.from(row(f.t[id].data, S.pos, sh[1])))] : undefined}
					shape={[sh[1]]}
					maxCells={40}
					cellSize={11}
					name={{ en: 'row of the selected token', zh: '所选词元这一行' }}
					selected={sh[1] === 76 ? S.dim : null}
					onselect={(i) => {
						if (sh[1] === 76) S.dim = i;
					}}
				/>
			{/if}
		{/if}
	{/snippet}
	{#snippet values(id)}
		{#if f?.t[id]}
			{#if isAttn(id)}
				<div class="hsel">
					<span class="segmented" role="group" aria-label={t({ en: 'Head', zh: '头' })}>
						{#each heads as h (h)}
							<button type="button" aria-pressed={S.head === h} onclick={() => ((S.head = h), (S.keySel = null))}>{t({ en: 'Head', zh: '头' })} {h + 1}</button>
						{/each}
					</span>
				</div>
				{#if id.endsWith('weights')}
					<AttentionMatrix weights={headOf(id)} queries={S.labels} causal row={S.pos} col={S.key} label={{ en: `Head ${S.head + 1} [${T}×${T}]`, zh: `第 ${S.head + 1} 头 [${T}×${T}]` }} />
				{:else}
					{@const dom = finiteDomain(headOf(id))}
					<TensorGrid
						data={headOf(id)}
						rows={T}
						cols={T}
						maxCell={30}
						domain={dom}
						masked={id.endsWith('masked') ? (r, c) => c > r : undefined}
						highlights={[{ row: S.pos, col: Math.min(S.key, S.pos), color: 'var(--k)' }]}
						label="{id} head {S.head + 1}"
						onselect={(c) => {
							S.selectPos(c.row);
							if (c.col <= c.row) S.selectKey(c.col);
						}}
					/>
					<ColorLegend kind="diverging" domain={dom} />
					{#if id.endsWith('masked')}<p class="small muted">{t({ en: 'Hatched cells hold −∞ (masked future keys).', zh: '阴影格里是 −∞（被掩码的后面的键）。' })}</p>{/if}
				{/if}
			{:else if id === 'logits' || id === 'probs'}
				{@const r = row(f.t[id].data, S.pos, V)}
				<TensorGrid data={f.t[id].data} rows={T} cols={V} kind={kindOf(id)} highlights={[{ row: S.pos, col: 0, h: 1, w: V, color: 'var(--q)' }]} label={id} onselect={(c) => S.selectPos(c.row)} />
				<p class="small muted">{t({ en: `${V} columns, one per vocabulary token; the grid scrolls sideways. Top tokens after the selected token:`, zh: `${V} 列，每列对应词表中的一个词元；网格可横向滚动。所选词元之后的前几名：` })}</p>
				<ol class="toplist">
					{#each topIndices(f.t.probs.data.subarray(S.pos * V, S.pos * V + V), 8) as k (k)}
						<li><span class="tok">{S.idLabel(k)}</span> <span class="num">#{k}</span> <span class="num">{id === 'logits' ? r[k].toFixed(3) : fmtPct(r[k])}</span></li>
					{/each}
				</ol>
			{:else}
				<div class="scope"><ScopeSwitch bind:value={scope} /></div>
				<TensorView tensor={f.t[id]} label={id} kind={kindOf(id)} {scope} scopeDomains={{ layer: domainOf(kindOf(id), f.t[id].data), global: globalDomain(id) }} />
			{/if}
		{:else}
			<p class="muted">{t({ en: 'No data yet.', zh: '暂无数据。' })}</p>
		{/if}
	{/snippet}
	{#snippet stats(id)}
		{#if f?.t[id]}
			{#if id.endsWith('masked')}
				<TensorStats tensor={{ shape: [finite(f.t[id].data).length], data: finite(f.t[id].data) }} kind="diverging" />
				<p class="small muted">{t({ en: 'Statistics of the unmasked cells only (the −∞ cells are left out).', zh: '只统计未被掩码的格子（−∞ 的格子不计入）。' })}</p>
			{:else}
				<TensorStats tensor={f.t[id]} kind={kindOf(id)} />
			{/if}
		{/if}
	{/snippet}
	{#snippet math(id)}
		<InspectorMath {S} {id} />
	{/snippet}
	{#snippet code(id)}
		<CodeBlock code={TRACE_CODE} lang="python" highlight={codeLines(id)} caption={{ en: 'pipeline/tinygpt/model.py · GPT.trace() (PyTorch reference)', zh: 'pipeline/tinygpt/model.py · GPT.trace()（PyTorch 参考实现）' }} />
		{#if jsLine(id)}
			<CodeBlock code={jsLine(id)} lang="ts" caption={{ en: 'src/lib/models/tinygpt.ts · what runs in your browser', zh: 'src/lib/models/tinygpt.ts · 在你浏览器里运行的代码' }} />
		{/if}
	{/snippet}
</Inspector>

<style>
	.hdr {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.tok {
		font-family: var(--font-mono);
		background: var(--surface-2);
		border-radius: 3px;
		padding: 0 4px;
		white-space: pre;
	}
	.hsel {
		margin-bottom: 8px;
	}
	.scope {
		margin-bottom: 8px;
	}
	.small {
		font-size: var(--fs-xs);
		margin: 4px 0;
	}
	.toplist {
		margin: 0;
		padding-left: 1.5em;
		font-size: var(--fs-xs);
		columns: 2;
	}
</style>
