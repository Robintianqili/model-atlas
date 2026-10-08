<!--
  The shared layer inspector (kit Inspector), wired to the page selection: Values (all channels of
  the selected layer + "what this channel responds to"), Stats, Math (formula with live numbers and
  the parameter count) and Code (the PyTorch line).
-->
<script lang="ts">
	import Inspector from '$lib/kit/Inspector.svelte';
	import TensorView from '$lib/kit/TensorView.svelte';
	import ScopeSwitch from '$lib/kit/ScopeSwitch.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import CodeBlock from '$lib/kit/CodeBlock.svelte';
	import Term from '$lib/kit/Term.svelte';
	import ChannelResponds from './ChannelResponds.svelte';
	import { specToLayers } from '$lib/kit/spec';
	import { sym } from '$lib/kit/tex';
	import { fmtInt } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { CODE_LINES, MODEL_SOURCE, MODEL_SOURCE_START, isSpatial, layerParams } from './data';
	import type { ModelSpec } from '$lib/kit/spec';
	import type { CnnStore } from './store.svelte';

	let { S }: { S: CnnStore } = $props();
	const layers = $derived(specToLayers(S.spec as unknown as ModelSpec));
	const kindOf = (id: string) => (id === 'input' || id === 'softmax' ? 'sequential' : 'diverging');

	function onCell(id: string, c: { row: number; col: number } | null) {
		if (!c) {
			S.hoverRF = null;
			return;
		}
		if (isSpatial(S.layer(id))) {
			S.setCell(c.row, c.col);
			S.hoverRF = S.receptiveField(id, c.row, c.col);
		}
	}

	function paramsTex(id: string): string | null {
		const p = layerParams(S.spec, id);
		if (!p) return null;
		const ws = p.wShape;
		if (ws.length === 4) return `${ws[0]} \\times (${ws[1]} \\times 3 \\times 3) + ${ws[0]} = ${fmtInt(p.weight + p.bias).replace(',', '{,}')}`;
		return `${ws[0]} \\times ${ws[1]} + ${ws[0]} = ${fmtInt(p.weight + p.bias).replace(',', '{,}')}`;
	}

	type M = { tex: string; values: Record<string, number | string>; caption: { en: string; zh: string } };
	function mathOf(id: string): M | null {
		const L = S.layer(id);
		const tr = S.trace;
		if (!tr || !S.model) return null;
		const { ch, y, x } = S.sel;
		const at = { en: `channel ${ch}, cell (${y}, ${x})`, zh: `通道 ${ch}，格子 (${y}, ${x})` };
		if (L.op === 'input') return { tex: `${sym('in', 'x_{ij}')} \\in [0, 1],\\quad ${sym('in', 'x')}_{${y},${x}} = {{v}}`, values: { v: S.input[y * 28 + x] }, caption: { en: `pixel (${y}, ${x})`, zh: `像素 (${y}, ${x})` } };
		if (L.op === 'conv2d') {
			const d = S.model.convDetail(tr, id, ch, y, x);
			return {
				tex: `\\begin{aligned} ${sym('out', 'y_{ij}')} &= \\sum_{c}\\sum_{u,v} ${sym('param', 'K^{(c)}_{uv}')}\\,${sym('in', 'x^{(c)}_{i+u,j+v}')} + ${sym('param', 'b')} \\\\ &= {{s}} + ({{b}}) = {{y}} \\end{aligned}`,
				values: { s: d.total - d.bias, b: d.bias, y: d.total },
				caption: at
			};
		}
		const H = L.outShape[1] ?? 1;
		const W = L.outShape[2] ?? 1;
		if (L.op === 'relu') {
			const pre = tr.t[L.in[0]].data[(ch * H + y) * W + x];
			return { tex: `${sym('act', 'a')} = \\max(0, ${sym('out', 'y')}) = \\max(0, {{v}}) = {{r}}`, values: { v: pre, r: Math.max(0, pre) }, caption: at };
		}
		if (L.op === 'maxpool2d') {
			const d = S.model.poolDetail(tr, id, ch, y, x);
			return { tex: `${sym('out', 'p_{ij}')} = \\max_{u,v \\in \\{0,1\\}} ${sym('in', 'a_{2i+u,\\,2j+v}')} = {{p}}`, values: { p: d.value }, caption: at };
		}
		if (L.op === 'flatten') return { tex: `${sym('act', 'v')}_{16c + 4y + x} = ${sym('in', 'p')}_{c,y,x}`, values: {}, caption: { en: '256 = 16 × 4 × 4', zh: '256 = 16 × 4 × 4' } };
		if (L.op === 'dense') {
			const d = S.model.denseDetail(tr, ch, 1);
			return {
				tex: `${sym('out', 'z_k')} = \\sum_{i=0}^{255} ${sym('param', 'W_{k,i}')}\\,${sym('in', 'x_i')} + ${sym('param', 'b_k')} = {{z}}`,
				values: { z: d.logit },
				caption: { en: `class “${S.spec.classes[ch].en}”`, zh: `类别“${S.spec.classes[ch].zh}”` }
			};
		}
		const sd = S.model.softmaxDetail(tr);
		return {
			tex: `${sym('prob', 'p_k')} = \\frac{e^{${sym('out', 'z_k')}}}{\\sum_j e^{${sym('out', 'z_j')}}} = {{p}}`,
			values: { p: sd.probs[ch] },
			caption: { en: `class “${S.spec.classes[ch].en}”`, zh: `类别“${S.spec.classes[ch].zh}”` }
		};
	}
	const rfOf = (id: string) => (S.layer(id) as { rf?: { size: number; jump: number } }).rf;
</script>

<Inspector
	{layers}
	bind:current={() => S.sel.layer, (v) => S.select(v, S.sel.ch)}
	tensorOf={(id) => S.trace?.t[id]}
	{kindOf}
	provenance="live"
	precision={S.spec.precision}
	tolerance={S.spec.parity?.max_abs_diff as number | undefined}
>
	{#snippet header(id)}
		{@const L = S.layer(id)}
		{@const rf = rfOf(id)}
		{@const p = layerParams(S.spec, id)}
		<div class="hdr">
			{#if L.desc}<p>{t(L.desc)}</p>{/if}
			<p class="facts">
				{#if rf}
					<span><Term id="receptive-field" />: <strong class="num">{rf.size}×{rf.size}</strong> {t({ en: 'pixels', zh: '像素' })}{#if rf.jump > 1}<span class="muted">, {t({ en: `neighbouring cells ${rf.jump} px apart`, zh: `相邻格子相距 ${rf.jump} 像素` })}</span>{/if}</span>
				{/if}
				<span><Term id="parameter">{t({ en: 'parameters', zh: '参数' })}</Term>: <strong class="num">{p ? fmtInt(p.weight + p.bias) : '0'}</strong></span>
			</p>
		</div>
	{/snippet}
	{#snippet values(id)}
		{@const tr = S.trace?.t[id]}
		{#if tr && S.domains}
			{#if isSpatial(S.layer(id)) && id !== 'input'}<div class="scope"><ScopeSwitch bind:value={S.scope} /></div>{/if}
			<TensorView
				tensor={tr}
				label={S.layer(id).label}
				kind={kindOf(id)}
				scope={id === 'input' ? 'unit' : S.scope}
				scopeDomains={{ layer: S.domains.layer[id], global: S.domains.global }}
				bind:channel={() => (S.layer(id).outShape.length === 3 ? S.sel.ch : 0), (c) => S.select(id, c)}
				onhover={(c) => onCell(id, c)}
			/>
			{#if isSpatial(S.layer(id)) && id !== 'input'}<ChannelResponds {S} layerId={id} />{/if}
		{:else}
			<p class="muted">{t({ en: 'Loading…', zh: '加载中…' })}</p>
		{/if}
	{/snippet}
	{#snippet math(id)}
		{@const m = mathOf(id)}
		{@const pt = paramsTex(id)}
		{#if m}<FormulaCard tex={m.tex} values={m.values} caption={t(m.caption)} provenance="live" />{/if}
		{#if pt}
			<FormulaCard
				tex={`\\text{${t({ en: 'parameters', zh: '参数' })}} = ${pt}`}
				caption={t({ en: 'Learned numbers in this layer (weights + biases)', zh: '这一层学到的数（权重 + 偏置）' })}
			/>
		{:else}
			<p class="muted small">{t({ en: 'This layer has no parameters: it only rearranges or compares numbers.', zh: '这一层没有参数：它只重排或比较数字。' })}</p>
		{/if}
	{/snippet}
	{#snippet code(id)}
		<CodeBlock code={MODEL_SOURCE} lang="python" start={MODEL_SOURCE_START} highlight={CODE_LINES[id] ?? []} caption="pipeline/tinycnn/common.py · PyTorch" />
	{/snippet}
</Inspector>

<style>
	.hdr p {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.facts {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 18px;
		margin-top: 4px !important;
	}
	.scope {
		margin-bottom: 8px;
	}
	.small {
		font-size: var(--fs-sm);
	}
</style>
