<!--
  Compare two inputs with the same model (brief §4.6): A = the current input, B = a preset.
  Cosine similarity of A and B at every layer (live), and A, B and A − B for one feature map.
-->
<script lang="ts">
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtPct } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { cosine } from './data';
	import type { CnnStore } from './store.svelte';

	let { S, bCls = $bindable('apple'), bIdx = $bindable(4) }: { S: CnnStore; bCls?: string; bIdx?: number } = $props();

	const groups = $derived(S.presets ? [...S.presets.classes, ...S.presets.heldOut] : []);
	const bInput = $derived(S.presetInput(bCls, bIdx));
	const bTrace = $derived(S.model && bInput ? S.model.forward(bInput) : null);
	/** a third drawing from another class, as a reference for "how similar is unrelated?" */
	const cCls = $derived(bCls === 'sun' ? 'fish' : 'sun');
	const cInput = $derived(S.presetInput(cCls, 0));
	const cTrace = $derived(S.model && cInput ? S.model.forward(cInput) : null);
	const ids = ['input', 'relu1_1', 'relu1_2', 'pool1', 'relu2_1', 'relu2_2', 'pool2', 'dense'];
	const sims = $derived(
		S.trace && bTrace && cTrace
			? ids.map((id) => ({ id, ab: cosine(S.trace!.t[id].data, bTrace.t[id].data), ac: cosine(S.trace!.t[id].data, cTrace.t[id].data) }))
			: []
	);
	const bTop = $derived.by(() => {
		if (!bTrace) return -1;
		const p = bTrace.t.softmax.data;
		let k = 0;
		for (let i = 1; i < p.length; i++) if (p[i] > p[k]) k = i;
		return k;
	});

	// one feature map side by side
	let layer = $state('relu2_1');
	let ch = $state(0);
	const L = $derived(S.layer(layer));
	const hw = $derived(L.outShape[1] * L.outShape[2]);
	const aMap = $derived(S.trace ? S.trace.t[layer].data.subarray(ch * hw, (ch + 1) * hw) : null);
	const bMap = $derived(bTrace ? bTrace.t[layer].data.subarray(ch * hw, (ch + 1) * hw) : null);
	const diff = $derived(aMap && bMap ? Float32Array.from(aMap, (v, i) => v - bMap[i]) : null);
	const dom = $derived.by<[number, number]>(() => {
		let m = 0;
		for (const arr of [aMap, bMap, diff]) if (arr) for (let i = 0; i < arr.length; i++) m = Math.max(m, Math.abs(arr[i]));
		return [-(m || 1), m || 1];
	});
	const meanAbsDiff = $derived(diff ? diff.reduce((s, v) => s + Math.abs(v), 0) / diff.length : 0);
	const spatialIds = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2'];
	const nCh = $derived(L.outShape[0]);
	$effect(() => {
		if (ch >= nCh) ch = 0;
	});
	const label = (id: string) => t(S.layer(id).label);
	const W = 340;
	const BH = 16;
</script>

{#if S.trace && bTrace && bInput}
	<div class="cmp">
		<div class="pair">
			<figure>
				<MapCanvas data={S.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={84} />
				<figcaption><strong>A</strong> · {t({ en: 'current input', zh: '当前输入' })} · {t(S.spec.classes[S.top])} {fmtPct(S.probs?.[S.top] ?? 0)}</figcaption>
			</figure>
			<figure>
				<MapCanvas data={bInput} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={84} />
				<figcaption>
					<strong>B</strong> ·
					<select bind:value={bCls} aria-label={t({ en: 'Class of drawing B', zh: '画 B 的类别' })}>
						{#each groups as g (g.id)}<option value={g.id}>{t(g)}</option>{/each}
					</select>
					<select bind:value={bIdx} aria-label={t({ en: 'Which drawing', zh: '第几幅' })}>
						{#each [0, 1, 2, 3, 4, 5] as i (i)}<option value={i}>#{i + 1}</option>{/each}
					</select>
					<span>· {t(S.spec.classes[bTop])} {fmtPct(bTrace.t.softmax.data[bTop])}</span>
				</figcaption>
			</figure>
		</div>

		<div class="sim">
			<div class="sh">
				<h4>{t({ en: 'How alike are A and B, layer by layer?', zh: 'A 和 B 在每一层有多相似？' })}</h4>
				<ProvenanceBadge mode="live" compact />
			</div>
			<p class="muted small">
				{t({ en: `Cosine similarity of the whole layer (1 = same direction, 0 = nothing in common). Gray: A against a ${cCls === 'sun' ? 'sun' : 'fish'} drawing, for reference.`, zh: `整层的余弦相似度（1 = 方向相同，0 = 毫无共同之处）。灰色：A 与一幅${cCls === 'sun' ? '太阳' : '鱼'}的画相比，作为参照。` })}
			</p>
			<svg viewBox="0 0 {W + 120} {sims.length * (BH * 2 + 6)}" width="100%" role="img" aria-label={sims.map((s) => `${label(s.id)} ${fmt3(s.ab)}`).join(', ')}>
				{#each sims as s, i (s.id)}
					{@const y = i * (BH * 2 + 6)}
					<text x="0" y={y + BH - 3} class="lbl">{label(s.id)}</text>
					<rect x="110" y={y} width={W} height={BH - 2} fill="var(--surface-2)" rx="2" />
					<rect x="110" y={y} width={Math.max(1, s.ab * W)} height={BH - 2} fill="var(--accent)" rx="2" />
					<text x={116 + Math.max(1, s.ab * W)} y={y + BH - 4} class="v">{fmt3(s.ab)}</text>
					<rect x="110" y={y + BH - 1} width={Math.max(1, s.ac * W)} height={BH - 6} fill="var(--rule-strong)" rx="2" />
					<text x={116 + Math.max(1, s.ac * W)} y={y + 2 * BH - 8} class="v muted">{fmt3(s.ac)}</text>
				{/each}
			</svg>
		</div>

		<div class="maps">
			<div class="ctl">
				<label>
					{t({ en: 'Layer', zh: '层' })}
					<select bind:value={layer}>
						{#each spatialIds as id (id)}<option value={id}>{label(id)}</option>{/each}
					</select>
				</label>
				<label>
					{t({ en: 'Channel', zh: '通道' })} <span class="num">{ch}</span>
					<input type="range" min="0" max={nCh - 1} step="1" bind:value={ch} />
				</label>
			</div>
			{#if aMap && bMap && diff}
				<div class="three">
					<figure><MapCanvas data={aMap} rows={L.outShape[1]} cols={L.outShape[2]} domain={dom} size={104} /><figcaption>A</figcaption></figure>
					<figure><MapCanvas data={bMap} rows={L.outShape[1]} cols={L.outShape[2]} domain={dom} size={104} /><figcaption>B</figcaption></figure>
					<figure><MapCanvas data={diff} rows={L.outShape[1]} cols={L.outShape[2]} domain={dom} size={104} /><figcaption>A − B · {t({ en: 'mean |diff|', zh: '平均 |差|' })} {fmt3(meanAbsDiff)}</figcaption></figure>
					<ColorLegend domain={dom} width={150} />
				</div>
			{/if}
		</div>
	</div>
{/if}

<style>
	.cmp {
		display: grid;
		gap: 14px;
	}
	.pair,
	.three {
		display: flex;
		flex-wrap: wrap;
		gap: 12px 20px;
		align-items: flex-end;
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		align-items: center;
	}
	select {
		font-size: var(--fs-xs);
		padding: 2px 4px;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r1);
		background: var(--surface);
		min-height: 28px;
	}
	.sh {
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.small {
		margin: 2px 0 6px;
		font-size: var(--fs-xs);
	}
	.sim svg {
		max-width: 520px;
	}
	.lbl {
		font-size: var(--t-xs);
		fill: var(--ink-2);
		font-family: var(--font-body);
	}
	.v {
		font-size: var(--t-xs);
		font-family: var(--font-mono);
		fill: var(--ink);
	}
	.v.muted {
		fill: var(--muted);
	}
	.ctl {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 20px;
		font-size: var(--fs-sm);
		margin-bottom: 8px;
	}
	.ctl label {
		display: inline-flex;
		gap: 6px;
		align-items: center;
	}
</style>
