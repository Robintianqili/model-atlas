<!--
  "What this channel responds to" (precomputed, channels.json): the 6 test drawings that excite the
  channel most, cropped to the receptive field, plus the mean peak activation per class. For
  conv1_1 the 3x3 kernels themselves are shown first: at that size they are the clearest picture.
-->
<script lang="ts">
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import NumGrid from './NumGrid.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { b64ToFloats, sourceConv } from './data';
	import type { CnnStore, SelView } from './store.svelte';

	interface Props {
		S: CnnStore;
		layerId: string;
		/** whose selection to show (default: the page store) */
		view?: SelView;
		/** show this channel instead of the selected one */
		channel?: number;
		/** only the top drawings (no kernels, no per-class bars) */
		compact?: boolean;
	}
	let { S, layerId, view, channel, compact = false }: Props = $props();
	const V = $derived(view ?? S);

	const conv = $derived(sourceConv(S.spec, layerId));
	const info = $derived(conv && S.channels ? S.channels.layers[conv] : null);
	const ch = $derived(channel ?? V.sel.ch);
	const cinfo = $derived(info?.channels[ch] ?? null);
	const patches = $derived(cinfo ? cinfo.top.map((p) => ({ ...p, img: b64ToFloats(p.patch) })) : []);
	const icon = $derived(cinfo?.icon ? b64ToFloats(cinfo.icon) : null);
	const ps = $derived(info?.patchSize ?? 7);
	const pad = $derived(S.channels?.pad ?? 2);
	const inner = $derived(info ? { y0: pad, x0: pad, y1: pad + info.rfSize - 1, x1: pad + info.rfSize - 1 } : null);
	const cmm = $derived(cinfo ? cinfo.classMeanMax : []);
	const cmmMax = $derived(Math.max(1e-6, ...cmm.map((v) => Math.abs(v))));
	const order = $derived(cmm.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v));

	// conv1_1: the kernels themselves
	const isFirst = $derived(conv === 'conv1_1');
	const kernels = $derived(S.model && isFirst ? S.model.weights['conv1_1.weight'].data : null);
	const kDom = $derived.by<[number, number]>(() => {
		let m = 0;
		if (kernels) for (let i = 0; i < kernels.length; i++) m = Math.max(m, Math.abs(kernels[i]));
		return [-(m || 1), m || 1];
	});
	const via = $derived(conv && conv !== layerId ? S.layer(conv) : null);
</script>

{#if conv && info && cinfo}
	<div class="cr">
		<div class="crh">
			<h4>{t({ en: 'What this channel responds to', zh: '这个通道对什么有反应' })}</h4>
			<ProvenanceBadge
				mode="precomputed"
				compact
				note={{ en: 'Top drawings: computed offline over the 15,000 test drawings with the same weights.', zh: '最强激活的画：用同一组权重在 15,000 幅测试画上离线算出。' }}
			/>
		</div>
		{#if via && !compact}
			<p class="muted small">
				{t({ en: `${t(S.layer(layerId).label)} channel ${ch} carries ${t(via.label)} channel ${ch}, so it responds to the same patterns.`, zh: `${t(S.layer(layerId).label)} 的通道 ${ch} 来自 ${t(via.label)} 的通道 ${ch}，所以对同样的模式有反应。` })}
			</p>
		{/if}
		{#if isFirst && kernels && !compact}
			<div class="kern">
				<NumGrid values={kernels.subarray(ch * 9, ch * 9 + 9)} rows={3} cols={3} domain={kDom} role="param" label={t({ en: `kernel ${ch} (3×3 weights)`, zh: `卷积核 ${ch}（3×3 权重）` })} cell={54} />
				<div class="allk">
					<span class="muted small">{t({ en: 'All 8 kernels of Conv 1-1', zh: 'Conv 1-1 的全部 8 个卷积核' })}</span>
					<div class="kk">
						{#each Array.from({ length: 8 }, (_, c) => c) as c (c)}
							<button type="button" class="kb" class:on={c === ch} aria-pressed={c === ch} aria-label="{t({ en: 'kernel', zh: '卷积核' })} {c}" onclick={() => V.select(layerId, c)}>
								<MapCanvas data={kernels.subarray(c * 9, c * 9 + 9)} rows={3} cols={3} domain={kDom} size={30} />
								<span class="num">{c}</span>
							</button>
						{/each}
					</div>
				</div>
			</div>
			<p class="muted small">
				{t({ en: 'Orange weights reward ink, blue weights punish it. A kernel with orange on one side and blue on the other is an edge detector.', zh: '橙色权重奖励墨迹，蓝色权重惩罚墨迹。一边橙一边蓝的卷积核就是边缘检测器。' })}
			</p>
		{/if}
		<p class="small">
			{t({ en: `Test drawings that excite ${t(S.layer(conv).label)} channel ${ch} the most (receptive field ${info.rfSize}×${info.rfSize}, outlined):`, zh: `最能激活 ${t(S.layer(conv).label)} 通道 ${ch} 的测试画（框出 ${info.rfSize}×${info.rfSize} 的感受野）：` })}
		</p>
		<div class="patches">
			{#each patches as p, i (i)}
				<figure>
					<MapCanvas data={p.img} rows={ps} cols={ps} domain={[0, 1]} paper size={64} boxes={inner ? [{ ...inner, style: 'soft', color: 'var(--accent)' }] : []} />
					<figcaption><span class="sw" style="background: {categorical(p.class, themeState.resolved)}"></span>{t(S.spec.classes[p.class])} · <span class="num">{fmt3(p.act)}</span></figcaption>
				</figure>
			{/each}
			<!-- the optimized-input picture needs its own explanation, so the compact (lesson) view leaves it out -->
			{#if icon && !isFirst && !compact}
				<figure class="icon">
					<MapCanvas data={icon} rows={ps} cols={ps} domain={[0, 1]} paper size={64} boxes={inner ? [{ ...inner, style: 'soft', color: 'var(--accent)' }] : []} />
					<figcaption>{t({ en: 'optimized input', zh: '优化出的输入' })}</figcaption>
				</figure>
			{/if}
		</div>
		{#if !compact}
		<div class="cmm">
			<span class="muted small">{t({ en: 'Average peak activation per class (test set)', zh: '各类别的平均峰值激活（测试集）' })}</span>
			{#each order as o (o.i)}
				<div class="cm">
					<span class="cn"><span class="sw" style="background: {categorical(o.i, themeState.resolved)}"></span>{t(S.spec.classes[o.i])}</span>
					<span class="bar"><span style="width: {Math.max(0, o.v / cmmMax) * 100}%"></span></span>
					<span class="num v">{fmt3(o.v)}</span>
				</div>
			{/each}
		</div>
		{/if}
	</div>
{/if}

<style>
	.cr {
		display: grid;
		gap: 8px;
		padding-top: 12px;
		margin-top: 12px;
		border-top: 1px solid var(--rule);
	}
	.crh {
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
		font-size: var(--fs-sm);
		margin: 0;
	}
	.kern {
		display: flex;
		flex-wrap: wrap;
		gap: 12px 24px;
		align-items: flex-end;
	}
	.allk {
		display: grid;
		gap: 4px;
	}
	.kk {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.kb {
		display: grid;
		justify-items: center;
		gap: 1px;
		padding: 3px;
		border: 1px solid var(--rule);
		border-radius: var(--r1);
		background: var(--surface);
		font-size: var(--t-xs);
	}
	.kb.on {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
	}
	.patches {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	figure {
		margin: 0;
		display: grid;
		gap: 2px;
	}
	figcaption {
		font-size: var(--t-xs);
		color: var(--ink-2);
		display: flex;
		align-items: center;
		gap: 3px;
		max-width: 80px;
		flex-wrap: wrap;
	}
	.icon figcaption {
		color: var(--muted);
	}
	.sw {
		width: 8px;
		height: 8px;
		border-radius: 2px;
		flex: 0 0 auto;
	}
	.cmm {
		display: grid;
		gap: 1px;
		max-width: 360px;
	}
	.cm {
		display: grid;
		grid-template-columns: 6.5em 1fr 3.5em;
		gap: 6px;
		align-items: center;
		font-size: var(--t-xs);
	}
	.cn {
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.bar {
		height: 8px;
		background: var(--surface-2);
		border-radius: 0 2px 2px 0;
	}
	.bar span {
		display: block;
		height: 100%;
		background: var(--seq-mid);
		border-radius: 0 2px 2px 0;
	}
	.v {
		text-align: right;
	}
</style>
