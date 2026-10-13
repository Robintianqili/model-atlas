<!--
  llava-toy: the page's Training section. Every number is read from spec.training (pipeline/llava-toy/train.py) or
  facts.json (pipeline/llava-toy/stats.py).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { AtlasSpec } from '../../../atlas/types/index.ts';
	import type { AtlasStore } from '../../../atlas/ui/runtime.svelte.ts';
	import { fmtInt, fmtPct } from '../../../atlas/core/format.ts';

	let { store, spec }: { store: AtlasStore | null; spec: AtlasSpec | null } = $props();
	type Tr = { epochs?: number; bestEpoch?: number; batchSize?: number; optimizer?: string; metrics?: { test_n?: number; trainable_params?: number } };
	const tr = $derived((spec?.training ?? {}) as unknown as Tr);
	const f = (k: string) => store?.rt.num(`fact:${k}`) ?? NaN;
	const n = $derived(fmtInt(tr.metrics?.test_n ?? NaN));
	const rows = $derived([
		{ en: 'Data', zh: '数据', v: t({ en: 'Quick, Draw! drawings (TinyCNN’s 10 classes and split); every epoch each training drawing gets a new question: “what is this ?”, or “is this a … ?” about its own class or another one', zh: 'Quick, Draw! 的画（TinyCNN 的 10 个类别和数据划分）；每一轮给每幅训练画配一个新问题：“what is this ?”，或者就它自己的类别或别的类别问“is this a … ?”' }) },
		{ en: 'What was trained', zh: '训练了什么', v: t({ en: `the projector and the language model, ${fmtInt(tr.metrics?.trainable_params ?? NaN)} weights; the ${fmtInt(f('vitRead'))} TinyViT weights read here stayed frozen`, zh: `投影层和语言模型，共 ${fmtInt(tr.metrics?.trainable_params ?? NaN)} 个权重；这里用到的 ${fmtInt(f('vitRead'))} 个 TinyViT 权重保持冻结` }) },
		{ en: 'Training', zh: '训练', v: t({ en: `${tr.epochs} epochs in batches of ${tr.batchSize}, the loss on the answer word only; the epoch that did best on the validation drawings, epoch ${tr.bestEpoch}, is the one shipped`, zh: `${tr.epochs} 轮，每批 ${tr.batchSize} 幅，损失只算答案那个词；发布的是在验证画上表现最好的第 ${tr.bestEpoch} 轮` }) },
		{ en: 'Recipe', zh: '训练方法', v: tr.optimizer ?? '—' },
		{ en: '“what is this ?” on the test drawings', zh: '测试画上的“what is this ?”', v: t({ en: `right on ${fmtPct(f('whatAcc'), 2)} of ${n} drawings; TinyViT’s own class head: ${fmtPct(f('vitAcc'), 2)}`, zh: `${n} 幅画中答对 ${fmtPct(f('whatAcc'), 2)}；TinyViT 自己的类别头：${fmtPct(f('vitAcc'), 2)}` }) },
		{ en: 'Yes or no on the test drawings', zh: '测试画上的是或不是', v: t({ en: `“yes” about its own class ${fmtPct(f('yesTrue'), 2)}; “no” about another class ${fmtPct(f('noOther'), 2)}`, zh: `问它自己的类别时答“yes”：${fmtPct(f('yesTrue'), 2)}；问别的类别时答“no”：${fmtPct(f('noOther'), 2)}` }) }
	]);
</script>

<dl class="facts">
	{#each rows as r (r.en)}
		<div><dt>{t(r)}</dt><dd>{r.v}</dd></div>
	{/each}
</dl>

<style>
	.facts {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 8px;
		margin: 0;
	}
	.facts div {
		display: grid;
		grid-template-columns: minmax(0, 14em) minmax(0, 1fr);
		gap: 4px 16px;
	}
	.facts dt {
		color: var(--ink-2);
		font-size: var(--fs-sm);
	}
	.facts dd {
		margin: 0;
	}
	@media (max-width: 560px) {
		.facts div {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
