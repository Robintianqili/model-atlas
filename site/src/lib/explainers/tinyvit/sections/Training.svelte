<!--
  tinyvit: the page's Training section (replaces the auto list of raw metric keys such as "test_acc 0.9715", teaching
  review R1). Every number is read from spec.training (pipeline/tinyvit/train.py) or facts.json (pipeline/tinyvit/stats.py).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { AtlasSpec } from '../../../atlas/types/index.ts';
	import type { AtlasStore } from '../../../atlas/ui/runtime.svelte.ts';
	import { fmtInt, fmtPct } from '../../../atlas/core/format.ts';

	let { store, spec }: { store: AtlasStore | null; spec: AtlasSpec | null } = $props();
	type Tr = { epochs?: number; bestEpoch?: number; batchSize?: number; optimizer?: string; metrics?: { test_acc?: number; val_acc?: number; test_n?: number } };
	const tr = $derived((spec?.training ?? {}) as unknown as Tr);
	const f = (k: string) => store?.rt.num(`fact:${k}`) ?? NaN;
	const rows = $derived([
		{ en: 'Data', zh: '数据', v: t({ en: 'Quick, Draw! drawings: TinyCNN’s 10 classes and its training, validation and test split', zh: 'Quick, Draw! 的画：TinyCNN 的 10 个类别，以及同样的训练、验证、测试划分' }) },
		{ en: 'Training', zh: '训练', v: t({ en: `${tr.epochs} epochs (passes over the training drawings) in batches of ${tr.batchSize}; the epoch that did best on the validation drawings, epoch ${tr.bestEpoch}, is the one shipped`, zh: `${tr.epochs} 轮（每轮把训练画全部过一遍），每批 ${tr.batchSize} 幅；发布的是在验证画上表现最好的第 ${tr.bestEpoch} 轮` }) },
		{ en: 'Recipe', zh: '训练方法', v: tr.optimizer ?? '—' },
		{ en: 'Right on the test drawings', zh: '测试画上的正确率', v: t({ en: `${fmtPct(tr.metrics?.test_acc ?? NaN, 2)} of ${fmtInt(tr.metrics?.test_n ?? NaN)} drawings it never saw in training`, zh: `${fmtPct(tr.metrics?.test_acc ?? NaN, 2)}（${fmtInt(tr.metrics?.test_n ?? NaN)} 幅训练时没见过的画）` }) },
		{ en: 'Right on the validation drawings', zh: '验证画上的正确率', v: fmtPct(tr.metrics?.val_acc ?? NaN, 2) },
		{ en: 'TinyCNN on the same test drawings', zh: '同一批测试画上的 TinyCNN', v: t({ en: `right ${fmtPct(f('cnnTestAcc'), 2)}; the two give the same answer on ${fmtPct(f('agree'), 2)}`, zh: `正确率 ${fmtPct(f('cnnTestAcc'), 2)}；两者在 ${fmtPct(f('agree'), 2)} 的画上答案相同` }) }
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
