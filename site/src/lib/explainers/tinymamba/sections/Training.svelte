<!-- tinymamba: the page's Training section (the auto section prints English keys and raw shares). Every number is read
     from facts.json (pipeline/tinymamba/train.py and stats.py). -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { AtlasStore } from '../../../atlas/ui/runtime.svelte.ts';
	import { fmtInt, fmtPct } from '../../../atlas/core/format.ts';

	let { store }: { store: AtlasStore | null } = $props();
	const f = (k: string) => store?.rt.num(`fact:${k}`) ?? NaN;
	// a share just under 1 keeps two decimals (fmtPct would print 0.999875 as "100.0%")
	const p = (k: string) => (f(k) < 1 && f(k) >= 0.9995 ? `${(100 * f(k)).toFixed(2)}%` : fmtPct(f(k)));
	const rows = $derived([
		{ k: { en: 'Data', zh: '数据' }, v: { en: `Made-up rows from task.ts: ${fmtInt(f('trainExamples'))} for training; every number below is measured on new rows. A row has ${f('bodyMin')} to ${f('bodyMax')} dots and letters, with 1 to ${f('kMax')} letters, then one ? per letter.`, zh: `由 task.ts 生成的行：${fmtInt(f('trainExamples'))} 行用于训练；下面每个数都是在新的行上量出来的。每行有 ${f('bodyMin')} 到 ${f('bodyMax')} 个点和字母，其中 1 到 ${f('kMax')} 个是字母，后面每个字母配一个 ?。` } },
		{ k: { en: 'Training', zh: '训练' }, v: { en: `${fmtInt(f('steps'))} steps of ${fmtInt(f('batch'))} rows (AdamW); the loss counts only the answers at the ?`, zh: `${fmtInt(f('steps'))} 步，每步 ${fmtInt(f('batch'))} 行（AdamW）；损失只算 ? 处的回答` } },
		{ k: { en: 'Answers right', zh: '回答正确' }, v: { en: `${p('valAcc')} of the answers on ${fmtInt(f('valExamples'))} new rows; every answer right in ${p('valExact')} of the rows`, zh: `${fmtInt(f('valExamples'))} 个新行中 ${p('valAcc')} 的回答正确；${p('valExact')} 的行全部答对` } },
		{ k: { en: 'More letters than in training', zh: '比训练时更多的字母' }, v: { en: `answers right with 4 letters (the most in training): ${p('accK4')}; with 5, 6, 7, 8 letters: ${p('accK5')}, ${p('accK6')}, ${p('accK7')}, ${p('accK8')}`, zh: `4 个字母（训练里最多的）时回答正确的比例：${p('accK4')}；5、6、7、8 个字母时：${p('accK5')}、${p('accK6')}、${p('accK7')}、${p('accK8')}` } },
		{ k: { en: 'One letter, far back', zh: '一个字母，离得很远' }, v: { en: `right after 16 dots: ${p('accGap16')}; after 38 dots: ${p('accGap38')}`, zh: `隔 16 个点：${p('accGap16')}；隔 38 个点：${p('accGap38')}` } }
	]);
</script>

<dl class="facts">
	{#each rows as r (r.k.en)}<div><dt>{t(r.k)}</dt><dd>{t(r.v)}</dd></div>{/each}
</dl>

<style>
	.facts { display: grid; grid-template-columns: minmax(0, 1fr); gap: 8px; margin: 0; }
	.facts div { display: grid; grid-template-columns: minmax(0, 14em) minmax(0, 1fr); gap: 4px 16px; }
	.facts dt { color: var(--ink-2); font-size: var(--fs-sm); }
	.facts dd { margin: 0; }
	@media (max-width: 560px) { .facts div { grid-template-columns: minmax(0, 1fr); } }
</style>
