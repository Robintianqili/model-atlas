<!-- char-rnn: the page's Training section, with labels in both languages (the auto section prints the raw metric keys). -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { AtlasSpec } from '../../../atlas/types/index.ts';

	let { spec }: { spec?: AtlasSpec | null } = $props();
	const tr = $derived((spec?.training ?? {}) as { steps?: number; chars?: number; metrics?: { val_loss?: number; val_bpc?: number } });
	const n = (v: number | undefined, d = 3) => (v === undefined ? '…' : v.toLocaleString('en', { maximumFractionDigits: d }));
	const rows = $derived([
		[{ en: 'Training text', zh: '训练文本' }, { en: `${n(tr.chars)} characters from the start of TinyStories’ train split`, zh: `TinyStories 训练集开头的 ${n(tr.chars)} 个字符` }],
		[{ en: 'Training', zh: '训练' }, { en: `${n(tr.steps)} steps, each on 256 windows of 128 characters (AdamW)`, zh: `${n(tr.steps)} 步，每步 256 段、每段 128 个字符（AdamW）` }],
		[{ en: 'Loss on held-out stories', zh: '在没见过的故事上的损失' }, { en: `${n(tr.metrics?.val_loss)} nats per character (${n(tr.metrics?.val_bpc)} bits per character)`, zh: `每个字符 ${n(tr.metrics?.val_loss)} 奈特（${n(tr.metrics?.val_bpc)} 比特）` }],
		[{ en: 'Parameters', zh: '参数' }, { en: n(spec?.params), zh: n(spec?.params) }]
	]);
</script>

<dl class="tr">
	{#each rows as [k, v], i (i)}<div><dt>{t(k)}</dt><dd>{t(v)}</dd></div>{/each}
</dl>

<style>
	.tr {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: 12px 24px;
		margin: 0;
	}
	dt {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	dd {
		margin: 2px 0 0;
	}
</style>
