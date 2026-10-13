<!-- sasrec Capture (local adapter 'clicks'): start histories and your clicks in order; outside the compact copy beside
     the lesson also the whole shop to click in. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { PRESETS, toClicks, type Clicks } from '../../world.ts';
	import Shop from '../../parts/Shop.svelte';

	let { raw, onchange, compact = false }: { raw: unknown; onchange: (raw: Clicks) => void; compact?: boolean } = $props();
	const key = (a: unknown) => toClicks(a).clicks.join(',');
</script>

<div class="cap">
	<div class="row" role="group" aria-label={t({ en: 'Start histories', zh: '起始历史' })}>
		{#each PRESETS as x (x.id)}<button type="button" class="btn" aria-pressed={key(x.raw) === key(raw)} onclick={() => onchange(toClicks(x.raw))}>{t(x.label)}</button>{/each}
	</div>
	<Shop {raw} {onchange} grid={!compact} />
	{#if compact}<p>{t({ en: 'Add clicks in the stage of step 1, or in the shop below the lesson.', zh: '可以在第 1 步的舞台上或课程下方的商店里继续点击。' })}</p>{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); } .btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	p { margin: 0; font-size: var(--fs-xs); }
</style>
