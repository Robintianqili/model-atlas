<!-- tinymamba: the kit's text-char input box with the preset rows, plus "A new row from the training task", which draws a
     row with the same generator training used (task.ts; seeded, so the n-th press always gives the same row). -->
<script lang="ts">
	import TextChar from '../../../../atlas/adapters/text-char/Capture.svelte';
	import type { CaptureProps } from '../../../../atlas/ui/capture.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import { createRng } from '../../../../atlas/core/rng.ts';
	import { example, textOf } from '../../task.ts';

	let p: CaptureProps = $props();
	let n = $state(0);
	const fresh = () => p.onchange(textOf(example(createRng(900 + n++)).ids));
</script>

<div class="cap">
	<TextChar {...p} />
	{#if !p.compact}
		<p class="hint">
			{t({ en: 'Use dots, the letters a to h and one ? per letter you want back.', zh: '用点、字母 a 到 h，想要找回几个字母就加几个 ?。' })}
			<button type="button" class="btn" onclick={fresh}>{t({ en: 'A new row from the training task', zh: '从训练任务里换一行' })}</button>
		</p>
	{/if}
</div>

<style>
	.cap { display: grid; gap: 6px; min-width: 0; }
	.hint { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
</style>
