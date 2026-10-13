<!-- self-supervised-objectives 'chips' (meta.ts): the text as tokens; in 'pick' mode a click hides another token. -->
<script lang="ts">
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { chipsOf } from './meta.ts';

	let P: StageProps = $props();
	const C = $derived(chipsOf(P));
	const LABEL = {
		pick: { en: 'Click a token to hide it', zh: '点一个词元把它藏起来' },
		clm: { en: 'What TinyGPT may read', zh: 'TinyGPT 能读到的内容' },
		plain: { en: 'Your text as tokens', zh: '你的文字变成的词元' },
		train: { en: 'What training shows the model', zh: '训练时模型看到的内容' }
	} as const;
	function pick(q: number) {
		const k = C.toText[q];
		if (C.mode !== 'pick' || k === undefined || k < 0) return;
		const raw = { ...((P.rt.input ?? {}) as object), mask: k, picked: true };
		if (P.setInput) P.setInput(raw);
		else P.rt.setInput(raw);
	}
</script>

<div class="ch">
	<TokenChips tokens={C.items} selected={C.selected} marked={C.marked} selectable={C.mode === 'pick'} showIds={C.mode === 'pick' || C.mode === 'plain'} label={LABEL[C.mode as keyof typeof LABEL] ?? LABEL.pick} onselect={pick} />
	<p class="small">{t(C.caption)}</p>
</div>

<style>
	.ch { display: grid; gap: 10px; min-width: 0; }
	.small { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
</style>
