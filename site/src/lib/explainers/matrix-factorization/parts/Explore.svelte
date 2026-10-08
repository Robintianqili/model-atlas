<!-- matrix-factorization: one of the package's views on the 'train' source run to its end, for the page's explore
     sections (Training: the map and the error curve; Output: the filled matrix). It reruns when the input changes. -->
<script lang="ts">
	import type { StageProps } from '../../../atlas/types/view.ts';
	import type { AtlasStore } from '../../../atlas/ui/runtime.svelte.ts';
	import ViewSlot from '../../../atlas/ui/ViewSlot.svelte';
	import { atFor } from '../../../atlas/runtime/sources.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { I18n } from '../../../atlas/types/core.ts';

	let { store, view, props, note }: { store: AtlasStore | null; view: string; props: Record<string, unknown>; note: I18n } = $props();
	$effect(() => {
		const s = store;
		if (!s) return;
		void s.inputGen;
		void s.rt.ensure('train', 'last').then(() => s.touch(), () => {});
	});
	const stage = $derived.by((): StageProps | null => {
		void store?.version;
		const tape = store?.r.sources.train?.tape;
		if (!store || !tape?.length || (tape as { done?: boolean }).done === false) return null;
		return { rt: store.r, source: 'train', at: atFor(store.r, 'train', tape.length - 1), focus: [], sel: {}, provenance: store.r.sources.train.provenance, autoplay: false, onend: () => {}, props, setInput: (raw) => store.setInput(raw) };
	});
</script>

<p class="muted">{t(note)}</p>
{#if stage}<ViewSlot {view} {stage} />{:else}<p class="muted">{t({ en: 'Training on your ratings…', zh: '正在用你的评分训练……' })}</p>{/if}
