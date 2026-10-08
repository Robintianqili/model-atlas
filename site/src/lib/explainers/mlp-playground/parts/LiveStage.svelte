<!-- mlp-playground: a view on one source run to `at` (default its end), for the explore sections; reruns on new input -->
<script lang="ts">
	import type { Cursor } from '../../../atlas/types/process.ts';
	import type { StageProps } from '../../../atlas/types/view.ts';
	import type { AtlasStore } from '../../../atlas/ui/runtime.svelte.ts';
	import ViewSlot from '../../../atlas/ui/ViewSlot.svelte';
	import { atFor } from '../../../atlas/runtime/sources.ts';
	import { resolveCursor } from '../../../atlas/runtime/selection.ts';
	import { say } from '../../../atlas/ui/strings.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { I18n } from '../../../atlas/types/core.ts';

	let { store, source, view, at = 'last', props = {}, note }: { store: AtlasStore | null; source: string; view: string; at?: Cursor; props?: Record<string, unknown>; note: I18n } = $props();
	$effect(() => {
		const s = store;
		if (!s) return;
		void s.inputGen;
		for (const name of [source, ...((props.compare as string[] | undefined) ?? [])]) void s.rt.ensure(name, name === source ? at : 'last').then(() => s.touch(), () => {});
	});
	// a new stage only when this source's tape changes (other sources' frames bump store.version too)
	let last: { tape: unknown; n: number; stage: StageProps } | null = null;
	const stage = $derived.by((): StageProps | null => {
		void store?.version;
		const tape = store?.r.sources[source]?.tape;
		if (!store || !tape?.length) return (last = null);
		if (last && last.tape === tape && last.n === tape.length) return last.stage;
		const at0 = atFor(store.r, source, resolveCursor(tape, at).index);
		last = { tape, n: tape.length, stage: { rt: store.r, source, at: at0, focus: [], sel: {}, provenance: store.r.sources[source].provenance, autoplay: false, onend: () => {}, props } };
		return last.stage;
	});
</script>

<p class="muted">{t(note)}</p>
{#if stage}<ViewSlot {view} {stage} />{:else}<p class="muted">{say(store?.r.pre?.empty ? 'emptyInput' : 'computing')}</p>{/if}
