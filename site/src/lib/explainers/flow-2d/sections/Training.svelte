<!-- flow-2d: the page's Training section. The kit's points its loss curve at the run's last frame, which here is an Euler
     sampling frame without a loss ("loss —"); this one shows the curve at the last training snapshot. Trains on request. -->
<script lang="ts">
	import type { StageProps } from '../../../atlas/types/view.ts';
	import type { AtlasStore } from '../../../atlas/ui/runtime.svelte.ts';
	import ViewSlot from '../../../atlas/ui/ViewSlot.svelte';
	import { atFor } from '../../../atlas/runtime/sources.ts';
	import { say } from '../../../atlas/ui/strings.ts';

	let { store }: { store: AtlasStore | null } = $props();
	const END = { phase: 'snapshot', last: true };
	let busy = $state(false);
	const stage = $derived.by((): StageProps | null => {
		void store?.version;
		const st = store?.r.sources.flow;
		const i = st?.tape && st.status === 'ready' ? st.tape.find(END) : -1;
		if (!store || !st || i < 0) return null;
		return { rt: store.r, source: 'flow', at: atFor(store.r, 'flow', i), focus: [], sel: {}, provenance: st.provenance, autoplay: false, onend: () => {}, props: {} };
	});
	async function train() {
		if (!store) return;
		busy = true;
		try {
			await store.rt.ensure('flow', END);
		} finally {
			busy = false;
			store.touch();
		}
	}
</script>

<p><button type="button" class="btn primary" onclick={train} disabled={busy || !store}>{busy ? say('computing') : say('runTraining')}</button></p>
{#if stage}<ViewSlot view="loss-curve" {stage} />{/if}
