<!--
  TinyCNN's Training view (redesign spec §5.9), loaded only when the view opens: how the 6,722 numbers were learned and
  how well they work (measured once, offline: precomputed), then the credits and licences (#credits opens them).
-->
<script lang="ts">
	import Section from '$lib/kit/Section.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { ui } from '$lib/i18n/i18n.svelte';
	import TrainingPanel from '../TrainingPanel.svelte';
	import Credits from '../Credits.svelte';
	import type { TinyCNNSpec } from '../data';

	let { vp }: { vp: { spec: TinyCNNSpec }; section?: string } = $props();
	const n = $derived(fmtInt(vp.spec.params));
</script>

<Section
	id="training-data"
	title={ui('secTraining')}
	intro={{
		en: `How these ${n} numbers were learned, and how well they work. These are measurements made once, offline.`,
		zh: `这 ${n} 个数是怎么学来的，效果如何。这些是离线测过一次的结果。`
	}}
	provenance="precomputed"
>
	<TrainingPanel spec={vp.spec} />
</Section>
<Section id="training-credits" title={ui('credits')}>
	<Credits />
</Section>
