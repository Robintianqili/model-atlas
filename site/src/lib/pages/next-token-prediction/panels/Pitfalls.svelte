<!--
  Pitfalls (redesign spec §5.9; a lazy panel of the next-token-prediction page, #pitfalls): the misconception checks with
  live evidence on the learner's sentence (#pitfalls-list), then the cover question revealed and three checks
  (#pitfalls-post, the old #post and #checkpoint).
-->
<script lang="ts">
	import Section from '$lib/kit/Section.svelte';
	import PredictReveal from '$lib/kit/PredictReveal.svelte';
	import type { PredictQuestion } from '$lib/kit/types';
	import type { Text } from '$lib/i18n/i18n.svelte';
	import Pitfalls from '../Pitfalls.svelte';
	import Quiz from '../Quiz.svelte';
	import type { NtpState } from '../state.svelte';

	type Check = { id: string; question: Text; options: Text[]; answer: number; explain: Text };
	let { vp }: { vp: { S: NtpState; prequestion: PredictQuestion; checks: Check[] }; section?: string } = $props();
</script>

<div class="pits">
	<Section id="pitfalls-list" title={{ en: 'Misconceptions and pitfalls', zh: '误区与陷阱' }}>
		<Pitfalls S={vp.S} />
	</Section>
	<Section id="pitfalls-post" title={{ en: 'Check yourself', zh: '检验一下' }}>
		<div class="post" id="checkpoint">
			<PredictReveal q={vp.prequestion} />
			<div class="checks">
				{#each vp.checks as q (q.id)}<Quiz {q} />{/each}
			</div>
		</div>
	</Section>
</div>

<style>
	.pits {
		display: grid;
		gap: var(--s7);
	}
	.post {
		display: grid;
		gap: var(--s4);
	}
	.checks {
		display: grid;
		gap: var(--s4);
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
		align-items: start;
	}
</style>
