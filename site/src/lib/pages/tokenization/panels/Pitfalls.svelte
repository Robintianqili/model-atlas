<!--
  Pitfalls (redesign spec §5.9; a lazy panel of the tokenization page, #pitfalls): two misconception checks with live
  evidence and other traps (#pitfalls-list), then the cover question revealed and two checkpoint questions
  (#pitfalls-post, the old #post and #checkpoint). The language summary is computed here (LangCounts), so the evidence
  has its numbers without Explore being opened first.
-->
<script lang="ts">
	import Section from '$lib/kit/Section.svelte';
	import PredictReveal from '$lib/kit/PredictReveal.svelte';
	import type { PredictQuestion } from '$lib/kit/types';
	import Pitfalls from '../Pitfalls.svelte';
	import Quiz from '../Quiz.svelte';
	import { LangCounts } from '../langcounts.svelte';
	import type { TokPage } from '../page.svelte';

	let { vp }: { vp: { P: TokPage; prequestion: PredictQuestion }; section?: string } = $props();
	const P = $derived(vp.P);
	const counts = new LangCounts(() => ({ tiny: P.tiny!, trace: P.trace, k: P.k }));
</script>

<div class="pits">
	<Section id="pitfalls-list" title={{ en: 'Misconceptions and pitfalls', zh: '误区与陷阱' }}>
		{#if P.tiny}<Pitfalls tiny={P.tiny} summary={counts.summary} />{:else}<p class="muted">{P.loadError || '…'}</p>{/if}
	</Section>
	<Section id="pitfalls-post" title={{ en: 'Checkpoint', zh: '检查点' }}>
		<div class="post" id="checkpoint">
			<PredictReveal q={vp.prequestion} />
			<Quiz
				question={{ en: 'With no merges at all (the vocabulary is just the 256 bytes), how many tokens is 我爱你?', zh: '如果一次合并都没有（词表只有 256 个字节），“我爱你”是多少个词元？' }}
				options={[{ en: '3', zh: '3' }, { en: '6', zh: '6' }, { en: '9', zh: '9' }]}
				answer={2}
				explain={{
					en: 'Each Chinese character is 3 UTF-8 bytes, and with no merges every byte is its own token: 3 × 3 = 9. Check it in Explore: Part 1 (drag the vocabulary to 256) or TinyGPT’s byte view in Part 2.',
					zh: '每个汉字占 3 个 UTF-8 字节；没有合并时，每个字节各是一个词元：3 × 3 = 9。可以在“探索”里验证：第 1 部分（把词表拖到 256）或第 2 部分 TinyGPT 的字节视图。'
				}}
			/>
			<Quiz
				question={{ en: 'Why does BERT turn 爱 into [UNK], while GPT-2 never produces [UNK]?', zh: '为什么 BERT 会把“爱”变成 [UNK]，而 GPT-2 从不产生 [UNK]？' }}
				options={[
					{ en: 'BERT’s vocabulary has no piece for 爱 and nothing smaller to fall back on; GPT-2 can always fall back to single bytes.', zh: 'BERT 的词表里没有“爱”这个片段，也没有更小的单位可退；GPT-2 总能退回到单个字节。' },
					{ en: 'GPT-2 was trained on more Chinese text than BERT.', zh: 'GPT-2 训练时用的中文比 BERT 多。' },
					{ en: 'BERT lowercases the text, which removes Chinese characters.', zh: 'BERT 会把文本转成小写，这会删掉汉字。' }
				]}
				answer={0}
				explain={{
					en: 'WordPiece can only use pieces that are in its vocabulary; when a word cannot be spelled, the whole word becomes [UNK]. Byte-level BPE starts from all 256 bytes, so any text can be written, at worst one byte per token (GPT-2 writes 爱 as 3 byte pieces, one per byte; GPT-4’s cl100k needs 2).',
					zh: 'WordPiece 只能用词表里有的片段；一个词拼不出来时，整个词就变成 [UNK]。字节级 BPE 从全部 256 个字节出发，所以任何文本都写得出来，最坏也就是一个字节一个词元（GPT-2 把“爱”写成 3 个字节碎片，每个字节一个；GPT-4 的 cl100k 需要 2 个）。'
				}}
			/>
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
</style>
