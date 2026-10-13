<!--
  What the model reads (live): the learner's sentence as characters → UTF-8 bytes → TinyGPT tokens, the tokens as chips
  with their ids, and the ids themselves. The lesson's cover and Explore's first section show it.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import Chips from './Chips.svelte';
	import type { ChipItem } from './types';
	import type { TokPage } from './page.svelte';

	let { P }: { P: TokPage } = $props();

	const ids = $derived(P.tiny ? P.tiny.encode(P.sentence) : []);
	const items = $derived<ChipItem[]>(
		P.tiny
			? ids.map((id) => {
					const tk = P.tiny!;
					const d = tk.tokenToDisplay(id);
					return {
						text: d,
						id,
						kind: tk.isSpecial(id) ? 'special' : d.includes('<0x') ? 'partial' : undefined,
						hex: Array.from(tk.tokenBytes(id), (b) => b.toString(16).padStart(2, '0')).join(' ')
					};
				})
			: []
	);
	const nChars = $derived(Array.from(P.sentence).length);
	const nBytes = $derived(new TextEncoder().encode(P.sentence).length);
</script>

{#if P.tiny}
	<div class="reads">
		<p class="flow" aria-live="polite">
			<span><b class="num">{nChars}</b> {t({ en: 'characters', zh: '个字符' })}</span>
			<span class="arrow" aria-hidden="true">→</span>
			<span><b class="num">{nBytes}</b> {t({ en: 'UTF-8 bytes', zh: '个 UTF-8 字节' })}</span>
			<span class="arrow" aria-hidden="true">→</span>
			<span><b class="num big">{ids.length}</b> {t({ en: 'TinyGPT tokens', zh: '个 TinyGPT 词元' })}</span>
		</p>
		<Chips {items} showIds limit={600} label={{ en: 'TinyGPT tokens of your sentence', zh: '你的句子的 TinyGPT 词元' }} />
		<p class="ids"><span class="k">{t({ en: 'What the model gets:', zh: '模型拿到的：' })}</span> <span class="num">[{ids.join(', ')}]</span></p>
	</div>
{:else if P.loadError}
	<p class="err">{P.loadError}</p>
{:else}
	<p class="muted">{t({ en: 'Loading TinyGPT’s tokenizer (27 KB)…', zh: '正在加载 TinyGPT 的分词器（27 KB）…' })}</p>
{/if}

<style>
	.reads {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s3);
	}
	.flow {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 10px;
		color: var(--ink-2);
	}
	.arrow {
		color: var(--muted);
	}
	.big {
		font-size: var(--t-xl);
		color: var(--ink);
	}
	.ids {
		margin: 0;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--ink-2);
		overflow-wrap: anywhere;
	}
	.ids .num {
		white-space: normal;
	}
	.k {
		font-weight: var(--w-ui);
	}
	.err {
		color: var(--no);
	}
</style>
