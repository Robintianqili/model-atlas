<!--
  Lesson step 6: encode new text with the learned merges (BPETokenizer.encodeWithSteps, the same code
  TinyGPT uses). Source: the learner's own merges (BPETokenizer.fromMerges) or TinyGPT's 767.
  Pick a pre-token; a StepPlayer walks its merges: every adjacent pair with its merge number, the
  lowest number wins, until no pair was ever learned. Below: the whole new text as token ids.
-->
<script lang="ts">
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtInt, visibleToken } from '$lib/kit/format';
	import type { BPETokenizer } from '$lib/models/bpe';
	import Chips from '../Chips.svelte';
	import Tok from '../Tok.svelte';
	import type { ChipItem } from '../types';
	import { plainToken } from '../lesson';

	type Src = 'yours' | 'tiny';
	interface Props {
		yours: BPETokenizer | null;
		tiny: BPETokenizer;
		/** which merges (bindable) */
		src?: Src;
		sentence: string;
		/** pair count when each of YOUR merges was learned (trace.steps[r].count) */
		yourCounts: number[];
		/** pair count in TinyStories when each of TinyGPT's merges was learned (precomputed) */
		tinyCounts: number[];
		autoplay?: boolean;
		onend?: () => void;
	}
	let { yours, tiny, src = $bindable('yours'), sentence, yourCounts, tinyCounts, autoplay = false, onend }: Props = $props();

	const tok = $derived(src === 'tiny' || !yours ? tiny : yours);
	const es = $derived(tok.encodeWithSteps(sentence));
	const nYours = $derived(yours ? yours.encode(sentence).length : 0);
	const nTiny = $derived(tiny.encode(sentence).length);
	const disp = (id: number) => tok.tokenToDisplay(id);
	const hex = (id: number) => Array.from(tok.tokenBytes(id), (b) => b.toString(16).padStart(2, '0')).join(' ');

	let sel = $state<number | null>(null);
	let userPicked = $state(false);
	let step = $state(0);
	// default pre-token: the one with the most merges (re-chosen when the text or the merges change, unless picked)
	$effect(() => {
		const n = es.pretokens.length;
		if (!n) {
			sel = null;
			return;
		}
		if (!userPicked || sel === null || sel >= n) {
			let best = 0;
			es.pretokens.forEach((p, i) => {
				if (p.steps.length > es.pretokens[best].steps.length) best = i;
			});
			sel = best;
		}
	});
	$effect(() => {
		void sel;
		void sentence;
		void src;
		step = 0;
	});
	const pt = $derived(sel !== null ? es.pretokens[sel] : null);
	const clampStep = (s: number) => (pt ? Math.max(0, Math.min(s, pt.steps.length)) : 0);
	const stateAt = (s0: number) => {
		const s = clampStep(s0);
		return pt ? (s === 0 ? pt.bytes : pt.steps[s - 1].tokens) : [];
	};
	const labels = $derived(
		pt
			? [
					{ en: `Start: ${pt.bytes.length} bytes, one token each`, zh: `起点：${pt.bytes.length} 个字节，各是一个词元` },
					...pt.steps.map((s) => ({
						en: `Merge #${s.rank + 1}: “${plainToken(disp(s.pair[0]))}” + “${plainToken(disp(s.pair[1]))}” → “${plainToken(disp(s.newId))}” (ID ${s.newId})`,
						zh: `第 ${s.rank + 1} 次合并：“${plainToken(disp(s.pair[0]))}” + “${plainToken(disp(s.pair[1]))}” → “${plainToken(disp(s.newId))}”（ID ${s.newId}）`
					}))
				]
			: []
	);
	const candidates = $derived.by(() => {
		const ids = stateAt(step);
		const out: { a: number; b: number; r: number | undefined }[] = [];
		for (let i = 0; i < ids.length - 1; i++) out.push({ a: ids[i], b: ids[i + 1], r: tok.rank(ids[i], ids[i + 1]) });
		return out;
	});
	const minRank = $derived(Math.min(...candidates.map((c) => c.r ?? Infinity)));

	// the whole new text as tokens (the selected pre-token's tokens marked)
	const flat = $derived.by(() => {
		const out: { id: number; pt: number }[] = [];
		es.pretokens.forEach((p, pi) => {
			for (const id of p.ids) out.push({ id, pt: pi });
		});
		return out;
	});
	const items = $derived<ChipItem[]>(
		flat.map((f, i) => ({
			text: disp(f.id),
			id: f.id,
			gap: i > 0 && flat[i - 1].pt !== f.pt,
			hex: hex(f.id),
			kind: tok.isSpecial(f.id) ? 'special' : disp(f.id).includes('<0x') ? 'partial' : undefined
		}))
	);
	const marks = $derived(flat.map((f) => (f.pt === sel ? 'link' : null)));
	const isZh = (s: string) => /[一-鿿]/.test(s);

	function pickKey(e: KeyboardEvent, i: number) {
		const n = es.pretokens.length;
		let j = i;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = Math.min(n - 1, i + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = Math.max(0, i - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = n - 1;
		else return;
		e.preventDefault();
		userPicked = true;
		sel = j;
		(e.currentTarget as HTMLElement).parentElement?.querySelectorAll<HTMLElement>('.pt')[j]?.focus();
	}
	function setSrc(s: Src) {
		src = s;
		userPicked = false;
	}
</script>

<div class="ls">
	<div class="srcs" role="group" aria-label={t({ en: 'Which merges', zh: '使用哪套合并规则' })}>
		<button type="button" class="sb" aria-pressed={src === 'yours'} onclick={() => setSrc('yours')} disabled={!yours}>
			<span class="sw" style:background="var(--cat-0)"></span>
			<span class="st">{t({ en: 'Your merges', zh: '你的合并规则' })}<span class="muted num">{t({ en: ` (${yours ? yours.merges.length : 0})`, zh: `（${yours ? yours.merges.length : 0}）` })}</span></span>
			<span class="sn"><b class="num">{fmtInt(nYours)}</b> {t({ en: 'tokens', zh: '个词元' })}</span>
		</button>
		<button type="button" class="sb" aria-pressed={src === 'tiny'} onclick={() => setSrc('tiny')}>
			<span class="sw" style:background="var(--cat-1)"></span>
			<span class="st">{t({ en: 'TinyGPT’s merges', zh: 'TinyGPT 的合并规则' })}<span class="muted num">{t({ en: ' (767)', zh: '（767）' })}</span></span>
			<span class="sn"><b class="num">{fmtInt(nTiny)}</b> {t({ en: 'tokens', zh: '个词元' })}</span>
		</button>
	</div>

	<h4>{t({ en: 'Pick a pre-token of the new text', zh: '在新文本里选一个预分词片段' })} <span class="muted sub">{t({ en: '(under each: bytes B → tokens)', zh: '（下面的小字：字节数 B → 词元数）' })}</span></h4>
	<div class="pts" role="radiogroup" aria-label={t({ en: 'Pre-tokens of the new text', zh: '新文本的预分词片段' })}>
		{#each es.pretokens as p, i (i)}
			<button
				type="button"
				class="pt"
				role="radio"
				aria-checked={sel === i}
				tabindex={sel === i ? 0 : -1}
				onclick={() => {
					userPicked = true;
					sel = i;
				}}
				onkeydown={(e) => pickKey(e, i)}
			>
				<span class="ptext" lang={isZh(p.text) ? 'zh-CN' : undefined}>{p.special ? p.text : visibleToken(p.text)}</span>
				<span class="pmeta num">{p.special ? '' : `${p.bytes.length} B → `}{p.ids.length}</span>
			</button>
		{/each}
		{#if !es.pretokens.length}<span class="muted">{t({ en: 'The new text is empty.', zh: '新文本是空的。' })}</span>{/if}
	</div>

	{#if pt}
		{#if pt.special}
			<p class="hint">{t({ en: 'This is TinyGPT’s special token <|endoftext|>: it is matched as a whole and never split.', zh: '这是 TinyGPT 的特殊词元 <|endoftext|>：整体匹配，从不拆分。' })}</p>
		{:else}
			{#key `${src}|${sel}|${pt.text}`}
				<StepPlayer count={pt.steps.length + 1} bind:step {labels} interval={1100} {autoplay} {onend} label={{ en: 'Inside this pre-token: the lowest merge number wins', zh: '片段内部：编号最小的合并先做' }}>
					{#snippet children(s0)}
						{@const s = clampStep(s0)}
						{@const ids = stateAt(s)}
						{@const made = s > 0 ? pt.steps[s - 1].newId : -1}
						<div class="stage">
							<div class="row-toks">
								{#each ids as id, j (j)}
									<span class="tk"><Tok text={disp(id)} mark={id === made ? 'new' : null} /><span class="num tid">{id}</span></span>
								{/each}
							</div>
							<div class="cands">
								<span class="muted small">{t({ en: 'Neighbouring pairs now, with their merge number (— = never learned):', zh: '当前的相邻对及其合并编号（— 表示从没学过）：' })}</span>
								{#each candidates as c, j (j)}
									<span class="cand" class:best={c.r !== undefined && c.r === minRank}>
										<Tok text={disp(c.a)} small /><Tok text={disp(c.b)} small />
										<span class="num">{c.r === undefined ? '—' : `#${c.r + 1}`}</span>
									</span>
								{/each}
								{#if !candidates.length}<span class="small">{t({ en: 'one token left: done', zh: '只剩一个词元：结束' })}</span>{/if}
								{#if candidates.length && minRank === Infinity}<span class="small done">{t({ en: 'no pair here was ever learned: done', zh: '这里没有学过的对：结束' })}</span>{/if}
							</div>
							{#if s > 0}
								{@const stp = pt.steps[s - 1]}
								{@const cnt = src === 'tiny' ? tinyCounts[stp.rank] : yourCounts[stp.rank]}
								<p class="why">
									{t({ en: 'Merged', zh: '合并了' })} <Tok text={disp(stp.pair[0])} small /> + <Tok text={disp(stp.pair[1])} small />{t({ en: ' because merge', zh: '：它的编号' })} <b class="num">#{stp.rank + 1}</b> {t({ en: 'was learned before every other pair here.', zh: '比这里其他任何一对都学得早。' })}
									{#if cnt !== undefined}
										{#if src === 'tiny'}
											{t({ en: 'TinyGPT learned it when the pair occurred', zh: 'TinyGPT 学到它时，这一对在 TinyStories 里出现了' })} <b class="num">{fmtInt(cnt)}</b> {t({ en: 'times in TinyStories.', zh: '次。' })}
											<ProvenanceBadge mode="precomputed" compact note={{ en: 'Pair count recorded when TinyGPT’s tokenizer was trained (tokenizer.json, mergeCounts).', zh: '训练 TinyGPT 分词器时记录下的相邻对计数（tokenizer.json 中的 mergeCounts）。' }} />
										{:else}
											{#if cnt === 1}
												{t({ en: 'You learned it when the pair occurred only once in your training text.', zh: '你学到它时，这一对在你的训练文本里只出现了 1 次。' })}
											{:else}
												{t({ en: 'You learned it when the pair occurred', zh: '你学到它时，这一对在你的训练文本里出现了' })} <b class="num">{fmtInt(cnt)}</b> {t({ en: 'times in your training text.', zh: '次。' })}
											{/if}
										{/if}
									{/if}
								</p>
							{/if}
						</div>
					{/snippet}
				</StepPlayer>
			{/key}
		{/if}
	{/if}

	{#if items.length}
		<div class="res">
			<p class="rh"><b class="num big">{fmtInt(items.length)}</b> {t({ en: 'tokens: what the model would receive', zh: '个词元：模型收到的就是这些' })}</p>
			<Chips {items} {marks} showIds limit={300} label={{ en: 'The new text as tokens', zh: '新文本的词元' }} onselect={(i) => { userPicked = true; sel = flat[i]?.pt ?? sel; }} />
		</div>
	{/if}
</div>

<style>
	.ls {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.srcs {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 210px), 1fr));
		gap: 6px;
	}
	.sb {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		align-items: center;
		gap: 8px;
		padding: 8px 12px;
		border: 1.5px solid var(--rule-strong);
		border-radius: var(--r2);
		background: var(--surface);
		color: inherit;
		text-align: left;
		cursor: pointer;
	}
	.sb[aria-pressed='true'] {
		border-color: var(--accent);
		background: var(--accent-soft);
		box-shadow: 0 0 0 1px var(--accent);
	}
	.sb:disabled {
		opacity: 0.5;
		cursor: default;
	}
	.sw {
		width: 10px;
		height: 10px;
		border-radius: 2px;
	}
	.st {
		font-weight: 650;
		font-size: var(--fs-sm);
	}
	.sn {
		font-size: var(--fs-sm);
		white-space: nowrap;
	}
	.pts {
		display: flex;
		flex-wrap: wrap;
		gap: 5px;
		max-height: 150px;
		overflow-y: auto;
		padding: 2px;
	}
	.pt {
		display: grid;
		gap: 1px;
		justify-items: start;
		padding: 3px 8px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		color: inherit;
		text-align: left;
		max-width: 100%;
		min-width: 0;
		cursor: pointer;
	}
	.pt:hover {
		background: var(--surface-2);
	}
	.pt[aria-checked='true'] {
		border-color: var(--accent);
		box-shadow: 0 0 0 1px var(--accent);
		background: var(--accent-soft);
	}
	.ptext {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		white-space: pre;
		max-width: 100%;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.pmeta {
		font-size: 10.5px;
		color: var(--muted);
	}
	.stage {
		display: grid;
		gap: 8px;
		padding: 10px 12px;
		border-radius: var(--r2);
		background: var(--surface-2);
		min-width: 0;
	}
	.row-toks {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.tk {
		display: inline-flex;
		flex-direction: column;
		align-items: center;
	}
	.tid {
		font-size: 10px;
		color: var(--muted);
	}
	.cands {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 10px;
	}
	.cand {
		display: inline-flex;
		align-items: center;
		gap: 2px;
		padding: 1px 4px;
		border-radius: 4px;
		font-size: var(--fs-xs);
	}
	.cand.best {
		outline: 2px solid var(--accent);
	}
	.sub {
		font-family: var(--font-body);
		font-size: var(--fs-xs);
		font-weight: 400;
	}
	.small {
		font-size: var(--fs-xs);
	}
	.done {
		color: var(--live);
		font-weight: 650;
	}
	.why,
	.hint {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.res {
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	.rh {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.big {
		font-size: var(--fs-lg);
		color: var(--ink);
	}
</style>
