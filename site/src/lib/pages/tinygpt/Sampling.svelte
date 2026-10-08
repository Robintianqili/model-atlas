<!--
  Output and sampling: the model's next-token probabilities (after the last token), the sampling
  distribution after temperature / top-k / top-p (recomputed instantly from the cached logits, no new
  forward pass), the seeded random draw, and generation one token at a time or automatically.
-->
<script lang="ts">
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import Num from '$lib/kit/Num.svelte';
	import Term from '$lib/kit/Term.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtPct, fmt3 } from '$lib/kit/format';
	import type { ProbItem } from '$lib/kit/types';
	import type { TGState } from './state.svelte';
	import { V } from './state.svelte';
	import { entropy, row, topIndices, type Pick } from './analysis';

	type Part = 'controls' | 'bars' | 'stats' | 'gen';
	type Knob = 'temperature' | 'topk' | 'topp' | 'seed';
	interface Props {
		S: TGState;
		dist: Float64Array | null;
		temperature: number;
		topK: number;
		topP: number;
		seed?: number;
		draws?: number;
		pick?: Pick | null;
		auto?: boolean;
		busy?: boolean;
		promptText?: string;
		genIds?: number[];
		ongenerate?: () => void;
		onauto?: () => void;
		onstop?: () => void;
		onreset?: () => void;
		onkeep?: () => void;
		onseed?: (s: number) => void;
		/** which pieces to show (the lesson shows controls, bars and stats); default all */
		parts?: Part[];
		/** which sliders to show; default all */
		knobs?: Knob[];
		/** rows per bar chart (default 10) */
		rows?: number;
		/** show each token's id next to it (default true; the lesson hides it, since "#" means rank there) */
		ids?: boolean;
	}
	const nop = () => {};
	let {
		S,
		dist,
		temperature = $bindable(),
		topK = $bindable(),
		topP = $bindable(),
		seed = 42,
		draws = 0,
		pick = null,
		auto = false,
		busy = false,
		promptText = '',
		genIds = [],
		ongenerate = nop,
		onauto = nop,
		onstop = nop,
		onreset = nop,
		onkeep = nop,
		onseed = nop,
		parts = ['controls', 'bars', 'stats', 'gen'],
		knobs = ['temperature', 'topk', 'topp', 'seed'],
		rows = 10,
		ids = true
	}: Props = $props();
	const has = (p: Part) => parts.includes(p);
	const knob = (k: Knob) => knobs.includes(k);

	const f = $derived(S.run?.forward);
	const T = $derived(S.T);
	const modelP = $derived(f ? row(f.t.probs.data, T - 1, V) : null);
	const order = $derived(modelP ? topIndices(modelP, rows) : []);
	const lab = (id: number) => S.tok?.tokenToDisplay(id) ?? String(id);
	const items = $derived<ProbItem[]>(modelP ? order.map((id) => (ids ? { label: lab(id), p: modelP[id], id } : { label: lab(id), p: modelP[id] })) : []);
	const sitems = $derived<ProbItem[]>(dist ? order.map((id) => (ids ? { label: lab(id), p: dist[id], id } : { label: lab(id), p: dist[id] })) : []);
	const kept = $derived(dist ? Array.from(dist).filter((x) => x > 0).length : V);
	const cutoffs = $derived.by(() => {
		const c: { afterRank: number; label: string }[] = [];
		if (temperature <= 0) c.push({ afterRank: 1, label: t({ en: 'greedy: only the top token', zh: '贪心：只取第一名' }) });
		else {
			if (topK > 0 && topK < rows) c.push({ afterRank: topK, label: `top-k = ${topK}` });
			if (topP < 1 && kept < rows && (topK === 0 || kept < topK)) c.push({ afterRank: kept, label: `top-p = ${topP.toFixed(2)}` });
		}
		return c;
	});
	const Hm = $derived(modelP ? entropy(modelP) : 0);
	const Hd = $derived(dist ? entropy(dist) : 0);
	// the bars show the distribution for the NEXT token; the last pick belongs to the previous one, so it is
	// described in words below, and only the greedy choice (what temperature 0 would take now) is highlighted
	const chosen = $derived(temperature <= 0 && order.length ? (ids ? order[0] : lab(order[0])) : null);
	const genText = $derived(S.tok && genIds.length ? S.tok.decode(genIds) : '');
</script>

{#if f && modelP}
	<div class="smp">
		{#if has('controls')}
		<div class="controls panel">
			{#if knob('temperature')}
			<label class="ctl">
				<span class="name"><Term id="temperature" /> <span class="num val">{temperature <= 0 ? t({ en: '0 (greedy)', zh: '0（贪心）' }) : temperature.toFixed(2)}</span></span>
				<input type="range" min="0" max="2" step="0.05" bind:value={temperature} aria-valuetext={temperature <= 0 ? 'greedy' : temperature.toFixed(2)} />
				<span class="help">{t({ en: 'divides the logits: < 1 sharper, > 1 flatter, 0 = always the top token', zh: '用来除 logits：< 1 更尖锐，> 1 更平坦，0 = 总选第一名' })}</span>
			</label>
			{/if}
			{#if knob('topk')}
			<label class="ctl">
				<span class="name">top-k <span class="num val">{topK === 0 ? t({ en: 'off', zh: '关' }) : topK}</span></span>
				<input type="range" min="0" max="50" step="1" bind:value={topK} disabled={temperature <= 0} />
				<span class="help">{t({ en: 'keep only the k most likely tokens (0 = off)', zh: '只保留最可能的 k 个词元（0 = 关闭）' })}</span>
			</label>
			{/if}
			{#if knob('topp')}
			<label class="ctl">
				<span class="name">top-p <span class="num val">{topP >= 1 ? t({ en: 'off', zh: '关' }) : topP.toFixed(2)}</span></span>
				<input type="range" min="0.05" max="1" step="0.01" bind:value={topP} disabled={temperature <= 0} />
				<span class="help">{t({ en: 'keep the most likely tokens until their probabilities add up to p (1 = off)', zh: '从最可能的词元开始保留，直到概率之和达到 p（1 = 关闭）' })}</span>
			</label>
			{/if}
			{#if knob('seed')}
			<label class="ctl seed">
				<span class="name">{t({ en: 'Random seed', zh: '随机种子' })}</span>
				<input type="number" min="0" max="999999" step="1" value={seed} onchange={(e) => onseed(Math.max(0, Math.floor(Number((e.currentTarget as HTMLInputElement).value) || 0)))} />
				<span class="help">{t({ en: 'same text + seed + settings = same story, in any browser', zh: '相同的文本、种子和设置 = 相同的故事，在任何浏览器里都一样' })}</span>
			</label>
			{/if}
		</div>
		{/if}

		{#if has('bars')}
		<div class="bars">
			<ProbBars items={items} sort={false} k={rows} tokens {chosen} label={{ en: 'Model probabilities (temperature 1)', zh: '模型的概率（温度 1）' }} />
			<ProbBars items={sitems} sort={false} k={rows} tokens {chosen} {cutoffs} label={{ en: 'Sampling distribution (your settings)', zh: '采样分布（按你的设置）' }} />
		</div>
		{/if}
		{#if has('stats')}
		<p class="stats muted">
			{t({ en: 'Next token after', zh: '接在' })} <span class="tok">{S.label(T - 1)}</span>{t({ en: '.', zh: ' 之后的下一个词元。' })}
			{t({ en: 'Spread (entropy):', zh: '分散程度（熵）：' })} <Num value={Hm} /> → <Num value={Hd} /> nats
			{t({ en: '(about', zh: '（约等于在' })} <Num value={Math.exp(Hm)} /> → <Num value={Math.exp(Hd)} /> {t({ en: 'equally likely choices).', zh: '个等可能的选项之间选择）。' })}
			{#if dist && kept < V}{t({ en: `Tokens kept: ${kept} of 1,024.`, zh: `保留的词元：1024 个中的 ${kept} 个。` })}{/if}
		</p>

		{/if}
		{#if has('gen')}
		<div class="genbox panel">
			<div class="buttons">
				<button type="button" class="btn primary" onclick={ongenerate} disabled={auto || busy}>{t({ en: 'Generate next token', zh: '生成下一个词元' })}</button>
				{#if auto}
					<button type="button" class="btn" onclick={onstop}>■ {t({ en: 'Stop', zh: '停止' })}</button>
				{:else}
					<button type="button" class="btn" onclick={onauto} disabled={busy}>▶ {t({ en: 'Auto-generate', zh: '自动生成' })}</button>
				{/if}
				<button type="button" class="btn ghost" onclick={onreset} disabled={auto || !genIds.length}>{t({ en: 'Remove generated', zh: '清除生成内容' })}</button>
				<button type="button" class="btn ghost" onclick={onkeep} disabled={auto || !genIds.length}>{t({ en: 'Use as new prompt', zh: '作为新的提示词' })}</button>
			</div>
			<p class="pick num" aria-live="polite">
				{#if pick}
					{#if pick.u === null}
						{t({ en: 'Temperature 0: took the most likely token', zh: '温度 0：直接取最可能的词元' })} <span class="tok gen">{S.idLabel(pick.token)}</span> ({fmtPct(pick.pModel)}). {t({ en: 'No random number used.', zh: '没有用随机数。' })}
					{:else}
						{t({ en: `Draw #${pick.draw} of seed ${pick.seed}: u =`, zh: `种子 ${pick.seed} 的第 ${pick.draw} 次抽取：u =` })} {pick.u.toFixed(4)}
						→ <span class="tok gen">{S.idLabel(pick.token)}</span>
						{t({ en: `(walking through the tokens in id order, its slice of the running sum is ${fmt3(pick.lo)}…${fmt3(pick.hi)}; p = ${fmtPct(pick.p)}, model p = ${fmtPct(pick.pModel)})`, zh: `（按 id 顺序累加概率时，它占据 ${fmt3(pick.lo)}…${fmt3(pick.hi)} 这一段；p = ${fmtPct(pick.p)}，模型原始 p = ${fmtPct(pick.pModel)}）` })}
					{/if}
				{:else}
					{t({ en: `Seed ${seed}, ${draws} random draws so far. Each sampled token uses the next number from the seeded generator.`, zh: `种子 ${seed}，目前已抽取 ${draws} 次。每采样一个词元，就从带种子的随机数生成器取下一个数。` })}
				{/if}
			</p>
			<p class="story">
				<span class="prompt">{promptText}</span><span class="genw">{genText}</span>{#if busy}<span class="cursor" aria-hidden="true">▍</span>{/if}
			</p>
			{#if genIds.length}
				<p class="muted small">
					{t({ en: `${genIds.length} generated token${genIds.length === 1 ? '' : 's'} (magenta). Every new token is appended and the whole model runs again on the last 64 tokens, so all views above update.`, zh: `已生成 ${genIds.length} 个词元（品红色）。每生成一个词元就把它接到末尾，整个模型在最后 64 个词元上重新运行，上面所有视图都会更新。` })}
				</p>
			{/if}
		</div>
		{/if}
	</div>
{/if}

<style>
	.smp {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 14px;
	}
	.controls {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
		gap: 12px 20px;
	}
	.ctl {
		display: grid;
		gap: 2px;
	}
	.name {
		font-weight: 650;
		font-size: var(--fs-sm);
		display: flex;
		justify-content: space-between;
		gap: 8px;
	}
	.val {
		font-weight: 500;
		color: var(--accent-strong);
	}
	.help {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.seed input {
		width: 9em;
		min-height: 36px;
		padding: 4px 8px;
		border: 1.5px solid var(--rule-strong);
		border-radius: var(--r2);
		background: var(--surface);
		font-family: var(--font-mono);
	}
	.bars {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr));
		gap: 16px 28px;
	}
	.stats {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.tok {
		font-family: var(--font-mono);
		background: var(--surface-2);
		border-radius: 3px;
		padding: 0 4px;
		white-space: pre;
		color: var(--ink);
	}
	.tok.gen {
		background: color-mix(in srgb, var(--generated) 16%, var(--surface));
		box-shadow: inset 0 0 0 1px var(--generated);
	}
	.genbox {
		display: grid;
		gap: 10px;
	}
	.buttons {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.pick {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		min-height: 1.5em;
	}
	.story {
		margin: 0;
		padding: 10px 12px;
		border-radius: var(--r2);
		background: var(--surface-2);
		font-family: var(--font-mono);
		font-size: 0.92rem;
		white-space: pre-wrap;
		word-break: break-word;
		max-height: 16em;
		overflow-y: auto;
	}
	.genw {
		color: var(--generated-text);
		font-weight: 600;
	}
	.cursor {
		color: var(--generated-text);
	}
	.small {
		font-size: var(--fs-xs);
		margin: 0;
	}
</style>
