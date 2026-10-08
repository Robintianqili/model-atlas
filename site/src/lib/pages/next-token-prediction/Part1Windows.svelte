<!--
  Part 1 "One sentence is many lessons": the learner's tokens become (context -> next token)
  training examples. A StepPlayer walks the sliding window; the staircase on the right shows all
  examples at once (it is the causal mask). The live probability TinyGPT gives each label is
  shown for the current example.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import { base } from '$app/paths';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import Term from '$lib/kit/Term.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { visibleToken, fmtInt } from '$lib/kit/format';
	import type { NtpState } from './state.svelte';
	import { FACTS, MAX_CHARS, MAX_TOKENS, PRESETS } from './facts';

	let { S }: { S: NtpState } = $props();

	const n = $derived(S.ids.length);
	const count = $derived(Math.max(1, n - 1));
	let step = $state(0);

	// the example the player shows <-> the page-wide selected token (focus = label position = step + 1)
	$effect(() => {
		const c = count;
		if (untrack(() => step) > c - 1) step = c - 1;
	});
	$effect(() => {
		const f = S.focus;
		if (f !== null && f >= 1 && f - 1 < count && f - 1 !== untrack(() => step)) step = f - 1;
	});

	/** first token the model sees when it predicts position pos (64-token windows, stride 32) */
	const ctxStart = (pos: number) => (pos < FACTS.context ? 0 : 32 * Math.floor((pos - 32) / 32));

	const disp = (id: number) => visibleToken(S.disp(id));
	const ctxText = (k: number) => {
		const ids = S.ids.slice(ctxStart(k + 1), k + 1);
		const s = S.decode(ids);
		return s.length > 46 ? '…' + s.slice(s.length - 45) : s;
	};
	const labels = $derived(
		n >= 2 ? Array.from({ length: n - 1 }, (_, k) => `“${ctxText(k).replace(/\n/g, '↵')}”  →  ${disp(S.ids[k + 1])}`) : []
	);

	function cls(i: number, k: number): string {
		const start = ctxStart(k + 1);
		if (i < start) return 'cut';
		if (i <= k) return 'ctx';
		if (i === k + 1) return 'tgt';
		return 'fut';
	}

	// the player's step can briefly point past the end when the text gets shorter (the effect above clamps it
	// after the next render), so everything that indexes S.ids uses a clamped copy
	const cur = $derived(Math.max(0, Math.min(step, count - 1)));
	const livePos = $derived(cur + 1);
	const liveP = $derived(S.fresh && S.score && livePos < S.score.p.length ? S.score.p[livePos] : null);

	// staircase geometry
	let boxW = $state(320);
	const cell = $derived(Math.max(3, Math.min(14, Math.floor((Math.min(boxW, 520) - 2) / Math.max(1, n)))));
	const gridW = $derived(cell * Math.max(1, n));
	const gridH = $derived(cell * Math.max(1, n - 1));
	const rows = $derived(Array.from({ length: Math.max(0, n - 1) }, (_, k) => k));

	function pickRow(e: MouseEvent) {
		const svg = e.currentTarget as SVGSVGElement;
		const r = svg.getBoundingClientRect();
		const k = Math.floor(((e.clientY - r.top) / r.height) * (n - 1));
		if (k >= 0 && k < n - 1) {
			step = k;
			S.focus = k + 1;
		}
	}
</script>

<div class="p1">
	<div class="prose">
		<Lang>
			{#snippet en()}
				<p>
					Nobody labels the text a language model learns from. So where does the training signal come from? From the
					text itself. Cut a sentence into <Term id="token">tokens</Term>, then at every position ask: <em>given the
					tokens so far, which token comes next?</em> The answer is already in the sentence. That is
					<Term id="next-token-prediction" />, and it is how GPT-style models are <Term id="pre-training">pre-trained</Term>.
					It is a form of <Term id="self-supervised">self-supervised learning</Term>: the data supplies its own
					<Term id="label">labels</Term>.
				</p>
				<p>
					Type any sentence. Every token after the first becomes the answer to one small exercise. (The tokens are TinyGPT's
					own: a byte-pair encoder with 1,024 of them. How text is cut into tokens has its own page:
					<a href="{base}/train/tokenization/">Tokenization</a>.)
				</p>
			{/snippet}
			{#snippet zh()}
				<p>
					语言模型学习用的文本，并没有人去给它打标签。那训练信号从哪里来？就来自文本本身。把一句话切成<Term id="token">词元</Term>，然后在每个位置都问一遍：<em>已知前面的词元，下一个是哪个？</em>答案就在句子里。这就是<Term id="next-token-prediction" />，GPT 一类的模型正是这样<Term id="pre-training">预训练</Term>的。它属于<Term id="self-supervised">自监督学习</Term>：数据自己提供了<Term id="label">标签</Term>。
				</p>
				<p>
					随便输入一句话。第一个词元之后的每一个词元，都是一道小练习的答案。（这里用的是 TinyGPT 自己的词元：一个有 1,024
					个词元的字节对编码器。文本如何被切成词元，见单独的一页：<a href="{base}/train/tokenization/">分词</a>。）
				</p>
			{/snippet}
		</Lang>
	</div>

	<TextInputBar
		bind:value={S.text}
		maxLength={MAX_CHARS}
		label={{ en: 'Your text (used by every part of this page)', zh: '你的文本（本页所有部分共用）' }}
		presets={PRESETS}
	/>

	{#if S.tok}
		<p class="tcount">
			{#if n === 0}
				{t({ en: 'Type something to begin.', zh: '先输入一点文字。' })}
			{:else}
				<strong class="num">{n}</strong>
				{t({ en: n === 1 ? 'token' : 'tokens', zh: '个词元' })} →
				<strong class="num">{Math.max(0, n - 1)}</strong>
				{t({ en: n - 1 === 1 ? 'training example' : 'training examples', zh: '个训练样本' })}
				{#if S.cut > 0}
					<span class="muted">
						· {t({ en: `only the first ${MAX_TOKENS} tokens are used (${S.cut} more were cut)`, zh: `只用前 ${MAX_TOKENS} 个词元（截掉了 ${S.cut} 个）` })}
					</span>
				{/if}
			{/if}
		</p>
	{:else if S.error}
		<p class="err">{S.error}</p>
	{:else}
		<p class="muted">{t({ en: 'Loading the tokenizer…', zh: '正在加载分词器…' })}</p>
	{/if}

	{#if n >= 2}
		<div class="grid">
			<StepPlayer
				{count}
				bind:step
				{labels}
				interval={900}
				label={{ en: 'The sliding window', zh: '滑动窗口' }}
				onstep={(s) => (S.focus = s + 1)}
			>
				{#snippet children(k0)}
					{@const k = Math.max(0, Math.min(k0, n - 2))}
					<div class="stage">
						<p class="ex num">
							{t({ en: `Example ${k + 1} of ${n - 1}`, zh: `第 ${k + 1} 个样本，共 ${n - 1} 个` })}
						</p>
						<div class="sent" aria-hidden="true">
							{#each S.ids as id, i (i)}
								<!-- hover only links views (the chips are aria-hidden; the StepPlayer is the keyboard path) -->
								<!-- svelte-ignore a11y_no_static_element_interactions -->
								<span
									class="tk {cls(i, k)}"
									class:hov={S.hover === i}
									onpointerenter={() => (S.hover = i >= 1 ? i : null)}
									onpointerleave={() => (S.hover = null)}>{disp(id)}</span
								>
							{/each}
						</div>
						<div class="pair">
							<div class="side">
								<span class="lab">{t({ en: 'Input: the context', zh: '输入：上下文' })}</span>
								<span class="ctxt">{visibleToken(ctxText(k))}</span>
								<span class="muted small num">
									{k + 1 - ctxStart(k + 1)}
									{t({ en: k + 1 - ctxStart(k + 1) === 1 ? 'token' : 'tokens', zh: '个词元' })}{#if ctxStart(k + 1) > 0}
										· {t({ en: `starts at token ${ctxStart(k + 1)} (64-token window)`, zh: `从第 ${ctxStart(k + 1)} 个词元开始（64 词元窗口）` })}{/if}
								</span>
							</div>
							<span class="arrow" aria-hidden="true">→</span>
							<div class="side">
								<span class="lab">{t({ en: 'Label: the next token', zh: '标签：下一个词元' })}</span>
								<span class="tk tgt big">{disp(S.ids[k + 1])}</span>
							</div>
						</div>
						<p class="live">
							{#if liveP !== null}
								{t({ en: 'TinyGPT gives this label', zh: 'TinyGPT 给这个标签的概率' })}
								<span class="sym-p">p</span> = <Num value={liveP} />
								<span class="muted">({t({ en: 'loss', zh: '损失' })} −ln p = <Num value={-Math.log(liveP)} />
									{t({ en: 'nats', zh: '奈特' })})</span>
								<ProvenanceBadge mode="live" compact />
							{:else}
								<span class="muted">{t({ en: 'TinyGPT is scoring your text…', zh: 'TinyGPT 正在给你的文本打分…' })}</span>
							{/if}
						</p>
					</div>
				{/snippet}
			</StepPlayer>

			<figure class="stair" bind:clientWidth={boxW}>
				<figcaption>
					{t({ en: 'All examples at once', zh: '所有样本一览' })}
					<span class="muted small">
						({t({ en: 'rows = examples, columns = tokens', zh: '行 = 样本，列 = 词元' })})
					</span>
				</figcaption>
				<!-- the StepPlayer is the keyboard path; a click on a row is a pointer shortcut -->
				<!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
				<svg width={gridW} height={gridH} viewBox="0 0 {gridW} {gridH}" onclick={pickRow} role="presentation">
					{#each rows as k (k)}
						{@const s0 = ctxStart(k + 1)}
						{#if s0 > 0}<rect class="cutc" x={0} y={k * cell} width={s0 * cell} height={cell} />{/if}
						<rect class="ctxc" class:cur={k === cur} x={s0 * cell} y={k * cell} width={(k + 1 - s0) * cell} height={cell} />
						<rect class="tgtc" x={(k + 1) * cell} y={k * cell} width={cell} height={cell} />
					{/each}
					<rect class="row" x="0.5" y={cur * cell + 0.5} width={gridW - 1} height={cell - 1} />
				</svg>
				<div class="legend small">
					<span><i class="sw ctxc"></i>{t({ en: 'context the model sees', zh: '模型看到的上下文' })}</span>
					<span><i class="sw tgtc"></i>{t({ en: 'label (next token)', zh: '标签（下一个词元）' })}</span>
					{#if n > FACTS.context}<span><i class="sw cutc"></i>{t({ en: 'outside the 64-token window', zh: '在 64 词元窗口之外' })}</span>{/if}
				</div>
			</figure>
		</div>

		<div class="prose after">
			<Lang>
				{#snippet en()}
					<p>
						The staircase is the whole lesson plan for your sentence: row <em>k</em> is one example, its shaded part
						is the context and the green cell is the label. A Transformer does not run these
						{fmtInt(n - 1)} examples one by one. It reads the sentence once, and a <Term id="causal-mask" /> stops
						each position from looking ahead, so all rows are computed in one pass. The staircase is exactly
						that mask.
					</p>
					<p>
						TinyGPT's <Term id="context-window" /> is {FACTS.context} tokens. In training it read windows of 64 tokens
						from {fmtInt(FACTS.trainTokens)} tokens of children's stories, so each training step (512 windows)
						held 32,768 of these examples. Over 30,000 steps it saw {fmtInt(FACTS.tokensSeen)} labels.
					</p>
				{/snippet}
				{#snippet zh()}
					<p>
						这个阶梯图就是你这句话的全部“练习题”：第 <em>k</em> 行是一个样本，带底色的部分是上下文，绿色格子是标签。Transformer 并不会把这 {fmtInt(n - 1)} 个样本一个一个地算。它把整句话读一遍，用<Term id="causal-mask" />禁止每个位置往后看，于是所有行在一次前向计算里同时得到。这个阶梯正是那个掩码。
					</p>
					<p>
						TinyGPT 的<Term id="context-window" />是 {FACTS.context} 个词元。训练时，它从 {fmtInt(FACTS.trainTokens)}
						个词元的儿童故事里截取 64 个词元长的窗口，所以每个训练步（512 个窗口）包含 32,768 个这样的样本。30,000 步下来，它一共见过 {fmtInt(FACTS.tokensSeen)} 个标签。
					</p>
				{/snippet}
			</Lang>
		</div>
	{/if}
</div>

<style>
	.p1 {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
	}
	.prose p {
		margin: 0 0 0.8em;
	}
	.tcount {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.err {
		color: var(--k-text);
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
		gap: var(--s5);
		align-items: start;
	}
	@media (max-width: 760px) {
		.grid {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.stage {
		display: grid;
		gap: 10px;
		padding: 14px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		min-width: 0;
	}
	.ex {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--muted);
		letter-spacing: 0.04em;
		text-transform: uppercase;
	}
	.sent {
		display: flex;
		flex-wrap: wrap;
		gap: 3px 2px;
	}
	.tk {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		padding: 1px 4px;
		border-radius: 4px;
		border: 1.5px solid transparent;
		white-space: pre;
		transition:
			background-color var(--dur-2) var(--ease),
			opacity var(--dur-2) var(--ease),
			border-color var(--dur-2) var(--ease);
	}
	.tk.ctx {
		background: var(--accent-soft);
		color: var(--ink);
	}
	.tk.cut {
		background: repeating-linear-gradient(135deg, var(--surface-2) 0 3px, var(--surface) 3px 6px);
		color: var(--muted);
	}
	.tk.tgt {
		border-color: var(--real);
		background: color-mix(in srgb, var(--real) 16%, var(--surface));
		color: var(--ink);
		font-weight: 700;
	}
	.tk.fut {
		color: var(--muted);
		border-style: dashed;
		border-color: var(--rule);
	}
	.tk.hov {
		outline: 2px solid var(--highlight);
		outline-offset: 1px;
	}
	.tk.big {
		font-size: var(--fs-md);
		padding: 3px 8px;
		justify-self: start;
	}
	.pair {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto auto;
		align-items: center;
		gap: 10px;
		padding: 10px;
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.side {
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	.lab {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		font-weight: 650;
	}
	.ctxt {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.arrow {
		font-size: var(--fs-xl);
		color: var(--muted);
	}
	@media (max-width: 480px) {
		.pair {
			grid-template-columns: minmax(0, 1fr);
		}
		.arrow {
			transform: rotate(90deg);
			justify-self: start;
		}
	}
	.live {
		margin: 0;
		font-size: var(--fs-sm);
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 6px;
	}
	.sym-p {
		color: var(--seq-hi);
		font-style: italic;
		font-family: var(--font-display);
	}
	.small {
		font-size: var(--fs-xs);
	}
	.stair {
		margin: 0;
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.stair figcaption {
		font-size: var(--fs-sm);
		font-weight: 650;
	}
	.stair svg {
		display: block;
		max-width: 100%;
		height: auto;
		cursor: pointer;
	}
	.ctxc {
		fill: var(--accent-soft);
		stroke: var(--surface);
		stroke-width: 0.5;
	}
	.ctxc.cur {
		fill: color-mix(in srgb, var(--accent) 45%, var(--accent-soft));
	}
	.tgtc {
		fill: var(--real);
	}
	.cutc {
		fill: var(--surface-3);
	}
	.row {
		fill: none;
		stroke: var(--highlight);
		stroke-width: 1;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		color: var(--ink-2);
	}
	.sw {
		display: inline-block;
		width: 11px;
		height: 11px;
		border-radius: 2px;
		margin-right: 5px;
		vertical-align: -1px;
	}
	.sw.ctxc {
		background: var(--accent-soft);
		border: 1px solid var(--accent);
	}
	.sw.tgtc {
		background: var(--real);
	}
	.sw.cutc {
		background: var(--surface-3);
	}
	.after {
		max-width: var(--measure);
	}
</style>
