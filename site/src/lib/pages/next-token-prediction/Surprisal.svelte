<!--
  Part 2b "The real thing": the learner's text colored by how surprised TinyGPT (final, fp32, live)
  was by each token: GLTR rank buckets or -log2 p. Hover / focus / tap a token for its arithmetic
  (semantic zoom: text -> one position's distribution -> -log2 p). The average becomes the loss and
  the perplexity, compared with TinyGPT's validation numbers.
-->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import Num from '$lib/kit/Num.svelte';
	import Term from '$lib/kit/Term.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import { sym } from '$lib/kit/tex';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtInt, fmtPct, visibleToken } from '$lib/kit/format';
	import { motion } from '$lib/kit/theme.svelte';
	import TryThis from './TryThis.svelte';
	import { bucketOf, type NtpState } from './state.svelte';
	import { FACTS, MAX_CHARS, PRESETS } from './facts';

	let { S }: { S: NtpState } = $props();

	let mode = $state<'rank' | 'bits'>('rank');
	const BITS_MAX = 14;
	const seq = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	const bitsScale = $derived(makeScale('sequential', [0, BITS_MAX], themeState.resolved));
	const BUCKET_POS = [0.06, 0.34, 0.62, 0.92];
	const BUCKETS = [
		{ en: 'top 1', zh: '第 1 名' },
		{ en: 'rank 2–10', zh: '第 2–10 名' },
		{ en: 'rank 11–100', zh: '第 11–100 名' },
		{ en: 'rank > 100', zh: '100 名之后' }
	];

	const sc = $derived(S.fresh ? S.score : null);
	const n = $derived(S.ids.length);

	function bg(i: number): string | null {
		if (!sc || i < 1 || i >= sc.p.length) return null;
		return mode === 'rank' ? seq.css(BUCKET_POS[bucketOf(sc.rank[i])]) : bitsScale.css(-Math.log2(sc.p[i]));
	}
	function ink(i: number): string | null {
		if (!sc || i < 1 || i >= sc.p.length) return null;
		return mode === 'rank' ? seq.ink(BUCKET_POS[bucketOf(sc.rank[i])]) : bitsScale.ink(-Math.log2(sc.p[i]));
	}

	const counts = $derived.by(() => {
		const c = [0, 0, 0, 0];
		if (sc) for (let i = 1; i < sc.rank.length; i++) c[bucketOf(sc.rank[i])]++;
		return c;
	});

	// the position shown in the detail panel: hover, else selection, else the most surprising token
	const worst = $derived.by(() => {
		if (!sc || sc.p.length < 2) return null;
		let w = 1;
		for (let i = 2; i < sc.p.length; i++) if (sc.p[i] < sc.p[w]) w = i;
		return w;
	});
	const pos = $derived.by(() => {
		const a = S.active;
		if (sc && a !== null && a >= 1 && a < sc.p.length) return a;
		return worst;
	});

	const detail = $derived.by(() => {
		if (!sc || pos === null) return null;
		const actual = sc.ids[pos];
		const items = sc.top[pos].map((x) => ({ label: S.disp(x.id), p: x.p, id: x.id }));
		if (!items.some((x) => x.id === actual)) items.push({ label: S.disp(actual), p: sc.p[pos], id: actual });
		const start = Math.max(0, pos - 8);
		return {
			actual,
			p: sc.p[pos],
			rank: sc.rank[pos],
			bits: -Math.log2(sc.p[pos]),
			nats: -Math.log(sc.p[pos]),
			entropy: sc.entropy[pos],
			items,
			ctx: (start > 0 ? '…' : '') + S.decode(sc.ids.slice(start, pos))
		};
	});

	// roving keyboard focus over the token spans
	let spans: HTMLElement[] = $state([]);
	function onKey(e: KeyboardEvent, i: number) {
		let j = i;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = Math.min(n - 1, i + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = Math.max(1, i - 1);
		else if (e.key === 'Home') j = 1;
		else if (e.key === 'End') j = n - 1;
		else return;
		e.preventDefault();
		S.focus = j;
		spans[j]?.focus();
	}
	const tabFor = (i: number) => (i === (S.focus !== null && S.focus >= 1 && S.focus < n ? S.focus : 1) ? 0 : -1);

	// perplexity ruler (log scale 1 .. 2048, values beyond are pinned to the right edge)
	const RULER_MAX = 2048;
	const rx = (v: number) => (Math.log(Math.min(Math.max(v, 1), RULER_MAX)) / Math.log(RULER_MAX)) * 100;
	const marks = $derived(
		[
			{ v: FACTS.valPpl, label: { en: 'TinyGPT on validation stories', zh: 'TinyGPT 在验证集故事上' }, kind: 'pre' },
			{ v: FACTS.vocab, label: { en: 'uniform guess (1,024)', zh: '均匀乱猜（1,024）' }, kind: 'ref' }
		] as { v: number; label: { en: string; zh: string }; kind: string }[]
	);

	let textBox: HTMLElement | undefined = $state();
	function setText(v: string) {
		S.text = v;
		S.focus = null;
		textBox?.scrollIntoView({ behavior: motion.reduced ? 'auto' : 'smooth', block: 'center' });
	}
	const tries = [
		{
			title: { en: 'A grammar slip', zh: '一个语法错误' },
			predict: { en: 'In “She were very happy because the sun was shining.”, which token will be darkest?', zh: '在“She were very happy because the sun was shining.”里，哪个词元颜色最深？' },
			action: { en: 'Load it', zh: '载入' },
			run: () => setText('She were very happy because the sun was shining.')
		},
		{
			title: { en: 'Say it twice', zh: '说两遍' },
			predict: { en: 'An odd sentence, repeated. TinyGPT has only 4 layers: will the second copy be much less surprising than the first?', zh: '一个奇怪的句子，重复两遍。TinyGPT 只有 4 层：第二遍会比第一遍明显不意外吗？' },
			action: { en: 'Load it', zh: '载入' },
			run: () => setText('The blue robot ate seven purple socks. The blue robot ate seven purple socks.')
		},
		{
			title: { en: 'A language it never read', zh: '它从没读过的语言' },
			predict: { en: 'Will TinyGPT’s loss on 我爱自然语言处理 be above or below the random-guess loss of 6.93 nats?', zh: 'TinyGPT 在“我爱自然语言处理”上的损失，会高于还是低于随机乱猜的 6.93 奈特？' },
			action: { en: 'Load it', zh: '载入' },
			run: () => setText('我爱自然语言处理')
		}
	];

	const lossTex = $derived(
		`\\begin{aligned} ${sym('out', '\\mathcal{L}')} &= \\frac{1}{N}\\sum_{i=1}^{N} -\\ln ${sym('prob', 'p_i')} \\\\ &= \\frac{ {{sum}} }{ ${sc ? sc.scored : 'N'} } = {{L}}\\ \\text{${t({ en: 'nats/token', zh: '奈特/词元' })}} \\end{aligned}`
	);
	const pplTex = `\\text{PPL} = e^{${sym('out', '\\mathcal{L}')}} = e^{ {{L}} } = {{ppl}}`;
	const tokTex = $derived(
		`\\begin{aligned} s &= -\\log_2 ${sym('prob', 'p')} = -\\log_2({{p}}) \\\\ &= {{s}}\\ \\text{${t({ en: 'bits', zh: '比特' })}} \\end{aligned}`
	);
</script>

<div class="real">
	<div class="prose">
		<Lang>
			{#snippet en()}
				<p>
					Now the real thing. TinyGPT (364,040 parameters, trained on children's stories) reads your text and, at every
					position, predicts the next token. Each token is colored by how surprised the model was to see it. In
					<strong>rank</strong> mode (the idea comes from GLTR) the color says where the real token was in the model's list
					of guesses; in <strong>bits</strong> mode it shows the surprisal −log₂ p. Hover, tap or use the arrow keys on a
					token to see what the model expected instead.
				</p>
			{/snippet}
			{#snippet zh()}
				<p>
					现在来看真家伙。TinyGPT（364,040 个参数，用儿童故事训练）读你的文本，在每个位置预测下一个词元。每个词元按模型看到它时有多“意外”来上色。<strong>排名</strong>模式（思路来自 GLTR）的颜色表示真实词元在模型猜测列表里排第几；<strong>比特</strong>模式显示意外度 −log₂ p。把鼠标移到某个词元上、点一下，或用方向键选中，就能看到模型原本期待的是什么。
				</p>
			{/snippet}
		</Lang>
	</div>

	<TextInputBar
		bind:value={S.text}
		maxLength={MAX_CHARS}
		label={{ en: 'Your text (shared with the whole page)', zh: '你的文本（全页共用）' }}
		presets={PRESETS}
	/>

	<div class="cols">
		<div class="left">
			<div class="bar-head">
				<span class="segmented" role="group" aria-label={t({ en: 'Color by', zh: '着色方式' })}>
					<button type="button" aria-pressed={mode === 'rank'} onclick={() => (mode = 'rank')}>{t({ en: 'rank (GLTR)', zh: '排名（GLTR）' })}</button>
					<button type="button" aria-pressed={mode === 'bits'} onclick={() => (mode = 'bits')}>{t({ en: 'bits −log₂ p', zh: '比特 −log₂ p' })}</button>
				</span>
				<ProvenanceBadge mode="live" precision="fp32" tolerance={FACTS.tolerance} against="PyTorch fp32" />
			</div>

			<div class="text" bind:this={textBox} role="listbox" aria-label={t({ en: 'Your text, colored by surprisal', zh: '按意外度着色的文本' })} aria-orientation="horizontal">
				{#each S.ids as id, i (i)}
					{@const s = S.disp(id)}
					{#if i === 0}
						<span
							class="t first"
							role="option"
							aria-selected="false"
							aria-disabled="true"
							title={t({ en: 'The first token has no context, so it is not scored.', zh: '第一个词元没有上下文，因此不计分。' })}>{s.replace(/\n/g, '↵\n')}</span
						>
					{:else}
						<span
							bind:this={spans[i]}
							class="t"
							class:sel={pos === i}
							style:background={bg(i)}
							style:color={ink(i)}
							role="option"
							aria-selected={S.focus === i}
							tabindex={tabFor(i)}
							aria-label={sc ? `${visibleToken(s)}: rank ${sc.rank[i]}, ${fmt3(-Math.log2(sc.p[i]))} bits` : visibleToken(s)}
							onpointerenter={() => (S.hover = i)}
							onpointerleave={() => (S.hover = null)}
							onclick={() => (S.focus = i)}
							onfocus={() => (S.focus = i)}
							onkeydown={(e) => onKey(e, i)}>{s.replace(/\n/g, '↵\n')}</span
						>
					{/if}
				{/each}
			</div>
			{#if !sc && n >= 2}<p class="pending muted small">{t({ en: 'TinyGPT is scoring your text…', zh: 'TinyGPT 正在给你的文本打分…' })}</p>{/if}

			<div class="legend small">
				{#if mode === 'rank'}
					{#each BUCKETS as b, k (k)}
						<span class="lg"><i class="sw" style:background={seq.css(BUCKET_POS[k])}></i>{t(b)} <span class="num muted">({counts[k]})</span></span>
					{/each}
				{:else}
					<span class="lg">0 {t({ en: 'bits', zh: '比特' })}</span>
					<span class="grad" style:background="linear-gradient(90deg, {bitsScale.css(0)}, {bitsScale.css(BITS_MAX / 2)}, {bitsScale.css(BITS_MAX)})"></span>
					<span class="lg">≥ {BITS_MAX}</span>
				{/if}
			</div>
			{#if sc && sc.scored > 0}
				<div class="hist" aria-label={t({ en: 'Share of tokens in each rank bucket', zh: '各排名区间的词元占比' })}>
					{#each counts as c, k (k)}
						{#if c > 0}<span class="hseg" style:flex-grow={c} style:background={seq.css(BUCKET_POS[k])} title="{t(BUCKETS[k])}: {c}"></span>{/if}
					{/each}
				</div>
				<p class="small muted hist-cap">
					{t({
						en: `${fmtPct(counts[0] / sc.scored, 0)} of your tokens were TinyGPT's first guess; ${fmtPct((counts[0] + counts[1]) / sc.scored, 0)} were in its top 10.`,
						zh: `你的词元里有 ${fmtPct(counts[0] / sc.scored, 0)} 正是 TinyGPT 的第一猜测；${fmtPct((counts[0] + counts[1]) / sc.scored, 0)} 在它的前 10 名里。`
					})}
				</p>
			{/if}
			{#if S.hasCjk}
				<p class="warn small">
					{t({
						en: 'TinyGPT was trained only on English children\'s stories. Its tokenizer has no Chinese tokens, so each Chinese character becomes 3 byte tokens it has almost never seen. Expect a loss far above a random guess: the model is confidently wrong.',
						zh: 'TinyGPT 只用英文儿童故事训练过。它的分词器里没有中文词元，每个汉字会变成 3 个它几乎从没见过的字节词元。损失会远高于随机乱猜：模型错得很自信。'
					})}
				</p>
			{/if}
		</div>

		<div class="right">
			{#if detail && pos !== null}
				<div class="zoom panel">
					<p class="eyebrow">{t({ en: `Position ${pos}: one prediction`, zh: `位置 ${pos}：一次预测` })}</p>
					<p class="ctxline">
						<span class="ctx">{visibleToken(detail.ctx)}</span><span class="act">{visibleToken(S.disp(detail.actual))}</span>
					</p>
					<ProbBars
						items={detail.items}
						k={6}
						tokens
						chosen={detail.actual}
						label={{ en: 'What TinyGPT expected', zh: 'TinyGPT 的预期' }}
					/>
					<p class="small muted">
						{t({ en: 'Highlighted: the token that really came next.', zh: '高亮的是真正出现的下一个词元。' })}
						{t({ en: 'Rank', zh: '排名' })} <span class="num">{detail.rank}</span> / 1,024 ·
						{t({ en: 'spread (entropy)', zh: '分散程度（熵）' })} <Num value={detail.entropy} /> {t({ en: 'bits', zh: '比特' })}
					</p>
					<FormulaCard tex={tokTex} values={{ p: detail.p, s: detail.bits }} caption={{ en: 'Surprisal of this token', zh: '这个词元的意外度' }} provenance="live" />
				</div>
			{:else if n < 2}
				<p class="muted">{t({ en: 'Type at least two tokens: the first token has nothing before it, so there is nothing to predict yet.', zh: '至少输入两个词元：第一个词元前面什么都没有，还没有东西可预测。' })}</p>
			{/if}
		</div>
	</div>

	{#if sc && sc.scored > 0}
		<div class="summary">
			<div class="stats">
				<div class="stat"><span class="k">{t({ en: 'tokens scored', zh: '计分词元' })}</span><span class="v num">{sc.scored}</span></div>
				<div class="stat"><span class="k">{t({ en: 'average surprisal', zh: '平均意外度' })}</span><span class="v"><Num value={sc.meanBits} /> <small>{t({ en: 'bits/token', zh: '比特/词元' })}</small></span></div>
				<div class="stat"><span class="k">{t({ en: 'loss', zh: '损失' })}</span><span class="v"><Num value={sc.meanNats} /> <small>{t({ en: 'nats/token', zh: '奈特/词元' })}</small></span></div>
				<div class="stat"><span class="k"><Term id="perplexity" /></span><span class="v"><Num value={sc.perplexity} /></span></div>
				<ProvenanceBadge mode="live" compact />
			</div>
			<div class="formulas">
				<FormulaCard
					tex={lossTex}
					values={{ sum: sc.meanNats * sc.scored, N: sc.scored, L: sc.meanNats }}
					caption={{ en: 'The loss is the average −ln p over your tokens', zh: '损失就是你的词元上 −ln p 的平均值' }}
					provenance="live"
				/>
				<FormulaCard
					tex={pplTex}
					values={{ L: sc.meanNats, ppl: sc.perplexity }}
					caption={{ en: 'Perplexity undoes the log', zh: '困惑度把对数还原回来' }}
					provenance="live"
				/>
			</div>

			<figure class="ruler">
				<figcaption class="small">
					{t({ en: 'Perplexity on a log scale (lower is better)', zh: '困惑度（对数刻度，越低越好）' }) }
				</figcaption>
				<div class="track">
					{#each [1, 4, 16, 64, 256, 1024] as v (v)}
						<span class="tk-mark num" style:left="{rx(v)}%">{v}</span>
					{/each}
					{#each marks as mk (mk.v)}
						<span class="mk {mk.kind}" class:flip={rx(mk.v) > 60} style:left="{rx(mk.v)}%"><span class="mlbl">{t(mk.label)}{#if mk.kind !== 'ref'}{' '}<span class="num">{fmt3(mk.v)}</span>{/if}</span></span>
					{/each}
					<span class="mk you" class:flip={rx(sc.perplexity) > 60} style:left="{rx(sc.perplexity)}%">
						<span class="mlbl">{t({ en: 'your text', zh: '你的文本' })} <span class="num">{fmt3(sc.perplexity)}</span>{#if sc.perplexity > RULER_MAX} →{/if}</span>
					</span>
				</div>
			</figure>
		</div>
	{/if}

	<TryThis items={tries} />

	<div class="prose val">
		<div class="valhead">
			<h3>{t({ en: 'What “validation perplexity 6.81” means', zh: '“验证困惑度 6.81”是什么意思' })}</h3>
			<ProvenanceBadge mode="precomputed" note={{ en: 'Measured offline in PyTorch on all 6,976,704 validation tokens.', zh: '离线用 PyTorch 在全部 6,976,704 个验证词元上测得。' }} />
		</div>
		<Lang>
			{#snippet en()}
				<p>
					The same calculation, run offline over every one of the {fmtInt(FACTS.valTokens)} tokens in TinyStories' validation
					stories (text TinyGPT never trained on), gives a loss of <strong>{FACTS.valLoss.toFixed(3)} nats per token</strong>. That
					is the number training pushed down. Its exponential is the perplexity: e<sup>{FACTS.valLoss.toFixed(3)}</sup> =
					<strong>{FACTS.valPpl.toFixed(2)}</strong>.
				</p>
				<p>
					One way to read it: on average TinyGPT is as unsure as if it were choosing evenly among about 7 tokens, out of 1,024.
					Equivalently, the typical (geometric-mean) probability it gives the right token is 1/{FACTS.valPpl.toFixed(2)} =
					{fmt3(1 / FACTS.valPpl)}. A model that knew nothing would score ln 1024 = 6.93 nats, perplexity 1,024. On a same-size
					slice of its training text the loss is {FACTS.trainLoss.toFixed(3)}, only 0.01 lower, so TinyGPT has not memorized
					its stories (<Term id="overfitting">no overfitting</Term>).
				</p>
				<p>
					Your text's number depends on your text: story-like English scores well below 6.81, unusual sentences far above it.
					A perplexity is only comparable between models that use the same tokenizer and the same test text.
				</p>
			{/snippet}
			{#snippet zh()}
				<p>
					同样的计算，离线跑遍 TinyStories 验证集故事里的全部 {fmtInt(FACTS.valTokens)} 个词元（TinyGPT 训练时从没见过这些文本），得到的损失是<strong>每个词元 {FACTS.valLoss.toFixed(3)} 奈特</strong>。训练要压低的就是这个数。对它取指数就是困惑度：e<sup>{FACTS.valLoss.toFixed(3)}</sup> = <strong>{FACTS.valPpl.toFixed(2)}</strong>。
				</p>
				<p>
					一种理解方式：平均来说，TinyGPT 的犹豫程度，相当于在 1,024 个词元里的约 7 个之间平均地挑一个。换句话说，它给正确词元的典型（几何平均）概率是 1/{FACTS.valPpl.toFixed(2)} = {fmt3(1 / FACTS.valPpl)}。什么都不懂的模型得分是 ln 1024 = 6.93 奈特，困惑度 1,024。在同样大小的训练文本切片上，损失是 {FACTS.trainLoss.toFixed(3)}，只低了 0.01，说明 TinyGPT 没有死记硬背它的故事（<Term id="overfitting">没有过拟合</Term>）。
				</p>
				<p>
					你的文本得到的数取决于你写了什么：像故事的英文会远低于 6.81，不寻常的句子会远高于它。只有使用同一个分词器、在同一份测试文本上算出的困惑度，才能互相比较。
				</p>
			{/snippet}
		</Lang>
	</div>
</div>

<style>
	.real {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
	}
	.prose {
		max-width: var(--measure);
	}
	.prose p {
		margin: 0 0 0.8em;
	}
	.cols {
		display: grid;
		grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
		gap: var(--s5);
		align-items: start;
	}
	@media (max-width: 860px) {
		.cols {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.left,
	.right {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.bar-head {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
		justify-content: space-between;
	}
	.text {
		padding: 14px 16px;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		font-family: var(--font-mono);
		font-size: var(--fs-md);
		line-height: 2;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.t {
		border-radius: 3px;
		padding: 2px 0;
		cursor: pointer;
		box-decoration-break: clone;
		-webkit-box-decoration-break: clone;
	}
	.t.first {
		cursor: help;
		outline: 1px dashed var(--rule-strong);
		outline-offset: -1px;
		color: var(--muted);
	}
	.t.sel {
		outline: 2px solid var(--highlight);
		outline-offset: 1px;
	}
	.t:focus-visible {
		outline: 3px solid var(--accent);
		outline-offset: 2px;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		align-items: center;
		color: var(--ink-2);
	}
	.lg {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}
	.sw {
		display: inline-block;
		width: 12px;
		height: 12px;
		border-radius: 3px;
		border: 1px solid var(--rule);
	}
	.grad {
		display: inline-block;
		width: 140px;
		height: 10px;
		border-radius: 3px;
		border: 1px solid var(--rule);
	}
	.hist {
		display: flex;
		gap: 2px;
		height: 10px;
		border-radius: 3px;
		overflow: hidden;
	}
	.hseg {
		display: block;
		flex-basis: 0;
		min-width: 3px;
	}
	.hist-cap,
	.pending {
		margin: 0;
	}
	.warn {
		margin: 0;
		padding: 8px 10px;
		border-left: 3px solid var(--illus);
		background: var(--illus-soft);
		border-radius: 0 var(--r1) var(--r1) 0;
	}
	.panel {
		padding: 14px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
	}
	.zoom {
		display: grid;
		gap: 10px;
	}
	.zoom .eyebrow {
		margin: 0;
	}
	.ctxline {
		margin: 0;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		overflow-wrap: anywhere;
	}
	.ctx {
		color: var(--ink-2);
		white-space: pre-wrap;
	}
	.act {
		white-space: pre;
		border: 1.5px solid var(--real);
		border-radius: 3px;
		padding: 0 3px;
		margin-left: 2px;
		font-weight: 700;
	}
	.small {
		font-size: var(--fs-xs);
	}
	.zoom p.small {
		margin: 0;
	}
	.summary {
		display: grid;
		gap: var(--s4);
		padding: var(--s4);
		border-radius: var(--r3);
		background: var(--surface-2);
	}
	.stats {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 10px 28px;
	}
	.stat {
		display: grid;
	}
	.stat .k {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.stat .v {
		font-size: var(--fs-xl);
		font-weight: 650;
	}
	.stat small {
		font-size: var(--fs-xs);
		font-weight: 400;
		color: var(--ink-2);
	}
	.formulas {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
		gap: 12px;
	}
	.ruler {
		margin: 0;
		display: grid;
		gap: 4px;
	}
	.track {
		position: relative;
		height: 76px;
		margin: 0 18px;
		border-bottom: 1px solid var(--rule-strong);
	}
	.tk-mark {
		position: absolute;
		bottom: -18px;
		transform: translateX(-50%);
		font-size: 10px;
		color: var(--muted);
	}
	.mk {
		position: absolute;
		bottom: 0;
		width: 0;
		height: 100%;
		border-left: 2px solid var(--muted);
	}
	.mk.pre {
		border-color: var(--pre);
		height: 64%;
	}
	.mk.ref {
		border-color: var(--rule-strong);
		height: 46%;
	}
	.mk.you {
		border-color: var(--highlight);
		height: 100%;
	}
	.mlbl {
		position: absolute;
		top: -2px;
		left: 4px;
		font-size: 11px;
		white-space: nowrap;
		color: var(--ink-2);
		background: color-mix(in srgb, var(--surface-2) 85%, transparent);
		padding: 0 3px;
		border-radius: 3px;
	}
	.mk.flip .mlbl {
		left: auto;
		right: 4px;
	}
	.mk.you .mlbl {
		color: var(--ink);
		font-weight: 650;
	}
	.mk.pre .mlbl {
		top: auto;
		bottom: 2px;
	}
	.val h3 {
		margin: 0;
		font-size: var(--fs-lg);
	}
	.valhead {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 14px;
		align-items: center;
		margin-bottom: 8px;
	}
</style>
