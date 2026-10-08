<!--
  /m/tinygpt/ — TinyGPT, a 364,040-parameter GPT, explained on the learner's own text.
  Every number is computed live in the browser (Web Worker) from the trained fp32 weights;
  the forward pass matches PyTorch within 1e-4. Page parts live in src/lib/pages/tinygpt/.
-->
<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { base } from '$app/paths';
	import ModelPageShell from '$lib/kit/ModelPageShell.svelte';
	import FactStrip from '$lib/kit/FactStrip.svelte';
	import Article from '$lib/kit/Article.svelte';
	import CodeBlock from '$lib/kit/CodeBlock.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { takeHandoff } from '$lib/kit/handoff';
	import { motion } from '$lib/kit/theme.svelte';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import { fmtBytes, fmtInt } from '$lib/kit/format';
	import type { ModelSpec } from '$lib/kit/spec';
	import type { PageProps } from './$types';
	import type { PredictQuestion } from '$lib/kit/types';
	import type { Engine } from '$lib/pages/tinygpt/engine';
	import type { TinyGPTSpec } from '$lib/models/tinygpt';
	import { TGState, V } from '$lib/pages/tinygpt/state.svelte';
	import { DEFAULT_TEXT } from '$lib/pages/tinygpt/presets';
	import { argmax, pickToken, row, type Pick, type HeadMetric } from '$lib/pages/tinygpt/analysis';
	import { BLOCK_CODE, USAGE_CODE } from '$lib/pages/tinygpt/code';
	import InputPanel from '$lib/pages/tinygpt/InputPanel.svelte';
	import Overview from '$lib/pages/tinygpt/Overview.svelte';
	import ParamTable from '$lib/pages/tinygpt/ParamTable.svelte';
	import InspectorPanel from '$lib/pages/tinygpt/InspectorPanel.svelte';
	import BlockView from '$lib/pages/tinygpt/BlockView.svelte';
	import AttnExplorer from '$lib/pages/tinygpt/AttnExplorer.svelte';
	import LogitLens from '$lib/pages/tinygpt/LogitLens.svelte';
	import Sampling from '$lib/pages/tinygpt/Sampling.svelte';
	import TryThis from '$lib/pages/tinygpt/TryThis.svelte';
	import Checkpoint from '$lib/pages/tinygpt/Checkpoint.svelte';
	import ArticleEn from '$lib/pages/tinygpt/ArticleEn.svelte';
	import ArticleZh from '$lib/pages/tinygpt/ArticleZh.svelte';
	import Lesson from '$lib/pages/tinygpt/Lesson.svelte';

	let { data }: PageProps = $props();

	const S = new TGState();
	/** the guided lesson's own selection (followed token, head, key, …); shares the run with S */
	const LS = new TGState(S);
	// this page's catalog entry and area name, from the root layout's server load (no catalog module in this page's JS)
	const entry = $derived(data.catalog?.model);
	const area = $derived(data.catalog?.area);
	const EOT = 1023;

	let eng: Engine | null = null;
	let spec = $state.raw<TinyGPTSpec | null>(null);
	let loadState = $state<'loading' | 'ready' | 'error'>('loading');
	let loadError = $state('');
	let fns = $state.raw<{
		samplingDistribution: (l: ArrayLike<number>, temp: number, k?: number, p?: number) => Float64Array;
		mulberry32: (seed: number) => () => number;
	} | null>(null);

	// ---------------------------------------------------------------- input
	let text = $state(DEFAULT_TEXT);
	let promptIds = $state.raw<number[]>([]);
	let genIds = $state.raw<number[]>([]);
	const allIds = $derived([...promptIds, ...genIds]);

	// ---------------------------------------------------------------- runs (Web Worker)
	let wanted: number[] = [];
	let running: Promise<void> | null = null;
	let busy = $state(false);
	let debounce: ReturnType<typeof setTimeout> | undefined;

	function pump(): Promise<void> {
		if (running) return running;
		busy = true;
		const p = (async () => {
			try {
				for (;;) {
					const ids = wanted;
					if (!eng || ids.length === 0) {
						S.run = null;
						S.labels = [];
						break;
					}
					const r = await eng.run(ids);
					if (ids !== wanted) continue; // the input changed meanwhile: run the newest
					S.labels = r.forward.ids.map((id) => S.idLabel(id));
					S.run = r;
					break;
				}
			} catch (err) {
				loadError = err instanceof Error ? err.message : String(err);
			}
		})();
		// cleared after the assignment below, also when the body finished synchronously (empty input)
		running = p;
		void p.finally(() => {
			if (running === p) {
				running = null;
				busy = false;
			}
		});
		return p;
	}
	function runNow(): Promise<void> {
		clearTimeout(debounce);
		wanted = allIds;
		return pump().then(() => (wanted !== allIds ? runNow() : undefined));
	}
	function schedule(ms = 60) {
		clearTimeout(debounce);
		debounce = setTimeout(() => void runNow(), ms);
	}

	function setPrompt(v: string, immediate = false): Promise<void> {
		auto = false;
		const before = allIds;
		text = v;
		promptIds = S.tok ? S.tok.encode(v) : [];
		genIds = [];
		S.keepSelection(before, promptIds);
		LS.keepSelection(before, promptIds);
		resetRng();
		if (immediate) return runNow();
		schedule();
		return Promise.resolve();
	}

	// ---------------------------------------------------------------- sampling and generation
	let temperature = $state(1);
	let topK = $state(0);
	let topP = $state(1);
	let seed = $state(42);
	let draws = $state(0);
	let rng: (() => number) | null = null;
	let pick = $state<Pick | null>(null);
	let auto = $state(false);

	function resetRng() {
		rng = fns ? fns.mulberry32(seed) : null;
		draws = 0;
		pick = null;
	}
	const lastLogits = $derived(S.run ? row(S.run.forward.t.logits.data, S.T - 1, V) : null);
	const modelProbs = $derived(S.run ? row(S.run.forward.t.probs.data, S.T - 1, V) : null);
	const dist = $derived.by(() => {
		if (!lastLogits || !fns) return null;
		if (temperature <= 0) {
			const d = new Float64Array(V);
			d[argmax(lastLogits)] = 1;
			return d;
		}
		return fns.samplingDistribution(lastLogits, temperature, topK, topP);
	});

	async function genStep(): Promise<number> {
		if (!S.run) return -1;
		// sample from the distribution of exactly the current text (finish a pending or debounced run first)
		if (running || wanted !== allIds) await runNow();
		if (!S.run || !dist || !lastLogits || !modelProbs) return -1;
		let token: number;
		let u: number | null = null;
		let lo = 0;
		let hi = 1;
		if (temperature <= 0) token = argmax(lastLogits);
		else {
			if (!rng) resetRng();
			u = rng!();
			draws += 1;
			({ token, lo, hi } = pickToken(dist, u));
		}
		pick = { token, u, lo, hi, p: dist[token], pModel: modelProbs[token], draw: draws, seed };
		genIds = [...genIds, token];
		await runNow();
		return token;
	}
	async function autoGenerate() {
		auto = true;
		let n = 0;
		while (auto && n < 80) {
			const tok = await genStep();
			n++;
			if (tok < 0 || tok === EOT) break;
			if (!motion.reduced) await new Promise((r) => setTimeout(r, 140));
		}
		auto = false;
	}
	/** the lesson's generation: append one token it picked (with its own settings) and run the model */
	function appendToken(token: number): Promise<void> {
		auto = false;
		pick = null;
		genIds = [...genIds, token];
		return runNow();
	}
	/** wait until the run matches the current text (a debounced or pending run finishes first) */
	async function settle(): Promise<void> {
		if (running || wanted !== allIds) await runNow();
	}
	function removeGenerated() {
		auto = false;
		genIds = [];
		resetRng();
		void runNow();
	}
	function keepAsPrompt() {
		if (!S.tok) return;
		const v = text + S.tok.decode(genIds);
		setPrompt([...v].slice(-400).join(''), true);
	}
	function onSeed(s: number) {
		seed = s;
		resetRng();
	}
	async function compare(n: number, hot: number, sd: number) {
		if (!eng || !S.tok || !allIds.length) return null;
		const a = await eng.generate(allIds, { maxNew: n, temperature: 0, stopAtEot: true });
		const b = await eng.generate(allIds, { maxNew: n, temperature: hot, seed: sd, stopAtEot: true });
		return { greedy: S.tok.decode(a.result.newIds), hot: S.tok.decode(b.result.newIds), ms: a.ms + b.ms };
	}

	// ---------------------------------------------------------------- navigation helpers
	let blockStep = $state(0);
	let metric = $state<HeadMetric>('prev');
	async function goto(id: string) {
		await tick(); // let the views render the new run first, so the target does not move
		const el = document.getElementById(id);
		if (!el) return;
		el.scrollIntoView({ behavior: motion.reduced ? 'auto' : 'smooth', block: 'start' });
		el.classList.remove('flash');
		void el.offsetWidth;
		el.classList.add('flash');
		setTimeout(() => el.classList.remove('flash'), 1800);
		// if something above changed size during a smooth scroll, correct the landing once
		if (!motion.reduced)
			setTimeout(() => {
				const top = el.getBoundingClientRect().top;
				const want = parseFloat(getComputedStyle(el).scrollMarginTop) || 0;
				if (Math.abs(top - want) > 40) el.scrollIntoView({ behavior: 'auto', block: 'start' });
			}, 900);
	}
	function openBlock(b: number) {
		S.block = b;
		void goto('unit');
	}

	// ---------------------------------------------------------------- load
	onMount(() => {
		let disposed = false;
		(async () => {
			try {
				const mod = await import('$lib/pages/tinygpt/engine');
				const url = new URL(`${base}/models/tinygpt/`, location.href).href;
				const e = await mod.loadEngine(url);
				if (disposed) {
					e.dispose();
					return;
				}
				eng = e;
				fns = { samplingDistribution: mod.samplingDistribution, mulberry32: mod.mulberry32 };
				S.model = e.model;
				S.tok = e.tok;
				spec = e.spec;
				let start = DEFAULT_TEXT;
				try {
					const q = new URLSearchParams(location.search).get('text');
					const h = takeHandoff('text');
					if (q) start = q;
					else if (h?.text) start = h.text;
				} catch {
					/* ignore */
				}
				setPrompt([...start].slice(0, 400).join(''), true);
				loadState = 'ready';
			} catch (err) {
				loadError = err instanceof Error ? err.message : String(err);
				loadState = 'error';
			}
		})();
		return () => {
			disposed = true;
			auto = false;
			clearTimeout(debounce);
			eng?.dispose();
		};
	});

	// ---------------------------------------------------------------- page text
	const goals = [
		{ en: 'Follow one token from its id through the embeddings, 4 blocks and the output head to a probability, with every number computed live.', zh: '跟踪一个词元：从它的 id 出发，经过嵌入、4 个块和输出头，直到变成概率——每个数字都实时计算。' },
		{ en: 'Explain attention with real numbers: queries and keys, the 1/√19 scaling, the causal mask, softmax and the weighted sum of values.', zh: '用真实数字解释注意力：查询与键、1/√19 缩放、因果掩码、softmax，以及对值的加权求和。' },
		{ en: 'Use the logit lens and the sampling controls to see when the model decides, and how temperature, top-k and top-p change what it writes.', zh: '用 logit lens 和采样控件，看模型何时做出决定，以及温度、top-k、top-p 如何改变它写出的内容。' }
	];
	const prequestion: PredictQuestion = {
		id: 'tinygpt-where-decided',
		question: {
			en: 'TinyGPT reads “Once upon a time, there was a little” and must guess the next token. Where does that guess take shape?',
			zh: 'TinyGPT 读到 “Once upon a time, there was a little”，要猜下一个词元。这个猜测是在哪里成形的？'
		},
		options: [
			{ en: 'Only in the attention weights of the last block', zh: '只在最后一个块的注意力权重里' },
			{ en: 'Step by step: each attention and MLP step adds to the residual stream, and the guess sharpens over the blocks', zh: '一步步成形：每个注意力和 MLP 步骤都往残差流里加东西，猜测随着块的推进越来越清晰' },
			{ en: 'Only in the final softmax; the numbers before it mean nothing yet', zh: '只在最后的 softmax 里；之前的数字还没有任何意义' }
		],
		answer: 1,
		explain: {
			en: 'Check the logit lens in the Output section: reading the residual stream after each step with the final LayerNorm and output head shows the guess forming over several steps, and both attention and MLP steps move it.',
			zh: '看“输出”部分的 logit lens：用最终层归一化和输出头读取每一步之后的残差流，可以看到猜测在好几步中逐渐成形，注意力和 MLP 步骤都会推动它。'
		}
	};
	const factList = $derived([
		{ label: ui('area'), value: area ?? entry?.area ?? '—' },
		{ label: ui('family'), value: entry?.family ?? '—' },
		{ label: ui('params'), value: fmtInt(364040), mono: true },
		{ label: ui('download'), value: fmtBytes(data.tinygptBytes), mono: true },
		{ label: { en: 'Shape', zh: '结构' }, value: { en: '4 blocks · 4 heads · d 76 · context 64 · vocab 1,024', zh: '4 块 · 4 头 · d 76 · 上下文 64 · 词表 1024' }, mono: true },
		{
			label: ui('runtime'),
			value: S.run
				? S.run.where === 'worker'
					? { en: `own JS forward pass in a Web Worker · ${Math.round(S.run.ms)} ms`, zh: `自研 JS 前向计算（Web Worker）· ${Math.round(S.run.ms)} 毫秒` }
					: { en: `own JS forward pass on the main thread · ${Math.round(S.run.ms)} ms`, zh: `自研 JS 前向计算（主线程）· ${Math.round(S.run.ms)} 毫秒` }
				: { en: 'own JS forward pass in a Web Worker', zh: '自研 JS 前向计算（Web Worker）' }
		},
		{ label: ui('precision'), value: 'fp32', mono: true }
	]);
	const parityNote = {
		en: 'Every one of the 66 traced tensors matches PyTorch within 1e-4 (the largest difference measured on fresh prompts was 4.2e-5, at a full 64-token context).',
		zh: '66 个被追踪的张量全部与 PyTorch 相差在 1e-4 以内（在新提示词上实测的最大差异为 4.2e-5，出现在满 64 个词元的上下文时）。'
	};
	const trainingMetrics = $derived((spec?.training as { metrics?: Record<string, number> } | undefined)?.metrics);
</script>

<ModelPageShell
	slug="tinygpt"
	spec={spec as unknown as ModelSpec | null}
	downloadBytes={data.tinygptBytes}
	provenance="live"
	{goals}
	{prequestion}
	lessonNote={{ en: '12 short steps from your text to the next token, with the real numbers', zh: '12 个小步骤，从你的文本一直到下一个词元，每一步都是真实的数' }}
	exploreIntro={{
		en: 'The whole model at once: every tensor, all 16 attention heads, the logit lens for every position, sampling and experiments. Click anything.',
		zh: '整个模型一次摆开：每个张量、全部 16 个注意力头、每个位置的 logit lens、采样和各种实验。点哪里都可以。'
	}}
	titles={{ unit: { en: 'Inside a block', zh: '块的内部' }, output: { en: 'Output: logit lens and sampling', zh: '输出：logit lens 与采样' } }}
	intros={{
		input: { en: 'Type the start of a story. The text is split into tokens on every keystroke and the whole model runs again on your tokens.', zh: '输入一个故事的开头。每敲一个键，文本都会重新切分成词元，整个模型在你的词元上重新运行。' },
		lesson: {
			en: 'Follow your text through TinyGPT one step at a time. Press Next (or the → key), or Play to let the lesson run; each view uses your text and the real model.',
			zh: '一步一步跟着你的文本穿过 TinyGPT。点“下一步”（或按 → 键），或点“自动播放”让课程自己走；每个视图都用你的文本和真实的模型。'
		},
		overview: { en: 'The whole model at a glance, drawn from your text. Each thumbnail is the real tensor (rows = your tokens); the outlined row is the token you are following.', zh: '一眼看完整个模型，全部由你的文本画出。每张缩略图都是真实的张量（行 = 你的词元）；带框的行是你正在跟踪的词元。' },
		inspector: { en: 'Any of the 66 tensors the model computed, with values, statistics, the arithmetic and the code. Use [ and ] to step through them.', zh: '模型算出的 66 个张量中的任意一个：数值、统计、算式和代码。用 [ 和 ] 键逐个浏览。' },
		unit: { en: 'Zoom into one block: play the steps, then explore all 16 attention heads and redo one attention score by hand.', zh: '放大一个块：播放各个步骤，再查看全部 16 个注意力头，并亲手重算一个注意力分数。' },
		output: { en: 'When does the model decide, and how does it pick the next token?', zh: '模型在什么时候做出决定？它又是怎样挑出下一个词元的？' }
	}}
	badges={{ input: 'live', lesson: 'live', overview: 'live', unit: 'live', output: 'live', training: 'precomputed' }}
>
	{#snippet facts()}
		<FactStrip facts={factList} provenance="live" precision="fp32" tolerance={1e-4} against="PyTorch fp32" />
		<p class="parity-note">{t(parityNote)}</p>
	{/snippet}

	{#snippet input()}
		{#if loadState === 'error'}
			<p class="err" role="alert">{t({ en: 'The model could not be loaded:', zh: '模型加载失败：' })} {loadError}</p>
		{/if}
		<InputPanel {S} bind:text {promptIds} {genIds} oninput={(v) => setPrompt(v)} />
		{#if loadState === 'loading'}<p class="muted">{t({ en: 'Loading TinyGPT (1.5 MB)…', zh: '正在加载 TinyGPT（1.5 MB）…' })}</p>{/if}
	{/snippet}

	{#snippet lesson()}
		<Lesson
			{S}
			{LS}
			bind:text
			{promptIds}
			{genIds}
			{busy}
			{fns}
			setPrompt={(v) => void setPrompt(v)}
			append={appendToken}
			{settle}
			clearGenerated={removeGenerated}
		/>
	{/snippet}

	{#snippet overview()}
		<Overview {S} onopen={openBlock} />
		{#if spec}
			<details class="params">
				<summary>{t({ en: 'Where the 364,040 parameters are', zh: '364,040 个参数分布在哪里' })}</summary>
				<ParamTable {spec} />
			</details>
		{/if}
	{/snippet}

	{#snippet inspector()}
		{#if spec}
			<InspectorPanel {S} spec={spec as unknown as ModelSpec} />
		{:else}
			<p class="muted">{t({ en: 'Loading…', zh: '加载中…' })}</p>
		{/if}
	{/snippet}

	{#snippet unit()}
		<BlockView {S} bind:step={blockStep} />
		<div id="attention" class="sub">
			<h3>{t({ en: 'All 16 attention heads', zh: '全部 16 个注意力头' })}</h3>
			<p class="muted">
				{t({
					en: 'Rows are blocks, columns are heads. Click a map to zoom in; hover the text to see one token’s attention row; click a cell of the big map to pick a query and a key for the arithmetic below.',
					zh: '行是块，列是头。点击小图可放大；在文本上悬停可看一个词元的注意力行；点击大图中的格子可为下面的算式选择查询和键。'
				})}
			</p>
			<AttnExplorer {S} bind:metric />
		</div>
	{/snippet}

	{#snippet output()}
		<div id="lens" class="sub first">
			<h3>{t({ en: 'Logit lens: when does it decide?', zh: 'Logit lens：它在什么时候做出决定？' })}</h3>
			<LogitLens {S} />
		</div>
		<div class="sub">
			<h3>{t({ en: 'Next token: probabilities and sampling', zh: '下一个词元：概率与采样' })}</h3>
			<Sampling
				{S}
				{dist}
				bind:temperature
				bind:topK
				bind:topP
				{seed}
				{draws}
				{pick}
				{auto}
				{busy}
				promptText={text}
				{genIds}
				ongenerate={() => void genStep()}
				onauto={() => void autoGenerate()}
				onstop={() => (auto = false)}
				onreset={removeGenerated}
				onkeep={keepAsPrompt}
				onseed={onSeed}
			/>
		</div>
	{/snippet}

	{#snippet tryThis()}
		<TryThis {S} load={(v) => setPrompt(v, true)} goto={(id) => void goto(id)} setMetric={(m) => (metric = m)} {compare} {seed} promptText={text} />
	{/snippet}

	{#snippet training()}
		<div class="train">
			<p>
				{t({
					en: 'TinyGPT was trained once, offline, on a laptop GPU. These numbers come from that run (precomputed); the page does not train.',
					zh: 'TinyGPT 是在一块笔记本 GPU 上离线训练好的。下面的数字来自那次训练（预计算）；本页不会进行训练。'
				})}
			</p>
			<dl class="tfacts">
				<div><dt>{t({ en: 'Data', zh: '数据' })}</dt><dd>TinyStories (Eldan &amp; Li 2023, CDLA-Sharing-1.0) · 32.5M {t({ en: 'tokens', zh: '词元' })}</dd></div>
				<div><dt>{t({ en: 'Training', zh: '训练' })}</dt><dd class="num">30,000 {t({ en: 'steps × 512 × 64 tokens ≈ 30 passes', zh: '步 × 512 × 64 词元 ≈ 30 遍' })}</dd></div>
				<div><dt>{t({ en: 'Validation loss', zh: '验证集损失' })}</dt><dd class="num">{#if trainingMetrics}<Num value={trainingMetrics.val_loss} digits={4} />{:else}1.919{/if} nats / {t({ en: 'token', zh: '词元' })}</dd></div>
				<div><dt><span class="term-like">{t({ en: 'Perplexity', zh: '困惑度' })}</span></dt><dd class="num">{#if trainingMetrics}<Num value={trainingMetrics.val_perplexity} digits={3} />{:else}6.81{/if}</dd></div>
				<div><dt>{t({ en: 'Guessing at random', zh: '随机乱猜' })}</dt><dd class="num">ln 1024 = 6.93 nats</dd></div>
			</dl>
			<p>
				<a href="{base}/train/next-token-prediction/">{t({ en: 'Watch TinyGPT learn: replay its 10 training checkpoints on the next-token prediction page →', zh: '看 TinyGPT 如何学会：在“下一个词元预测”页回放它的 10 个训练检查点 →' })}</a>
			</p>
		</div>
	{/snippet}

	{#snippet article()}
		<Article en={ArticleEn} zh={ArticleZh} />
	{/snippet}

	{#snippet code()}
		<div class="codes">
			<CodeBlock code={BLOCK_CODE} lang="python" caption={{ en: 'pipeline/tinygpt/model.py · one block and the forward pass (PyTorch, used for training)', zh: 'pipeline/tinygpt/model.py · 一个块与前向计算（PyTorch，用于训练）' }} />
			<CodeBlock code={USAGE_CODE} lang="ts" caption={{ en: 'How this page runs the model (TypeScript)', zh: '本页如何运行模型（TypeScript）' }} />
			<p class="muted small">
				{t({ en: 'The Inspector’s Code tab shows the exact line of the PyTorch reference for each of the 66 tensors. Further reading:', zh: '检查器的“代码”标签页会为 66 个张量逐一显示 PyTorch 参考实现中对应的那一行。延伸阅读：' })}
				<a href="https://github.com/karpathy/nanoGPT" rel="noopener">nanoGPT</a>,
				<a href="https://zh.d2l.ai/chapter_attention-mechanisms/index.html" rel="noopener">{t({ en: 'd2l.ai attention (中文)', zh: '《动手学深度学习》注意力机制' })}</a>,
				<a href="https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html" rel="noopener">d2l.ai (English)</a>.
			</p>
		</div>
	{/snippet}

	{#snippet moreLinks()}
		<div class="ext">
			<h3>{t({ en: 'Elsewhere', zh: '站外资源' })}</h3>
			<ul>
				<li><a href="https://poloclub.github.io/transformer-explainer/" rel="noopener">Transformer Explainer</a> — {t({ en: 'the same steps on GPT-2 small (124M parameters), in your browser', zh: '在浏览器里演示 GPT-2 small（1.24 亿参数）的同样步骤' })}</li>
				<li><a href="https://bbycroft.net/llm" rel="noopener">LLM Visualization</a> — {t({ en: 'a 3D walk through every weight of a GPT', zh: '用 3D 走一遍 GPT 的每一个权重' })}</li>
				<li><a href="https://github.com/jessevig/bertviz" rel="noopener">BertViz</a> — {t({ en: 'the layers × heads attention grid, for big models', zh: '大模型的“层 × 头”注意力网格' })}</li>
				<li><a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" rel="noopener">interpreting GPT: the logit lens</a> — {t({ en: 'where the logit lens comes from (nostalgebraist, 2020)', zh: 'logit lens 的出处（nostalgebraist，2020）' })}</li>
				<li><a href="https://arxiv.org/abs/1902.10186" rel="noopener">Attention is not Explanation</a> — {t({ en: 'Jain & Wallace (2019), on the misconception above', zh: 'Jain 与 Wallace（2019），关于上面提到的误区' })}</li>
			</ul>
			<p>
				<a href="{base}/train/tokenization/">{t({ en: 'Tokenization', zh: '分词' })}</a> ·
				<a href="{base}/train/next-token-prediction/">{t({ en: 'Next-token prediction', zh: '下一个词元预测' })}</a> ·
				<a href="{base}/m/tinycnn/">TinyCNN</a>
			</p>
		</div>
	{/snippet}

	{#snippet checkpoint()}
		<Checkpoint />
	{/snippet}

	{#snippet credits()}
		<p class="small">
			{t({
				en: 'Model, tokenizer and page by Model Atlas. Training data: TinyStories (Eldan & Li 2023, CDLA-Sharing-1.0). Ideas from nanoGPT/minGPT (MIT), Transformer Explainer (MIT), BertViz (Apache-2.0) and the logit lens post; no code copied. Full list on the About page.',
				zh: '模型、分词器和页面由 Model Atlas 制作。训练数据：TinyStories（Eldan 与 Li 2023，CDLA-Sharing-1.0）。借鉴了 nanoGPT/minGPT（MIT）、Transformer Explainer（MIT）、BertViz（Apache-2.0）和 logit lens 一文的思路，未复制代码。完整列表见“关于”页面。'
			})}
			<a href="{base}/about/">{t({ en: 'About and licenses', zh: '关于与许可' })}</a>
		</p>
	{/snippet}
</ModelPageShell>

<style>
	.parity-note {
		margin: -8px 0 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
	}
	.err {
		color: var(--illus);
		font-weight: 650;
	}
	.params {
		margin-top: 14px;
	}
	.params summary,
	.sub h3 {
		font-size: var(--fs-md);
	}
	.params summary {
		cursor: pointer;
		font-weight: 650;
		margin-bottom: 8px;
	}
	.sub {
		margin-top: var(--s6);
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.sub.first {
		margin-top: 0;
	}
	.sub > .muted {
		font-size: var(--fs-sm);
		max-width: var(--measure);
	}
	.train {
		max-width: var(--measure);
	}
	.tfacts {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
		gap: 8px 20px;
		margin: 0 0 12px;
	}
	.tfacts dt {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.tfacts dd {
		margin: 0;
		font-weight: 600;
	}
	.codes {
		display: grid;
		gap: 12px;
	}
	.small {
		font-size: var(--fs-sm);
	}
	.ext ul {
		padding-left: 1.2em;
	}
	/* token text: no programming ligatures ("<|" would turn into a triangle in JetBrains Mono) */
	:global(.mps) {
		font-variant-ligatures: none;
	}
	/* formulas: a little smaller on phones so most lines fit (long ones still scroll in their card) */
	@media (max-width: 560px) {
		:global(.mps .fc .katex) {
			font-size: 1em;
		}
	}
	:global(.flash) {
		animation: flash 1.8s ease-out;
		border-radius: var(--r3);
	}
	@keyframes flash {
		0%,
		40% {
			box-shadow: 0 0 0 3px var(--accent);
		}
		100% {
			box-shadow: 0 0 0 3px transparent;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		:global(.flash) {
			animation: none;
			box-shadow: 0 0 0 3px var(--accent);
		}
	}
</style>
