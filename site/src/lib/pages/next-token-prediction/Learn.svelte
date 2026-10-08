<!--
  Part 3 "Watch it learn": a scrubber over TinyGPT's 10 saved checkpoints (step 0 = random weights …
  30,000 = final), the loss curve with a marker, the precomputed probe prompts (replay.json) and the
  learner's own text run LIVE through the selected fp16 checkpoint (fetched lazily, cached).
-->
<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { base } from '$app/paths';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import Num from '$lib/kit/Num.svelte';
	import Term from '$lib/kit/Term.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import { motion } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtBytes, fmtCount, fmtInt, visibleToken } from '$lib/kit/format';
	import LossChart from './LossChart.svelte';
	import TryThis from './TryThis.svelte';
	import type { NtpState } from './state.svelte';
	import type { NextResult } from './protocol';
	import { FACTS, MAX_CHARS, REPLAY_SAMPLING } from './facts';

	let { S }: { S: NtpState } = $props();

	type I18n = { en: string; zh: string };
	type ProbeOut = {
		top: { id: number; text: string; p: number }[];
		greedy: { ids: number[]; text: string };
		sampled: { ids: number[]; text: string };
	};
	type ReplayCk = {
		step: number;
		file: string;
		tokens: number;
		trainLoss: number;
		valLoss: number;
		label?: I18n;
		heldOut: { loss: number; perplexity: number; n: number };
		probes: ProbeOut[];
	};
	type Replay = { probes: string[]; probeIds: number[][]; checkpoints: ReplayCk[] };
	type Curve = { points: { step: number; tokens: number; train: number; val: number; lr: number }[] };

	let replay = $state.raw<Replay | null>(null);
	let curve = $state.raw<Curve | null>(null);
	let dataErr = $state<string | null>(null);
	let ci = $state(0);
	let probe = $state(0);
	let xMode = $state<'log' | 'linear'>('log');
	let armed = $state(false);
	let visible = $state(false);
	let root: HTMLElement | undefined = $state();
	let playing = $state(false);
	let panel: HTMLElement | undefined = $state();
	let scrubEl: HTMLElement | undefined = $state();

	const cks = $derived(replay?.checkpoints ?? []);
	const ck = $derived(cks[ci]);
	const step = $derived(ck?.step ?? 0);
	const short = (s: number) => (s >= 1000 ? `${s / 1000}k` : String(s));

	onMount(() => {
		const url = (f: string) => `${base}/models/tinygpt/${f}`;
		Promise.all([fetch(url('replay.json')), fetch(url('checkpoints/curve.json'))])
			.then(async ([a, b]) => {
				if (!a.ok || !b.ok) throw new Error('could not load replay.json / curve.json');
				replay = (await a.json()) as Replay;
				curve = (await b.json()) as Curve;
			})
			.catch((e) => (dataErr = String(e?.message ?? e)));
		let io: IntersectionObserver | null = null;
		if (root && 'IntersectionObserver' in window) {
			io = new IntersectionObserver(
				(entries) => {
					for (const e of entries) {
						visible = e.isIntersecting;
						if (e.isIntersecting) armed = true;
					}
				},
				{ rootMargin: '200px 0px' }
			);
			io.observe(root);
		} else {
			armed = visible = true;
		}
		return () => io?.disconnect();
	});

	// ---------------------------------------------------------------- live runs (cached per checkpoint + text)
	type LiveRun = {
		next: NextResult | null;
		greedy: number[];
		greedyDone: boolean;
		sampled: number[];
		sampledDone: boolean;
		loss: number | null;
		error: string | null;
	};
	let runs = $state.raw<Record<string, LiveRun>>({});
	let yours = $state.raw<Record<number, { key: string; loss: number }>>({});
	const idsKey = $derived(S.ids.join(','));
	const runKey = $derived(`${step}|${idsKey}`);
	const run = $derived(runs[runKey] as LiveRun | undefined);
	const complete = (r: LiveRun | undefined) => !!r && !r.error && r.greedyDone && r.sampledDone && !!r.next;

	function patch(key: string, f: (r: LiveRun) => Partial<LiveRun>) {
		const cur = runs[key];
		if (!cur) return;
		const keys = Object.keys(runs);
		const next: Record<string, LiveRun> = {};
		// keep the cache small: the 40 most recent runs
		for (const k of keys.slice(Math.max(0, keys.length - 40))) next[k] = runs[k];
		next[key] = { ...cur, ...f(cur) };
		runs = next;
	}

	function startRun(st: number, ids: number[]) {
		const c = S.client;
		if (!c || ids.length === 0) return;
		const key = `${st}|${ids.join(',')}`;
		if (complete(runs[key])) return;
		runs = { ...runs, [key]: { next: null, greedy: [], greedyDone: false, sampled: [], sampledDone: false, loss: null, error: null } };
		const fail = (e: unknown) => patch(key, () => ({ error: String((e as Error)?.message ?? e) }));
		c.next(st, ids, 'ck-next', 10).then((r) => r && patch(key, () => ({ next: r })), fail);
		if (ids.length >= 2) {
			c.score(st, ids, 'ck-score', 1).then((r) => {
				if (!r) return;
				patch(key, () => ({ loss: r.meanNats }));
				yours = { ...yours, [st]: { key: ids.join(','), loss: r.meanNats } };
			}, fail);
		}
		c.generate(st, ids, { maxNew: REPLAY_SAMPLING.maxNew, temperature: 0, topN: 1 }, 'ck-greedy', (s) =>
			patch(key, (r) => ({ greedy: [...r.greedy, s.token] }))
		).then((r) => r && patch(key, () => ({ greedy: r.newIds, greedyDone: true })), fail);
		c.generate(
			st,
			ids,
			{ maxNew: REPLAY_SAMPLING.maxNew, temperature: REPLAY_SAMPLING.temperature, seed: REPLAY_SAMPLING.seed, topN: 1 },
			'ck-sampled',
			(s) => patch(key, (r) => ({ sampled: [...r.sampled, s.token] }))
		).then((r) => r && patch(key, () => ({ sampled: r.newIds, sampledDone: true })), fail);
	}

	// run the learner's text through the selected checkpoint (only once the learner reached this part)
	$effect(() => {
		const st = step;
		const ids = S.ids;
		const ok = armed && visible && !!S.client && !!ck;
		if (!ok) return;
		const h = setTimeout(() => untrack(() => startRun(st, ids)), 220);
		return () => clearTimeout(h);
	});

	// the loss of the learner's text at every other checkpoint already loaded (dots on the chart)
	const loaded = $derived(
		cks.filter((c) => S.statusOf(c.step)?.state === 'ready').map((c) => c.step)
	);
	$effect(() => {
		const ids = S.ids;
		const key = idsKey;
		const steps = loaded;
		if (!visible || !S.client || ids.length < 2) return;
		const h = setTimeout(() => {
			for (const st of steps) {
				if (untrack(() => yours[st]?.key) === key) continue;
				S.client?.score(st, ids, `yt:${st}`, 1).then((r) => {
					if (r) yours = { ...yours, [st]: { key, loss: r.meanNats } };
				});
			}
		}, 500);
		return () => clearTimeout(h);
	});
	const yourDots = $derived(
		Object.entries(yours)
			.filter(([, v]) => v.key === idsKey)
			.map(([s, v]) => ({ step: Number(s), loss: v.loss }))
	);

	// ---------------------------------------------------------------- playback
	$effect(() => {
		if (!playing) return;
		const r = run;
		const st = S.statusOf(step)?.state;
		const ready = (r && r.greedyDone) || st === 'error' || !!r?.error || S.ids.length === 0;
		if (!ready) return;
		const h = setTimeout(() => {
			if (ci < cks.length - 1) ci++;
			else playing = false;
		}, 1500);
		return () => clearTimeout(h);
	});
	function play() {
		armed = true;
		if (motion.reduced) {
			if (ci < cks.length - 1) ci++;
			return;
		}
		if (playing) {
			playing = false;
			return;
		}
		if (ci >= cks.length - 1) ci = 0;
		playing = true;
	}
	function select(i: number) {
		armed = true;
		playing = false;
		ci = i;
	}
	function onScrubKey(e: KeyboardEvent, i: number) {
		let j = i;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = Math.min(cks.length - 1, i + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = Math.max(0, i - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = cks.length - 1;
		else return;
		e.preventDefault();
		select(j);
		(scrubEl?.querySelectorAll('button[role=radio]')[j] as HTMLElement | undefined)?.focus();
	}

	// ---------------------------------------------------------------- display helpers
	const statusText = (s: number) => {
		const st = S.statusOf(s);
		if (!st) return t({ en: 'not loaded', zh: '未加载' });
		if (st.state === 'ready') return t({ en: 'loaded', zh: '已加载' });
		if (st.state === 'error') return t({ en: 'failed', zh: '加载失败' });
		return st.total ? `${Math.round((100 * st.loaded) / st.total)}%` : t({ en: 'loading', zh: '加载中' });
	};
	const loadedBytes = $derived(loaded.length * FACTS.ckptBytes);
	const promptTail = $derived.by(() => {
		const s = S.decode(S.ids);
		return s.length > 60 ? '…' + s.slice(s.length - 59) : s;
	});
	const probeTail = (i: number) => replay?.probes[i] ?? '';
	const nextItems = $derived(run?.next ? run.next.top.map((x) => ({ label: S.disp(x.id), p: x.p, id: x.id })) : []);
	const probeOut = $derived(ck?.probes[probe]);
	const probeItems = $derived(probeOut ? probeOut.top.map((x) => ({ label: S.disp(x.id), p: x.p, id: x.id })) : []);
	const epochs = (tokens: number) => tokens / FACTS.trainTokens;

	function focusPanel() {
		panel?.scrollIntoView({ behavior: motion.reduced ? 'auto' : 'smooth', block: 'start' });
	}

	const tries = $derived([
		{
			title: { en: 'Step 0 on your text', zh: '第 0 步跑你的文本' },
			predict: {
				en: 'Before you look: at step 0 the weights are random. Will the greedy continuation of your text be a sentence, random pieces, or something else?',
				zh: '先别看：第 0 步时权重是随机的。你的文本的贪心续写会是一句话、一堆随机碎片，还是别的什么？'
			},
			action: { en: 'Show step 0', zh: '显示第 0 步' },
			run: () => {
				select(0);
				focusPanel();
			}
		},
		{
			title: { en: 'When do words appear?', zh: '什么时候开始出现单词？' },
			predict: {
				en: 'Predict the first checkpoint where the greedy continuation is real English words. Then watch the whole run.',
				zh: '预测一下：贪心续写从哪个检查点开始出现真正的英文单词？然后看完整个训练过程。'
			},
			action: { en: 'Play from step 0', zh: '从第 0 步开始播放' },
			run: () => {
				select(0);
				if (!motion.reduced) playing = true;
				focusPanel();
			}
		},
		{
			title: { en: 'Where does the learning happen?', zh: '学习发生在哪里？' },
			predict: {
				en: 'Switch the curve to a linear axis. What fraction of the 30,000 steps does it take to get most of the way down?',
				zh: '把曲线换成线性坐标。要走完大部分下降，需要 30,000 步中的多大比例？'
			},
			action: { en: 'Linear axis', zh: '线性坐标' },
			run: () => (xMode = 'linear')
		}
	]);
</script>

<div class="learn" bind:this={root}>
	<div class="prose">
		<Lang>
			{#snippet en()}
				<p>
					Training repeats the lesson from “One sentence is many lessons” about a billion times. TinyGPT started with random weights, read 512 windows
					of 64 tokens per step, and after each step nudged its weights to lower the average loss. Ten
					<Term id="checkpoint">checkpoints</Term> of its weights were saved along the way, from step 0 (random) to step 30,000
					(the final model). Each one is a complete model you can run.
				</p>
				<p>
					Pick a checkpoint. The <strong>live</strong> panel downloads that checkpoint (728 KB, only when you ask for it) and
					runs your own text through it in your browser. The <strong>precomputed</strong> panel shows six fixed probe prompts
					that were run through the same checkpoint files offline.
				</p>
			{/snippet}
			{#snippet zh()}
				<p>
					训练就是把“一句话，许多道练习题”里的练习重复大约十亿次。TinyGPT 从随机权重出发，每一步读 512 个 64 词元长的窗口，然后微调权重，让平均损失降低一点。训练途中保存了 10 个<Term id="checkpoint">检查点</Term>，从第 0 步（随机）到第 30,000 步（最终模型）。每一个都是可以运行的完整模型。
				</p>
				<p>
					选一个检查点。<strong>实时</strong>面板会下载这个检查点（728 KB，只在你需要时才下载），并在你的浏览器里用它运行你的文本。<strong>预计算</strong>面板展示 6 个固定的探针提示，它们是离线用同一批检查点文件跑出来的。
				</p>
			{/snippet}
		</Lang>
	</div>

	{#if dataErr}<p class="err">{dataErr}</p>{/if}

	<div class="scrub-row">
		<button type="button" class="btn primary play" onclick={play} disabled={!cks.length}>
			{#if motion.reduced}
				{t({ en: 'Next checkpoint', zh: '下一个检查点' })}
			{:else if playing}
				<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="3.5" y="3" width="3" height="10" rx="1" fill="currentColor" /><rect x="9.5" y="3" width="3" height="10" rx="1" fill="currentColor" /></svg>
				{t({ en: 'Pause', zh: '暂停' })}
			{:else}
				<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4.5 2.8v10.4L13 8z" fill="currentColor" /></svg>
				{t({ en: 'Play training', zh: '播放训练过程' })}
			{/if}
		</button>
		<div class="scrub" role="radiogroup" aria-label={t({ en: 'Training checkpoint', zh: '训练检查点' })} bind:this={scrubEl}>
			{#each cks as c, i (c.step)}
				{@const st = S.statusOf(c.step)?.state}
				<button
					type="button"
					role="radio"
					aria-checked={i === ci}
					tabindex={i === ci ? 0 : -1}
					class="ckb"
					class:on={i === ci}
					onclick={() => select(i)}
					onkeydown={(e) => onScrubKey(e, i)}
					aria-label="{t({ en: 'step', zh: '第' })} {fmtInt(c.step)}{t({ en: '', zh: ' 步' })}, {statusText(c.step)}"
				>
					<span class="lbl num">{short(c.step)}</span>
					<span class="st {st ?? 'none'}" aria-hidden="true"></span>
				</button>
			{/each}
		</div>
	</div>

	{#if ck}
		<div class="ckhead">
			<div class="big">
				<span class="eyebrow">{t({ en: 'Checkpoint', zh: '检查点' })}</span>
				<strong class="num">{t({ en: 'step', zh: '第' })} {fmtInt(ck.step)}{t({ en: '', zh: ' 步' })}</strong>
				{#if ck.label}<span class="tag">{t(ck.label)}</span>{/if}
			</div>
			<dl class="facts small">
				<div><dt>{t({ en: 'tokens seen', zh: '已读词元' })}</dt><dd class="num">{fmtCount(ck.tokens)} <span class="muted">({fmt3(epochs(ck.tokens))} {t({ en: 'epochs', zh: '轮' })})</span></dd></div>
				<div><dt>{t({ en: 'validation loss', zh: '验证损失' })}</dt><dd class="num">{fmt3(ck.valLoss)} <span class="muted">(ppl {fmt3(Math.exp(ck.valLoss))})</span></dd></div>
				<div>
					<dt>{t({ en: 'file', zh: '文件' })}</dt>
					<dd><span class="num">{ck.file}</span> · <span class="num">{fmtBytes(FACTS.ckptBytes)}</span> · {statusText(ck.step)}</dd>
				</div>
				<div><dt>{t({ en: 'loaded so far', zh: '已加载' })}</dt><dd class="num">{loaded.length} / {cks.length} ({fmtBytes(loadedBytes)})</dd></div>
			</dl>
		</div>
	{/if}

	{#if curve && cks.length}
		<div class="chart-row">
			<div class="chart-head">
				<span class="segmented" role="group" aria-label={t({ en: 'x-axis scale', zh: 'x 轴刻度' })}>
					<button type="button" aria-pressed={xMode === 'log'} onclick={() => (xMode = 'log')}>{t({ en: 'log steps', zh: '对数步数' })}</button>
					<button type="button" aria-pressed={xMode === 'linear'} onclick={() => (xMode = 'linear')}>{t({ en: 'linear steps', zh: '线性步数' })}</button>
				</span>
				<ProvenanceBadge mode="precomputed" note={{ en: 'Loss curve logged during training (40 batches of 512 windows per point). Your-text dots are live.', zh: '训练时记录的损失曲线（每个点 40 个批次、每批 512 个窗口）。“你的文本”的点是实时计算的。' }} />
			</div>
			<LossChart
				points={curve.points}
				ckpts={cks.map((c) => ({ step: c.step, val: c.valLoss }))}
				selected={step}
				yours={yourDots}
				{xMode}
				uniform={FACTS.uniformLoss}
				onselect={(s) => select(cks.findIndex((c) => c.step === s))}
			/>
			<p class="small muted">
				{t({
					en: 'The two lines lie on top of each other: training and validation loss differ by about 0.01 nats, so TinyGPT generalizes to stories it never saw.',
					zh: '两条线几乎重合：训练损失与验证损失只差约 0.01 奈特，说明 TinyGPT 能推广到没见过的故事。'
				})}
			</p>
		</div>
	{/if}

	<div class="panels" bind:this={panel}>
		<section class="pn live">
			<header>
				<h3>{t({ en: 'Your text at this checkpoint', zh: '你的文本在这个检查点上' })}</h3>
				<ProvenanceBadge
					mode="live"
					precision="fp16"
					approximate
					note={{
						en: 'Runs the fp16 checkpoint file in your browser. On the same fp16 weights the JS forward pass matches PyTorch within 1.2e-5.',
						zh: '在你的浏览器里运行 fp16 检查点文件。在同样的 fp16 权重上，JS 前向计算与 PyTorch 的差距在 1.2e-5 以内。'
					}}
				/>
			</header>
			<TextInputBar bind:value={S.text} maxLength={MAX_CHARS} label={{ en: 'Your prompt (shared with the whole page)', zh: '你的提示词（全页共用）' }} />
			{#if !armed}
				<p class="muted small">{t({ en: 'The checkpoint loads when you reach this part.', zh: '到达这一部分时才加载检查点。' })}</p>
			{:else if S.statusOf(step)?.state === 'error' || run?.error}
				<p class="err small">{t({ en: 'Could not load this checkpoint.', zh: '无法加载这个检查点。' })} {S.statusOf(step)?.message ?? run?.error ?? ''}</p>
			{:else if S.statusOf(step)?.state !== 'ready'}
				<p class="muted small" aria-live="polite">
					{t({ en: 'Downloading', zh: '正在下载' })} <span class="num">{ck?.file}</span> ({fmtBytes(FACTS.ckptBytes)}) … {statusText(step)}
				</p>
			{/if}
			{#if S.ids.length === 0}
				<p class="muted small">{t({ en: 'Type a prompt above.', zh: '先在上面输入提示词。' })}</p>
			{:else}
				<div class="cont">
					<p class="kind">{t({ en: 'Greedy (always the top token)', zh: '贪心（每次取概率最高的词元）' })}</p>
					<p class="gen"><span class="prompt">{promptTail}</span><span class="new">{run ? S.decode(run.greedy) : ''}</span>{#if run && !run.greedyDone && !run.error}<span class="caret" aria-hidden="true">▍</span>{/if}</p>
				</div>
				<div class="cont">
					<p class="kind">{t({ en: `Sampled (temperature ${REPLAY_SAMPLING.temperature}, seed ${REPLAY_SAMPLING.seed})`, zh: `采样（温度 ${REPLAY_SAMPLING.temperature}，种子 ${REPLAY_SAMPLING.seed}）` })}</p>
					<p class="gen"><span class="prompt">{promptTail}</span><span class="new">{run ? S.decode(run.sampled) : ''}</span>{#if run && !run.sampledDone && !run.error}<span class="caret" aria-hidden="true">▍</span>{/if}</p>
				</div>
				{#if nextItems.length}
					<ProbBars items={nextItems} k={10} tokens label={{ en: 'Next token after your text', zh: '你的文本之后的下一个词元' }} />
				{/if}
				{#if run?.loss != null}
					<p class="small">
						{t({ en: 'Loss on your text at this checkpoint:', zh: '这个检查点在你的文本上的损失：' })}
						<strong><Num value={run.loss} /></strong> {t({ en: 'nats/token', zh: '奈特/词元' })}
						(ppl <Num value={Math.exp(run.loss)} />){' '}{#if S.fresh && S.score && step !== cks[cks.length - 1]?.step}
							· {t({ en: 'final model (fp32):', zh: '最终模型（fp32）：' })} <Num value={S.score.meanNats} />{/if}
					</p>
				{/if}
			{/if}
		</section>

		<section class="pn pre">
			<header>
				<h3>{t({ en: 'Probe prompts', zh: '探针提示' })}</h3>
				<ProvenanceBadge
					mode="precomputed"
					precision="fp16"
					note={{ en: 'replay.json: computed offline from the same 10 checkpoint files; the browser reproduces all of it exactly.', zh: 'replay.json：离线用同样的 10 个检查点文件算出；浏览器能精确复现其中每一项。' }}
				/>
			</header>
			{#if replay}
				<div class="probes" role="group" aria-label={t({ en: 'Probe prompt', zh: '探针提示' })}>
					{#each replay.probes as pr, i (i)}
						<button type="button" class="chip" aria-pressed={probe === i} onclick={() => (probe = i)}>{pr}</button>
					{/each}
				</div>
				{#if probeOut}
					<div class="cont">
						<p class="kind">{t({ en: 'Greedy', zh: '贪心' })}</p>
						<p class="gen"><span class="prompt">{probeTail(probe)}</span><span class="new">{probeOut.greedy.text}</span></p>
					</div>
					<div class="cont">
						<p class="kind">{t({ en: `Sampled (temperature ${REPLAY_SAMPLING.temperature}, seed ${REPLAY_SAMPLING.seed})`, zh: `采样（温度 ${REPLAY_SAMPLING.temperature}，种子 ${REPLAY_SAMPLING.seed}）` })}</p>
						<p class="gen"><span class="prompt">{probeTail(probe)}</span><span class="new">{probeOut.sampled.text}</span></p>
					</div>
					<ProbBars items={probeItems} k={10} tokens label={{ en: 'Next token', zh: '下一个词元' }} />
					<p class="small">
						{t({ en: 'Held-out loss (first 64 predictions of 200 validation stories):', zh: '留出集损失（200 个验证故事的前 64 次预测）：' })}
						<strong><Num value={ck.heldOut.loss} /></strong> (ppl <Num value={ck.heldOut.perplexity} />)
					</p>
				{/if}
			{:else}
				<p class="muted small">{t({ en: 'Loading…', zh: '加载中…' })}</p>
			{/if}
		</section>
	</div>

	{#if cks.length}
		<details class="table">
			<summary>{t({ en: 'All checkpoints as a table', zh: '以表格查看所有检查点' })}</summary>
			<div class="tscroll" use:keyScroll>
				<table>
					<thead>
						<tr>
							<th>{t({ en: 'step', zh: '步' })}</th>
							<th>{t({ en: 'tokens seen', zh: '已读词元' })}</th>
							<th>{t({ en: 'train loss', zh: '训练损失' })}</th>
							<th>{t({ en: 'val loss', zh: '验证损失' })}</th>
							<th>{t({ en: 'held-out loss', zh: '留出集损失' })}</th>
							<th>{t({ en: 'your text (live)', zh: '你的文本（实时）' })}</th>
						</tr>
					</thead>
					<tbody>
						{#each cks as c, i (c.step)}
							<tr class:cur={i === ci}>
								<td class="num">{fmtInt(c.step)}</td>
								<td class="num">{fmtCount(c.tokens)}</td>
								<td class="num">{fmt3(c.trainLoss)}</td>
								<td class="num">{fmt3(c.valLoss)}</td>
								<td class="num">{fmt3(c.heldOut.loss)}</td>
								<td class="num">{yours[c.step]?.key === idsKey ? fmt3(yours[c.step].loss) : '—'}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</details>
	{/if}

	<TryThis items={tries} />

	<div class="prose">
		<Lang>
			{#snippet en()}
				<p><strong>What changes, step by step.</strong></p>
				<ul>
					<li>
						<strong>Step 0.</strong> Every token gets about 1/1,024 of the probability (the top one barely 0.3%), so the loss is
						ln 1024 ≈ 6.93. Sampling gives byte soup. Greedy decoding almost always just repeats your last token (it did for 290 of 300
						story snippets we tried): TinyGPT uses the same
						matrix to read tokens in and to score them out (tied embeddings), so with random weights the input token's own
						vector matches itself best.
					</li>
					<li>
						<strong>Steps 10–30.</strong> It learns which tokens are common: “.”, “ a”, “ and”, “ to” rise to the top whatever
						the context. Greedy text is “........” or “ and and and”.
					</li>
					<li>
						<strong>Steps 100–400.</strong> Words and short phrases (“girl named Tim”, “was very happy”), then grammar. By step
						400 the validation loss is 2.66.
					</li>
					<li>
						<strong>Steps 1,000–30,000.</strong> Whole sentences, characters that stay the same, simple plots. Most of the drop
						happened in the first 3% of training (val loss 2.28 at step 1,000); the remaining 29,000 steps shave it to 1.92.
					</li>
				</ul>
				<p>
					Nothing in the training changed along the way: the same −ln p, averaged over 32,768 labels per step. Everything you
					see, from spelling to plot, is what it took to lower that one number.
				</p>
			{/snippet}
			{#snippet zh()}
				<p><strong>一步一步，变化了什么。</strong></p>
				<ul>
					<li>
						<strong>第 0 步。</strong>每个词元大约分到 1/1,024 的概率（最高的也只有 0.3% 左右），所以损失是 ln 1024 ≈ 6.93。采样出来是一堆字节碎片。贪心解码几乎总是重复你的最后一个词元（我们试的 300 个故事片段里有 290 个如此）：TinyGPT 用同一个矩阵来读入词元和给输出打分（共享嵌入），权重随机时，输入词元的向量和它自己最匹配。
					</li>
					<li>
						<strong>第 10–30 步。</strong>它学会了哪些词元常见：“.”、“ a”、“ and”、“ to”不管上下文是什么都排到最前面。贪心续写是“........”或“ and and and”。
					</li>
					<li>
						<strong>第 100–400 步。</strong>出现了单词和短语（“girl named Tim”“was very happy”），接着是语法。到第 400 步，验证损失已降到 2.66。
					</li>
					<li>
						<strong>第 1,000–30,000 步。</strong>完整的句子、前后一致的角色、简单的情节。大部分下降发生在训练的前 3%
						（第 1,000 步时验证损失为 2.28）；剩下的 29,000 步只把它磨到 1.92。
					</li>
				</ul>
				<p>
					训练方法从头到尾都没变：同一个 −ln p，每步在 32,768 个标签上取平均。你看到的一切，从拼写到情节，都是为了把这一个数降下来而学会的。
				</p>
			{/snippet}
		</Lang>
	</div>
</div>

<style>
	.learn {
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
	.prose ul {
		margin: 0 0 0.8em;
		padding-left: 1.2em;
	}
	.prose li + li {
		margin-top: 6px;
	}
	.err {
		color: var(--k-text);
	}
	.scrub-row {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
		align-items: center;
	}
	.play {
		min-width: 9em;
		display: inline-flex;
		gap: 6px;
		align-items: center;
		justify-content: center;
	}
	.scrub {
		display: grid;
		grid-template-columns: repeat(10, minmax(0, 1fr));
		gap: 4px;
		flex: 1 1 420px;
		min-width: 0;
	}
	@media (max-width: 560px) {
		.scrub {
			grid-template-columns: repeat(5, minmax(0, 1fr));
			flex-basis: 100%;
		}
	}
	.ckb {
		display: grid;
		justify-items: center;
		gap: 3px;
		padding: 6px 2px 5px;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r2);
		background: var(--surface);
		color: var(--ink);
		cursor: pointer;
		min-height: 44px;
		font: inherit;
	}
	.ckb:hover {
		background: var(--surface-2);
	}
	.ckb.on {
		background: var(--accent);
		border-color: var(--accent);
		color: var(--accent-ink);
	}
	.lbl {
		font-size: var(--fs-sm);
		font-weight: 650;
	}
	.st {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		border: 1.5px solid currentColor;
		opacity: 0.7;
	}
	.st.ready {
		background: currentColor;
		opacity: 1;
	}
	.st.loading {
		border-style: dotted;
		opacity: 1;
	}
	.st.error {
		border-color: var(--k);
	}
	.ckhead {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 28px;
		align-items: center;
		padding: 12px 14px;
		background: var(--surface-2);
		border-radius: var(--r3);
	}
	.big {
		display: flex;
		align-items: baseline;
		gap: 10px;
		flex-wrap: wrap;
	}
	.big strong {
		font-size: var(--fs-xl);
	}
	.big .eyebrow {
		margin: 0;
	}
	.tag {
		font-size: var(--fs-xs);
		padding: 1px 8px;
		border-radius: var(--r-pill);
		background: var(--surface);
		border: 1px solid var(--rule-strong);
	}
	.facts {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 22px;
		margin: 0;
	}
	.facts div {
		display: grid;
	}
	.facts dt {
		color: var(--ink-2);
	}
	.facts dd {
		margin: 0;
	}
	.chart-row {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.chart-row p {
		margin: 0;
	}
	.chart-head {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
		justify-content: space-between;
	}
	.panels {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr));
		gap: var(--s4);
		align-items: start;
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.pn {
		display: grid;
		gap: 10px;
		padding: 14px;
		border-radius: var(--r3);
		background: var(--surface);
		border: 1px solid var(--rule);
		min-width: 0;
	}
	.pn.live {
		border-top: 3px solid var(--live);
	}
	.pn.pre {
		border-top: 3px solid var(--pre);
	}
	.pn header {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 10px;
		align-items: center;
		justify-content: space-between;
	}
	.pn h3 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.pn p {
		margin: 0;
	}
	.cont {
		display: grid;
		gap: 2px;
	}
	.kind {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		font-weight: 650;
	}
	.gen {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		line-height: 1.55;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		padding: 8px 10px;
		background: var(--surface-2);
		border-radius: var(--r2);
		min-height: 3.2em;
	}
	.prompt {
		color: var(--ink-2);
	}
	.new {
		color: var(--generated-text);
		font-weight: 600;
	}
	.caret {
		color: var(--generated-text);
		animation: blink 1s steps(2) infinite;
	}
	@keyframes blink {
		50% {
			opacity: 0;
		}
	}
	.probes {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.probes .chip {
		cursor: pointer;
		font: inherit;
		font-size: var(--fs-xs);
	}
	.probes .chip[aria-pressed='true'] {
		background: var(--accent-soft);
		border-color: var(--accent);
	}
	.small {
		font-size: var(--fs-xs);
	}
	.table summary {
		cursor: pointer;
		font-size: var(--fs-sm);
	}
	.tscroll {
		overflow-x: auto;
		margin-top: 8px;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-sm);
		min-width: 100%;
	}
	th,
	td {
		text-align: right;
		padding: 4px 10px;
		border-bottom: 1px solid var(--rule);
		white-space: nowrap;
	}
	th {
		font-weight: 650;
		color: var(--ink-2);
	}
	tr.cur td {
		background: var(--accent-soft);
	}
</style>
