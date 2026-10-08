<!--
  Part 4 "Decoding": temperature / top-k / top-p applied to the final model's live logits for the
  learner's text (computed once in the worker; the cut-offs are recomputed instantly on the main
  thread with the same samplingDistribution() that generate() uses). Sample one token at a time or
  generate 20.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import Term from '$lib/kit/Term.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import { sym } from '$lib/kit/tex';
	import { motion } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import TryThis from './TryThis.svelte';
	import type { NtpState } from './state.svelte';
	import { FACTS } from './facts';

	let { S }: { S: NtpState } = $props();

	type Sampler = (logits: ArrayLike<number>, temperature: number, topK?: number, topP?: number) => Float64Array;
	let sampler = $state.raw<Sampler | null>(null);
	let rng = $state.raw<((seed: number) => () => number) | null>(null);
	let logits = $state.raw<Float32Array | null>(null);
	let logitsKey = $state('');
	let greedy = $state(false);
	let temp = $state(1);
	let topK = $state(0);
	let topP = $state(1);
	let gen = $state<number[]>([]);
	let seed = $state(1);
	let lastU = $state<number | null>(null);
	let busy = $state(false);
	let visible = $state(false);
	let root: HTMLElement | undefined = $state();
	let controls: HTMLElement | undefined = $state();

	const ctx = $derived([...S.ids, ...gen]);
	const ctxKey = $derived(ctx.join(','));

	onMount(() => {
		void import('$lib/models/tinygpt').then((m) => {
			sampler = m.samplingDistribution;
			rng = m.mulberry32;
		});
		let io: IntersectionObserver | null = null;
		if (root && 'IntersectionObserver' in window) {
			io = new IntersectionObserver((es) => es.forEach((e) => (visible = e.isIntersecting)), { rootMargin: '200px 0px' });
			io.observe(root);
		} else visible = true;
		return () => io?.disconnect();
	});

	// the learner's text changed: start a new continuation (and drop a generation still running)
	let lastText = '';
	let genRun = 0;
	$effect(() => {
		const k = S.ids.join(',');
		if (k !== lastText) {
			lastText = k;
			genRun++;
			S.client?.cancel('dec-gen');
			busy = false;
			gen = [];
			lastU = null;
		}
	});
	const stale = $derived(logitsKey !== ctxKey);

	$effect(() => {
		const ids = ctx;
		const key = ctxKey;
		if (!visible || !S.client || ids.length === 0 || key === logitsKey) return;
		const h = setTimeout(() => {
			S.client?.next('final', ids, 'dec-next', 1).then((r) => {
				if (r) {
					logits = r.logits;
					logitsKey = key;
				}
			});
		}, 150);
		return () => clearTimeout(h);
	});

	const T = $derived(greedy ? 0 : temp);
	const dist = $derived.by(() => {
		if (!logits || !sampler) return null;
		if (greedy) {
			let a = 0;
			for (let i = 1; i < logits.length; i++) if (logits[i] > logits[a]) a = i;
			const d = new Float64Array(logits.length);
			d[a] = 1;
			return d;
		}
		return sampler(logits, temp, topK, topP);
	});
	const plain = $derived(logits && sampler ? sampler(logits, 1) : null);
	const order = $derived.by(() => {
		if (!plain) return [] as number[];
		return Array.from({ length: plain.length }, (_, i) => i).sort((a, b) => plain[b] - plain[a] || a - b);
	});
	const kept = $derived(dist ? dist.reduce((a, v) => a + (v > 0 ? 1 : 0), 0) : 0);
	const entropy = $derived(dist ? dist.reduce((a, v) => a - (v > 0 ? v * Math.log2(v) : 0), 0) : 0);
	const items = $derived(
		dist ? order.slice(0, 12).map((id) => ({ label: S.disp(id), p: dist[id], id })) : []
	);
	const count = (d: ArrayLike<number>) => {
		let c = 0;
		for (let i = 0; i < d.length; i++) if (d[i] > 0) c++;
		return c;
	};
	// size of the top-p "nucleus" on its own (before top-k), to place its cut-off line
	const nucleus = $derived(topP < 1 && !greedy && logits && sampler ? count(sampler(logits, temp, 0, topP)) : null);
	const cutoffs = $derived.by(() => {
		const out: { afterRank: number; label: string }[] = [];
		if (greedy) return out;
		if (topK > 0 && topK < 12) out.push({ afterRank: topK, label: `top-k = ${topK}` });
		if (nucleus !== null && nucleus < 12) out.push({ afterRank: nucleus, label: `top-p = ${fmt3(topP)}` });
		return out;
	});
	const top0 = $derived(order.length && logits && plain ? { id: order[0], z: logits[order[0]], pPlain: plain[order[0]] } : null);
	// the formula is the temperature softmax over all 1,024 tokens, before any top-k / top-p cut
	const tempOnly = $derived(logits && sampler && !greedy ? sampler(logits, temp) : null);
	const cutActive = $derived(!greedy && (topK > 0 || topP < 1));

	function sampleOne() {
		if (!dist || !rng || stale) return;
		const u = rng(seed)();
		seed++;
		let tok = -1;
		let cum = 0;
		for (let i = 0; i < dist.length; i++) {
			if (dist[i] === 0) continue;
			cum += dist[i];
			tok = i;
			if (cum > u) break;
		}
		lastU = greedy ? null : u;
		if (tok >= 0) gen = [...gen, tok];
	}
	async function generate20() {
		if (!S.client || busy) return;
		busy = true;
		const my = ++genRun;
		const start = ctx;
		const r = await S.client.generate(
			'final',
			start,
			{ maxNew: 20, temperature: T, topK, topP, seed, topN: 1 },
			'dec-gen',
			(s) => {
				if (my === genRun) gen = [...gen, s.token];
			}
		);
		if (my !== genRun) return;
		if (r) seed++;
		busy = false;
		lastU = null;
	}
	function reset() {
		genRun++;
		S.client?.cancel('dec-gen');
		busy = false;
		gen = [];
		lastU = null;
	}

	const tex = $derived(
		`\\begin{aligned} ${sym('prob', 'p_{\\text{top}}')} &= \\frac{e^{ ${sym('act', 'z_{\\text{top}}')} / T}}{\\sum_{j=1}^{1024} e^{ ${sym('act', 'z_j')} / T}} = {{p}} \\\\ T &= {{T}},\\quad ${sym('act', 'z_{\\text{top}}')} = {{z}} \\end{aligned}`
	);
	const promptTail = $derived.by(() => {
		const s = S.decode(S.ids);
		return s.length > 70 ? '…' + s.slice(s.length - 69) : s;
	});

	function flash() {
		controls?.scrollIntoView({ behavior: motion.reduced ? 'auto' : 'smooth', block: 'center' });
		controls?.classList.add('flash');
		setTimeout(() => controls?.classList.remove('flash'), 1200);
	}
	const tries = [
		{
			title: { en: 'Cold', zh: '低温' },
			predict: { en: 'Set the temperature to 0.2 and sample 5 tokens. Predict: how often will it pick something other than the top token?', zh: '把温度设为 0.2，采样 5 个词元。预测：它有多少次不选概率最高的那个？' },
			action: { en: 'Set T = 0.2', zh: '设 T = 0.2' },
			run: () => {
				greedy = false;
				temp = 0.2;
				topK = 0;
				topP = 1;
				flash();
			}
		},
		{
			title: { en: 'Hot', zh: '高温' },
			predict: { en: 'Set the temperature to 1.8 and generate 20 tokens. Predict: will the text stay a story?', zh: '把温度设为 1.8，生成 20 个词元。预测：文本还会像故事吗？' },
			action: { en: 'Set T = 1.8', zh: '设 T = 1.8' },
			run: () => {
				greedy = false;
				temp = 1.8;
				topK = 0;
				topP = 1;
				flash();
			}
		},
		{
			title: { en: 'Nucleus', zh: '核采样' },
			predict: { en: 'Keep T = 1.8 but set top-p = 0.9. Predict: how many of the 1,024 tokens stay in the running?', zh: '保持 T = 1.8，再把 top-p 设为 0.9。预测：1,024 个词元里还剩多少个有机会被选中？' },
			action: { en: 'Set top-p = 0.9', zh: '设 top-p = 0.9' },
			run: () => {
				greedy = false;
				temp = 1.8;
				topP = 0.9;
				topK = 0;
				flash();
			}
		}
	];
</script>

<div class="dec" bind:this={root}>
	<h3 class="sub">{t({ en: 'Last step: from probabilities to text (decoding)', zh: '最后一步：从概率到文字（解码）' })}</h3>
	<div class="prose">
		<Lang>
			{#snippet en()}
				<p>
					Training only teaches the model to output good probabilities. To write text, something must <em>pick</em> a token
					from them, append it and run the model again (<Term id="autoregressive">autoregressive generation</Term>). That
					picking rule is decoding, and it is not learned. Greedy decoding always takes the top token, which is why the greedy
					texts above often repeat themselves. Sampling draws at random from the probabilities, usually after reshaping them with a
					<Term id="temperature" /> and cutting off the unlikely tail with <Term id="sampling">top-k or top-p</Term>.
				</p>
				<p>
					Below are the final model's real logits for the token after your text. Move the controls: the model is not run again,
					only the rule changes. The loss you met above always uses the plain probabilities (T = 1, no cut-offs).
				</p>
			{/snippet}
			{#snippet zh()}
				<p>
					训练只教会模型输出好的概率。要写出文字，还得有人从这些概率里<em>挑</em>一个词元，接到末尾，再把模型跑一遍（<Term id="autoregressive">自回归生成</Term>）。这个挑选规则就是解码，它不是学出来的。贪心解码永远取概率最高的词元，所以上面的贪心续写常常会自我重复。采样则按概率随机抽取，通常会先用<Term id="temperature" />重新调整分布，再用 <Term id="sampling">top-k 或 top-p</Term> 砍掉不太可能的长尾。
				</p>
				<p>
					下面是最终模型在你的文本之后、对下一个词元给出的真实 logits。拖动这些控件时，模型并不会重新运行，变的只是挑选规则。前面讲的损失，始终用的是原始概率（T = 1，不截断）。
				</p>
			{/snippet}
		</Lang>
	</div>

	<div class="grid">
		<div class="ctl" bind:this={controls}>
			<label class="row check">
				<input type="checkbox" bind:checked={greedy} />
				{t({ en: 'Greedy (always the top token, T = 0)', zh: '贪心（总取最高者，T = 0）' })}
			</label>
			<label class="row" class:off={greedy}>
				<span class="name">{t({ en: 'temperature T', zh: '温度 T' })} <span class="num val">{fmt3(temp)}</span></span>
				<input type="range" min="0.1" max="2" step="0.05" bind:value={temp} disabled={greedy} />
			</label>
			<label class="row" class:off={greedy}>
				<span class="name">top-k <span class="num val">{topK === 0 ? t({ en: 'off', zh: '关' }) : topK}</span></span>
				<input type="range" min="0" max="50" step="1" bind:value={topK} disabled={greedy} />
			</label>
			<label class="row" class:off={greedy}>
				<span class="name">top-p <span class="num val">{topP >= 1 ? t({ en: 'off', zh: '关' }) : fmt3(topP)}</span></span>
				<input type="range" min="0.05" max="1" step="0.05" bind:value={topP} disabled={greedy} />
			</label>
			{#if dist}<p class="small stats">
				{t({ en: 'tokens still possible', zh: '仍可能被选中的词元' })}: <strong class="num">{kept}</strong> / {FACTS.vocab} ·
				{t({ en: 'spread', zh: '分散程度' })} <Num value={entropy} /> {t({ en: 'bits', zh: '比特' })}
			</p>{/if}
			{#if top0 && !greedy}
				<FormulaCard {tex} values={{ T: temp, z: top0.z, p: tempOnly ? tempOnly[top0.id] : 0 }} caption={{ en: 'Temperature reshapes the softmax', zh: '温度重塑 softmax' }} provenance="live" />
				<p class="small muted">
					{t({ en: 'With T = 1 (the model as trained) the top token has', zh: 'T = 1（训练时的模型）时，最高的词元概率是' })}
					<Num value={top0.pPlain} />.
					{#if cutActive && dist}
						{t({ en: 'After the top-k / top-p cut, the tokens that are kept are rescaled to add up to 1 again, so its bar shows', zh: '经过 top-k / top-p 截断后，留下的词元会重新缩放、使总和回到 1，所以它的条形显示为' })}
						<Num value={dist[top0.id]} />.
					{/if}
				</p>
			{/if}
		</div>

		<div class="out">
			<div class="out-head">
				<span class="small">{t({ en: 'Final model (step 30,000), next token', zh: '最终模型（第 30,000 步），下一个词元' })}</span>
				<ProvenanceBadge mode="live" precision="fp32" tolerance={FACTS.tolerance} against="PyTorch fp32" />
			</div>
			{#if items.length}
				<div class:stale>
				<ProbBars {items} k={12} tokens {cutoffs} label={{ en: 'Sampling distribution', zh: '采样分布' }} />
				</div>
			{:else}
				<p class="muted small">{S.ids.length ? t({ en: 'Computing…', zh: '计算中…' }) : t({ en: 'Type some text first.', zh: '请先输入文本。' })}</p>
			{/if}
			<div class="btns">
				<button type="button" class="btn primary" onclick={sampleOne} disabled={!dist || busy || stale}>{t({ en: 'Pick next token', zh: '挑下一个词元' })}</button>
				<button type="button" class="btn" onclick={generate20} disabled={!S.client || busy || S.ids.length === 0}>{busy ? t({ en: 'Generating…', zh: '生成中…' }) : t({ en: 'Generate 20 tokens', zh: '生成 20 个词元' })}</button>
				<button type="button" class="btn ghost" onclick={reset} disabled={gen.length === 0 && !busy}>{t({ en: 'Reset', zh: '重置' })}</button>
			</div>
			<p class="gen" aria-live="polite"><span class="prompt">{promptTail}</span><span class="new">{S.decode(gen)}</span></p>
			<p class="small muted">
				{#if lastU !== null}
					{t({ en: 'Last pick: random number u =', zh: '上一次：随机数 u =' })} <span class="num">{fmt3(lastU)}</span>;
					{t({ en: 'walk down the list adding probabilities until the sum passes u.', zh: '沿着列表累加概率，直到总和超过 u。' })}
				{:else}
					{t({ en: `Seed ${seed} (mulberry32), so every pick can be reproduced.`, zh: `种子 ${seed}（mulberry32），每一次挑选都能复现。` })}
				{/if}
			</p>
			<p class="small">
				<a href="{base}/m/tinygpt/">{t({ en: 'See every block of TinyGPT for your prompt →', zh: '查看 TinyGPT 每一层如何处理你的提示词 →' })}</a>
			</p>
		</div>
	</div>
	<TryThis items={tries} />
</div>

<style>
	.dec {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
		padding-top: var(--s5);
		border-top: 1px dashed var(--rule);
	}
	.sub {
		margin: 0;
		font-size: var(--fs-lg);
	}
	.prose {
		max-width: var(--measure);
	}
	.prose p {
		margin: 0 0 0.8em;
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr);
		gap: var(--s5);
		align-items: start;
	}
	@media (max-width: 820px) {
		.grid {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.ctl,
	.out {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.ctl {
		padding: 12px;
		border-radius: var(--r3);
		transition: box-shadow var(--dur-3) var(--ease);
	}
	:global(.ctl.flash) {
		box-shadow: 0 0 0 3px var(--accent);
	}
	.row {
		display: grid;
		gap: 4px;
		font-size: var(--fs-sm);
	}
	.row.check {
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.row.off {
		opacity: 0.5;
	}
	.name {
		display: flex;
		justify-content: space-between;
	}
	.val {
		font-weight: 650;
	}
	input[type='range'] {
		width: 100%;
	}
	.stats {
		margin: 0;
	}
	.out-head {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 10px;
		align-items: center;
		justify-content: space-between;
	}
	.btns {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.gen {
		margin: 0;
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
	.small {
		font-size: var(--fs-xs);
	}
	.out p {
		margin: 0;
	}
	.stale {
		opacity: 0.55;
		transition: opacity var(--dur-2) var(--ease);
	}
</style>
