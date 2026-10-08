<!-- tinyseq2seq 'attend' (meta.ts): one decoder step. The input characters as columns with their score (mode scores and
     later), their weight as a purple bar (weights and later) and, in mode context, their encoder state faded by its
     weight next to the context vector. Under it, the step's arithmetic worked out with the real numbers. While a
     predict question is open (the stage is inert) the answer parts (.ans) are hidden. -->
<script lang="ts">
	import TensorStrip from '$lib/kit/TensorStrip.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import type { I18n, StageProps } from '../../../../atlas/types/index.ts';
	import { TGT } from '../../data.ts';
	import { dec, fx, restOf, sym, tr } from '../../look.ts';
	import { attendOf } from './meta.ts';

	let P: StageProps = $props();
	const A = $derived(attendOf(P));
	const S = $derived(A.S);
	const level = $derived(['start', 'scores', 'weights', 'context', 'next'].indexOf(A.mode));
	const enc = $derived(tr(P.rt, 'enc'));
	const H = $derived(enc?.shape[1] ?? 0);
	const div = $derived(makeScale('diverging', [-1, 1], themeState.resolved));
	const cw = 36;
	const bar = 56;
	const hy = $derived(level >= 2 ? 78 + bar : level === 1 ? 50 : 30);
	const ch = 8;
	const wMax = $derived(S.a ? Math.max(...S.a) : 1);
	const n = $derived(A.chars.length + (level === 3 ? 2 : 0));
	const pc = (v: number) => (v < 0.0005 ? '0' : fmtPct(v).replace('%', '')); // as the narration prints it
	const L = (en: string, zh: string): I18n => ({ en, zh });
	const par = (v: number, d = 2) => (v < 0 ? `(${fx(v, d)})` : fx(v, d));
	const strips = $derived(A.score ? [
		{ k: 'q', name: L('query = W_q × the decoder’s state', '查询 = W_q × 解码器的状态'), d: A.score.q },
		{ k: 'k', name: L(`key of “${A.chars[S.j]}” = W_k × its encoder state + b`, `“${A.chars[S.j]}”的键 = W_k × 它的编码器状态 + b`), d: A.score.k },
		{ k: 'th', name: L('tanh(query + key), number by number', 'tanh(查询 + 键)，逐个计算'), d: A.score.th },
		{ k: 'v', name: L('v, learned weights', 'v，学来的权重'), d: A.score.v },
		{ k: 'p', name: L('v × tanh(query + key): add these up', 'v × tanh(查询 + 键)：把这些加起来'), d: A.score.prod }
	] : []);
</script>

<div class="at">
	<div class="so" aria-label={t(L('The output, one slot per decoder step', '输出：每个解码步一格'))}>
		{#each { length: S.ids.length } as _, i (i)}<b class="slot num" class:cur={i === S.t} class:tpl={i > S.t}>{i < S.t ? sym(S.ids[i]) : i === S.t ? (level === 4 ? sym(S.y) : '?') : 'YYYY-MM-DD■'[i]}</b>{/each}
		<span class="lab">{t(L(`step ${S.t} reads “${sym(S.prev)}”${S.t === 0 ? ' (the start symbol)' : ''}`, `第 ${S.t} 步读入“${sym(S.prev)}”${S.t === 0 ? '（起始符号）' : ''}`))}</span>
	</div>
	{#if S.e}
		<div class="box" use:keyScroll>
			<svg viewBox="0 0 {n * cw} {level === 3 ? hy + H * ch + 6 : hy}" style:width="min(100%, {n * cw}px)" style:min-width="{n * (level === 3 ? 20 : 24)}px" role="img" aria-label={t(L('Scores and weights per input character', '每个输入字符的分数和权重'))}>
				{#each A.chars as c, j (j)}
					{@const x = j * cw + cw / 2}
					<text {x} y="20" class="ch" class:top={j === S.j && level > 1}>{c}</text>
					{#if level >= 1}<text {x} y="42" class="sc ans" class:top={j === S.j}>{fmt3(S.e[j], 2)}</text>{/if}
					{#if level >= 2 && S.a}
						<rect x={j * cw + 7} y={54 + bar - (S.a[j] / wMax) * bar} width={cw - 14} height={Math.max(1.5, (S.a[j] / wMax) * bar)} class="w" class:top={j === S.j} />
						<text {x} y={72 + bar} class="pc">{pc(S.a[j])}</text>
					{/if}
					{#if level === 3 && enc}
						{#each { length: H } as _, d (d)}<rect x={j * cw + 6} y={hy + d * ch} width={cw - 12} height={ch - 1} fill={div.css(enc.data[j * H + d])} opacity={0.1 + (0.9 * (S.a?.[j] ?? 0)) / wMax} />{/each}
					{/if}
				{/each}
				{#if level <= 2}<rect class="ring ans" x={(level ? S.j : A.chars.length - 1) * cw + 2} y="2" width={cw - 4} height={hy - 4} rx="5" />{/if}
				{#if level === 3 && S.c}
					<text x={(A.chars.length + 1) * cw} y="20" class="ch top">c</text>
					<text x={(A.chars.length + 1) * cw} y={hy - 8} class="pc">=</text>
					{#each S.c as v, d (d)}<rect x={(A.chars.length + 0.5) * cw + 6} y={hy + d * ch} width={cw - 12} height={ch - 1} fill={div.css(v)} />{/each}
					{#if A.mix}<rect class="rowf" x="2" y={hy + A.mix.f * ch - 2} width={n * cw - 4} height={ch + 3} rx="2" />{/if}
				{/if}
			</svg>
		</div>
		<p class="small">{t(level === 0 ? L('Circled: your last character. The encoder’s state after it is where the decoder starts.', '圈出的是你的最后一个字符。编码器读完它之后的状态，就是解码器的起点。') : level === 1 ? L('Under each character: its score (circled: the largest).', '每个字符下面是它的分数（圈出的是最大的）。') : L('Bars: the weights in % (they add up to 100).', '条形：权重，单位 %（加起来是 100）。'))}{#if level === 3}{t(L(' Below: each encoder state, faded by its weight; on the right, the context vector c.', ' 下面：每个编码器状态（按权重变淡）；右边是上下文向量 c。'))}{/if}</p>
	{/if}

	<div class="work ans">
		{#if level === 0 && S.sPrev}
			<span class="lab">{t(S.t === 0 ? L('The decoder’s state = the encoder’s last state', '解码器的状态 = 编码器的最后一个状态') : L('The decoder’s state', '解码器的状态'))}</span>
			<TensorStrip data={S.sPrev} name={t(L('decoder state', '解码器状态'))} shape={[S.sPrev.length]} maxCells={64} cellSize={9} domain={[-1, 1]} />
		{:else if level === 1 && A.score}
			{#each strips as s (s.k)}
				<span class="lab">{t(s.name)}</span>
				<TensorStrip data={s.d} name={t(s.name)} shape={[s.d.length]} maxCells={64} cellSize={9} />
			{/each}
			<p class="eq num">{t(L(`score of “${A.chars[S.j]}” = `, `“${A.chars[S.j]}”的分数 = `))}{[0, 1, 2].map((a) => par(A.score!.prod[a])).join(' + ')} + ⋯ + {par(A.score.prod.at(-1)!)} = <Num value={A.score.sum} /></p>
			<p class="small">{t(L(`The ${A.score.prod.length} numbers of the last strip, added up.`, `把最后一条的 ${A.score.prod.length} 个数加起来。`))}</p>
		{:else if level === 2 && S.e}
			{#each A.soft as x (x.j)}<p class="eq num">“{A.chars[x.j]}”: e<sup>{fmt3(S.e[x.j])} − {fmt3(x.m)}</sup> = {dec(x.ex)}; &nbsp; {dec(x.ex)} ÷ {dec(x.sum)} = {fmtPct(x.w)}</p>{/each}
			<p class="small">{t(L(`The two largest weights: e to the power (the score − the largest score), divided by the sum of that over all ${A.chars.length} characters. Subtracting the largest score keeps the powers small; the weights come out the same.`, `最大的两个权重：e 的（分数 − 最大分数）次方，除以所有 ${A.chars.length} 个字符这个值的和。减去最大分数只是让乘方不至于太大，算出的权重不变。`))}</p>
		{:else if level === 3 && A.mix}
			{@const m = A.mix.parts[0]}
			<p class="eq num">c[{A.mix.f}] = {fx(m.w)} × {fx(m.h)} (“{A.chars[m.j]}”) + {par(restOf(A.mix.cf, m.wh), 4)} ({t(L('the others', '其他字符'))}) = {fx(A.mix.cf)}</p>
			<p class="small">{t(L(`Number ${A.mix.f} of the context vector (the outlined row): each state’s number ${A.mix.f} times its weight, added up; the largest part first, then the other ${A.chars.length - 1} characters together.`, `上下文向量的第 ${A.mix.f} 个数（框出的那一行）：每个状态的第 ${A.mix.f} 个数乘以它的权重，再加起来；先列最大的一项，再把其他 ${A.chars.length - 1} 个字符合在一起。`))}</p>
		{:else if level === 4 && S.p}
			<ProbBars items={Array.from(S.p, (p, i) => ({ label: i === 1 ? t(L('end ■', '结束 ■')) : i === 0 ? t(L('start ▶', '起始 ▶')) : TGT[i], p, id: i }))} k={5} chosen={S.y} label={t(L('Chances of the next symbol', '下一个符号的概率'))} />
			{#if S.s}<span class="lab">{t(L('The decoder’s new state', '解码器的新状态'))}</span><TensorStrip data={S.s} name={t(L('new decoder state', '新的解码器状态'))} shape={[S.s.length]} maxCells={64} cellSize={9} domain={[-1, 1]} />{/if}
		{/if}
	</div>
</div>

<style>
	.at { display: grid; gap: 10px; min-width: 0; }
	.so { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
	.lab { font-size: var(--fs-sm); color: var(--ink-2); }
	.slot { display: inline-grid; place-items: center; width: 1.5em; height: 1.8em; border: 1px solid var(--rule-strong); border-radius: 4px; font-size: var(--fs-lg); }
	.slot.tpl { color: var(--muted); opacity: 0.5; font-size: var(--fs-sm); }
	.slot.cur { border: 2px solid var(--accent); color: var(--accent-strong); background: var(--accent-soft); }
	.box { overflow-x: auto; max-width: 100%; }
	svg { display: block; }
	.ch { font: 600 18px var(--font-mono); text-anchor: middle; fill: var(--ink); }
	.sc, .pc { font: 12px var(--font-mono); text-anchor: middle; fill: var(--ink-2); }
	.top { fill: var(--accent-strong); font-weight: 700; }
	.w { fill: var(--seq-mid); }
	.w.top { fill: var(--seq-hi); }
	.ring { fill: none; stroke: var(--accent); stroke-width: 2.5; }
	.rowf { fill: none; stroke: var(--ink); stroke-width: 1.5; }
	.work { display: grid; gap: 4px; min-width: 0; overflow-x: auto; }
	.eq { margin: 4px 0 0; overflow-wrap: anywhere; }
	.small { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
	:global([inert]) .ans { visibility: hidden; }
</style>
