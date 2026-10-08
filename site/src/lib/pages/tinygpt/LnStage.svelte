<!--
  Lesson step 3: LayerNorm on the followed token, as three vectors on one scale: the input x
  (block 1: the embedding), the normalized z = (x − μ)/√(σ² + ε) (mean 0, spread 1, computed here
  from x) and the output y = γ·z + β (the traced L{b}.ln1). One dimension worked out with live numbers.
-->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { sym } from '$lib/kit/tex';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import VecBars from './VecBars.svelte';
	import type { TGState } from './state.svelte';
	import { D } from './state.svelte';
	import { row } from './analysis';
	import { meanSd } from './lesson';

	let { LS }: { LS: TGState } = $props();

	const EPS = 1e-5;
	const f = $derived(LS.run?.forward);
	const pos = $derived(LS.pos);
	const b = $derived(LS.block);
	const inId = $derived(b === 0 ? 'embed' : `L${b - 1}.resid2`);
	const v = $derived.by(() => {
		if (!f || !LS.model) return null;
		const x = row(f.t[inId].data, pos, D);
		const y = row(f.t[`L${b}.ln1`].data, pos, D);
		const st = meanSd(x);
		const den = Math.sqrt(st.sd * st.sd + EPS);
		const z = new Float32Array(D);
		for (let i = 0; i < D; i++) z[i] = (x[i] - st.mean) / den;
		const g = LS.model.weight(`L${b}.ln1.weight`).data;
		const be = LS.model.weight(`L${b}.ln1.bias`).data;
		let max = 0;
		for (let i = 0; i < D; i++) max = Math.max(max, Math.abs(x[i]), Math.abs(z[i]), Math.abs(y[i]));
		// spread of every token in the window, before and after
		const T = f.ids.length;
		let bLo = Infinity, bHi = 0, aLo = Infinity, aHi = 0;
		for (let r = 0; r < T; r++) {
			const sb = meanSd(row(f.t[inId].data, r, D)).sd;
			const sa = meanSd(row(f.t[`L${b}.ln1`].data, r, D)).sd;
			bLo = Math.min(bLo, sb);
			bHi = Math.max(bHi, sb);
			aLo = Math.min(aLo, sa);
			aHi = Math.max(aHi, sa);
		}
		return { x, y, z, mu: st.mean, sd: st.sd, den, g, be, max, yst: meanSd(y), T, bLo, bHi, aLo, aHi };
	});
	const d = $derived(Math.min(D - 1, Math.max(0, LS.dim)));
	const pick = (i: number) => (LS.dim = i);
</script>

{#if v}
	<div class="ln">
		<p class="head">
			{t({ en: `Block ${b + 1}, LayerNorm 1, token`, zh: `第 ${b + 1} 块，层归一化 1，词元` })} <span class="tok">{LS.label(pos)}</span>
			<span class="muted">· {t({ en: `one scale for all three (±${fmt3(v.max)}); the band is mean ± spread`, zh: `三行使用同一个刻度（±${fmt3(v.max)}）；色带表示均值 ± 标准差` })}</span>
		</p>
		<div class="rows">
			<div class="r">
				<VecBars data={v.x} max={v.max} band={{ mean: v.mu, sd: v.sd }} selected={d} onselect={pick} name="x" height={60} label={{ en: 'Input vector x', zh: '输入向量 x' }} />
				<p class="st num">μ = <Num value={v.mu} />, σ = <Num value={v.sd} /></p>
			</div>
			<div class="r">
				<VecBars data={v.z} max={v.max} band={{ mean: 0, sd: 1 }} selected={d} onselect={pick} name="(x−μ)/σ" height={60} label={{ en: 'Normalized vector', zh: '归一化后的向量' }} />
				<p class="st num">μ = 0, σ = 1 <span class="muted">· {t({ en: 'subtract the mean, divide by the spread', zh: '减去均值，除以标准差' })}</span></p>
			</div>
			<div class="r">
				<VecBars data={v.y} max={v.max} band={{ mean: v.yst.mean, sd: v.yst.sd }} selected={d} onselect={pick} name="γ·z + β" height={60} label={{ en: 'LayerNorm output', zh: '层归一化的输出' }} />
				<p class="st num">μ = <Num value={v.yst.mean} />, σ = <Num value={v.yst.sd} /> <span class="muted">· {t({ en: 'learned gain γ and shift β, one pair per dimension', zh: '学到的增益 γ 和偏移 β，每个维度一对' })}</span></p>
			</div>
		</div>
		<FormulaCard
			tex={`\\begin{aligned}${sym('out', 'y_{' + d + '}')} &= ${sym('param', '\\gamma_{' + d + '}')}\\,\\frac{${sym('in', 'x_{' + d + '}')} - \\mu}{\\sqrt{\\sigma^2+\\epsilon}} + ${sym('param', '\\beta_{' + d + '}')} \\\\ &= {{g}} \\cdot \\frac{ {{x}} - ({{mu}}) }{ {{den}} } \\\\ &\\quad + ({{be}}) = {{y}}\\end{aligned}`}
			values={{ g: v.g[d], x: v.x[d], mu: v.mu, den: v.den, be: v.be[d], y: v.y[d] }}
			caption={{ en: `Dimension ${d} (click a bar to pick another)`, zh: `第 ${d} 维（点击柱子换一维）` }}
			where={[
				{ tex: '\\mu, \\sigma', text: { en: 'mean and spread of this token’s 76 numbers', zh: '这个词元 76 个数的均值与标准差' } },
				{ tex: '\\epsilon', text: { en: '10⁻⁵, only to avoid dividing by 0', zh: '10⁻⁵，只是为了避免除以 0' } }
			]}
			provenance="live"
		/>
		{#if v.T > 1}
			<p class="all muted">
				{t({
					en: `All ${v.T} tokens of your text: spread before ${fmt3(v.bLo)}–${fmt3(v.bHi)}, after ${fmt3(v.aLo)}–${fmt3(v.aHi)}.`,
					zh: `你的文本的全部 ${v.T} 个词元：之前的标准差 ${fmt3(v.bLo)}–${fmt3(v.bHi)}，之后 ${fmt3(v.aLo)}–${fmt3(v.aHi)}。`
				})}
			</p>
		{/if}
	</div>
{/if}

<style>
	.ln {
		display: grid;
		gap: 10px;
		grid-template-columns: minmax(0, 1fr);
	}
	.head {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.head .muted {
		font-size: var(--fs-xs);
	}
	.tok {
		font-family: var(--font-mono);
		padding: 0 4px;
		border-radius: 3px;
		white-space: pre;
		background: var(--surface-2);
		box-shadow: inset 0 0 0 2px var(--q);
		font-variant-ligatures: none;
	}
	.rows {
		display: grid;
		gap: 4px;
	}
	.r {
		display: grid;
		gap: 0;
	}
	.st {
		margin: -2px 0 0;
		padding-left: calc(4.5em + 8px);
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.all {
		margin: 0;
		font-size: var(--fs-xs);
	}
	@media (max-width: 480px) {
		.st {
			padding-left: 0;
		}
	}
</style>
