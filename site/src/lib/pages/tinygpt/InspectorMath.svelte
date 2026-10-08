<!--
  Inspector "Math" tab: the operation that produced the selected tensor, redone by hand for the
  selected token (and dimension / head / key), with the live numbers.
-->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { sym } from '$lib/kit/tex';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { TGState } from './state.svelte';
	import { D, DH, DMLP, V } from './state.svelte';
	import { argmax, mean, row, std } from './analysis';
	import { linearTerms, linearTex } from './linear';

	let { S, id }: { S: TGState; id: string } = $props();

	const f = $derived(S.run?.forward);
	const T = $derived(S.T);
	const pos = $derived(S.pos);
	const i = $derived(S.dim);

	type Card = { tex: string; values: Record<string, number>; caption: { en: string; zh: string }; where?: { tex: string; text: { en: string; zh: string } }[] };

	const card = $derived.by((): Card | null => {
		if (!f || !S.model) return null;
		const W = (n: string) => S.model!.weight(n).data;
		const tn = (n: string) => f.t[n].data;
		const m = /^L(\d)\.(.+)$/.exec(id);
		const p = m ? `L${m[1]}.` : '';
		const op = m ? m[2] : id;
		const blockIn = m ? (m[1] === '0' ? 'embed' : `L${Number(m[1]) - 1}.resid2`) : '';
		const cap = (en: string, zh: string) => ({ en, zh });
		const tokId = f.ids[pos];
		if (op === 'tok_emb')
			return {
				tex: `${sym('out', 'x^{tok}_{' + pos + ',' + i + '}')} = ${sym('param', 'E')}[${tokId}, ${i}] = {{v}}`,
				values: { v: tn('tok_emb')[pos * D + i] },
				caption: cap(`Token ${pos} has id ${tokId}: copy row ${tokId} of E [1024×76]`, `词元 ${pos} 的 id 是 ${tokId}：复制 E [1024×76] 的第 ${tokId} 行`)
			};
		if (op === 'pos_emb')
			return {
				tex: `${sym('out', 'x^{pos}_{' + pos + ',' + i + '}')} = ${sym('param', 'P')}[${pos}, ${i}] = {{v}}`,
				values: { v: tn('pos_emb')[pos * D + i] },
				caption: cap(`Position ${pos}: copy row ${pos} of P [64×76]`, `位置 ${pos}：复制 P [64×76] 的第 ${pos} 行`)
			};
		if (op === 'embed' || op === 'resid1' || op === 'resid2') {
			const [a, b] = op === 'embed' ? ['tok_emb', 'pos_emb'] : op === 'resid1' ? [blockIn, `${p}attn.proj`] : [`${p}resid1`, `${p}mlp.proj`];
			return {
				tex: `${sym('out', 'x_{' + pos + ',' + i + '}')} = ${sym('in', 'a_{' + pos + ',' + i + '}')} + ${sym('accent', 'b_{' + pos + ',' + i + '}')} = {{a}} + ({{b}}) = {{c}}`,
				values: { a: tn(a)[pos * D + i], b: tn(b)[pos * D + i], c: tn(id)[pos * D + i] },
				caption: cap(`${a} + ${b}`, `${a} + ${b}`)
			};
		}
		if (op === 'ln1' || op === 'ln2' || op === 'ln_f') {
			const src = op === 'ln1' ? blockIn : op === 'ln2' ? `${p}resid1` : 'L3.resid2';
			const x = row(tn(src), pos, D);
			const mu = mean(x);
			const sd = Math.sqrt(std(x) ** 2 + 1e-5);
			return {
				tex: `\\begin{aligned}${sym('out', 'y_{' + i + '}')} &= ${sym('param', '\\gamma_{' + i + '}')}\\frac{${sym('in', 'x_{' + i + '}')} - \\mu}{\\sqrt{\\sigma^2+\\epsilon}} + ${sym('param', '\\beta_{' + i + '}')} \\\\ &= {{g}}\\cdot\\frac{ {{x}} - ({{mu}}) }{ {{sd}} } + ({{b}}) \\\\ &= {{y}}\\end{aligned}`,
				values: { g: W(`${id}.weight`)[i], x: x[i], mu, sd, b: W(`${id}.bias`)[i], y: tn(id)[pos * D + i] },
				caption: cap(`Token ${pos}, dimension ${i}`, `词元 ${pos}，第 ${i} 维`)
			};
		}
		if (op === 'attn.q' || op === 'attn.k' || op === 'attn.v') {
			const L = op.slice(-1).toUpperCase();
			const r = linearTerms(row(tn(`${p}ln1`), pos, D), W(`${id}.weight`), W(`${id}.bias`), D, i);
			const lt = linearTex(`${L.toLowerCase()}_{${pos},${i}}`, 'h_i', `W^${L}_{${i},i}`, `b^${L}_{${i}}`, r, L.toLowerCase() as 'q' | 'k' | 'v');
			return { ...lt, caption: cap(`Token ${pos}, column ${i} (head ${Math.floor(i / DH) + 1})`, `词元 ${pos}，第 ${i} 列（第 ${Math.floor(i / DH) + 1} 头）`) };
		}
		if (op === 'attn.scores' || op === 'attn.masked' || op === 'attn.weights') {
			const h = S.head;
			const j = Math.min(S.key, pos);
			const o = (h * T + pos) * T + j;
			if (op === 'attn.scores')
				return {
					tex: `${sym('accent', 's_{' + pos + ',' + j + '}')} = \\frac{${sym('q', 'q_{' + pos + '}')}\\cdot ${sym('k', 'k_{' + j + '}')}}{\\sqrt{19}} = {{s}}`,
					values: { s: tn(id)[o] },
					caption: cap(`Head ${h + 1}, query ${pos}, key ${j} (full arithmetic in "Inside a block")`, `第 ${h + 1} 头，查询 ${pos}，键 ${j}（完整算式见“块的内部”）`)
				};
			if (op === 'attn.masked')
				return {
					tex: `s'_{ij} = \\begin{cases} s_{ij} & j \\le i \\\\ -\\infty & j > i \\end{cases} \\qquad s'_{${pos},${j}} = {{s}}`,
					values: { s: tn(id)[o] },
					caption: cap(`Head ${h + 1}: keys after the query are set to −∞`, `第 ${h + 1} 头：查询之后的键被设为 −∞`)
				};
			return {
				tex: `${sym('prob', 'a_{' + pos + ',' + j + '}')} = \\frac{e^{s_{${pos},${j}}}}{\\sum_{j' \\le ${pos}} e^{s_{${pos},j'}}} = {{a}}`,
				values: { a: tn(id)[o] },
				caption: cap(`Head ${h + 1}, row ${pos} sums to 1`, `第 ${h + 1} 头，第 ${pos} 行之和为 1`)
			};
		}
		if (op === 'attn.heads') {
			const h = Math.floor(i / DH);
			const c = i % DH;
			let s = 0;
			const w = tn(`${p}attn.weights`);
			const v = tn(`${p}attn.v`);
			for (let j = 0; j <= pos; j++) s += w[(h * T + pos) * T + j] * v[j * D + h * DH + c];
			return {
				tex: `${sym('out', 'o_{' + pos + ',' + i + '}')} = \\sum_{j \\le ${pos}} ${sym('prob', 'a^{(' + (h + 1) + ')}_{' + pos + ',j}')}\\, ${sym('v', 'v_{j,' + i + '}')} = {{s}}`,
				values: { s },
				caption: cap(`Column ${i} belongs to head ${h + 1}`, `第 ${i} 列属于第 ${h + 1} 头`)
			};
		}
		if (op === 'attn.proj' || op === 'mlp.fc' || op === 'mlp.proj') {
			const src = op === 'attn.proj' ? `${p}attn.heads` : op === 'mlp.fc' ? `${p}ln2` : `${p}mlp.gelu`;
			const nIn = op === 'mlp.proj' ? DMLP : D;
			const o = op === 'mlp.fc' ? Math.min(DMLP - 1, i) : i;
			const r = linearTerms(row(tn(src), pos, nIn), W(`${id}.weight`), W(`${id}.bias`), nIn, o);
			const lt = linearTex(`y_{${pos},${o}}`, 'x_i', `W_{${o},i}`, `b_{${o}}`, r);
			return { ...lt, caption: cap(`Token ${pos}, output ${o} (input: ${src})`, `词元 ${pos}，第 ${o} 个输出（输入：${src}）`) };
		}
		if (op === 'mlp.gelu') {
			const u = Math.min(DMLP - 1, i);
			return {
				tex: `\\mathrm{GELU}(x) = \\tfrac12 x\\big(1+\\tanh(\\sqrt{2/\\pi}(x+0.044715x^3))\\big) \\qquad \\mathrm{GELU}({{x}}) = {{y}}`,
				values: { x: tn(`${p}mlp.fc`)[pos * DMLP + u], y: tn(id)[pos * DMLP + u] },
				caption: cap(`Token ${pos}, unit ${u}`, `词元 ${pos}，第 ${u} 个单元`)
			};
		}
		if (op === 'logits' || op === 'probs') {
			const lg = row(tn('logits'), pos, V);
			const best = argmax(lg);
			if (op === 'logits') {
				const r = linearTerms(row(tn('ln_f'), pos, D), W('tok_emb.weight'), null, D, best);
				const lt = linearTex(`z_{${best}}`, 'x_i', `E_{${best},i}`, null, r);
				return { ...lt, caption: cap(`Logit of the top token "${S.idLabel(best)}" (id ${best}) after token ${pos}`, `词元 ${pos} 之后最高分词元 “${S.idLabel(best)}”（id ${best}）的 logit`) };
			}
			let mx = -Infinity;
			for (let k = 0; k < V; k++) mx = Math.max(mx, lg[k]);
			let Z = 0;
			for (let k = 0; k < V; k++) Z += Math.exp(lg[k] - mx);
			return {
				tex: `\\begin{aligned}${sym('prob', 'p_{' + best + '}')} &= \\frac{e^{z_{${best}} - m}}{\\sum_{v=0}^{1023} e^{z_v - m}} \\\\ &= \\frac{ {{num}} }{ {{Z}} } = {{p}}\\end{aligned}`,
				values: { num: Math.exp(lg[best] - mx), Z, p: tn('probs')[pos * V + best] },
				caption: cap(`Top token "${S.idLabel(best)}" after token ${pos}`, `词元 ${pos} 之后的第一名 “${S.idLabel(best)}”`)
			};
		}
		return null;
	});
</script>

{#if card}
	<FormulaCard tex={card.tex} values={card.values} caption={card.caption} where={card.where ?? []} provenance="live" />
	<p class="hint">
		{t({
			en: 'Numbers are for the selected token and dimension. Pick another token above (token chips) or another dimension by clicking a cell in the views.',
			zh: '数字对应当前选中的词元和维度。可在上方点词元换一个词元，或在视图中点击格子换一个维度。'
		})}
	</p>
{:else}
	<p class="muted">{t({ en: 'No data yet.', zh: '暂无数据。' })}</p>
{/if}

<style>
	.hint {
		font-size: var(--fs-xs);
		color: var(--muted);
		margin: 8px 0 0;
	}
</style>
