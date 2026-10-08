<!--
  Inside one transformer block, step by step (StepPlayer): LayerNorm → Q/K/V → scores → causal mask →
  softmax → weighted sum of V → concat heads → output projection → residual add → LayerNorm → MLP expand →
  GELU → MLP project → residual add. Each step shows the real [T×…] tensors for the learner's text, the
  selected token's row, and a formula card filled with the live numbers.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import TensorStrip from '$lib/kit/TensorStrip.svelte';
	import AttentionMatrix from '$lib/kit/AttentionMatrix.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import Num from '$lib/kit/Num.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import GeluPlot from './GeluPlot.svelte';
	import { sym } from '$lib/kit/tex';
	import { domainOf } from '$lib/kit/color';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import { fmtShape, fmt3, fmtPct } from '$lib/kit/format';
	import type { CellValue } from '$lib/kit/types';
	import type { TGState } from './state.svelte';
	import { D, DH, DMLP, H, NL } from './state.svelte';
	import { headSlice, mean, norm, row, std, topIndices } from './analysis';
	import { linearTerms, linearTex } from './linear';

	interface Props {
		S: TGState;
		step?: number;
		/** lesson use: play only these steps (indices into the 14 steps); a single step shows no player */
		only?: number[];
		/** lesson use: no block / head / step controls (the lesson owns them); followed-token strips in the Q, K, V step */
		focus?: boolean;
		/** start playing (e.g. when a lesson step is shown) */
		autoplay?: boolean;
		/** ms per step */
		interval?: number;
		/** playback reached the last step */
		onend?: () => void;
	}
	let { S, step = $bindable(0), only, focus = false, autoplay = false, interval = 1800, onend }: Props = $props();

	type StepDef = { layer: string; title: Text; label: Text };
	const STEPS: StepDef[] = [
		{ layer: 'ln1', title: { en: 'LayerNorm 1', zh: '层归一化 1' }, label: { en: 'LayerNorm: each token’s 76 numbers are shifted to mean 0 and scaled to spread 1, then a learned gain γ and bias β are applied.', zh: '层归一化：每个词元的 76 个数先平移到均值 0、缩放到标准差 1，再乘以学到的增益 γ、加上偏置 β。' } },
		{ layer: 'attn.q', title: { en: 'Q, K, V', zh: 'Q、K、V' }, label: { en: 'Three linear maps make a query, a key and a value for every token. Each head uses its own 19 of the 76 columns.', zh: '三个线性变换为每个词元生成查询、键和值。每个头使用 76 列中属于自己的 19 列。' } },
		{ layer: 'attn.scores', title: { en: 'Scores', zh: '注意力分数' }, label: { en: 'Scores: every query is compared with every key by a dot product, divided by √19 ≈ 4.36.', zh: '分数：每个查询与每个键做点积，再除以 √19 ≈ 4.36。' } },
		{ layer: 'attn.masked', title: { en: 'Causal mask', zh: '因果掩码' }, label: { en: 'Causal mask: a token may not look at later tokens, so their scores become −∞ (hatched).', zh: '因果掩码：词元不能看后面的词元，所以那些分数被设为 −∞（阴影）。' } },
		{ layer: 'attn.weights', title: { en: 'Softmax', zh: 'Softmax' }, label: { en: 'Softmax turns each row of scores into weights that are ≥ 0 and add up to 1.', zh: 'Softmax 把每一行分数变成非负、且总和为 1 的权重。' } },
		{ layer: 'attn.heads', title: { en: 'Weighted sum of V', zh: 'V 的加权和' }, label: { en: 'The head’s output for a token is a mix of the value vectors, weighted by its attention row.', zh: '一个头对某个词元的输出，是按它那一行注意力权重对值向量的加权混合。' } },
		{ layer: 'attn.heads', title: { en: 'Concat heads', zh: '拼接各头' }, label: { en: 'The 4 head outputs (19 numbers each) are placed side by side: 4 × 19 = 76.', zh: '4 个头的输出（各 19 个数）并排拼接：4 × 19 = 76。' } },
		{ layer: 'attn.proj', title: { en: 'Output projection', zh: '输出投影' }, label: { en: 'A 76×76 linear map mixes the heads into one update for the stream.', zh: '一个 76×76 的线性变换把各头混合成对残差流的一次更新。' } },
		{ layer: 'resid1', title: { en: 'Residual add 1', zh: '残差相加 1' }, label: { en: 'Residual add: the attention result is added to the vector that came in (x + attention).', zh: '残差相加：把注意力的结果加到进来的向量上（x + 注意力）。' } },
		{ layer: 'ln2', title: { en: 'LayerNorm 2', zh: '层归一化 2' }, label: { en: 'LayerNorm again, before the MLP.', zh: '在 MLP 之前再做一次层归一化。' } },
		{ layer: 'mlp.fc', title: { en: 'MLP expand', zh: 'MLP 扩展' }, label: { en: 'MLP, part 1: each token alone goes through a 76 → 304 linear layer (4× wider).', zh: 'MLP 第 1 步：每个词元单独经过 76 → 304 的线性层（宽 4 倍）。' } },
		{ layer: 'mlp.gelu', title: { en: 'GELU', zh: 'GELU' }, label: { en: 'GELU: a smooth curve that keeps positive inputs almost as they are and squashes negative inputs toward 0.', zh: 'GELU：一条平滑的曲线，正的输入基本保持不变，负的输入被压向 0。' } },
		{ layer: 'mlp.proj', title: { en: 'MLP project', zh: 'MLP 投影' }, label: { en: 'MLP, part 2: 304 → 76, back to the width of the stream.', zh: 'MLP 第 2 步：304 → 76，回到残差流的宽度。' } },
		{ layer: 'resid2', title: { en: 'Residual add 2', zh: '残差相加 2' }, label: { en: 'Residual add: x + MLP. This is the block’s output and the next block’s input.', zh: '残差相加：x + MLP。这是本块的输出，也是下一块的输入。' } }
	];
	const labels = STEPS.map((s) => s.label);
	/** the player's position inside `only` (lesson use) */
	let ps = $state(0);
	const onlyLabels = $derived(only ? only.map((i) => STEPS[i].label) : labels);
	/** the step shown now (index into STEPS) */
	const cur = $derived(only ? only[Math.min(ps, only.length - 1)] : step);

	const f = $derived(S.run?.forward);
	const T = $derived(S.T);
	const b = $derived(S.block);
	const h = $derived(S.head);
	const pos = $derived(S.pos);
	const p = $derived(`L${b}.`);
	const inId = $derived(b === 0 ? 'embed' : `L${b - 1}.resid2`);
	const tn = (id: string) => f!.t[id].data;
	// lesson (focus) captions name the token instead of its 0-based index and hide internal tensor ids
	const tokEn = (i: number) => (focus ? `“${S.label(i)}”` : `token ${i}`);
	const tokZh = (i: number) => (focus ? `“${S.label(i)}”` : `词元 ${i} `);
	const PART = { en: 'Part {i} of {n}', zh: '第 {i} / {n} 部分' };
	const W = (name: string) => S.model!.weight(name).data;

	// --- keep the inspector in step with the player, and the player with the inspector ---
	let synced = '';
	$effect(() => {
		const id = `L${b}.${STEPS[cur].layer}`;
		if (id === synced) return;
		synced = id;
		untrack(() => (S.layerId = id));
	});
	$effect(() => {
		if (only) return;
		const id = S.layerId;
		if (id === synced) return;
		const m = /^L(\d)\.(.+)$/.exec(id);
		if (!m) return;
		const i = STEPS.findIndex((s) => s.layer === m[2]);
		if (i < 0) return;
		synced = id;
		untrack(() => {
			S.block = Number(m[1]);
			step = i;
		});
	});

	// --- selection inside the step views ---
	let hc = $state(0); // channel inside one head (0..18)
	let unit = $state(0); // MLP unit (0..303)
	function pickRow(c: CellValue, cols: number) {
		S.selectPos(c.row);
		if (cols === D) S.dim = c.col;
		else if (cols === DH) hc = c.col;
		else if (cols === DMLP) unit = c.col;
	}
	// default MLP unit: the one with the largest GELU output for the selected token
	$effect(() => {
		if (!f) return;
		const g = row(tn(`L${b}.mlp.gelu`), pos, DMLP);
		untrack(() => (unit = topIndices(g, 1)[0] ?? 0));
	});
	const rowHL = $derived([{ row: pos, col: 0, h: 1, w: DMLP, style: 'solid' as const, color: 'var(--q)' }]);
	const setDim = (i: number) => (S.dim = i);

	// --- per-step numbers ---
	const ln = $derived.by(() => {
		if (!f || !S.model) return null;
		const which = cur >= 9 ? 'ln2' : 'ln1';
		const src = which === 'ln1' ? inId : `${p}resid1`;
		const x = row(tn(src), pos, D);
		const y = row(tn(`${p}${which}`), pos, D);
		const mu = mean(x);
		const sd = Math.sqrt(std(x) ** 2 + 1e-5);
		const i = S.dim;
		const g = W(`${p}${which}.weight`)[i];
		const be = W(`${p}${which}.bias`)[i];
		return { which, src, x, y, mu, sd, i, g, be, xi: x[i], yi: y[i], muY: mean(y), sdY: std(y) };
	});
	const qkv = $derived.by(() => {
		if (!f) return null;
		const sl = (n: string) => headSlice(tn(`${p}attn.${n}`), T, D, h, DH);
		return { q: sl('q'), k: sl('k'), v: sl('v') };
	});
	const qkvDom = $derived(qkv ? domainOf('diverging', qkv.q, qkv.k, qkv.v) : undefined);
	const qkvList = $derived(
		qkv
			? [
					{ r: 'q', name: 'Q', data: qkv.q },
					{ r: 'k', name: 'K', data: qkv.k },
					{ r: 'v', name: 'V', data: qkv.v }
				]
			: []
	);
	const scoresH = $derived(f ? tn(`${p}attn.scores`).subarray(h * T * T, (h + 1) * T * T) : null);
	const scoreDom = $derived.by(() => {
		if (!scoresH) return undefined;
		const fin: number[] = [];
		for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) fin.push(scoresH[i * T + j]);
		return domainOf('diverging', fin);
	});
	const weightsH = $derived(f ? tn(`${p}attn.weights`).subarray(h * T * T, (h + 1) * T * T) : null);
	const key = $derived(Math.min(S.key, pos));
	const mix = $derived.by(() => {
		if (!f || !weightsH || !qkv) return null;
		const ws = Array.from({ length: pos + 1 }, (_, j) => ({ j, a: weightsH[pos * T + j] })).sort((x, y) => y.a - x.a);
		const shown = ws.slice(0, 6);
		const rest = ws.slice(6).reduce((s, x) => s + x.a, 0);
		const out = row(headSlice(tn(`${p}attn.heads`), T, D, h, DH), pos, DH);
		let sum = 0;
		for (const { j, a } of ws) sum += a * qkv.v[j * DH + hc];
		return { shown, rest, nRest: ws.length - shown.length, out, sum, vDom: domainOf('diverging', qkv.v) };
	});
	const res = $derived.by(() => {
		if (!f) return null;
		const second = cur >= 9;
		const xin = row(tn(second ? `${p}resid1` : inId), pos, D);
		const delta = row(tn(second ? `${p}mlp.proj` : `${p}attn.proj`), pos, D);
		const out = row(tn(second ? `${p}resid2` : `${p}resid1`), pos, D);
		const dom = domainOf('diverging', xin, delta, out);
		return { second, xin, delta, out, dom, nx: norm(xin), nd: norm(delta), no: norm(out) };
	});
	const lin = $derived.by(() => {
		if (!f || !S.model) return null;
		const s = STEPS[cur].layer;
		if (s === 'attn.q') {
			const c = h * DH + hc;
			const r = linearTerms(row(tn(`${p}ln1`), pos, D), W(`${p}attn.q.weight`), W(`${p}attn.q.bias`), D, c);
			return linearTex(`q_{${pos},${c}}`, 'x_i', `W^Q_{${c},i}`, `b^Q_{${c}}`, r, 'q');
		}
		if (s === 'attn.proj') {
			const r = linearTerms(row(tn(`${p}attn.heads`), pos, D), W(`${p}attn.proj.weight`), W(`${p}attn.proj.bias`), D, S.dim);
			return linearTex(`a_{${pos},${S.dim}}`, 'o_i', `W^O_{${S.dim},i}`, `b^O_{${S.dim}}`, r);
		}
		if (s === 'mlp.fc') {
			const r = linearTerms(row(tn(`${p}ln2`), pos, D), W(`${p}mlp.fc.weight`), W(`${p}mlp.fc.bias`), D, unit);
			return linearTex(`h_{${pos},${unit}}`, 'x_i', `W^{(1)}_{${unit},i}`, `b^{(1)}_{${unit}}`, r);
		}
		if (s === 'mlp.proj') {
			const r = linearTerms(row(tn(`${p}mlp.gelu`), pos, DMLP), W(`${p}mlp.proj.weight`), W(`${p}mlp.proj.bias`), DMLP, S.dim);
			return linearTex(`m_{${pos},${S.dim}}`, 'g_i', `W^{(2)}_{${S.dim},i}`, `b^{(2)}_{${S.dim}}`, r);
		}
		return null;
	});
	const blocks = Array.from({ length: NL }, (_, i) => i);
	const heads = Array.from({ length: H }, (_, i) => i);
	const lensAt = (id: string) => {
		const L = S.run?.lens;
		if (!L) return null;
		const pi = L.points.indexOf(id);
		if (pi < 0) return null;
		const o = (pi * L.T + pos) * L.K;
		return { id: L.topIds[o], p: L.topP[o] };
	};
</script>

{#if f && S.model}
	<div class="bv" class:focus>
		{#if !focus}
		<div class="controls">
			<span class="ctl">
				<span class="lbl">{t({ en: 'Block', zh: '块' })}</span>
				<span class="segmented" role="group" aria-label={t({ en: 'Block', zh: '块' })}>
					{#each blocks as i (i)}
						<button type="button" aria-pressed={b === i} onclick={() => (S.block = i)}>{i + 1}</button>
					{/each}
				</span>
			</span>
			<span class="ctl">
				<span class="lbl">{t({ en: 'Head', zh: '头' })}</span>
				<span class="segmented" role="group" aria-label={t({ en: 'Head', zh: '头' })}>
					{#each heads as i (i)}
						<button type="button" aria-pressed={h === i} onclick={() => ((S.head = i), (S.keySel = null))}>{i + 1}</button>
					{/each}
				</span>
			</span>
			<span class="ctl crumbs num" aria-label={t({ en: 'Where you are', zh: '当前位置' })}>
				{t({ en: 'Model', zh: '模型' })} › {t({ en: `Block ${b + 1}`, zh: `第 ${b + 1} 块` })} › {t(STEPS[step].title)}
				› <span class="tok">{S.label(pos)}</span>
			</span>
		</div>

		<nav class="steps" aria-label={t({ en: 'Steps inside the block', zh: '块内的步骤' })}>
			{#each STEPS as st, i (i)}
				<button type="button" class:cur={i === step} class:mlp={i >= 9} aria-current={i === step ? 'step' : undefined} onclick={() => (step = i)}>
					<span class="sn num">{i + 1}</span>{t(st.title)}
				</button>
				{/each}
			</nav>
		{/if}

		{#snippet body(s: number)}
					<div class="stage">
					{#if s === 0 || s === 9}
						{#if ln}
							<div class="cols">
								<div class="vis">
									<div class="pair">
										<div>
											<p class="cap">{t({ en: 'before', zh: '之前' })}: {#if focus}{ln.which === 'ln1' ? t({ en: 'x, the block’s input', zh: 'x，本块的输入' }) : t({ en: 'x after attention', zh: '注意力之后的 x' })}{:else}<span class="mono">{ln.src}</span>{/if} {fmtShape([T, D])}</p>
											<TensorGrid data={tn(ln.src)} rows={T} cols={D} maxCell={10} highlights={rowHL} label={ln.src} onselect={(c) => pickRow(c, D)} />
										</div>
										<div>
											<p class="cap">{t({ en: 'after', zh: '之后' })}: {#if focus}LN(x){:else}<span class="mono">{p}{ln.which}</span>{/if} {fmtShape([T, D])}</p>
											<TensorGrid data={tn(`${p}${ln.which}`)} rows={T} cols={D} maxCell={10} highlights={rowHL} label="{p}{ln.which}" onselect={(c) => pickRow(c, D)} />
										</div>
									</div>
									<p class="cap">{t({ en: 'token', zh: '词元' })} <span class="tok">{S.label(pos)}</span>: x {t({ en: 'has', zh: '的' })} μ = <Num value={ln.mu} />, σ = <Num value={ln.sd} /> → LN(x) {t({ en: 'has', zh: '的' })} μ = <Num value={ln.muY} />, σ = <Num value={ln.sdY} /></p>
									<div class="srow"><span class="sl">x</span><TensorStrip data={ln.x} shape={[D]} maxCells={38} cellSize={11} name={{ en: 'x before', zh: '归一化前' }} selected={S.dim} onselect={setDim} /></div>
									<div class="srow"><span class="sl">LN(x)</span><TensorStrip data={ln.y} shape={[D]} maxCells={38} cellSize={11} name={{ en: 'after', zh: '归一化后' }} selected={S.dim} onselect={setDim} /></div>
								</div>
								<FormulaCard
									tex={`\\begin{aligned}${sym('out', 'y_{' + ln.i + '}')} &= ${sym('param', '\\gamma_{' + ln.i + '}')}\\,\\frac{${sym('in', 'x_{' + ln.i + '}')} - \\mu}{\\sqrt{\\sigma^2+\\epsilon}} + ${sym('param', '\\beta_{' + ln.i + '}')} \\\\ &= {{g}} \\cdot \\frac{ {{x}} - ({{mu}}) }{ {{sd}} } + ({{be}}) \\\\ &= {{y}}\\end{aligned}`}
									values={{ g: ln.g, x: ln.xi, mu: ln.mu, sd: ln.sd, be: ln.be, y: ln.yi }}
									caption={{ en: `Dimension ${ln.i} of ${tokEn(pos)} (click a cell to pick another)`, zh: `${tokZh(pos)}的第 ${ln.i} 维（点击格子换一维）` }}
									where={[
										{ tex: '\\mu, \\sigma', text: { en: 'mean and spread of this token’s 76 numbers', zh: '这个词元 76 个数的均值与标准差' } },
										{ tex: sym('param', '\\gamma, \\beta'), text: { en: 'learned gain and bias (76 each)', zh: '学到的增益和偏置（各 76 个）' } }
									]}
									provenance="live"
								/>
							</div>
							<p class="note">
								{t({
									en: 'LayerNorm works on each token separately, across its 76 numbers (not across tokens). It keeps the scale of the stream steady, so the next layer always gets inputs of a similar size.',
									zh: '层归一化对每个词元单独进行，作用在它自己的 76 个数上（而不是跨词元）。它让残差流的数值尺度保持稳定，下一层总能拿到大小相近的输入。'
								})}
							</p>
						{/if}
						{:else if s === 1}
							{#if qkv}
								{#if focus}
									<div class="follow">
										<p class="cap">
											{t({ en: 'The followed token', zh: '跟踪的词元' })} <span class="tok">{S.label(pos)}</span>:
											{t({ en: `its normalized vector, times head ${h + 1}’s own 19 columns of each weight matrix`, zh: `它归一化后的向量，分别乘以三个权重矩阵中属于第 ${h + 1} 头的 19 列` })}
										</p>
										<div class="srow"><span class="sl">LN(x)</span><TensorStrip data={row(tn(`${p}ln1`), pos, D)} shape={[D]} maxCells={38} cellSize={9} name="LN(x)" /></div>
										{#each qkvList as m (m.r)}
											<div class="srow">
												<span class="sl">→ {m.name.toLowerCase()}</span>
												<span class="frame {m.r} inl"><TensorStrip data={m.data.subarray(pos * DH, pos * DH + DH)} domain={qkvDom} shape={[DH]} maxCells={DH} cellSize={12} name={m.name.toLowerCase()} selected={hc} onselect={(i) => (hc = i)} /></span>
											</div>
										{/each}
									</div>
								{/if}
								<div class="cols">
								<div class="vis qkv">
									{#each qkvList as m (m.r)}
										<div class="frame {m.r}">
											<p class="cap"><b>{m.name}</b> {t({ en: `head ${h + 1}`, zh: `第 ${h + 1} 头` })} {fmtShape([T, DH])}</p>
											<TensorGrid data={m.data} rows={T} cols={DH} maxCell={14} domain={qkvDom} highlights={[{ row: pos, col: hc, style: 'solid', color: `var(--${m.r})` }]} label="{m.name} head {h + 1}" onselect={(c) => pickRow(c, DH)} />
										</div>
									{/each}
								</div>
								{#if lin}
									<FormulaCard tex={lin.tex} values={lin.values} caption={focus
											? { en: `Number ${hc} of the query of ${tokEn(pos)} in head ${h + 1} (column ${h * DH + hc} of Q; click a cell to pick another)`, zh: `${tokZh(pos)}在第 ${h + 1} 头的查询中的第 ${hc} 个数（Q 的第 ${h * DH + hc} 列；点击格子换一个）` }
											: { en: `Query of token ${pos}, column ${h * DH + hc} (head ${h + 1}, channel ${hc})`, zh: `词元 ${pos} 的查询，第 ${h * DH + hc} 列（第 ${h + 1} 头第 ${hc} 通道）` }} provenance="live"
										where={[{ tex: sym('param', 'W^Q'), text: { en: 'learned [76×76] weights; K and V have their own', zh: '学到的 [76×76] 权重；K 和 V 各有自己的一组' } }]} />
								{/if}
							</div>
							<p class="note">
								{t({
									en: 'Intuition: the query says what this token is looking for, the key says what it offers to others, and the value is what it passes on if someone attends to it. All three are computed from the same normalized vector with different learned weights. Q, K and V each have one row per token and 76 columns: head 1 uses columns 0–18, head 2 columns 19–37, and so on.',
									zh: '直观理解：查询表示“这个词元在找什么”，键表示“它能提供什么”，值是“被关注时要传递出去的内容”。三者都由同一个归一化向量乘以不同的学到的权重得到。Q、K、V 都是每个词元一行、共 76 列：第 1 头用第 0–18 列，第 2 头用第 19–37 列，依此类推。'
								})}
							</p>
						{/if}
					{:else if s === 2 || s === 3}
						{#if scoresH}
							<div class="cols">
								<div class="vis">
									<p class="cap">{s === 2 ? 'Q Kᵀ / √19' : t({ en: 'masked scores', zh: '掩码后的分数' })} · {t({ en: `head ${h + 1}`, zh: `第 ${h + 1} 头` })} {fmtShape([T, T])}</p>
									<TensorGrid
										data={scoresH}
										rows={T}
										cols={T}
										maxCell={30}
										domain={scoreDom}
										masked={s === 3 ? (r, c) => c > r : undefined}
										highlights={[{ row: pos, col: key, color: 'var(--k)' }]}
										label="scores head {h + 1}"
										onselect={(c) => {
											S.selectPos(c.row);
											if (c.col <= c.row) S.selectKey(c.col);
										}}
									/>
									<ColorLegend kind="diverging" domain={scoreDom ?? [-1, 1]} label={{ en: 'score', zh: '分数' }} />
								</div>
								<FormulaCard
									tex={s === 2
										? `${sym('accent', 's_{' + pos + ',' + key + '}')} = \\frac{${sym('q', 'q_{' + pos + '}')}\\cdot ${sym('k', 'k_{' + key + '}')}}{\\sqrt{19}} = {{s}}`
										: `s'_{ij} = \\begin{cases} s_{ij} & j \\le i \\\\ -\\infty & j > i \\end{cases}`}
									values={{ s: scoresH[pos * T + key] }}
									caption={s === 2
											? focus
												? { en: `Row ${pos} = the query of “${S.label(pos)}”, column ${key} = the key of “${S.label(key)}”`, zh: `第 ${pos} 行是“${S.label(pos)}”的查询，第 ${key} 列是“${S.label(key)}”的键` }
												: { en: `Row ${pos} (query), column ${key} (key)`, zh: `第 ${pos} 行（查询），第 ${key} 列（键）` }
											: { en: 'The causal mask', zh: '因果掩码' }}
									provenance="live"
								/>
							</div>
							<p class="note">
								{s === 2
									? focus
										? t({ en: 'Row i holds how well query i matches every key. The upper triangle is computed too, but it compares a token with tokens that come after it. Only the differences within a row will matter, so negative scores are fine.', zh: '第 i 行是查询 i 与每个键的匹配程度。右上三角也被算出来了，但它是在拿词元和它后面的词元比较。之后只有同一行内的差别才重要，所以分数为负也没关系。' })
										: t({ en: 'Row i holds how well query i matches every key. The upper triangle is computed too, but it compares a token with tokens that come after it. The arithmetic for one cell is worked out below the attention maps.', zh: '第 i 行是查询 i 与每个键的匹配程度。右上三角也被算出来了，但它是在拿词元和它后面的词元比较。某一格的具体算式见下方注意力图之后的“手算”部分。' })
									: t({ en: 'TinyGPT predicts the next token, so while reading token i it must not peek at tokens after i. Setting those scores to −∞ makes their softmax weight exactly 0.', zh: 'TinyGPT 要预测下一个词元，所以读第 i 个词元时不能偷看它后面的词元。把这些分数设为 −∞，它们的 softmax 权重就恰好为 0。' })}
							</p>
						{/if}
					{:else if s === 4}
						{#if weightsH}
							<div class="cols">
								<div class="vis">
									<AttentionMatrix weights={weightsH} queries={S.labels} causal row={pos} col={key} maxCell={30} label={{ en: `Attention weights, head ${h + 1} [${T}×${T}]`, zh: `注意力权重，第 ${h + 1} 头 [${T}×${T}]` }} />
									<p class="cap">{t({ en: 'row of', zh: '这一行：' })} <span class="tok">{S.label(pos)}</span> · Σ = <Num value={Array.from(weightsH.subarray(pos * T, pos * T + T)).reduce((a, x) => a + x, 0)} digits={4} /></p>
									<TensorStrip data={weightsH.subarray(pos * T, pos * T + pos + 1)} kind="sequential" domain={[0, 1]} shape={[pos + 1]} maxCells={32} cellSize={14} name={{ en: 'attention row', zh: '注意力行' }} />
								</div>
								<FormulaCard
									tex={`${sym('prob', 'a_{' + pos + ',' + key + '}')} = \\frac{e^{s_{${pos},${key}}}}{\\sum_{j \\le ${pos}} e^{s_{${pos},j}}} = {{a}}`}
									values={{ a: weightsH[pos * T + key] }}
									caption={{ en: 'Softmax of one row', zh: '对一行做 softmax' }}
									provenance="live"
								/>
							</div>
						{/if}
					{:else if s === 5}
						{#if mix && qkv}
							<div class="cols">
								<div class="vis">
									<p class="cap">
										{t({ en: 'output of head', zh: '第' })} {h + 1}{t({ en: ' for', zh: ' 头对' })} <span class="tok">{S.label(pos)}</span>{t({ en: '', zh: ' 的输出' })} = Σ a<sub>j</sub> · v<sub>j</sub>
										<span class="muted">({t({ en: 'largest weights first', zh: '按权重从大到小' })})</span>
									</p>
									<div class="mixrows">
										{#each mix.shown as m (m.j)}
											<div class="mixrow">
												<span class="tok">{S.label(m.j)}</span>
												<span class="w num">{fmtPct(m.a)}</span>
												<span class="wbar" aria-hidden="true"><span style="width: {m.a * 100}%"></span></span>
												<span class="frame v inl"><TensorStrip data={qkv.v.subarray(m.j * DH, m.j * DH + DH)} domain={mix.vDom} shape={[DH]} maxCells={DH} cellSize={10} name="v" /></span>
											</div>
										{/each}
										{#if mix.nRest > 0}
											<p class="muted small">+ {mix.nRest} {t({ en: 'more keys with total weight', zh: '个其余的键，权重合计' })} {fmtPct(mix.rest)}</p>
										{/if}
										<div class="mixrow out">
											<span class="tok">=</span>
											<span class="w num">100%</span>
											<span></span>
											<TensorStrip data={mix.out} domain={mix.vDom} shape={[DH]} maxCells={DH} cellSize={10} name="o" selected={hc} onselect={(i) => (hc = i)} />
										</div>
									</div>
								</div>
								<FormulaCard
									tex={`${sym('out', 'o_{' + pos + ',' + hc + '}')} = \\sum_{j \\le ${pos}} ${sym('prob', 'a_{' + pos + ',j}')}\\, ${sym('v', 'v_{j,' + hc + '}')} = {{sum}}`}
									values={{ sum: mix.sum }}
									caption={focus ? { en: `Number ${hc} of the 19 (click a cell of o to pick another)`, zh: `19 个数中的第 ${hc} 个（点击 o 的格子换一个）` } : { en: `Channel ${hc} (click a cell of o to pick another)`, zh: `第 ${hc} 通道（点击 o 的格子换一个）` }}
									provenance="live"
								/>
							</div>
						{/if}
					{:else if s === 6}
						<div class="vis">
							<p class="cap">{t({ en: 'heads joined', zh: '各头拼接' })} {#if !focus}<span class="mono">{p}attn.heads</span>{/if} {fmtShape([T, D])} = [{T}×19 | {T}×19 | {T}×19 | {T}×19]</p>
							<TensorGrid data={tn(`${p}attn.heads`)} rows={T} cols={D} maxCell={10} highlights={[{ row: 0, col: h * DH, h: T, w: DH, style: 'dashed', color: 'var(--accent)' }, ...rowHL]} label="{p}attn.heads" onselect={(c) => pickRow(c, D)} />
							<div class="hstrips">
								{#each heads as hh (hh)}
									<div class:curh={hh === h}>
										<span class="small muted">{t({ en: 'head', zh: '头' })} {hh + 1}</span>
										<TensorStrip data={row(tn(`${p}attn.heads`), pos, D).subarray(hh * DH, hh * DH + DH)} shape={[DH]} maxCells={DH} cellSize={9} name="head {hh + 1}" />
									</div>
								{/each}
							</div>
							<p class="note">
								{t({
									en: 'Each head can attend to different tokens for different reasons. Concatenating keeps all four results; nothing is averaged yet.',
									zh: '每个头可以出于不同原因关注不同的词元。拼接保留了四个头的全部结果，这时还没有做任何平均。'
								})}
							</p>
						</div>
					{:else if s === 7 || s === 10 || s === 12}
						{@const id = s === 7 ? `${p}attn.proj` : s === 10 ? `${p}mlp.fc` : `${p}mlp.proj`}
						{@const cols = s === 10 ? DMLP : D}
						<div class="cols">
							<div class="vis">
								<p class="cap">{#if focus}{s === 7 ? t({ en: 'Output projection, every token', zh: '输出投影，每个词元' }) : s === 10 ? t({ en: 'MLP expand: 304 numbers per token', zh: 'MLP 扩展：每个词元 304 个数' }) : t({ en: 'MLP project: back to 76 numbers per token', zh: 'MLP 投影：回到每个词元 76 个数' })}{:else}<span class="mono">{id}</span>{/if} {fmtShape([T, cols])}</p>
								<TensorGrid data={tn(id)} rows={T} {cols} maxCell={10} highlights={rowHL} label={id} onselect={(c) => pickRow(c, cols)} />
								<TensorStrip data={row(tn(id), pos, cols)} shape={[cols]} maxCells={38} cellSize={11} name={{ en: 'selected token', zh: '所选词元' }} />
							</div>
							{#if lin}
								<FormulaCard tex={lin.tex} values={lin.values} caption={focus ? { en: `“${S.label(pos)}”: output number ${s === 10 ? unit : S.dim}`, zh: `“${S.label(pos)}”：第 ${s === 10 ? unit : S.dim} 个输出` } : { en: `Token ${pos}, output ${s === 10 ? unit : S.dim}`, zh: `词元 ${pos}，第 ${s === 10 ? unit : S.dim} 个输出` }} provenance="live" />
							{/if}
						</div>
						{#if s === 10}
							<p class="note">{t({ en: 'The MLP works on each token on its own: no information moves between tokens here. Attention moves information between tokens; the MLP processes what each token now holds.', zh: 'MLP 对每个词元单独计算：这里词元之间不交换信息。注意力负责在词元之间搬运信息；MLP 负责加工每个词元当前持有的信息。' })}</p>
						{/if}
					{:else if s === 11}
						<div class="cols">
							<GeluPlot xs={row(tn(`${p}mlp.fc`), pos, DMLP)} ys={row(tn(`${p}mlp.gelu`), pos, DMLP)} bind:unit label={S.label(pos)} />
							<FormulaCard
								tex={`\\begin{aligned}\\mathrm{GELU}(${sym('in', 'x')}) &= \\tfrac12\\,x\\,(1 + \\tanh u) \\\\ u &= \\sqrt{2/\\pi}\\,(x + 0.044715\\,x^3) \\\\ \\mathrm{GELU}({{x}}) &= {{y}}\\end{aligned}`}
								values={{ x: row(tn(`${p}mlp.fc`), pos, DMLP)[unit], y: row(tn(`${p}mlp.gelu`), pos, DMLP)[unit] }}
								caption={{ en: `Unit ${unit} of 304`, zh: `304 个单元中的第 ${unit} 个` }}
								provenance="live"
							/>
						</div>
					{:else if s === 8 || s === 13}
						{#if res}
							<div class="cols">
								<div class="vis">
									<div class="srow"><span class="sl">x</span><TensorStrip data={res.xin} domain={res.dom} shape={[D]} maxCells={38} cellSize={11} name="x" selected={S.dim} onselect={setDim} /></div>
									<div class="srow"><span class="sl">+ {res.second ? 'MLP' : t({ en: 'attention', zh: '注意力' })}</span><TensorStrip data={res.delta} domain={res.dom} shape={[D]} maxCells={38} cellSize={11} name="delta" selected={S.dim} onselect={setDim} /></div>
									<div class="srow"><span class="sl">=</span><TensorStrip data={res.out} domain={res.dom} shape={[D]} maxCells={38} cellSize={11} name="sum" selected={S.dim} onselect={setDim} /></div>
									<p class="cap">
										‖x‖ = <Num value={res.nx} />, ‖{res.second ? 'MLP' : t({ en: 'attention', zh: '注意力' })}‖ = <Num value={res.nd} /> → ‖x′‖ = <Num value={res.no} />
										<span class="muted">· {t({ en: 'the update is', zh: '这次更新的大小是输入的' })} {fmtPct(res.nd / Math.max(1e-9, res.nx))} {t({ en: 'of the input size (‖v‖ is the length of a vector)', zh: '（‖v‖ 表示向量的长度）' })}</span>
									</p>
									{#if s === 13}
										{@const a = lensAt(`${p}resid1`)}
										{@const c = lensAt(`${p}resid2`)}
										{#if a && c}
											<p class="cap">
												{t({ en: 'Logit lens for this token: after attention it would predict', zh: '这个词元的 logit lens：注意力之后会预测' })}
												<span class="tok">{S.idLabel(a.id)}</span> ({fmtPct(a.p)}),
												{t({ en: 'after the MLP', zh: 'MLP 之后会预测' })} <span class="tok">{S.idLabel(c.id)}</span> ({fmtPct(c.p)}).
												<a href="#output">{t({ en: 'See the full logit lens', zh: '查看完整的 logit lens' })}</a>
											</p>
										{/if}
									{/if}
								</div>
								<FormulaCard
									tex={`\\begin{aligned}${sym('out', "x'_{" + S.dim + '}')} &= ${sym('in', 'x_{' + S.dim + '}')} + ${sym('accent', '\\Delta_{' + S.dim + '}')} \\\\ &= {{x}} + ({{d}}) = {{y}}\\end{aligned}`}
									values={{ x: res.xin[S.dim], d: res.delta[S.dim], y: res.out[S.dim] }}
									caption={{ en: `Dimension ${S.dim} of ${tokEn(pos)}`, zh: `${tokZh(pos)}的第 ${S.dim} 维` }}
									provenance="live"
								/>
							</div>
							<p class="note">
								{focus
									? t({
											en: 'The block never replaces the stream; it only adds to it. That is why information from the embedding can survive all 4 blocks.',
											zh: '块从不替换残差流，只是往里加东西。所以嵌入里的信息能穿过全部 4 个块保留下来。'
										})
									: t({
											en: 'The block never replaces the stream; it only adds to it. That is why information from the embedding can survive all 4 blocks, and why gradients flow easily during training.',
											zh: '块从不替换残差流，只是往里加东西。所以嵌入里的信息能穿过全部 4 个块保留下来，训练时梯度也容易传回去。'
										})}
							</p>
							{/if}
						{/if}
					</div>
		{/snippet}

		{#if only && only.length === 1}
			{@render body(only[0])}
		{:else if only}
			<StepPlayer count={only.length} bind:step={ps} labels={onlyLabels} {interval} {autoplay} {onend} counter={focus ? PART : undefined} label={only.map((i) => t(STEPS[i].title)).join(' → ')}>
				{#snippet children(s)}{@render body(only?.[s] ?? s)}{/snippet}
			</StepPlayer>
		{:else}
			<StepPlayer count={STEPS.length} bind:step {labels} {interval} {autoplay} {onend} label={{ en: `Inside block ${b + 1}, step by step`, zh: `第 ${b + 1} 块内部，逐步演示` }}>
				{#snippet children(s)}{@render body(s)}{/snippet}
			</StepPlayer>
		{/if}
		</div>
	{/if}

<style>
	.bv {
		container: bv / inline-size;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 12px;
	}
	.bv.focus .stage {
		padding: 0;
		border: 0;
		background: none;
		min-height: 0;
	}
	.follow {
		display: grid;
		gap: 2px;
		margin-bottom: 12px;
		padding-bottom: 10px;
		border-bottom: 1px solid var(--rule);
	}
	.follow .frame,
	.mixrow .frame {
		justify-self: start;
		max-width: 100%;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px 18px;
	}
	.ctl {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.lbl {
		font-size: var(--fs-sm);
		font-weight: 650;
	}
	.steps {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.steps button {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		min-height: 30px;
		padding: 0 8px;
		border: 1px solid var(--rule);
		border-left: 3px solid var(--q);
		border-radius: var(--r1);
		background: var(--surface);
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.steps button.mlp {
		border-left-color: var(--div-pos);
	}
	.steps button.cur {
		background: var(--accent-soft);
		border-color: var(--accent);
		color: var(--ink);
		font-weight: 700;
	}
	.sn {
		color: var(--muted);
		font-size: 10px;
	}
	.crumbs {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		flex-wrap: wrap;
	}
	.stage {
		padding: 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		min-height: 220px;
		min-width: 0;
	}
	.cols {
		display: grid;
		grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
		gap: 16px;
		align-items: start;
	}
	.vis {
		display: grid;
		gap: 6px;
		min-width: 0;
	}
	.pair {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		gap: 10px;
	}
	.qkv {
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 8px;
	}
	.frame {
		padding: 4px 6px;
		border-radius: var(--r1);
		border: 2px solid transparent;
		min-width: 0;
	}
	.frame.q {
		border-color: var(--q);
	}
	.frame.k {
		border-color: var(--k);
	}
	.frame.v {
		border-color: var(--v);
	}
	.frame.inl {
		padding: 1px 2px;
		display: inline-block;
	}
	.cap {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.mono {
		font-family: var(--font-mono);
	}
	.tok {
		font-family: var(--font-mono);
		background: var(--surface-2);
		border-radius: 3px;
		padding: 0 3px;
		white-space: pre;
	}
	.note {
		margin: 10px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
	.small {
		font-size: var(--fs-xs);
	}
	.mixrows {
		display: grid;
		gap: 2px;
	}
	.mixrow {
		display: grid;
		grid-template-columns: minmax(3em, 7em) 3.6em 50px minmax(0, 1fr);
		align-items: center;
		gap: 6px;
		font-size: var(--fs-xs);
	}
	.mixrow.out {
		border-top: 1.5px solid var(--ink-2);
		padding-top: 4px;
	}
	.mixrow :global(.readout) {
		display: none;
	}
	.w {
		text-align: right;
	}
	.wbar {
		height: 8px;
		background: var(--surface-3);
		border-radius: 3px;
		overflow: hidden;
	}
	.wbar span {
		display: block;
		height: 100%;
		background: var(--seq-mid);
	}
	.hstrips {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 12px;
	}
	.hstrips > div {
		padding: 2px 4px;
		border-radius: var(--r1);
		border: 1px dashed transparent;
	}
	.hstrips .curh {
		border-color: var(--accent);
	}
	.srow {
		display: grid;
		grid-template-columns: 6.5em minmax(0, 1fr);
		gap: 6px;
		align-items: start;
	}
	.sl {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		padding-top: 2px;
	}
	@media (max-width: 640px) {
		.steps {
			flex-wrap: nowrap;
			overflow-x: auto;
			padding-bottom: 4px;
		}
		.steps button {
			flex: 0 0 auto;
		}
	}
	/* by the block view's own width (it also sits inside the lesson stage) */
	@container bv (max-width: 820px) {
		.cols {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	@container bv (max-width: 540px) {
		.pair,
		.qkv {
			grid-template-columns: minmax(0, 1fr);
		}
		.stage {
			padding: 8px;
		}
		.srow {
			grid-template-columns: minmax(0, 1fr);
			gap: 0;
		}
	}
</style>
