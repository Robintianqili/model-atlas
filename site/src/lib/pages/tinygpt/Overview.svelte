<!--
  Overview: the whole model on one screen, drawn from the live trace.
  Lane 1 embedding (token + position = input), lane 2 the 4 blocks on the residual stream,
  lane 3 the output (final LayerNorm -> logits -> probabilities -> next token).
  Every thumbnail is the real [T×76] (or [T×1024]) tensor for the learner's text; the selected token's
  row is outlined. Click a card to inspect it; "Look inside" opens the block view.
-->
<script lang="ts">
	import MiniHeat from './MiniHeat.svelte';
	import TensorStrip from '$lib/kit/TensorStrip.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtShape, fmtPct, fmt3 } from '$lib/kit/format';
	import { domainOf } from '$lib/kit/color';
	import type { TGState } from './state.svelte';
	import { D, H, NL, V } from './state.svelte';
	import { row, topIndices } from './analysis';

	interface Props {
		S: TGState;
		onopen: (block: number) => void;
	}
	let { S, onopen }: Props = $props();

	const T = $derived(S.T);
	const f = $derived(S.run?.forward);
	const thumbH = $derived(Math.max(18, Math.min(72, T * 4)));
	const W = 116;
	// each thumbnail uses its own color range (shown as ±max under it): the stream grows block by block
	const maxAbs = (id: string) => (f ? domainOf('diverging', f.t[id].data)[1] : 0);
	const pos = $derived(S.pos);
	const idAt = $derived(f ? f.ids[pos] : 0);
	const logitsRow = $derived(f ? row(f.t.logits.data, pos, V) : null);
	const probsRow = $derived(f ? row(f.t.probs.data, pos, V) : null);
	const top3 = $derived(probsRow ? topIndices(probsRow, 3) : []);

	function pick(id: string, block?: number) {
		S.layerId = id;
		if (block !== undefined) S.block = block;
	}
	const isCur = (id: string) => S.layerId === id;
	const inBlock = (b: number) => S.layerId.startsWith(`L${b}.`);
</script>

{#if f}
	<div class="ov">
		<!-- lane 1: embedding -->
		<section class="lane" aria-label={t({ en: 'Embedding', zh: '嵌入' })}>
			<h3 class="lane-t"><span class="n">1</span>{t({ en: 'Embedding: one vector of 76 numbers per token', zh: '嵌入：每个词元一个 76 维向量' })}</h3>
			<div class="row">
				<button type="button" class="card" class:cur={isCur('tok_emb')} onclick={() => pick('tok_emb')}>
					<span class="ttl">{t({ en: 'Token embedding', zh: '词元嵌入' })}</span>
					<span class="op">{t({ en: 'row id of E [1024×76]', zh: '查表：E [1024×76] 的第 id 行' })}</span>
					<MiniHeat data={f.t.tok_emb.data} rows={T} cols={D} width={W} height={thumbH} hlRow={pos} hlColor="--accent" label="tok_emb {fmtShape([T, D])}" />
					<span class="shape num">{fmtShape([T, D])} · ±{fmt3(maxAbs('tok_emb'), 2)}</span>
				</button>
				<span class="glyph" aria-hidden="true">+</span>
				<button type="button" class="card" class:cur={isCur('pos_emb')} onclick={() => pick('pos_emb')}>
					<span class="ttl">{t({ en: 'Position embedding', zh: '位置嵌入' })}</span>
					<span class="op">{t({ en: 'row 0…T−1 of P [64×76]', zh: '查表：P [64×76] 的第 0…T−1 行' })}</span>
					<MiniHeat data={f.t.pos_emb.data} rows={T} cols={D} width={W} height={thumbH} hlRow={pos} hlColor="--accent" label="pos_emb {fmtShape([T, D])}" />
					<span class="shape num">{fmtShape([T, D])} · ±{fmt3(maxAbs('pos_emb'), 2)}</span>
				</button>
				<span class="glyph" aria-hidden="true">=</span>
				<button type="button" class="card" class:cur={isCur('embed')} onclick={() => pick('embed')}>
					<span class="ttl">{t({ en: 'Input vectors x', zh: '输入向量 x' })}</span>
					<span class="op">{t({ en: 'start of the residual stream', zh: '残差流的起点' })}</span>
					<MiniHeat data={f.t.embed.data} rows={T} cols={D} width={W} height={thumbH} hlRow={pos} hlColor="--accent" label="embed {fmtShape([T, D])}" />
					<span class="shape num">{fmtShape([T, D])} · ±{fmt3(maxAbs('embed'), 2)}</span>
				</button>
			</div>
		</section>

		<div class="down" aria-hidden="true"><span>↓</span> <span class="num">{fmtShape([T, D])}</span></div>

		<!-- lane 2: blocks -->
		<section class="lane blocks" aria-label={t({ en: 'Transformer blocks', zh: 'Transformer 块' })}>
			<h3 class="lane-t"><span class="n">2</span>{t({ en: '4 transformer blocks, each adds to the residual stream', zh: '4 个 Transformer 块，每块都往残差流里“加”东西' })}</h3>
			<div class="row stream">
				{#each Array.from({ length: NL }, (_, i) => i) as b (b)}
					<div class="card block" class:cur={inBlock(b)}>
						<button type="button" class="blk-main" onclick={() => pick(`L${b}.resid2`, b)} aria-label={t({ en: `Block ${b + 1}: inspect its output`, zh: `第 ${b + 1} 块：查看输出` })}>
							<span class="ttl">{t({ en: `Block ${b + 1}`, zh: `第 ${b + 1} 块` })}</span>
							<span class="op">{t({ en: 'x + attention, then + MLP', zh: 'x + 注意力，再 + MLP' })}</span>
							<MiniHeat data={f.t[`L${b}.resid2`].data} rows={T} cols={D} width={W} height={thumbH} hlRow={pos} hlColor="--accent" label="L{b}.resid2 {fmtShape([T, D])}" />
							<span class="heads" aria-hidden="true">
								{#each Array.from({ length: H }, (_, h) => h) as h (h)}
									<MiniHeat
										data={f.t[`L${b}.attn.weights`].data.subarray(h * T * T, (h + 1) * T * T)}
										rows={T}
										cols={T}
										width={26}
										height={26}
										kind="sequential"
										domain={[0, 1]}
										causal
										label="head {h + 1}"
									/>
								{/each}
							</span>
							<span class="shape num">{fmtShape([T, D])} · ±{fmt3(maxAbs(`L${b}.resid2`), 2)}</span>
						</button>
						<button type="button" class="btn ghost open" onclick={() => onopen(b)}>{t({ en: 'Look inside', zh: '展开看内部' })} ▸</button>
					</div>
					{#if b < NL - 1}<span class="glyph arrow" aria-hidden="true">→</span>{/if}
				{/each}
			</div>
			<p class="stream-note">
				<span class="band" aria-hidden="true"></span>
				{t({
					en: 'Residual stream: the [T×76] vectors flow straight through; each block reads them and adds its result.',
					zh: '残差流：[T×76] 的向量一路直通；每个块读取它们，再把自己的结果加回去。'
				})}
				{t({ en: 'Each thumbnail has its own color range (±max): the numbers in the stream grow from block to block.', zh: '每张缩略图有自己的颜色范围（±最大值）：残差流里的数值一块比一块大。' })}
			</p>
		</section>

		<div class="down" aria-hidden="true"><span>↓</span> <span class="num">{fmtShape([T, D])}</span></div>

		<!-- lane 3: output -->
		<section class="lane" aria-label={t({ en: 'Output', zh: '输出' })}>
			<h3 class="lane-t"><span class="n">3</span>{t({ en: 'Output: scores for all 1,024 tokens', zh: '输出：给全部 1024 个词元打分' })}</h3>
			<div class="row">
				<button type="button" class="card" class:cur={isCur('ln_f')} onclick={() => pick('ln_f')}>
					<span class="ttl">{t({ en: 'Final LayerNorm', zh: '最终层归一化' })}</span>
					<span class="op">LN_f</span>
					<MiniHeat data={f.t.ln_f.data} rows={T} cols={D} width={W} height={thumbH} hlRow={pos} hlColor="--accent" label="ln_f {fmtShape([T, D])}" />
					<span class="shape num">{fmtShape([T, D])} · ±{fmt3(maxAbs('ln_f'), 2)}</span>
				</button>
				<span class="glyph arrow" aria-hidden="true">→</span>
				<button type="button" class="card" class:cur={isCur('logits')} onclick={() => pick('logits')}>
					<span class="ttl">{t({ en: 'Logits', zh: 'Logits（未归一化分数）' })}</span>
					<span class="op">{t({ en: 'x · Eᵀ (tied head)', zh: 'x · Eᵀ（共享输出头）' })}</span>
					<span class="toplist">
						{#if logitsRow}
							{#each top3 as id (id)}
								<span class="ti"><span class="tk">{S.idLabel(id)}</span> <Num value={logitsRow[id]} plain /></span>
							{/each}
						{/if}
					</span>
					<span class="shape num">{fmtShape([T, V])}</span>
				</button>
				<span class="glyph arrow" aria-hidden="true">→</span>
				<button type="button" class="card" class:cur={isCur('probs')} onclick={() => pick('probs')}>
					<span class="ttl">{t({ en: 'Probabilities', zh: '概率' })}</span>
					<span class="op">softmax</span>
					<span class="toplist">
						{#if probsRow}
							{#each top3 as id (id)}
								<span class="ti">
									<span class="tk">{S.idLabel(id)}</span>
									<span class="pbar" aria-hidden="true"><span style="width: {probsRow[id] * 100}%"></span></span>
									<span class="num">{fmtPct(probsRow[id])}</span>
								</span>
							{/each}
						{/if}
					</span>
					<span class="shape num">{fmtShape([T, V])}</span>
				</button>
				<span class="glyph arrow" aria-hidden="true">→</span>
				<a class="card next" href="#output">
					<span class="ttl">{t({ en: 'Next token', zh: '下一个词元' })}</span>
					<span class="op">{t({ en: 'pick one (sampling)', zh: '选一个（采样）' })}</span>
					{#if probsRow && top3.length}
						<span class="big">{S.idLabel(top3[0])}</span>
						<span class="muted small">{t({ en: 'most likely after', zh: '最可能接在后面：' })} <span class="tk">{S.label(pos)}</span></span>
					{/if}
				</a>
			</div>
		</section>

		<!-- follow one token -->
		<section class="follow" aria-label={t({ en: 'Follow one token', zh: '跟踪一个词元' })}>
			<h3 class="lane-t">
				{t({ en: 'Follow one token:', zh: '跟踪一个词元：' })}
				<span class="tk big2">{S.label(pos)}</span>
				<span class="muted small">{t({ en: `position ${pos}, id ${idAt}`, zh: `位置 ${pos}，id ${idAt}` })}</span>
			</h3>
			<div class="sum">
				<div class="srow">
					<span class="sl">{t({ en: `E[${idAt}]  token`, zh: `E[${idAt}]  词元` })}</span>
					<TensorStrip data={row(f.t.tok_emb.data, pos, D)} shape={[D]} maxCells={28} cellSize={13} name={{ en: 'token embedding row', zh: '词元嵌入行' }} />
				</div>
				<div class="srow">
					<span class="sl">+ P[{pos}] {t({ en: 'position', zh: '位置' })}</span>
					<TensorStrip data={row(f.t.pos_emb.data, pos, D)} shape={[D]} maxCells={28} cellSize={13} name={{ en: 'position embedding row', zh: '位置嵌入行' }} />
				</div>
				<div class="srow">
					<span class="sl">= x[{pos}]</span>
					<TensorStrip data={row(f.t.embed.data, pos, D)} shape={[D]} maxCells={28} cellSize={13} name={{ en: 'input vector', zh: '输入向量' }} />
				</div>
			</div>
			<p class="muted small">
				{t({
					en: 'Looking up an embedding is just copying one row of a learned table. The model adds a second learned row for the position, because attention by itself does not know the order of tokens. Hover a cell for its value.',
					zh: '查嵌入就是从一张学到的表里复制一行。模型再加上表示“位置”的另一行，因为注意力本身不知道词元的先后顺序。悬停在格子上可看具体数值。'
				})}
			</p>
		</section>
	</div>
{:else}
	<p class="muted">
		{S.model
			? t({ en: 'Type some text (Edit your text) to run the model.', zh: '输入一些文字（编辑你的文本）来运行模型。' })
			: t({ en: 'Loading the model…', zh: '正在加载模型…' })}
	</p>
{/if}

<style>
	.ov {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 6px;
	}
	.lane {
		padding: 12px 14px 14px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		min-width: 0;
	}
	.lane-t {
		font-family: var(--font-body);
		font-size: var(--fs-sm);
		font-weight: 700;
		margin: 0 0 10px;
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 6px 8px;
	}
	.n {
		display: inline-grid;
		place-items: center;
		width: 1.5em;
		height: 1.5em;
		border-radius: 50%;
		background: var(--accent);
		color: var(--accent-ink);
		font-size: var(--fs-xs);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px;
	}
	.card {
		display: grid;
		justify-items: center;
		gap: 4px;
		min-width: 0;
		width: 140px;
		padding: 8px 8px 6px;
		border: 1.5px solid var(--rule-strong);
		border-radius: var(--r2);
		background: var(--surface);
		color: var(--ink);
		text-align: center;
		text-decoration: none;
		font: inherit;
	}
	button.card:hover,
	a.card:hover,
	.blk-main:hover {
		border-color: var(--ink-2);
	}
	.card.cur {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
	}
	.block {
		padding: 0;
		overflow: hidden;
	}
	.blk-main {
		display: grid;
		justify-items: center;
		gap: 4px;
		width: 100%;
		padding: 8px 8px 4px;
		border: 0;
		background: none;
		color: var(--ink);
		font: inherit;
	}
	.open {
		width: 100%;
		min-height: 32px;
		border-radius: 0;
		border-top: 1px solid var(--rule);
		font-size: var(--fs-xs);
	}
	.ttl {
		font-weight: 700;
		font-size: var(--fs-sm);
		line-height: 1.2;
	}
	.op {
		font-size: var(--t-xs);
		color: var(--muted);
		line-height: 1.25;
	}
	.shape {
		font-size: var(--t-xs);
		color: var(--ink-2);
	}
	.heads {
		display: flex;
		gap: 3px;
	}
	.glyph {
		font-size: 1.4rem;
		font-weight: 700;
		color: var(--ink-2);
		width: 1em;
		text-align: center;
	}
	.down {
		display: flex;
		justify-content: center;
		gap: 8px;
		align-items: baseline;
		color: var(--ink-2);
		font-size: var(--fs-sm);
	}
	.down span:first-child {
		font-size: 1.3rem;
		font-weight: 700;
	}
	.stream {
		position: relative;
	}
	.stream-note {
		margin: 8px 0 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.band {
		flex: 0 0 auto;
		width: 36px;
		height: 8px;
		border-radius: 4px;
		background: linear-gradient(90deg, var(--div-neg), var(--div-mid), var(--div-pos));
	}
	.toplist {
		display: grid;
		gap: 2px;
		width: 100%;
		min-height: 54px;
		align-content: center;
		font-size: var(--t-xs);
	}
	.ti {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		gap: 4px;
		align-items: center;
		text-align: left;
	}
	.ti:has(.pbar) {
		grid-template-columns: minmax(0, 1fr) 30px auto;
	}
	.tk {
		font-family: var(--font-mono);
		background: var(--surface-2);
		border-radius: 3px;
		padding: 0 3px;
		white-space: pre;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.pbar {
		height: 6px;
		background: var(--surface-3);
		border-radius: 3px;
		overflow: hidden;
	}
	.pbar span {
		display: block;
		height: 100%;
		background: var(--seq-mid);
	}
	.next .big {
		font-family: var(--font-mono);
		font-size: 1.15rem;
		padding: 2px 8px;
		border-radius: 4px;
		background: color-mix(in srgb, var(--generated) 14%, var(--surface));
		border: 1px solid var(--generated);
		white-space: pre;
		max-width: 100%;
		overflow: hidden;
	}
	.small {
		font-size: var(--fs-xs);
	}
	.follow {
		margin-top: 10px;
		padding: 12px 14px;
		border: 1px dashed var(--rule-strong);
		border-radius: var(--r3);
		min-width: 0;
	}
	.big2 {
		font-size: var(--fs-md);
	}
	.sum {
		display: grid;
		gap: 2px;
	}
	.srow {
		display: grid;
		grid-template-columns: 9.5em minmax(0, 1fr);
		align-items: start;
		gap: 8px;
	}
	.sl {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--ink-2);
		padding-top: 2px;
		white-space: pre;
	}
	@media (max-width: 640px) {
		.card {
			width: calc(50% - 22px);
		}
		.glyph.arrow {
			display: none;
		}
		.srow {
			grid-template-columns: minmax(0, 1fr);
			gap: 0;
		}
	}
</style>
