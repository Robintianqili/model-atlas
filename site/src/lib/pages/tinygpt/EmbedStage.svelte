<!--
  Lesson step 2: the followed token's id picks row `id` of the token-embedding table E [1024×76];
  its position picks row `pos` of the position table P [64×76]; the sum is the vector that enters
  block 1 (the traced `embed`). Same y range for the three vectors, one dimension worked out.
-->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { sym } from '$lib/kit/tex';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtInt } from '$lib/kit/format';
	import MiniHeat from './MiniHeat.svelte';
	import VecBars from './VecBars.svelte';
	import type { TGState } from './state.svelte';
	import { CTX, D, V } from './state.svelte';
	import { row } from './analysis';

	let { LS }: { LS: TGState } = $props();

	const f = $derived(LS.run?.forward);
	const pos = $derived(LS.pos);
	const E = $derived(LS.model?.weight('tok_emb.weight').data ?? null);
	const P = $derived(LS.model?.weight('pos_emb.weight').data ?? null);
	const maxAbs = (a: ArrayLike<number>) => {
		let m = 0;
		for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
		return m;
	};
	// one colour range per table (computed once per model)
	const eDom = $derived(E ? ([-maxAbs(E), maxAbs(E)] as [number, number]) : undefined);
	const pDom = $derived(P ? ([-maxAbs(P), maxAbs(P)] as [number, number]) : undefined);
	const v = $derived.by(() => {
		if (!f || !E || !P) return null;
		const id = f.ids[pos];
		const e = E.subarray(id * D, id * D + D);
		const p = P.subarray(pos * D, pos * D + D);
		const x = row(f.t.embed.data, pos, D);
		return { id, e, p, x, max: Math.max(maxAbs(e), maxAbs(p), maxAbs(x)) };
	});
	const d = $derived(Math.min(D - 1, Math.max(0, LS.dim)));
	const pick = (i: number) => (LS.dim = i);
</script>

{#if v && E && P}
	<div class="es">
		<p class="head">
			{t({ en: 'Token', zh: '词元' })} <span class="tok">{LS.label(pos)}</span>
			· id <span class="num">{v.id}</span> · {t({ en: 'position', zh: '位置' })} <span class="num">{pos}</span>
		</p>
		<div class="row2">
			<div class="tables">
				<figure>
					<MiniHeat data={E} rows={V} cols={D} width={76} height={170} domain={eDom} hlRow={v.id} hlColor="--q" label={t({ en: `Token embedding table, 1,024 rows × 76; row ${v.id} outlined`, zh: `词元嵌入表，1024 行 × 76；第 ${v.id} 行带框` })} />
					<figcaption>
						<b>E</b> {t({ en: 'tokens', zh: '词元表' })}<br /><span class="num">{fmtInt(V)} × {D}</span><br />{t({ en: 'row', zh: '第' })} <span class="num">{v.id}</span>{t({ en: '', zh: ' 行' })}
					</figcaption>
				</figure>
				<span class="plus" aria-hidden="true">+</span>
				<figure>
					<MiniHeat data={P} rows={CTX} cols={D} width={76} height={64} domain={pDom} hlRow={pos} hlColor="--q" label={t({ en: `Position table, 64 rows × 76; row ${pos} outlined`, zh: `位置表，64 行 × 76；第 ${pos} 行带框` })} />
					<figcaption>
						<b>P</b> {t({ en: 'positions', zh: '位置表' })}<br /><span class="num">{CTX} × {D}</span><br />{t({ en: 'row', zh: '第' })} <span class="num">{pos}</span>{t({ en: '', zh: ' 行' })}
					</figcaption>
				</figure>
			</div>
			<div class="vecs">
				<VecBars data={v.e} max={v.max} selected={d} onselect={pick} name="E[{v.id}]" label={{ en: `Token embedding of id ${v.id}`, zh: `id ${v.id} 的词元嵌入` }} />
				<VecBars data={v.p} max={v.max} selected={d} onselect={pick} name="+ P[{pos}]" label={{ en: `Position embedding of position ${pos}`, zh: `位置 ${pos} 的位置嵌入` }} />
				<VecBars data={v.x} max={v.max} selected={d} onselect={pick} name="= x" label={{ en: 'Input vector of block 1', zh: '第 1 块的输入向量' }} />
				<p class="note muted">
					{t({ en: `76 numbers each, drawn on one scale (±${fmt3(v.max)}): orange is positive, blue negative. Click a bar or use the arrow keys to pick a dimension.`, zh: `每个都是 76 个数，画在同一个刻度上（±${fmt3(v.max)}）：橙色为正，蓝色为负。点击一根柱子或用方向键选择一个维度。` })}
				</p>
			</div>
		</div>
		<FormulaCard
			tex={`\\begin{aligned}${sym('out', `x_{${pos},${d}}`)} &= ${sym('param', `E_{${v.id},${d}}`)} + ${sym('param', `P_{${pos},${d}}`)} \\\\ &= {{e}} + ({{p}}) = {{x}}\\end{aligned}`}
			values={{ e: v.e[d], p: v.p[d], x: v.x[d] }}
			caption={{ en: `Dimension ${d} of the vector for “${LS.label(pos)}”`, zh: `“${LS.label(pos)}”的向量的第 ${d} 维` }}
			where={[
				{ tex: sym('param', 'E'), text: { en: 'learned token table: one row of 76 numbers per token', zh: '学到的词元表：每个词元一行 76 个数' } },
				{ tex: sym('param', 'P'), text: { en: 'learned position table: one row per position 0–63', zh: '学到的位置表：位置 0–63 各一行' } }
			]}
			provenance="live"
		/>
		<p class="check">
			{t({ en: 'The model’s own trace for this number:', zh: '模型自己的计算结果：' })} <Num value={v.x[d]} />
			<span class="muted">· {t({ en: 'the same token at another position gets a different vector, so word order is not lost', zh: '同一个词元放在别的位置会得到不同的向量，所以词序不会丢失' })}</span>
		</p>
	</div>
{/if}

<style>
	.es {
		container: es / inline-size;
		display: grid;
		gap: 12px;
		grid-template-columns: minmax(0, 1fr);
	}
	.head {
		margin: 0;
		font-size: var(--fs-sm);
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
	.row2 {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr);
		gap: 16px;
		align-items: start;
	}
	.tables {
		display: flex;
		align-items: flex-start;
		gap: 8px;
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
		justify-items: start;
	}
	figcaption {
		font-size: 11px;
		color: var(--ink-2);
		line-height: 1.35;
	}
	.plus {
		font-size: 1.2rem;
		color: var(--muted);
		padding-top: 20px;
	}
	.vecs {
		display: grid;
		gap: 2px;
		min-width: 0;
	}
	.note {
		margin: 2px 0 0;
		font-size: var(--fs-xs);
	}
	.check {
		margin: 0;
		font-size: var(--fs-sm);
	}
	@container es (max-width: 560px) {
		.row2 {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
