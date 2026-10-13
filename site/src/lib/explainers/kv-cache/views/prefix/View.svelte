<!--
  'kv-cache/prefix' (meta.ts has the props): prompts A and B as token chips (B's shared prefix marked as reused from A's
  cache, the rest as computed), two bars for B's prefill work (to scale), and with props.layers a table of the key and value
  differences per layer for the tokens both prompts have in the same place after the first difference.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { fmt3, fmtInt, fmtPct } from '$lib/kit/format';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { tokText } from '../grid/meta.ts';

	let { at, props }: StageProps = $props();
	const m = $derived(at.frame?.metrics);
	const st = $derived(at.state as Record<string, { data: ArrayLike<number> }> | undefined);
	const ids = (k: string) => Array.from(st?.[k]?.data ?? []).filter((v) => v >= 0);
	const A = $derived(ids('a'));
	const B = $derived(ids('b'));
	const dK = $derived(Array.from(st?.dK?.data ?? []));
	const dV = $derived(Array.from(st?.dV?.data ?? []));
	const col = $derived.by(() => {
		void themeState.resolved;
		return { reuse: categorical(0), comp: categorical(1) };
	});
	const bmax = $derived(m ? Math.max(1, m.macsB) : 1);
</script>

{#if m}
	<div class="prefix">
		{#each [{ name: { en: 'Prompt A', zh: '提示词 A' }, toks: A, b: false }, { name: { en: 'Prompt B', zh: '提示词 B' }, toks: B, b: true }] as row (row.b)}
			<div class="prow">
				<span class="who">{t(row.name)}</span>
				<p class="toks">
					{#each row.toks as id, i (i)}
						<span class="tk" class:shared={i < m.s} class:reuse={row.b && i < m.reused} class:comp={row.b && i >= m.reused} style:border-color={row.b ? (i < m.reused ? col.reuse : col.comp) : undefined}>{tokText(id)}</span>
					{/each}
				</p>
			</div>
		{/each}
		<div class="legend">
			<span class="key"><span class="sw" style:background={col.reuse}></span>{t({ en: `reused from A’s cache: ${m.reused} tokens`, zh: `从 A 的缓存复用：${m.reused} 个词元` })}</span>
			<span class="key"><span class="sw" style:background={col.comp}></span>{t({ en: `computed for B: ${m.computed} tokens`, zh: `为 B 计算：${m.computed} 个词元` })}</span>
		</div>
		<div class="bars">
			<div class="bar">
				<span class="lab">{t({ en: 'B from scratch', zh: 'B 从头计算' })}</span>
				<span class="track"><span class="fill" style:width="{(m.macsB / bmax) * 100}%" style:background={col.comp}></span></span>
				<span class="num">{fmtInt(m.macsB)}</span>
			</div>
			<div class="bar">
				<span class="lab">{t({ en: 'B with A’s prefix', zh: 'B 复用 A 的前缀' })}</span>
				<span class="track"><span class="fill" style:width="{(m.macsReuse / bmax) * 100}%" style:background={col.comp}></span></span>
				<span class="num">{fmtInt(m.macsReuse)}</span>
			</div>
			<p class="cap">{t({ en: 'Multiply-adds of B’s prefill (bars to scale).', zh: 'B 预填充的乘加次数（条形按比例）。' })}</p>
		</div>
		{#if props.layers}
			<p class="cap">
				{t(m.nSame > 0 ? { en: `After the first difference, ${m.nSame} tokens are the same token in the same place in both prompts. Their keys and values, largest difference per layer:`, zh: `第一个不同之后，有 ${m.nSame} 个词元在两个提示词里是同一位置的同一词元。它们的键和值在每一层的最大差异：` } : { en: 'After the first difference, no token is the same token in the same place in both prompts.', zh: '第一个不同之后，两个提示词里没有同一位置的同一词元。' })}
			</p>
			{#if m.nSame > 0}
				<div class="tbl" use:keyScroll>
					<table>
						<thead><tr><th scope="col">{t({ en: 'Layer', zh: '层' })}</th><th scope="col">{t({ en: 'keys: largest |Δ|', zh: '键：最大 |Δ|' })}</th><th scope="col">{t({ en: 'values: largest |Δ|', zh: '值：最大 |Δ|' })}</th></tr></thead>
						<tbody>
							{#each dK as k, l (l)}<tr><th scope="row">{l}</th><td class="num">{fmt3(k)}</td><td class="num">{fmt3(dV[l])}</td></tr>{/each}
						</tbody>
					</table>
				</div>
				<p class="cap">
					{t({ en: `Reusing A’s rows for those tokens would move B’s next-token logits by up to ${fmt3(m.dWrong)}: B’s top token “${tokText(m.tokB)}” would get ${fmtPct(m.pWrong)} instead of ${fmtPct(m.pB)}.`, zh: `如果这些词元也复用 A 的缓存，B 的下一词元 logit 最多会偏移 ${fmt3(m.dWrong)}：B 排第一的词元“${tokText(m.tokB)}”的概率会变成 ${fmtPct(m.pWrong)}，而不是 ${fmtPct(m.pB)}。` })}
				</p>
			{/if}
		{/if}
	</div>
{/if}

<style>
	.prefix {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.prow {
		display: grid;
		grid-template-columns: 5.5rem minmax(0, 1fr);
		gap: 8px;
		align-items: start;
	}
	.who {
		font-size: var(--fs-sm);
		font-weight: 600;
	}
	.toks {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 2px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		min-width: 0;
	}
	.tk {
		padding: 1px 3px;
		border: 1px solid var(--rule);
		border-radius: 3px;
		white-space: pre;
	}
	.tk.shared {
		background: var(--surface-2);
	}
	.tk.reuse,
	.tk.comp {
		border-width: 2px;
	}
	.tk.comp {
		border-style: dashed;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.key {
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.sw {
		width: 12px;
		height: 12px;
		border-radius: 2px;
		display: inline-block;
		flex: none;
	}
	.bars {
		display: grid;
		gap: 4px;
	}
	.bar {
		display: grid;
		grid-template-columns: 9rem minmax(0, 1fr) 6.5rem;
		gap: 8px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.track {
		height: 12px;
		background: var(--surface-2);
		border-radius: 2px;
		overflow: hidden;
	}
	.fill {
		display: block;
		height: 100%;
	}
	.cap {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.tbl {
		overflow-x: auto;
		min-width: 0;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-sm);
	}
	th,
	td {
		padding: 2px 10px;
		text-align: right;
		border-bottom: 1px solid var(--rule);
	}
	thead th {
		color: var(--ink-2);
		font-weight: 500;
	}
	@media (max-width: 520px) {
		.prow {
			grid-template-columns: 1fr;
			gap: 2px;
		}
		.bar {
			grid-template-columns: 7rem minmax(0, 1fr) 5.5rem;
		}
	}
</style>
