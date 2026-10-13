<!--
  'backprop/net': the 2-2-1 network at a frame of the 'net' process (meta.ts has the props): the graph of its units
  (the toy graph's Diagram) and the table of its nine parameters, in a box that scrolls on its own on a narrow screen.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { NET_GRAPH } from '../../data.ts';
	import Diagram from '../graph/Diagram.svelte';
	import { netOf } from './meta.ts';

	let { at, props }: StageProps = $props();
	const pic = $derived(netOf(at?.frame, at?.state, props));
	const L = {
		fig: { en: 'The 2-2-1 network as a graph of its units', zh: '把 2-2-1 网络画成由单元组成的图' },
		param: { en: 'parameter', zh: '参数' },
		value: { en: 'value', zh: '数值' },
		grad: { en: 'gradient (backprop)', zh: '梯度（反向传播）' },
		num: { en: 'numerical', zh: '数值估计' },
		after: { en: 'after one update', zh: '更新一次之后' },
		cap: { en: 'Nine parameters: the first layer\'s weights W and biases b, the output layer\'s weights v and bias c.', zh: '九个参数：第一层的权重 W 和偏置 b，输出层的权重 v 和偏置 c。' }
	};
</script>

{#if pic}
	<figure class="net">
		<Diagram def={NET_GRAPH} vals={pic.vals} grads={pic.grads} edges={pic.edges} label={L.fig} />
		<div class="scroll" use:keyScroll>
			<table>
				<caption>{t(L.cap)}</caption>
				<thead>
					<tr>
						<th scope="col">{t(L.param)}</th>
						<th scope="col">{t(L.value)}</th>
						<th scope="col">{t(L.grad)}</th>
						{#if pic.showNum}<th scope="col">{t(L.num)}</th>{/if}
						{#if pic.showAfter}<th scope="col">{t(L.after)}</th>{/if}
					</tr>
				</thead>
				<tbody>
					{#each pic.rows as r (r.name)}
						<tr>
							<th scope="row">{r.name}</th>
							<td class="num">{fmt3(r.value)}</td>
							<td class="num g" class:unknown={!Number.isFinite(r.grad)}>{Number.isFinite(r.grad) ? fmt3(r.grad) : '?'}</td>
							{#if pic.showNum}<td class="num">{Number.isFinite(r.num) ? fmt3(r.num) : '?'}</td>{/if}
							{#if pic.showAfter}<td class="num">{fmt3(r.after)}</td>{/if}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</figure>
{/if}

<style>
	.net {
		margin: 0;
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.scroll {
		overflow-x: auto;
		min-width: 0;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-sm);
	}
	caption {
		text-align: left;
		color: var(--ink-2);
		font-size: var(--fs-xs);
		padding-bottom: 4px;
	}
	th,
	td {
		padding: 2px 12px 2px 0;
		text-align: right;
		border-bottom: 1px solid var(--rule);
		white-space: nowrap;
	}
	th[scope='row'],
	thead th:first-child {
		text-align: left;
	}
	thead th {
		color: var(--ink-2);
		font-weight: 600;
	}
	.g {
		color: var(--grad);
	}
	.g.unknown {
		color: var(--muted);
	}
</style>
