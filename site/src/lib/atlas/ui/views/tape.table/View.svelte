<!--
  tape.table: the frames of a process or training tape as a table, like `atlas run` prints it in Node: frame, phase,
  step, then the frame's metrics and the schema's scalar fields (props.columns picks them). A window of rows around the
  stage's frame (props.window, default 30); that frame's row is outlined and kept in view. Changed values are marked
  against the row above, so what a frame did is visible without animation.
-->
<script module lang="ts">
	let instances = 0;
</script>

<script lang="ts">
	import { fmt3 } from '../../../core/format.ts';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import Notice from '../kit.parts/Notice.svelte';
	import { tableOf } from './rows.ts';

	let P: StageProps = $props();
	const D = $derived(tableOf(P));
	const show = (v: number | string | undefined) => (v === undefined ? '' : typeof v === 'number' ? (Number.isInteger(v) && Math.abs(v) < 1e6 ? String(v) : fmt3(v)) : v);
	// (no bind:this: an id keeps the view off a Svelte runtime helper that the flagship pages' chunks share)
	const uid = `tape-table-${++instances}`;
	$effect(() => {
		// keep the current row in view inside the table's own scroll box (never scroll the page)
		const at = D?.at;
		const wrap = document.getElementById(uid);
		if (!wrap || at === undefined) return;
		const row = wrap.querySelector<HTMLTableRowElement>(`tr[data-i="${at}"]`);
		if (!row) return;
		const top = row.offsetTop - wrap.clientHeight / 2;
		wrap.scrollTop = Math.max(0, top);
	});
	$effect(() => {
		if (P.autoplay && D) P.onend();
	});
</script>

{#if !D}
	<Notice text={{ en: 'No frames yet: the process has not run.', zh: '还没有帧：过程还没有运行。' }} />
{:else}
	<div class="tt">
		<p class="hd">
			{t({ en: `Frames ${D.from + 1}–${D.to} of ${D.tape.length}`, zh: `第 ${D.from + 1}–${D.to} 帧，共 ${D.tape.length} 帧` })}
			<span class="muted num">· {D.tape.process}</span>
		</p>
		<div class="wrap" id={uid} use:keyScroll>
			<table class="num">
				<caption class="visually-hidden">{t({ en: 'Recorded frames of the process', zh: '过程记录下的各帧' })}</caption>
				<thead>
					<tr>
						<th scope="col">#</th>
						<th scope="col" class="l">{t({ en: 'phase', zh: '阶段' })}</th>
						<th scope="col">{t({ en: 'step', zh: '步' })}</th>
						{#each D.cols as c (c.key)}<th scope="col">{c.label}</th>{/each}
					</tr>
				</thead>
				<tbody>
					{#each D.rows as r, k (r.i)}
						{@const prev = k > 0 ? D.rows[k - 1] : null}
						<tr data-i={r.i} class:on={r.i === D.at} aria-current={r.i === D.at ? 'step' : undefined}>
							<th scope="row">{r.i + 1}</th>
							<td class="l">{t(r.phaseLabel)}</td>
							<td>{r.step === undefined ? '' : r.step}</td>
							{#each D.cols as c (c.key)}
								{@const v = r.cells[c.key]}
								<td class:chg={prev !== null && v !== undefined && prev.cells[c.key] !== undefined && prev.cells[c.key] !== v}>{show(v)}</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="muted small">{t({ en: 'Bold numbers changed since the row above. Steps count from 1.', zh: '粗体数字表示与上一行相比有变化。步数从 1 开始数。' })}</p>
	</div>
{/if}

<style>
	.tt {
		display: grid;
		gap: 8px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.hd {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.wrap {
		max-height: 360px;
		max-width: 100%;
		overflow: auto;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		position: relative;
	}
	table {
		border-collapse: collapse;
		font-size: 12px;
		min-width: 100%;
	}
	th,
	td {
		padding: 2px 10px;
		text-align: right;
		white-space: nowrap;
		border-bottom: 1px solid var(--surface-2);
	}
	thead th {
		position: sticky;
		top: 0;
		z-index: 1;
		background: var(--surface-2);
		font-weight: 600;
	}
	tbody th {
		color: var(--muted);
		font-weight: 400;
	}
	.l {
		text-align: left;
	}
	tr.on td,
	tr.on th {
		background: var(--accent-soft);
		color: var(--ink);
	}
	tr.on th {
		box-shadow: inset 3px 0 0 var(--accent);
	}
	td.chg {
		font-weight: 700;
	}
	.small {
		margin: 0;
		font-size: var(--fs-xs);
	}
</style>
