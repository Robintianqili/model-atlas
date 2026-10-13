<!--
  'number-formats/bits': one number in several formats, bit by bit (meta.ts has the props). Each row: the format, its bits
  coloured by role, the value stored, the error, the gap and what happened. With props.detail a worked line and a ruler
  for one format; with props.flip every bit is a button that flips it and sets x to the new value.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical, hexToRgb, inkOn } from '$lib/kit/color';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { I18n } from '../../../../atlas/types/core.ts';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { decode, flipBit, type FmtId, type Role, type Row } from '../../fmt.ts';
	import { dataOf, DIGITS, NAMES, neighbours, show, splitText, STATUS, workedText } from './meta.ts';

	let { rt, at, props, setInput }: StageProps = $props();
	const d = $derived(dataOf({ at, props }));
	const canFlip = $derived(!!props.flip && !!setInput);
	const detail = $derived(d?.rows.find((r) => r.id === (props.detail as FmtId | undefined)) ?? null);
	let note = $state<I18n | null>(null);

	const ROLE_INDEX: Record<Role, number> = { sign: 0, exp: 1, frac: 2, int: 3, idx: 3 };
	const ROLE_NAME: Record<Role, I18n> = {
		sign: { en: 'sign', zh: '符号' },
		exp: { en: 'exponent', zh: '指数' },
		frac: { en: 'fraction', zh: '尾数' },
		int: { en: 'integer code', zh: '整数编码' },
		idx: { en: 'NF4 index', zh: 'NF4 索引' }
	};
	const colors = $derived.by(() => {
		void themeState.resolved;
		return [0, 1, 2, 3].map((k) => {
			const bg = categorical(k);
			return { bg, ink: inkOn(hexToRgb(bg)) };
		});
	});
	const legendRoles = $derived.by(() => {
		const seen = new Set<Role>();
		for (const r of d?.rows ?? []) for (const g of r.groups) seen.add(g.role);
		return (['sign', 'exp', 'frac', 'int', 'idx'] as Role[]).filter((r) => seen.has(r));
	});

	function flip(r: Row, bit: number) {
		if (!setInput) return;
		const code = flipBit(r.id, r.code, bit);
		const v = decode(r.id, code);
		const bits = code.toString(2).padStart(r.total, '0');
		if (!Number.isFinite(v)) {
			const what = Number.isNaN(v) ? 'NaN' : v > 0 ? '+∞' : '−∞';
			note = { en: `In ${NAMES[r.id].en}, the pattern ${bits} is ${what}, not a number x can be: x stays as it is.`, zh: `在 ${NAMES[r.id].zh} 里，位模式 ${bits} 是 ${what}，不能作为 x：x 保持不变。` };
			return;
		}
		note = null;
		const raw = (rt.input ?? {}) as Record<string, unknown>;
		setInput({ ...raw, x: v });
	}
	const pos = (r: Row, gi: number, bi: number) => r.groups.slice(0, gi).reduce((a, g) => a + g.bits.length, 0) + bi;

	// the ruler of the detail format: the stored value, its two neighbours and x
	const RW = 340;
	const RH = 64;
	const RM = 24;
	const ruler = $derived.by(() => {
		if (!detail || !d || !Number.isFinite(detail.value)) return null;
		const [lo, hi] = neighbours(detail);
		const pts = [lo, detail.value, hi, d.x].filter(Number.isFinite);
		const a = Math.min(...pts);
		const b = Math.max(...pts);
		if (!(b > a)) return null;
		const sx = (v: number) => RM + ((v - a) / (b - a)) * (RW - 2 * RM);
		return { lo, hi, v: detail.value, x: d.x, sx };
	});
</script>

{#if d}
	<div class="bitsview">
		<p class="x">x = <strong class="num">{show(d.x, 9)}</strong></p>
		<div class="legend" aria-hidden="true">
			{#each legendRoles as role (role)}
				<span class="key"><span class="sw" style:background={colors[ROLE_INDEX[role]].bg}></span>{t(ROLE_NAME[role])}</span>
			{/each}
		</div>
		{#if note}<p class="note" role="status">{t(note)}</p>{/if}
		<div class="rows" use:keyScroll>
			{#each d.rows as r (r.id)}
				<div class="row" class:sel={detail?.id === r.id}>
					<span class="name">{t(NAMES[r.id])} <span class="split">{t(splitText(r.id))}</span></span>
					<span class="cells" role={canFlip ? 'group' : undefined} aria-label={canFlip ? t({ en: `${NAMES[r.id].en} bits: press one to flip it`, zh: `${NAMES[r.id].zh} 的各位：按一下就翻转这一位` }) : undefined}>
						{#each r.groups as g, gi (gi)}
							<span class="group">
								{#each g.bits.split('') as b, bi (bi)}
									{@const c = colors[ROLE_INDEX[g.role]]}
									{#if canFlip}
										<button
											type="button"
											class="bit"
											style:background={c.bg}
											style:color={c.ink}
											aria-label={t({ en: `${NAMES[r.id].en} bit ${pos(r, gi, bi)} (${ROLE_NAME[g.role].en}) is ${b}: flip it`, zh: `${NAMES[r.id].zh} 第 ${pos(r, gi, bi)} 位（${ROLE_NAME[g.role].zh}）是 ${b}：翻转它` })}
											onclick={() => flip(r, pos(r, gi, bi))}>{b}</button
										>
									{:else}
										<span class="bit" style:background={c.bg} style:color={c.ink}>{b}</span>
									{/if}
								{/each}
							</span>
						{/each}
					</span>
					<span class="out">
						<span class="val num">{show(r.value, DIGITS[r.id])}</span>
						<span class="meta num">{t({ en: 'error', zh: '误差' })} {show(r.err, 3)}</span>
						<span class="meta num">{t({ en: 'gap', zh: '间隔' })} {Number.isFinite(r.gap) ? show(r.gap, 3) : t({ en: 'none (largest)', zh: '无（已是最大）' })}</span>
						<span class="status" class:warn={r.status !== 'exact' && r.status !== 'rounded'}>{t(STATUS[r.status])}</span>
					</span>
				</div>
			{/each}
		</div>
		{#if detail}
			<p class="worked">{t(workedText(detail))}</p>
			{#if ruler}
				<svg class="ruler" viewBox="0 0 {RW} {RH}" role="img" aria-label={t({ en: `Ruler: ${NAMES[detail.id].en} stores ${show(ruler.v, DIGITS[detail.id])} between its neighbours ${show(ruler.lo, DIGITS[detail.id])} and ${show(ruler.hi, DIGITS[detail.id])}; x is ${show(ruler.x, 9)}`, zh: `尺子：${NAMES[detail.id].zh} 存成 ${show(ruler.v, DIGITS[detail.id])}，两侧的邻居是 ${show(ruler.lo, DIGITS[detail.id])} 和 ${show(ruler.hi, DIGITS[detail.id])}；x 是 ${show(ruler.x, 9)}` })}>
					<line class="axis" x1={RM / 2} x2={RW - RM / 2} y1="30" y2="30" />
					{#each [ruler.lo, ruler.hi] as n, i (i)}
						{#if Number.isFinite(n)}
							<line class="tick" x1={ruler.sx(n)} x2={ruler.sx(n)} y1="22" y2="38" />
							<text class="lab" x={ruler.sx(n)} y="52" text-anchor={i === 0 ? 'start' : 'end'}>{show(n, DIGITS[detail.id])}</text>
						{/if}
					{/each}
					<line class="tick v" x1={ruler.sx(ruler.v)} x2={ruler.sx(ruler.v)} y1="18" y2="42" />
					<text class="lab v" x={ruler.sx(ruler.v)} y="52" text-anchor="middle">{show(ruler.v, DIGITS[detail.id])}</text>
					<path class="xmark" d="M {ruler.sx(ruler.x)} 24 l -5 -9 h 10 z" />
					<text class="lab xl" x={ruler.sx(ruler.x)} y="11" text-anchor="middle">x</text>
				</svg>
			{/if}
		{/if}
	</div>
{/if}

<style>
	.bitsview {
		container-type: inline-size;
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.x {
		margin: 0;
		font-size: var(--fs-md);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
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
	}
	.note {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		border-left: 3px solid var(--highlight);
		padding-left: 8px;
	}
	.rows {
		display: grid;
		gap: 6px;
		overflow-x: auto;
		min-width: 0;
	}
	.row {
		display: grid;
		grid-template-columns: 9rem minmax(0, 1fr);
		align-items: center;
		gap: 2px 8px;
		border-bottom: 1px solid var(--rule);
		padding: 2px 4px 4px;
		font-size: var(--fs-sm);
	}
	.row.sel {
		outline: 2px solid var(--highlight);
		outline-offset: -1px;
		border-radius: 4px;
	}
	.name {
		color: var(--ink);
	}
	.split {
		display: block;
		color: var(--ink-2);
		font-size: var(--fs-xs);
	}
	.cells {
		display: inline-flex;
		gap: 3px;
		font-family: var(--font-mono);
	}
	.group {
		display: inline-flex;
	}
	.bit {
		display: inline-block;
		width: 11px;
		text-align: center;
		font-size: 11px;
		line-height: 20px;
		padding: 0;
		border: 0;
		border-right: 1px solid var(--bg);
		font-family: var(--font-mono);
	}
	button.bit {
		cursor: pointer;
		min-height: 24px;
	}
	button.bit:hover,
	button.bit:focus-visible {
		outline: 2px solid var(--highlight);
		outline-offset: -1px;
	}
	.out {
		grid-column: 2;
		display: flex;
		flex-wrap: wrap;
		gap: 2px 14px;
		align-items: baseline;
	}
	.val {
		font-weight: 600;
	}
	.meta {
		color: var(--ink-2);
		font-size: var(--fs-xs);
	}
	.status {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		border: 1px solid var(--rule-strong);
		border-radius: 10px;
		padding: 0 6px;
	}
	.status.warn {
		color: var(--ink);
		border: 1px dashed var(--ink-2);
	}
	.worked {
		margin: 0;
		font-size: var(--fs-sm);
		font-family: var(--font-mono);
		overflow-wrap: anywhere;
	}
	.ruler {
		width: 100%;
		max-width: 420px;
		height: auto;
	}
	.axis,
	.tick {
		stroke: var(--ink-2);
		stroke-width: 1.2;
	}
	.tick.v {
		stroke: var(--ink);
		stroke-width: 2.5;
	}
	.xmark {
		fill: var(--highlight);
	}
	.lab {
		font-size: 11px;
		fill: var(--ink-2);
		font-family: var(--font-mono);
	}
	.lab.v,
	.lab.xl {
		fill: var(--ink);
	}
	@container (max-width: 520px) {
		.row {
			grid-template-columns: minmax(0, 1fr);
		}
		.out {
			grid-column: 1;
		}
		.bit {
			width: 9px;
			font-size: 10px;
		}
	}
</style>
