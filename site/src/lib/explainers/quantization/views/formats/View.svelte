<!--
  'quantization/formats': one number in fp32, fp16, bf16, int8 and int4 (meta.ts has the props). Each row: the format,
  its bits in groups (sign, exponent, fraction for floats; the code for integers), the value it stores and the error.
  With props.flip the fp32 bits are buttons that flip one bit of x.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical, hexToRgb, inkOn } from '$lib/kit/color';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { flipFp32 } from '../../quant.ts';
	import { NAMES, rowsOf, show } from './meta.ts';

	let { rt, at, props, setInput }: StageProps = $props();
	const r = $derived(rowsOf({ at }));
	const ROLE = { sign: 0, exp: 1, frac: 2, int: 3 } as const;
	const ROLE_NAME = {
		sign: { en: 'sign', zh: '符号' },
		exp: { en: 'exponent', zh: '指数' },
		frac: { en: 'fraction', zh: '尾数' },
		int: { en: 'integer code', zh: '整数编码' }
	};
	const colors = $derived.by(() => {
		void themeState.resolved;
		return [0, 1, 2, 3].map((k) => {
			const bg = categorical(k);
			return { bg, ink: inkOn(hexToRgb(bg)) };
		});
	});
	function flip(bit: number) {
		if (!r || !setInput) return;
		const raw = (rt.input ?? {}) as Record<string, unknown>;
		setInput({ ...raw, x: flipFp32(r.x, bit) });
	}
	const canFlip = $derived(!!props.flip && !!setInput);
</script>

{#if r}
	<div class="formats">
		<p class="x">x = <strong class="num">{show(r.x, 8)}</strong></p>
		<div class="legend" aria-hidden="true">
			{#each ['sign', 'exp', 'frac', 'int'] as const as role (role)}
				<span class="key"><span class="sw" style:background={colors[ROLE[role]].bg}></span>{t(ROLE_NAME[role])}</span>
			{/each}
		</div>
		<div class="rows" use:keyScroll>
			{#each r.rows as f (f.id)}
				<div class="row">
					<span class="name">{t(NAMES[f.id])}</span>
					<span class="bits" role={f.id === 'fp32' && canFlip ? 'group' : undefined} aria-label={f.id === 'fp32' && canFlip ? t({ en: 'fp32 bits: press one to flip it', zh: 'fp32 的各位：按一下就翻转这一位' }) : undefined}>
						{#each f.groups as g, gi (gi)}
							<span class="group">
								{#each g.bits.split('') as b, bi (bi)}
									{@const pos = f.groups.slice(0, gi).reduce((a, x) => a + x.bits.length, 0) + bi}
									{#if f.id === 'fp32' && canFlip}
										<button type="button" class="bit" style:background={colors[ROLE[g.role]].bg} style:color={colors[ROLE[g.role]].ink} aria-label={t({ en: `bit ${pos} (${ROLE_NAME[g.role].en}) is ${b}: flip it`, zh: `第 ${pos} 位（${ROLE_NAME[g.role].zh}）是 ${b}：翻转它` })} onclick={() => flip(pos)}>{b}</button>
									{:else}
										<span class="bit" style:background={colors[ROLE[g.role]].bg} style:color={colors[ROLE[g.role]].ink}>{b}</span>
									{/if}
								{/each}
							</span>
						{/each}
					</span>
					<span class="out"><span class="val num">{show(f.value, 10)}{#if f.code !== undefined}<span class="code"> (q = {f.code})</span>{/if}</span><span class="err num">{t({ en: 'error', zh: '误差' })} {show(f.err, 3)}</span></span>
				</div>
			{/each}
		</div>
	</div>
{/if}

<style>
	.formats {
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
	.rows {
		display: grid;
		gap: 6px;
		overflow-x: auto;
		min-width: 0;
	}
	.row {
		display: grid;
		grid-template-columns: 8.5rem minmax(0, 1fr);
		align-items: center;
		gap: 2px 8px;
		border-bottom: 1px solid var(--rule);
		padding-bottom: 4px;
		font-size: var(--fs-sm);
	}
	.name {
		color: var(--ink-2);
	}
	.bits {
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
		line-height: 18px;
		padding: 0;
		border: 0;
		border-right: 1px solid var(--bg);
		font-family: var(--font-mono);
	}
	button.bit {
		cursor: pointer;
	}
	button.bit:hover,
	button.bit:focus-visible {
		outline: 2px solid var(--highlight);
		outline-offset: -1px;
	}
	.code {
		color: var(--ink-2);
	}
	.err {
		color: var(--ink-2);
	}
	.out {
		grid-column: 2;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 16px;
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
