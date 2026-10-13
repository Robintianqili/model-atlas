<!--
  init-normalization Capture for the local 'setup' adapter: the toy network's setup. Five buttons for the named
  initializations, a slider for σ on a log scale (0.001 to 2), two buttons for the activation function and three for the
  normalization. Reports the whole Setup on every change (live while the slider moves).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { CaptureProps } from '../../../../atlas/types/explainer.ts';
	import { ACT_NAME, INITS, INIT_NAME, INIT_STD, NORM_NAME, STD_RANGE, initOf, setupOf, type Act, type Norm, type Setup } from '../../data.ts';

	let { raw, onchange, compact = false }: CaptureProps<Setup> = $props();

	const s = $derived(setupOf(raw));
	const named = $derived(initOf(s.std));
	const put = (o: Partial<Setup>, live = false) => onchange?.({ ...s, ...o }, { live });
	const [U0, U1] = [Math.log10(STD_RANGE[0]), Math.log10(STD_RANGE[1])];
	const uOf = (std: number) => (std <= 0 ? U0 : Math.log10(std));
	const stdOfU = (u: number) => Number((10 ** u).toPrecision(3));
	const ACTS: Act[] = ['tanh', 'relu'];
	const NORMS: Norm[] = ['none', 'batch', 'layer'];
	const L = {
		init: { en: 'Starting weights', zh: '初始权重' },
		std: { en: 'Spread σ', zh: '标准差 σ' },
		act: { en: 'Activation function', zh: '激活函数' },
		norm: { en: 'Normalization after every layer', zh: '每层之后的归一化' }
	};
</script>

<div class="setup" class:compact>
	<div class="group" role="group" aria-label={t(L.init)}>
		<span class="cap">{t(L.init)}</span>
		<div class="btns">
			{#each INITS as id (id)}
				<button type="button" class="btn" aria-pressed={named === id} onclick={() => put({ std: INIT_STD[id] })}>{t(INIT_NAME[id])}</button>
			{/each}
		</div>
	</div>
	<label class="row">
		<span class="name">{t(L.std)}</span>
		<input
			type="range"
			min={U0}
			max={U1}
			step="0.01"
			value={uOf(s.std)}
			aria-valuetext={fmt3(s.std)}
			oninput={(e) => put({ std: stdOfU(Number((e.currentTarget as HTMLInputElement).value)) }, true)}
			onchange={(e) => put({ std: stdOfU(Number((e.currentTarget as HTMLInputElement).value)) })}
		/>
		<output class="val num">{s.std === 0 ? '0' : fmt3(s.std)}</output>
	</label>
	<div class="group" role="group" aria-label={t(L.act)}>
		<span class="cap">{t(L.act)}</span>
		<div class="btns">
			{#each ACTS as a (a)}
				<button type="button" class="btn" aria-pressed={s.act === a} onclick={() => put({ act: a })}>{t(ACT_NAME[a])}</button>
			{/each}
		</div>
	</div>
	<div class="group" role="group" aria-label={t(L.norm)}>
		<span class="cap">{t(L.norm)}</span>
		<div class="btns">
			{#each NORMS as n (n)}
				<button type="button" class="btn" aria-pressed={s.norm === n} onclick={() => put({ norm: n })}>{t(NORM_NAME[n])}</button>
			{/each}
		</div>
	</div>
</div>

<style>
	.setup {
		display: grid;
		gap: 8px;
		min-width: 0;
		max-width: 520px;
		font-size: var(--fs-sm);
	}
	.group {
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	.cap,
	.name {
		color: var(--ink-2);
	}
	.btns {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.btns .btn {
		font-size: var(--fs-xs);
	}
	.btn[aria-pressed='true'] {
		border-color: var(--accent);
		box-shadow: inset 0 0 0 1px var(--accent);
		font-weight: 600;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(0, 5.5rem) minmax(0, 1fr) 3.6rem;
		align-items: center;
		gap: 8px;
	}
	input[type='range'] {
		width: 100%;
		min-width: 0;
		accent-color: var(--accent);
	}
	.val {
		text-align: right;
	}
</style>
