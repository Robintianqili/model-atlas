<!-- fno 'field' (meta.ts lists the modes). Green: the solver; magenta dashes: the network; channels in the categorical colors. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import Chart from '../../parts/Chart.svelte';
	import { GOAL } from '../../fno.ts';
	import { rawOf } from '../../adapters/series/adapter.ts';
	import { K, TALL, fieldData } from './meta.ts';

	let P: StageProps = $props();
	const d = $derived(fieldData(P));
	const s = $derived(d.st);
	const X = { en: 'point around the ring (0 to 63)', zh: '环上的点（0 到 63）' };
	const XM = { en: 'mode k (waves around the ring)', zh: '模式 k（绕环的波数）' };
	const H = { en: 'height', zh: '高度' };
	const sol = { en: 'solver, 50 steps', zh: '求解器，50 步' }, pred = { en: 'network, one pass', zh: '网络，一次前向' }, start = { en: 'starting curve', zh: '起始曲线' }; // prettier-ignore
	const ch = (c: number) => ({ en: `channel ${c}`, zh: `通道 ${c}` });
	let draft = $state<number | null>(null);
	const keep = $derived(draft ?? s?.keep ?? K);
	const commit = (v: number) => {
		draft = null;
		(P.setInput ?? ((x: unknown) => P.rt.setInput(x)))({ ...rawOf(P.rt.input), keep: v });
	};
</script>

<div class="field">
	{#if !s}
		<p class="note">{t({ en: 'No curve yet: pick one or draw one in the input box.', zh: '还没有曲线：在输入框里选一条或画一条。' })}</p>
	{:else if d.mode === 'input'}
		<Chart lines={[{ y: s.u0, cls: 'u0', name: start }]} xLabel={X} yLabel={H} lo={-1.5} hi={1.5} label={start} />
	{:else if d.mode === 'solver'}
		<Chart lines={[{ y: s.u0, cls: 'faint', name: start }, { y: d.frameU ?? [], cls: 'sol', name: { en: `solver after step ${d.step}`, zh: `求解器第 ${d.step} 步之后` } }]} xLabel={X} yLabel={H} lo={-1.5} hi={1.5} label={sol} />
	{:else if d.mode === 'lift'}
		<Chart lines={d.chans.map((y, c) => ({ y, cls: `c${c}`, name: ch(c) }))} xLabel={X} yLabel={{ en: 'channel value', zh: '通道值' }} label={{ en: 'The four channels', zh: '四个通道' }} />
	{:else if d.mode === 'fft' || d.mode === 'keep'}
		<Chart bars={{ v: d.spec, name: { en: `wave height, channel ${d.ch}`, zh: `波高，通道 ${d.ch}` }, cut: d.mode === 'keep' ? s.keep : undefined }} mark={d.mode === 'keep' ? s.keep : null} xLabel={XM} yLabel={{ en: 'wave height', zh: '波高' }} h={180} label={{ en: 'Wave heights by mode', zh: '各模式的波高' }} />
		{#if d.mode === 'keep'}
			<div class="answer"><Chart lines={[{ y: d.lifted, cls: 'faint', name: ch(d.ch) }, { y: d.rebuilt, cls: 'c0', name: { en: `rebuilt from the ${s.keep} kept modes`, zh: `用保留的 ${s.keep} 个模式重建` } }]} xLabel={X} yLabel={{ en: 'channel value', zh: '通道值' }} h={180} label={{ en: 'The channel and its rebuild from the kept modes', zh: '通道和用保留模式重建的结果' }} /></div>
		{/if}
	{:else if d.mode === 'back'}
		<Chart lines={[{ y: d.back[0], cls: 'c1', name: { en: 'inverse FFT (spectral path)', zh: '逆 FFT（频域通路）' } }, { y: d.back[1], cls: 'c2', name: { en: 'pointwise path', zh: '逐点通路' } }, { y: d.back[2], cls: 'faint', name: { en: 'their sum', zh: '两者之和' } }, { y: d.back[3], cls: 'c0', name: { en: 'after GELU', zh: 'GELU 之后' } }]} xLabel={X} yLabel={{ en: `channel ${d.ch} value`, zh: `通道 ${d.ch} 的值` }} label={{ en: 'One channel through the rest of the layer', zh: '一个通道走完这一层的其余部分' }} />
	{:else if d.mode === 'gain' && d.gain}
		<Chart bars={{ v: d.gain.solver, name: { en: 'solver: share of the wave left after 50 steps', zh: '求解器：50 步后这个波剩下的比例' } }} dots={{ v: d.gain.learned, name: { en: 'network: share it keeps', zh: '网络：它保留的比例' } }} mark={K} xLabel={XM} yLabel={{ en: 'share kept', zh: '保留的比例' }} lo={0} hi={1} label={{ en: 'The learned filter and the solver’s', zh: '学到的滤波器和求解器的滤波器' }} />
	{:else if d.mode === 'limits' && d.tall}
		<Chart lines={[{ y: d.tall.sol, cls: 'sol', name: sol }, { y: d.tall.pred, cls: 'pred', name: pred }]} xLabel={X} yLabel={H} label={{ en: `Your curve ${TALL} times taller`, zh: `你的曲线加高到 ${TALL} 倍` }} />
		<p class="cap">{t({ en: `${TALL} times taller: largest error ${fmt3(d.tall.err)} (your curve: ${fmt3(s.err)}).`, zh: `加高到 ${TALL} 倍：最大误差 ${fmt3(d.tall.err)}（你的曲线：${fmt3(s.err)}）。` })}</p>
	{:else}
		{#if d.mode === 'modes'}
			<label class="slider"><span>{t({ en: `modes kept: ${keep} of ${K}`, zh: `保留的模式：${keep} / ${K}` })}</span><input type="range" min="1" max={K} step="1" value={keep} oninput={(e) => (draft = +e.currentTarget.value)} onchange={(e) => commit(+e.currentTarget.value)} /></label>
		{/if}
		<Chart lines={[{ y: s.u0, cls: 'faint', name: start }, { y: s.sol, cls: 'sol', name: sol }, { y: s.pred, cls: 'pred', name: pred }]} xLabel={X} yLabel={H} lo={-1.5} hi={1.5} h={d.mode === 'modes' ? 180 : 210} label={{ en: 'Prediction and solver', zh: '预测和求解器' }} />
		<p class="cap">{t({ en: `Largest difference ${fmt3(s.err)}, at point ${s.at}.`, zh: `最大差距 ${fmt3(s.err)}，在点 ${s.at}。` })}</p>
		{#if d.mode === 'modes'}
			<Chart bars={{ v: d.errs, name: { en: 'largest error with k modes kept (k = 1 to 16)', zh: '保留 k 个模式时的最大误差（k = 1 到 16）' } }} lines={[{ y: d.errs.map(() => GOAL), cls: 'faint', name: { en: `target: ${GOAL}`, zh: `目标：${GOAL}` } }]} hot={keep - 1} x0={1} xLabel={{ en: 'modes kept', zh: '保留的模式数' }} yLabel={{ en: 'largest error (log scale)', zh: '最大误差（对数刻度）' }} log lo={0.001} h={160} label={{ en: 'Largest error by modes kept', zh: '按保留模式数的最大误差' }} />
		{/if}
	{/if}
</div>

<style>
	.field { display: grid; gap: var(--s2); grid-template-columns: minmax(0, 1fr); }
	.cap, .note { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); font-variant-numeric: tabular-nums; }
	.slider { display: grid; gap: 4px; font-size: var(--fs-sm); color: var(--ink); }
	.slider input { width: 100%; max-width: 420px; }
	/* the predict gate leaves the stage readable: hide the answer (the rebuild) until the learner has answered */
	:global([inert]) .answer { visibility: hidden; }
</style>
