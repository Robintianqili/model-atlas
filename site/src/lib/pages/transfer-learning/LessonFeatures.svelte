<!--
  Lesson step 2: the frozen body (two conv blocks, locked) turns the learner's drawing into the
  256 features the head reads. All live; the old 10-way head is shown set aside.
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import RoleChip from '$lib/kit/RoleChip.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { domainOf } from '$lib/kit/color';
	import { fmtInt } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import Thumb from './Thumb.svelte';
	import { LAYER_PARAMS, PARAMS } from './data';
	import type { LessonState } from './lessonState.svelte';

	let { L, tolerance }: { L: LessonState; tolerance?: number } = $props();

	/** the channel with the largest total activation, for the small block previews */
	function best(x: { shape: number[]; data: Float32Array }) {
		const [C, H, W] = x.shape;
		let bc = 0;
		let bs = -Infinity;
		for (let c = 0; c < C; c++) {
			let s = 0;
			for (let i = 0; i < H * W; i++) s += x.data[c * H * W + i];
			if (s > bs) {
				bs = s;
				bc = c;
			}
		}
		return x.data.subarray(bc * H * W, (bc + 1) * H * W);
	}
	const f = $derived(L.feat);
	const dom = $derived(f ? domainOf('diverging', f) : ([-1, 1] as [number, number]));
	const stats = $derived.by(() => {
		if (!f) return null;
		let zeros = 0;
		let d = 0;
		for (let i = 0; i < f.length; i++) {
			if (f[i] === 0) zeros++;
			if (f[i] > f[d]) d = i;
		}
		return { zeros, d, v: f[d] };
	});
	let cell = $state<{ row: number; col: number; value: number } | null>(null);
	const b1 = LAYER_PARAMS.conv1_1 + LAYER_PARAMS.conv1_2;
	const b2 = LAYER_PARAMS.conv2_1 + LAYER_PARAMS.conv2_2;
</script>

{#if L.trace && f}
	{@const tr = L.trace}
	<div class="feat">
		<ol class="pipe" aria-label={t({ en: 'The frozen body', zh: '冻结的主体' })}>
			<li class="step"><div class="blk">
				<Thumb data={tr.t.input.data} size={44} alt="" />
				<span class="bl">{t({ en: 'Input', zh: '输入' })}</span>
				<span class="bs num">28×28</span>
			</div></li>
			<li class="step"><span class="arr" aria-hidden="true">→</span><div class="blk frozen">
				<TensorGrid data={best(tr.t.pool1)} rows={12} cols={12} cell={4} interactive={false} readout={false} gridLines={false} label={{ en: "Conv block 1 output, its strongest map", zh: "卷积块 1 的输出（最强的一张图）" }} />
				<span class="bl"><RoleChip role="frozen" iconOnly /> {t({ en: 'Conv block 1', zh: '卷积块 1' })}</span>
				<span class="bs num">8×12×12</span>
				<span class="bs num">{fmtInt(b1)} {t({ en: 'params', zh: '个参数' })}</span>
			</div></li>
			<li class="step"><span class="arr" aria-hidden="true">→</span><div class="blk frozen">
				<TensorGrid data={best(tr.t.pool2)} rows={4} cols={4} cell={12} interactive={false} readout={false} gridLines={false} label={{ en: "Conv block 2 output, its strongest map", zh: "卷积块 2 的输出（最强的一张图）" }} />
				<span class="bl"><RoleChip role="frozen" iconOnly /> {t({ en: 'Conv block 2', zh: '卷积块 2' })}</span>
				<span class="bs num">16×4×4</span>
				<span class="bs num">{fmtInt(b2)} {t({ en: 'params', zh: '个参数' })}</span>
			</div></li>
			<li class="step"><span class="arr" aria-hidden="true">→</span><div class="blk out">
				<span class="big num">256</span>
				<span class="bl"><RoleChip role="frozen" iconOnly /> {t({ en: 'Features', zh: '特征' })}</span>
				<span class="bs">{t({ en: 'flattened', zh: '展平' })}</span>
			</div></li>
			<li class="step"><span class="arr" aria-hidden="true">⇢</span><div class="blk old">
				<span class="bl">{t({ en: 'Old head', zh: '旧输出头' })}</span>
				<span class="bs num">256→10 · {fmtInt(PARAMS.oldHead)}</span>
				<span class="bs">{t({ en: 'set aside', zh: '放在一边' })}</span>
			</div></li>
		</ol>

		<div class="main">
			<div class="grid">
				<div class="hd">
					<h4>{t({ en: 'The 256 features of this drawing', zh: '这张画的 256 个特征' })}</h4>
					<ProvenanceBadge mode="live" precision="fp32" {tolerance} against="PyTorch fp32" compact />
				</div>
				<TensorGrid
					data={f}
					rows={16}
					cols={16}
					kind="diverging"
					domain={dom}
					maxCell={20}
					label={{ en: 'features: 16 channels (rows) × 16 positions (columns)', zh: '特征：16 个通道（行）× 16 个位置（列）' }}
					onhover={(c) => (cell = c)}
				/>
				<ColorLegend kind="diverging" domain={dom} />
			</div>
			<div class="side">
				<p class="note">
					{t({
						en: 'Each row is one of the 16 channels of Conv block 2 (one pattern detector each), each column one of its 4 × 4 positions, read row by row. Feature d = 16 × channel + position.',
						zh: '每一行是卷积块 2 的 16 个通道之一（每个通道是一个图案探测器），每一列是它 4 × 4 个位置之一（按行读）。特征序号 d = 16 × 通道 + 位置。'
					})}
				</p>
				{#if stats}
					<p class="stat">
						<span class="num big2">{stats.zeros}</span>
						{t({ en: 'of 256 are exactly 0: a feature is 0 when its pattern was not found in that part of the drawing (the network’s ReLU step turns every negative value into 0).', zh: '个特征（共 256 个）正好是 0：如果画的那一部分没有出现对应的图案，这个特征就是 0（网络里的 ReLU 步骤把所有负数都变成 0）。' })}
					</p>
					<p class="stat">
						{t({ en: 'Largest:', zh: '最大的：' })}
						<span class="num">f<sub>{stats.d}</sub> = <Num value={stats.v} /></span>
						{t({ en: `(channel ${Math.floor(stats.d / 16)}, position ${stats.d % 16})`, zh: `（通道 ${Math.floor(stats.d / 16)}，位置 ${stats.d % 16}）` })}
					</p>
				{/if}
				{#if cell}
					<p class="stat hov num">d = {cell.row * 16 + cell.col}: <Num value={cell.value} /></p>
				{/if}
				<p class="lock">
					<RoleChip role="frozen" label={{ en: `${fmtInt(PARAMS.frozenBody)} parameters, locked`, zh: `${fmtInt(PARAMS.frozenBody)} 个参数，已锁定` }} />
				</p>
			</div>
		</div>
	</div>
{:else}
	<p class="muted">{t({ en: 'Draw something on the pad, or pick a ladder or an envelope.', zh: '在画板上画点什么，或者选一张梯子或信封。' })}</p>
{/if}

<style>
	.feat {
		container: lf / inline-size;
		display: grid;
		gap: 14px;
		min-width: 0;
	}
	.pipe {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
	}
	.step {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.blk {
		display: grid;
		justify-items: center;
		gap: 3px;
		padding: 6px 8px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		min-width: 70px;
	}
	.blk.frozen,
	.blk.out {
		background: var(--surface-2);
	}
	.blk.old {
		border-style: dashed;
		color: var(--muted);
	}
	.blk.old .bl {
		text-decoration: line-through;
	}
	.bl {
		display: inline-flex;
		align-items: center;
		gap: 2px;
		font-size: var(--fs-xs);
		font-weight: 650;
		white-space: nowrap;
	}
	.bs {
		font-size: 11px;
		color: var(--muted);
		white-space: nowrap;
	}
	.big {
		font-size: var(--fs-xl);
		font-weight: 700;
		line-height: 1.2;
	}
	.arr {
		color: var(--muted);
	}
	.main {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 12px 18px;
		align-items: start;
	}
	@container lf (min-width: 560px) {
		.main {
			grid-template-columns: minmax(0, auto) minmax(0, 1fr);
		}
	}
	.grid {
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	.hd {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 4px 10px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.side {
		display: grid;
		gap: 8px;
		align-content: start;
	}
	.note,
	.stat {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.stat {
		color: var(--ink);
	}
	.big2 {
		font-size: var(--fs-xl);
		font-weight: 700;
	}
	.hov {
		color: var(--accent);
	}
	.lock {
		margin: 0;
	}
</style>
