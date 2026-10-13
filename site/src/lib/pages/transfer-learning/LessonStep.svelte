<!--
  Lesson step 5: one step of full-batch gradient descent, by hand, for one training example:
  forward → softmax → loss → gradient → update. Every number is computed live from the lesson's
  head (step 0 → step 1) and checked against the trainer's own weights after step 1.
-->
<script lang="ts">
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import RoleFrame from '$lib/kit/RoleFrame.svelte';
	import Tex from '$lib/kit/Tex.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { domainOf } from '$lib/kit/color';
	import { sym } from '$lib/kit/tex';
	import { t } from '$lib/i18n/i18n.svelte';
	import Thumb from './Thumb.svelte';
	import { L2, LESSON_N, LR, clsName, type LessonState } from './lessonState.svelte';

	interface Props {
		L: LessonState;
		autoplay?: boolean;
		ondone?: () => void;
	}
	let { L, autoplay = false, ondone }: Props = $props();

	let sub = $state(0);
	// the side panel follows: the head after this step once the update has been applied
	$effect(() => {
		L.headStep = sub >= 4 ? 1 : 0;
	});

	const STAGES = [
		{ en: 'Forward: two scores', zh: '前向：两个分数' },
		{ en: 'Softmax: probabilities', zh: 'Softmax：概率' },
		{ en: 'Loss: how wrong', zh: '损失：错了多少' },
		{ en: 'Gradient: which way', zh: '梯度：往哪边改' },
		{ en: 'Update: one small step', zh: '更新：迈一小步' }
	];

	const d = $derived(L.detail);
	const run = $derived(L.run);
	const Y = $derived(d ? (d.y === 0 ? 'A' : 'B') : 'A');
	const zA = sym('out', 'z_A');
	const zB = sym('out', 'z_B');
	const fTex = `\\tfrac{${sym('act', 'f_d')}}{s}`;
	// narrow stages (phones, or a narrow lesson column) get the same formulas broken over more lines
	let w = $state(600);
	const nw = $derived(w < 540);
	const br = (narrow: boolean) => (narrow ? ' \\\\ &' : ' ');
	const fwdTex = (num: boolean, n: boolean) =>
		`\\begin{aligned} ${zA} &= \\textstyle\\sum_{d} ${sym('param', 'W_{A,d}')}\\,${fTex} + ${sym('param', 'b_A')}${num ? `${br(n)}= {{sA}} + ({{bA}}) = {{zA}}` : ''} \\\\ ${zB} &= \\textstyle\\sum_{d} ${sym('param', 'W_{B,d}')}\\,${fTex} + ${sym('param', 'b_B')}${num ? `${br(n)}= {{sB}} + ({{bB}}) = {{zB}}` : ''} \\end{aligned}`;
	const smTex = (num: boolean, n: boolean) =>
		`\\begin{aligned} ${sym('prob', 'p_A')} &= \\frac{e^{${zA}}}{e^{${zA}} + e^{${zB}}}${num ? ' = {{pA}}' : ''}${n ? ' \\\\ ' : ',\\quad '}${sym('prob', 'p_B')} ${n ? '&' : ''}= 1 - ${sym('prob', 'p_A')}${num ? ' = {{pB}}' : ''} \\end{aligned}`;
	const lossTex = (num: boolean) => `L = -\\ln ${sym('prob', `p_${Y}`)}${num ? ` = -\\ln {{py}} = {{L}}` : ''}`;
	const gradTex = (num: boolean, n: boolean) =>
		`\\begin{aligned} ${sym('grad', '\\tfrac{\\partial L}{\\partial z_A}')} &= ${sym('prob', 'p_A')} - ${Y === 'A' ? 1 : 0}${num ? ' = {{gA}}' : ''}${n ? ' \\\\ ' : ' \\qquad '}${sym('grad', '\\tfrac{\\partial L}{\\partial z_B}')} ${n ? '&' : ''}= ${sym('prob', 'p_B')} - ${Y === 'B' ? 1 : 0}${num ? ' = {{gB}}' : ''} \\\\ ${sym('grad', `\\tfrac{\\partial L}{\\partial W_{${Y},d}}`)} &= ${sym('grad', `\\tfrac{\\partial L}{\\partial z_${Y}}`)}\\cdot ${fTex} \\end{aligned}`;
	const updTex = (num: boolean, n: boolean) =>
		`\\begin{aligned} ${sym('param', `W_{${Y},d}`)} &\\leftarrow ${sym('param', `W_{${Y},d}`)} - \\eta\\,\\big(${sym('grad', `\\bar g_{${Y},d}`)} + \\lambda\\, ${sym('param', `W_{${Y},d}`)}\\big)${num ? (n ? ` \\\\ &= {{w0}} - ${LR}\\times\\big({{gbar}} \\\\ &\\qquad + ${L2}\\times({{w0}})\\big) \\\\ &= {{w1}}` : ` \\\\ &= {{w0}} - ${LR}\\,\\big({{gbar}} + ${L2}\\times({{w0}})\\big) \\\\ &= {{w1}}`) : ''} \\end{aligned}`;

	const vals = $derived<Record<string, number>>(
		d
			? {
					sA: d.sum[0],
					sB: d.sum[1],
					bA: d.b[0],
					bB: d.b[1],
					zA: d.z[0],
					zB: d.z[1],
					pA: d.p[0],
					pB: d.p[1],
					py: d.p[d.y],
					L: d.loss,
					gA: d.dz[0],
					gB: d.dz[1],
					w0: d.w0,
					gbar: d.gMean[d.y * 256 + d.d],
					w1: d.w1
				}
			: {}
	);
	const gDom = $derived(d ? domainOf('diverging', d.gEx) : ([-1, 1] as [number, number]));
	const dwDom = $derived(d ? domainOf('diverging', d.dW) : ([-1, 1] as [number, number]));
	const exLabel = $derived(d ? `${t(clsName(d.y))} 1` : '');
	const pct = (v: number) => (v * 100).toFixed(1) + '%';
	const hl = $derived(d ? [{ row: Math.floor(d.d / 16), col: d.d % 16, color: 'var(--highlight)' }] : []);
</script>

{#if run && d}
	<div class="ls" bind:clientWidth={w}>
		<div class="pick">
			<span class="muted small" id="ls-ex">{t({ en: 'Training example:', zh: '训练样本：' })}</span>
			<div class="segmented" role="group" aria-labelledby="ls-ex">
				{#each [0, LESSON_N] as ex (ex)}
					{@const k = ex === 0 ? 0 : 1}
					<button type="button" aria-pressed={L.example === ex} onclick={() => (L.example = ex)}>
						<Thumb data={run.inputs[ex]} size={22} alt="" />
						<span class="dot {k === 0 ? 'a' : 'b'}"></span>{t(clsName(k))} 1
					</button>
				{/each}
			</div>
		</div>

		<StepPlayer count={5} bind:step={sub} interval={2800} labels={STAGES} {autoplay} onend={ondone} label={{ en: 'One training step, stage by stage', zh: '一步训练，逐个阶段' }}>
			{#snippet children(s)}
				<ol class="rows">
					<li class:cur={s === 0} class:future={s < 0}>
						<p class="rt"><span class="n">1</span>{t(STAGES[0])}</p>
						<Tex display tex={fwdTex(s >= 0, nw)} values={vals} />
						<p class="note">{t({ en: `Σ_d means: add up over all 256 features d (d = 0 … 255). f = the 256 frozen features of ${exLabel}; s = ${run.scale.toFixed(2)}, their typical size over the 16 training examples (the head divides by it so one learning rate fits).`, zh: `Σ_d 的意思是：把全部 256 个特征（d = 0 … 255）的结果加起来。f = ${exLabel} 的 256 个冻结特征；s = ${run.scale.toFixed(2)}，是 16 个训练样本特征的典型大小（输出头先除以它，这样一个学习率就够用）。` })}</p>
					</li>
					<li class:cur={s === 1} class:future={s < 1}>
						<p class="rt"><span class="n">2</span>{t(STAGES[1])}</p>
						<Tex display tex={smTex(s >= 1, nw)} values={vals} />
					</li>
					<li class:cur={s === 2} class:future={s < 2}>
						<p class="rt"><span class="n">3</span>{t(STAGES[2])}</p>
						<Tex display tex={lossTex(s >= 2)} values={vals} />
						<p class="note">{t({ en: `The right answer is ${t(clsName(d.y))} (${Y}). −ln 0.5 = 0.693 is the loss of a coin flip; a sure, right answer (p = 1) would give 0.`, zh: `正确答案是${t(clsName(d.y))}（${Y}）。−ln 0.5 = 0.693 是抛硬币的损失；如果有把握地答对（p = 1），损失就是 0。` })}</p>
					</li>
					<li class:cur={s === 3} class:future={s < 3}>
						<p class="rt"><span class="n">4</span>{t(STAGES[3])}</p>
						<Tex display tex={gradTex(s >= 3, nw)} values={vals} />
						<p class="note">{t({ en: `Read ∂L/∂z_${Y} as “how much the loss changes when z_${Y} goes up a little”. Negative means: raising z_${Y} lowers the loss. For one weight, multiply by its feature f_d/s (the chain rule).`, zh: `∂L/∂z_${Y} 读作“z_${Y} 稍微变大一点时，损失变化多少”。负数表示：让 z_${Y} 变大，损失就会变小。对单个权重，再乘上它对应的特征 f_d/s（链式法则）。` })}</p>
						{#if s >= 3}
							<div class="vis">
								<RoleFrame role="grad" label={`∂L/∂W_${Y} · ${exLabel}`}>
									<TensorGrid data={d.gEx.subarray(d.y * 256, d.y * 256 + 256)} rows={16} cols={16} domain={gDom} maxCell={8} readout={false} highlights={hl} label={{ en: 'gradient of this example for the right output’s weights', zh: '这个样本对正确输出权重的梯度' }} />
								</RoleFrame>
								<p class="note">
									{t({ en: `Every cell is blue (negative) or gray (0): stepping against the gradient raises these weights, so z_${Y} grows for drawings like this one. Where a feature is 0, its weight gets no push. Framed: d = ${d.d}, this drawing's largest feature, f/s = `, zh: `每个格子不是蓝色（负值）就是灰色（0）：逆着梯度走，这些权重就会变大，于是像这张画一样的输入，z_${Y} 会变大。特征为 0 的地方，权重不受影响。方框：d = ${d.d}，这张画最大的特征，f/s = ` })}<Num value={d.fs[d.d]} />{t({ en: ', gradient ', zh: '，梯度 ' })}<Num value={d.gEx[d.y * 256 + d.d]} signed />{t({ en: '.', zh: '。' })}
								</p>
							</div>
						{/if}
					</li>
					<li class:cur={s === 4} class:future={s < 4}>
						<p class="rt"><span class="n">5</span>{t(STAGES[4])}</p>
						<Tex display tex={updTex(s >= 4, nw)} values={vals} />
						<p class="note">{t({ en: `η = ${LR} is the learning rate, λ = ${L2} a small weight decay, and ḡ the gradient averaged over all 16 training examples (this is full-batch gradient descent). Shown for d = ${d.d}; every one of the 514 parameters moves the same way.`, zh: `η = ${LR} 是学习率，λ = ${L2} 是很小的权重衰减，ḡ 是全部 16 个训练样本的平均梯度（这就是全批量梯度下降）。这里只写出 d = ${d.d}；514 个参数每一个都这样更新。` })}</p>
						{#if s >= 4}
							<div class="vis">
								<div class="dwbox">
									<RoleFrame role="param" status="trainable" label={`ΔW_${Y}`}>
										<TensorGrid data={d.dW.subarray(d.y * 256, d.y * 256 + 256)} rows={16} cols={16} domain={dwDom} maxCell={8} readout={false} highlights={hl} label={{ en: 'change of the right output’s weights in this step', zh: '这一步中正确输出权重的变化' }} />
									</RoleFrame>
									<p class="note">{t(d.y === 0 ? { en: 'ΔW_A, the change from this step, mixes all 16 examples: orange = the weight went up, blue = it went down. It goes up where ladders have bigger features than envelopes, and down where envelopes do.', zh: 'ΔW_A 是这一步带来的变化，综合了全部 16 个样本：橙色 = 权重变大，蓝色 = 变小。梯子的特征比信封大的地方，它变大；信封的特征更大的地方，它变小。' } : { en: 'ΔW_B, the change from this step, mixes all 16 examples: orange = the weight went up, blue = it went down. It goes up where envelopes have bigger features than ladders, and down where ladders do.', zh: 'ΔW_B 是这一步带来的变化，综合了全部 16 个样本：橙色 = 权重变大，蓝色 = 变小。信封的特征比梯子大的地方，它变大；梯子的特征更大的地方，它变小。' })}</p>
								</div>
								<ul class="after">
									<li>{t({ en: `${exLabel}: p(${t(clsName(d.y))})`, zh: `${exLabel}：p（${t(clsName(d.y))}）` })} <span class="num">{d.p[d.y].toFixed(3)} → <strong>{d.p1.toFixed(3)}</strong></span></li>
									<li>{t({ en: 'mean loss over the 16 examples', zh: '16 个样本的平均损失' })} <span class="num">{run.loss[0].toFixed(3)} → <strong>{run.loss[1].toFixed(3)}</strong></span></li>
									<li>{t({ en: 'held-out test drawings right', zh: '留出测试图判对' })} <span class="num">{pct(run.testAcc[0])} → <strong>{pct(run.testAcc[1])}</strong></span></li>
									<li class="muted">{t({ en: 'Checked: equals the trainer’s own weights after step 1 (largest difference ', zh: '已核对：与训练器自己第 1 步之后的权重一致（最大差 ' })}<span class="num">{d.check.toExponential(1)}</span>{t({ en: ').', zh: '）。' })}</li>
								</ul>
							</div>
						{/if}
					</li>
				</ol>
			{/snippet}
		</StepPlayer>
	</div>
{:else if L.presetError}
	<p class="muted">{t({ en: 'Could not compute the features. Please reload the page.', zh: '特征计算失败，请刷新页面。' })}</p>
{:else}
	<p class="muted">{t({ en: 'Computing the frozen features of the training drawings…', zh: '正在计算训练图的冻结特征…' })}</p>
{/if}

<style>
	.ls {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.pick {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 10px;
	}
	.pick button {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.small {
		font-size: var(--fs-sm);
	}
	.dot {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 50%;
	}
	.dot.a {
		background: var(--div-pos);
	}
	.dot.b {
		background: var(--div-neg);
		border-radius: 2px;
	}
	.rows {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 6px;
	}
	.rows > li {
		padding: 6px 10px;
		border-left: 3px solid var(--rule);
		border-radius: 0 var(--r1) var(--r1) 0;
		min-width: 0;
	}
	.rows > li.cur {
		border-left-color: var(--accent);
		background: var(--accent-soft);
	}
	.rows > li.future {
		color: var(--muted);
		border-left-style: dashed;
	}
	.rt {
		margin: 0;
		font-weight: 650;
		font-size: var(--fs-sm);
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.n {
		display: inline-grid;
		place-items: center;
		width: 1.35em;
		height: 1.35em;
		border-radius: 50%;
		background: var(--surface-3);
		font-size: var(--fs-xs);
	}
	.cur .n {
		background: var(--accent);
		color: var(--accent-ink);
	}
	.rows :global(.tex-block) {
		font-size: 0.92em;
		min-width: 0;
	}
	.note {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		max-width: var(--measure);
	}
	.vis {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 14px;
		align-items: flex-start;
		margin-top: 4px;
	}
	.vis .note {
		flex: 1 1 200px;
	}
	.dwbox {
		display: grid;
		gap: 4px;
		flex: 0 1 260px;
		min-width: 0;
	}
	.after {
		flex: 1 1 220px;
		margin: 0;
		padding: 0;
		list-style: none;
		display: grid;
		gap: 3px;
		font-size: var(--fs-sm);
	}
</style>
