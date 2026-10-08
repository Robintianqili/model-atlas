<!--
  /dev/kit/ — the component gallery. Every kit component rendered with clearly labeled
  ILLUSTRATIVE data (a hand-designed filter bank + a random dense layer; see demo.ts).
-->
<script lang="ts">
	import { base } from '$app/paths';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import TensorStrip from '$lib/kit/TensorStrip.svelte';
	import TileStack from '$lib/kit/TileStack.svelte';
	import TensorView from '$lib/kit/TensorView.svelte';
	import TensorStats from '$lib/kit/TensorStats.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import AttentionMatrix from '$lib/kit/AttentionMatrix.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import Tex from '$lib/kit/Tex.svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import Walkthrough from '$lib/kit/Walkthrough.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import StatusBadge from '$lib/kit/StatusBadge.svelte';
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import Inspector from '$lib/kit/Inspector.svelte';
	import ModelOverview from '$lib/kit/ModelOverview.svelte';
	import Card from '$lib/kit/Card.svelte';
	import ModelCard from '$lib/kit/ModelCard.svelte';
	import ConceptCard from '$lib/kit/ConceptCard.svelte';
	import Term from '$lib/kit/Term.svelte';
	import LangToggle from '$lib/kit/LangToggle.svelte';
	import ThemeToggle from '$lib/kit/ThemeToggle.svelte';
	import RoleChip from '$lib/kit/RoleChip.svelte';
	import RoleFrame from '$lib/kit/RoleFrame.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import ScopeSwitch from '$lib/kit/ScopeSwitch.svelte';
	import Num from '$lib/kit/Num.svelte';
	import Tooltip from '$lib/kit/Tooltip.svelte';
	import FactStrip from '$lib/kit/FactStrip.svelte';
	import PredictFirst from '$lib/kit/PredictFirst.svelte';
	import PredictReveal from '$lib/kit/PredictReveal.svelte';
	import CodeBlock from '$lib/kit/CodeBlock.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import InputIcon from '$lib/kit/InputIcon.svelte';
	import { sym } from '$lib/kit/tex';
	import { CATEGORICAL, domainOf, type ColorScope } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { rasterize, toInk, polylines, type Stroke } from '$lib/kit/draw';
	import { softmax, topK } from '$lib/kit/stats';
	import { MODEL_BY_SLUG } from '$lib/catalog/models';
	import { CONCEPT_BY_SLUG } from '$lib/catalog/concepts';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { CellValue, PredictQuestion, TokenItem, WalkStep } from '$lib/kit/types';
	import {
		DEMO_KERNELS,
		DEMO_BIAS,
		DEMO_CLASSES,
		DEMO_LAYERS,
		defaultInput,
		downsample,
		demoForward,
		demoAttention,
		demoVector
	} from '$lib/demo/cnn-demo';

	// ------------------------------------------------ input -> illustrative forward pass
	let strokes = $state<Stroke[]>([]);
	let input12 = $state<Float32Array>(defaultInput());
	function onDraw(s: Stroke[]) {
		input12 = s.length ? downsample(toInk(rasterize(s, 120)), 120, 12) : defaultInput();
	}
	const trace = $derived(demoForward(input12));
	const convLayerDom = $derived(domainOf('diverging', trace.t.conv.data));

	// ------------------------------------------------ grid hover drives the formula
	let hover = $state<CellValue | null>(null);
	let scope = $state<ColorScope>('layer');
	let channel = $state(0);
	const convCh = $derived(trace.t.conv.data.subarray(channel * 100, (channel + 1) * 100));
	const relCh = $derived(trace.t.relu.data.subarray(channel * 100, (channel + 1) * 100));
	const pos = $derived(hover ?? { row: 3, col: 4, value: convCh[34] });
	const formulaValues = $derived.by(() => {
		const k = DEMO_KERNELS[channel].k;
		let sum = 0;
		for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) sum += k[u][v] * input12[(pos.row + u) * 12 + pos.col + v];
		return { sum, b: DEMO_BIAS[channel], y: sum + DEMO_BIAS[channel], r: Math.max(0, sum + DEMO_BIAS[channel]) };
	});
	const convTex = `\\begin{aligned} ${sym('out', 'y_{ij}')} &= \\sum_{u,v=0}^{2} ${sym('param', 'K_{uv}')}\\, ${sym('in', 'x_{i+u,\\,j+v}')} + ${sym('param', 'b')} \\\\ &= {{sum}} + ({{b}}) = {{y}} \\end{aligned}`;
	const reluTex = `${sym('act', 'a_{ij}')} = \\max(0, ${sym('out', 'y_{ij}')}) = {{r}}`;

	// ------------------------------------------------ step player: the kernel sliding
	let step = $state(0);
	const stepLabels = Array.from({ length: 100 }, (_, i) => ({
		en: `Kernel at row ${Math.floor(i / 10)}, column ${i % 10}`,
		zh: `卷积核位于第 ${Math.floor(i / 10)} 行，第 ${i % 10} 列`
	}));

	// ------------------------------------------------ walkthrough: a 4-step guided lesson over the demo network
	let walkIndex = $state(0);
	let walkSlide = $state(0);
	const walkSteps: WalkStep[] = [
		{
			id: 'input',
			title: { en: 'The input is a grid of numbers', zh: '输入是一格一格的数' },
			narration: {
				en: 'The demo network sees 12 × 12 numbers between 0 and 1. Draw in the DrawPad demo further down and this grid changes; every later step follows.',
				zh: '演示网络看到的是 12 × 12 个介于 0 和 1 之间的数。在下方 DrawPad 演示里作画，这个网格就会变化，后面的每一步也跟着变。'
			},
			look: { en: 'The darker a cell, the more ink.', zh: '格子越深，墨迹越多。' }
		},
		{
			id: 'slide',
			title: { en: 'A kernel slides over it', zh: '卷积核在上面滑动' },
			narration: {
				en: 'This step autoplays when it is shown and tells the lesson when it is done (waitForDone), so Play moves on after the slide.',
				zh: '这一步显示时会自动播放，播放完会通知课程（waitForDone），所以“自动播放”会在滑动结束后才进入下一步。'
			},
			waitForDone: true,
			linger: 1500,
			onEnter: () => (walkSlide = 0)
		},
		{
			id: 'relu',
			title: { en: 'ReLU drops the negatives', zh: 'ReLU 去掉负数' },
			narration: { en: 'Before and after max(0, x), on the same color scale.', zh: 'max(0, x) 之前和之后，用同一个色阶。' }
		},
		{
			id: 'scores',
			title: { en: 'Scores for each class', zh: '每个类别的分数' },
			narration: { en: 'A random dense layer and softmax: illustrative numbers, not a trained model.', zh: '随机全连接层加 softmax：示意数字，不是训练好的模型。' }
		}
	];

	// ------------------------------------------------ text -> tokens (characters; real code points)
	let text = $state('Hello, 世界!');
	const tokens = $derived<TokenItem[]>([
		{ text: '<bos>', id: 0, special: true },
		...[...text].map((ch) => ({ text: ch, id: ch.codePointAt(0) ?? 0, bytes: [...new TextEncoder().encode(ch)] }))
	]);
	let selTok = $state<number | null>(1);
	const attTokens = $derived(tokens.slice(0, 12).map((x) => x.text));
	const att = $derived(demoAttention(attTokens.length));
	let attRow = $state<number | null>(null);

	// next-character distribution (illustrative): softmax of fixed random logits
	const charLogits = [2.1, 1.7, 1.2, 0.9, 0.6, 0.4, 0.1, -0.2, -0.6, -1.0, -1.5, -2.0];
	const charLabels = ['e', ' ', 'a', 'o', 't', 'n', '\n', 's', '的', 'i', ',', 'r'];
	let temperature = $state(1);
	const charProbs = $derived(softmax(charLogits, temperature));
	const topKcut = 3;
	const topPcut = $derived.by(() => {
		const idx = topK(charProbs, charProbs.length);
		let c = 0;
		for (let i = 0; i < idx.length; i++) {
			c += charProbs[idx[i]];
			if (c >= 0.9) return i + 1;
		}
		return idx.length;
	});

	// ------------------------------------------------ inspector + overview
	let current = $state('conv');

	// ------------------------------------------------ misc
	const strip = demoVector();
	const presets: { name: { en: string; zh: string }; strokes: Stroke[] }[] = [
		{ name: { en: 'House', zh: '房子' }, strokes: polylines([[20, 50, 50, 20, 80, 50], [28, 45, 28, 82, 72, 82, 72, 45], [44, 82, 44, 62, 56, 62, 56, 82]], 0.07) },
		{ name: { en: 'Cup', zh: '杯子' }, strokes: polylines([[25, 25, 30, 80, 70, 80, 75, 25], [75, 38, 88, 42, 88, 58, 72, 62]], 0.07) },
		{ name: { en: 'Seven', zh: '数字 7' }, strokes: polylines([[22, 20, 78, 20, 40, 85]], 0.09) }
	];
	const question: PredictQuestion = {
		id: 'dev-kit-demo',
		question: { en: 'After ReLU, what happens to a negative value?', zh: '经过 ReLU 后，负数会变成什么？' },
		options: [
			{ en: 'It becomes positive', zh: '变成正数' },
			{ en: 'It becomes zero', zh: '变成 0' },
			{ en: 'It stays the same', zh: '保持不变' }
		],
		answer: 1,
		explain: { en: 'ReLU is max(0, x): negative inputs become exactly 0.', zh: 'ReLU 就是 max(0, x)：负数输入恰好变成 0。' }
	};

	const sections = [
		['grammar', 'Visual grammar'],
		['num', 'Num · Tooltip'],
		['badges', 'ProvenanceBadge · StatusBadge · RoleChip · RoleFrame'],
		['tensorgrid', 'TensorGrid · ScopeSwitch · ColorLegend'],
		['tensorstrip', 'TensorStrip'],
		['tilestack', 'TileStack · TensorView · TensorStats'],
		['formula', 'FormulaCard · Tex'],
		['player', 'StepPlayer'],
		['walkthrough', 'Walkthrough'],
		['drawpad', 'DrawPad'],
		['text', 'TextInputBar · TokenChips'],
		['attention', 'AttentionMatrix'],
		['probbars', 'ProbBars'],
		['overview', 'ModelOverview · Inspector'],
		['cards', 'Card · ModelCard · ConceptCard · InputIcon'],
		['text-tools', 'Term · Lang · CodeBlock · FactStrip'],
		['predict', 'PredictFirst · PredictReveal'],
		['chrome', 'LangToggle · ThemeToggle · shells']
	];
</script>

<svelte:head><title>{ui('navKit')} · Model Atlas</title></svelte:head>

<div class="container kit">
	<header class="head">
		<p class="eyebrow">/dev/kit/</p>
		<h1>{t({ en: 'Component kit', zh: '组件库' })}</h1>
		<p class="lede">
			{t({
				en: 'Every shared component, rendered with demo data. The numbers come from a hand-designed filter bank (edge detectors and a blur) and a random dense layer, not from a trained model, so every panel is badged Illustrative. Draw in the pad below and all views update.',
				zh: '所有共享组件，用演示数据渲染。数字来自手工设计的滤波器组（边缘检测器和模糊核）以及一个随机全连接层，并非训练好的模型，因此每个面板都标为“示意”。在下面的画板上作画，所有视图都会更新。'
			})}
		</p>
		<p class="muted small">
			{t({ en: 'API docs: site/KIT.md. Theme and language toggles are in the header.', zh: 'API 文档：site/KIT.md。主题和语言切换在页头。' })}
		</p>
		<nav class="toc" aria-label="Components">
			{#each sections as [id, name] (id)}<a href="#{id}">{name}</a>{/each}
		</nav>
	</header>

	<!-- ============================================================ grammar -->
	<section id="grammar" class="demo">
		<h2>Visual grammar</h2>
		<p class="desc">{t({ en: 'One diverging scale for signed values, one purple ramp for probabilities and attention, ten fixed class colors.', zh: '有正负的数值用一个发散色阶，概率和注意力用一个紫色渐变，类别用十种固定颜色。' })}</p>
		<div class="row">
			<ColorLegend kind="diverging" domain={[-1, 1]} label={{ en: 'diverging (activations, weights, gradients, diffs)', zh: '发散色阶（激活、权重、梯度、差值）' }} width={280} />
			<ColorLegend kind="sequential" domain={[0, 1]} label={{ en: 'sequential (probabilities, attention)', zh: '顺序色阶（概率、注意力）' }} width={280} />
		</div>
		<div class="swatches" aria-label="categorical palette">
			{#each CATEGORICAL[themeState.resolved] as c, i (i)}
				<span class="sw"><span class="chip-c" style="background: {c}"></span><span class="num">{i} {DEMO_CLASSES[i]}</span></span>
			{/each}
		</div>
		<div class="row">
			<span class="pair"><span class="dot" style="background: var(--real)"></span>{t({ en: 'real data', zh: '真实数据' })}</span>
			<span class="pair"><span class="dot" style="background: var(--generated)"></span>{t({ en: 'generated samples', zh: '生成样本' })}</span>
			<span class="pair"><span class="frame" style="border-color: var(--q)"></span>Q</span>
			<span class="pair"><span class="frame" style="border-color: var(--k)"></span>K</span>
			<span class="pair"><span class="frame" style="border-color: var(--v)"></span>V</span>
		</div>
	</section>

	<!-- ============================================================ num -->
	<section id="num" class="demo">
		<h2>Num · Tooltip</h2>
		<p class="desc">{t({ en: 'Numbers: monospace, 3 significant digits, full precision on hover, focus or tap.', zh: '数字：等宽字体，3 位有效数字；悬停、聚焦或点按时显示完整精度。' })}</p>
		<p class="row">
			<Num value={0.123456789} /> <Num value={-12.3456} /> <Num value={0.000123456} /> <Num value={123456.7} />
			<Num value={-0.5} signed /> <Num value={0.731} signed />
			<Tooltip tip="A plain tooltip: hover, focus, or tap.">{t({ en: 'hover me', zh: '悬停这里' })}</Tooltip>
		</p>
	</section>

	<!-- ============================================================ badges -->
	<section id="badges" class="demo">
		<h2>ProvenanceBadge · StatusBadge · RoleChip · RoleFrame</h2>
		<div class="row">
			<ProvenanceBadge mode="live" precision="fp32" tolerance={3.2e-6} against="PyTorch fp32" />
			<ProvenanceBadge mode="precomputed" precision="fp16" />
			<ProvenanceBadge mode="precomputed" precision="q8" approximate />
			<ProvenanceBadge mode="illustrative" />
		</div>
		<div class="row">
			<StatusBadge status="live" />
			<StatusBadge status="precomputed" tier="P1" />
			<StatusBadge status="planned" tier="P2" tooltip />
		</div>
		<div class="row">
			<RoleChip role="activation" />
			<RoleChip role="param" />
			<RoleChip role="grad" />
			<RoleChip role="cache" />
			<RoleChip role="optimizer" />
			<RoleChip role="frozen" />
			<RoleChip role="trainable" />
			<RoleChip role="lora" rank={8} />
		</div>
		<div class="row top">
			{#each DEMO_KERNELS.slice(0, 2) as k, i (i)}
				<RoleFrame role="param" status={i === 0 ? 'frozen' : 'trainable'} label="conv.weight[{i}] [3×3]">
					<TensorGrid data={k.k.flat()} rows={3} cols={3} cell={34} domain={[-1, 1]} label="kernel {i}" readout={false} />
				</RoleFrame>
			{/each}
			<RoleFrame role="grad" label="∂L/∂K (illustrative)">
				<TensorGrid data={[0.1, -0.2, 0.05, 0.3, -0.4, 0.2, -0.1, 0.0, 0.15]} rows={3} cols={3} cell={34} label="gradient" readout={false} />
			</RoleFrame>
		</div>
	</section>

	<!-- ============================================================ tensorgrid -->
	<section id="tensorgrid" class="demo">
		<h2>TensorGrid · ScopeSwitch · ColorLegend <ProvenanceBadge mode="illustrative" compact /></h2>
		<p class="desc">{t({ en: 'Hover or tap a cell (or Tab to the grid and use the arrow keys). The formula below follows the cell. The highlight is the 3×3 input window.', zh: '悬停或点按单元格（或用 Tab 聚焦后用方向键）。下方公式随单元格变化。高亮框是 3×3 的输入窗口。' })}</p>
		<div class="row top">
			<figure>
				<figcaption>input [12×12]</figcaption>
				<TensorGrid data={input12} rows={12} cols={12} kind="sequential" domain={[0, 1]} cell={18} label="input" highlights={[{ row: pos.row, col: pos.col, h: 3, w: 3 }]} readout={false} />
			</figure>
			<figure>
				<figcaption>conv · {DEMO_KERNELS[channel].name} [10×10]</figcaption>
				<TensorGrid data={convCh} rows={10} cols={10} cell={21} {scope} scopeDomains={{ layer: convLayerDom }} label="conv output channel {channel}" onhover={(c) => (hover = c)} />
			</figure>
			<figure>
				<figcaption>{t({ en: 'kernel (large cells show values)', zh: '卷积核（大单元格显示数值）' })}</figcaption>
				<TensorGrid data={DEMO_KERNELS[channel].k.flat()} rows={3} cols={3} cell={40} domain={[-1, 1]} label="kernel" readout={false} />
			</figure>
		</div>
		<div class="row">
			<ScopeSwitch bind:value={scope} />
			<span class="segmented" role="group" aria-label="channel">
				{#each DEMO_KERNELS as k, i (i)}<button type="button" aria-pressed={channel === i} onclick={() => (channel = i)}>{i}</button>{/each}
			</span>
			<ColorLegend kind="diverging" domain={scope === 'unit' ? domainOf('diverging', convCh) : convLayerDom} />
		</div>
		<pre class="usage"><code>{`<TensorGrid data={fmap} rows={10} cols={10} scope="layer" scopeDomains={{ layer }} onhover={(c) => (cell = c)} />`}</code></pre>
	</section>

	<!-- ============================================================ tensorstrip -->
	<section id="tensorstrip" class="demo">
		<h2>TensorStrip <ProvenanceBadge mode="illustrative" compact /></h2>
		<p class="desc">{t({ en: 'A 768-long vector, elided in the middle; the exact shape label is always visible.', zh: '一个长度 768 的向量，中间省略；精确形状标签始终可见。' })}</p>
		<TensorStrip data={strip} name="embedding" shape={['T', 768]} maxCells={28} />
		<TensorStrip data={trace.t.probs.data} kind="sequential" domain={[0, 1]} name="probs" cellSize={36} values />
	</section>

	<!-- ============================================================ tilestack -->
	<section id="tilestack" class="demo">
		<h2>TileStack · TensorView · TensorStats <ProvenanceBadge mode="illustrative" compact /></h2>
		<p class="desc">{t({ en: 'A [4×10×10] activation as small multiples (arrow keys change the channel). TensorView picks the view from the shape.', zh: '把 [4×10×10] 的激活值画成小图阵列（方向键切换通道）。TensorView 根据形状自动选择视图。' })}</p>
		<TileStack data={trace.t.relu.data} channels={4} height={10} width={10} bind:selected={channel} label="relu" channelLabels={DEMO_KERNELS.map((k) => k.name)} />
		<div class="cols2">
			<TensorView tensor={trace.t.pool} label="pool" />
			<TensorStats tensor={trace.t.relu} />
		</div>
	</section>

	<!-- ============================================================ formula -->
	<section id="formula" class="demo">
		<h2>FormulaCard · Tex <ProvenanceBadge mode="illustrative" compact /></h2>
		<div class="cols2">
			<FormulaCard
				tex={convTex}
				values={formulaValues}
				caption={{ en: `Conv at row ${pos.row}, column ${pos.col}`, zh: `第 ${pos.row} 行第 ${pos.col} 列的卷积` }}
				where={[
					{ tex: sym('param', 'K'), text: { en: '3×3 kernel (parameter)', zh: '3×3 卷积核（参数）' } },
					{ tex: sym('in', 'x'), text: { en: 'input window', zh: '输入窗口' } },
					{ tex: sym('param', 'b'), text: { en: 'bias', zh: '偏置' } }
				]}
				provenance="illustrative"
			/>
			<FormulaCard tex={reluTex} values={formulaValues} caption="ReLU" />
		</div>
		<p>
			{t({ en: 'Inline math:', zh: '行内公式：' })}
			<Tex tex={`\\mathrm{softmax}(${sym('out', 'z')})_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}`} />,
			<Tex tex={`${sym('q', 'Q')}${sym('k', 'K')}^\\top / \\sqrt{d}`} /> · <Tex tex={`${sym('v', 'V')}`} />
		</p>
	</section>

	<!-- ============================================================ player -->
	<section id="player" class="demo">
		<h2>StepPlayer <ProvenanceBadge mode="illustrative" compact /></h2>
		<p class="desc">{t({ en: 'The kernel slides over the input; output cells appear as they are computed. Space plays or pauses, arrow keys step. With reduced motion, Play becomes Next.', zh: '卷积核在输入上滑动，输出单元格随计算逐个出现。空格播放/暂停，方向键单步。开启减少动态效果时，“播放”变为“下一步”。' })}</p>
		<StepPlayer count={100} bind:step labels={stepLabels} interval={260} label={{ en: 'Convolution, one position at a time', zh: '卷积：一次一个位置' }}>
			{#snippet children(s)}
				<div class="row top">
					<TensorGrid data={input12} rows={12} cols={12} kind="sequential" domain={[0, 1]} cell={16} label="input" highlights={[{ row: Math.floor(s / 10), col: s % 10, h: 3, w: 3 }]} readout={false} />
					<TensorGrid data={convCh} rows={10} cols={10} cell={19} scopeDomains={{ layer: convLayerDom }} scope="layer" label="output so far" masked={(r, c) => r * 10 + c > s} highlights={[{ row: Math.floor(s / 10), col: s % 10 }]} readout={false} />
				</div>
			{/snippet}
		</StepPlayer>
	</section>

	<!-- ============================================================ walkthrough -->
	<section id="walkthrough" class="demo">
		<h2>Walkthrough <ProvenanceBadge mode="illustrative" compact /></h2>
		<p class="desc">
			{t({
				en: 'A guided lesson: one focused view per step, narration, Back / Next / Play, a step list, keyboard ← → Home End, deep links (#demo-relu) and a polite live region. Wide: stage left, narration right; narrow: stage on top, progress dots.',
				zh: '分步课程：每步一个聚焦的视图、讲解、上一步 / 下一步 / 自动播放、步骤列表、键盘 ← → Home End、深链接（#demo-relu）和礼貌的实时播报区。宽屏：左边舞台、右边讲解；窄屏：舞台在上、进度圆点。'
			})}
		</p>
		<Walkthrough steps={walkSteps} bind:index={walkIndex} id="demo" syncHash={false} label={{ en: 'Demo lesson', zh: '演示课程' }} stageMinHeight={240}>
			{#snippet stage(step, ctx)}
				{#if step.id === 'input'}
					<TensorGrid data={input12} rows={12} cols={12} kind="sequential" domain={[0, 1]} maxCell={22} label="input" />
				{:else if step.id === 'slide'}
					<StepPlayer count={100} bind:step={walkSlide} interval={70} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} label={{ en: 'Kernel 0 sliding', zh: '卷积核 0 滑动' }}>
						{#snippet children(st)}
							<div class="row top">
								<TensorGrid data={input12} rows={12} cols={12} kind="sequential" domain={[0, 1]} cell={14} label="input" highlights={[{ row: Math.floor(st / 10), col: st % 10, h: 3, w: 3 }]} readout={false} />
								<TensorGrid data={trace.t.conv.data.subarray(0, 100)} rows={10} cols={10} cell={16} scopeDomains={{ layer: convLayerDom }} scope="layer" label="conv 0" masked={(r, c) => r * 10 + c > st} highlights={[{ row: Math.floor(st / 10), col: st % 10 }]} readout={false} />
							</div>
						{/snippet}
					</StepPlayer>
				{:else if step.id === 'relu'}
					<div class="row top">
						<TensorGrid data={trace.t.conv.data.subarray(0, 100)} rows={10} cols={10} cell={16} scopeDomains={{ layer: convLayerDom }} scope="layer" label="conv 0" readout={false} />
						<TensorGrid data={trace.t.relu.data.subarray(0, 100)} rows={10} cols={10} cell={16} scopeDomains={{ layer: convLayerDom }} scope="layer" label="relu 0" readout={false} />
					</div>
				{:else}
					<ProbBars items={DEMO_CLASSES.map((c, i) => ({ label: c, p: trace.t.probs.data[i], id: i, color: CATEGORICAL[themeState.resolved][i] }))} k={5} chosen={topK(trace.t.probs.data, 1)[0]} />
				{/if}
			{/snippet}
		</Walkthrough>
	</section>

	<!-- ============================================================ drawpad -->
	<section id="drawpad" class="demo">
		<h2>DrawPad</h2>
		<p class="desc">{t({ en: 'Mouse, touch and pen. The 12×12 grid is what the demo network receives (area-averaged ink).', zh: '支持鼠标、触摸和触控笔。右侧 12×12 网格就是演示网络收到的输入（按面积平均的墨迹）。' })}</p>
		<div class="row top">
			<DrawPad bind:strokes onchange={onDraw} ondraw={onDraw} {presets} size={260} />
			<figure>
				<figcaption>{t({ en: 'what the network sees [1×12×12] (a preset "7" until you draw)', zh: '网络看到的输入 [1×12×12]（作画前为预设的“7”）' })}</figcaption>
				<TensorGrid data={input12} rows={12} cols={12} kind="sequential" domain={[0, 1]} cell={14} label="input" />
			</figure>
		</div>
	</section>

	<!-- ============================================================ text + tokens -->
	<section id="text" class="demo">
		<h2>TextInputBar · TokenChips</h2>
		<p class="desc">{t({ en: 'Character tokens with Unicode code points as ids (real values, not a model). Hover a chip for its bytes; arrow keys move, Enter selects.', zh: '字符级词元，用 Unicode 码位作为 ID（真实值，不是模型输出）。悬停查看字节；方向键移动，回车选择。' })}</p>
		<TextInputBar
			bind:value={text}
			maxLength={40}
			label={{ en: 'Text', zh: '文本' }}
			presets={[{ text: 'Hello, 世界!' }, { text: '我爱自然语言处理', label: { en: 'Chinese preset', zh: '中文预设' } }, { text: 'unbelievable!' }]}
		/>
		<TokenChips {tokens} bind:selected={selTok} showIds />
	</section>

	<!-- ============================================================ attention -->
	<section id="attention" class="demo">
		<h2>AttentionMatrix <ProvenanceBadge mode="illustrative" compact /></h2>
		<p class="desc">{t({ en: 'Random causal attention over the tokens above (first 12). Upper triangle hatched = future tokens a query may not see.', zh: '对上面前 12 个词元的随机因果注意力。阴影上三角 = 查询看不到的未来词元。' })}</p>
		<div class="att">
			<AttentionMatrix weights={att} queries={attTokens} causal bind:row={attRow} label={{ en: 'head 0 (illustrative)', zh: '第 0 个头（示意）' }} />
		</div>
	</section>

	<!-- ============================================================ probbars -->
	<section id="probbars" class="demo">
		<h2>ProbBars <ProvenanceBadge mode="illustrative" compact /></h2>
		<div class="cols2">
			<ProbBars
				label={{ en: 'Classes (random dense layer)', zh: '类别（随机全连接层）' }}
				items={DEMO_CLASSES.map((c, i) => ({ label: c, p: trace.t.probs.data[i], id: i, color: CATEGORICAL[themeState.resolved][i] }))}
				k={5}
				chosen={topK(trace.t.probs.data, 1)[0]}
			/>
			<div>
				<ProbBars
					label={{ en: 'Next character', zh: '下一个字符' }}
					items={charLabels.map((c, i) => ({ label: c, p: charProbs[i], id: i }))}
					k={8}
					tokens
					chosen={0}
					cutoffs={[
						{ afterRank: topKcut, label: `top-k = ${topKcut}` },
						{ afterRank: topPcut, label: `top-p = 0.9 (${topPcut})` }
					]}
				/>
				<label class="temp">{t({ en: 'temperature', zh: '温度' })} <input type="range" min="0.2" max="3" step="0.05" bind:value={temperature} /> <Num value={temperature} plain /></label>
			</div>
		</div>
	</section>

	<!-- ============================================================ overview + inspector -->
	<section id="overview" class="demo">
		<h2>ModelOverview · Inspector <ProvenanceBadge mode="illustrative" compact /></h2>
		<p class="desc">{t({ en: 'The pipeline comes from a layer list (spec.layers). Click a node, or press [ and ] anywhere on the page.', zh: '流水线由层列表（spec.layers）生成。点击节点，或在页面任意位置按 [ 和 ]。' })}</p>
		<ModelOverview layers={DEMO_LAYERS} bind:current>
			{#snippet thumb(l)}
				{@const tt = trace.t[l.id]}
				{#if tt.shape.length === 3}
					<TensorGrid data={tt.data.subarray(0, tt.shape[1] * tt.shape[2])} rows={tt.shape[1]} cols={tt.shape[2]} kind={l.id === 'input' ? 'sequential' : 'diverging'} cell={Math.max(2, Math.floor(40 / tt.shape[2]))} interactive={false} gridLines={false} label={l.id} />
				{:else}
					<TensorGrid data={tt.data} rows={1} cols={tt.shape[0]} kind={l.id === 'probs' ? 'sequential' : 'diverging'} cell={7} interactive={false} gridLines={false} label={l.id} />
				{/if}
			{/snippet}
		</ModelOverview>
		<Inspector
			layers={DEMO_LAYERS}
			bind:current
			tensorOf={(id) => trace.t[id]}
			kindOf={(id) => (id === 'input' || id === 'probs' ? 'sequential' : 'diverging')}
			provenance="illustrative"
		>
			{#snippet math(id)}
				{#if id === 'conv'}
					<FormulaCard tex={convTex} values={formulaValues} />
				{:else if id === 'relu'}
					<FormulaCard tex={reluTex} values={formulaValues} />
				{:else if id === 'probs'}
					<FormulaCard tex={`p_i = \\frac{e^{${sym('out', 'z_i')}}}{\\sum_j e^{z_j}}`} />
				{:else}
					<p class="muted">{t({ en: 'No formula for this layer in the demo.', zh: '演示中此层没有公式。' })}</p>
				{/if}
			{/snippet}
			{#snippet code(id)}
				<CodeBlock
					caption="demo.ts"
					lang="ts"
					code={`const conv = conv3x3(input12, 12, 12);   // [4, 10, 10]\nconst act = relu(conv);                  // max(0, x)\nconst pool = maxpool2(act, 4, 10, 10);   // [4, 5, 5]\nconst logits = dense(pool);              // [10]\nconst probs = softmax(logits);           // [10]`}
					highlight={[{ conv: 1, relu: 2, pool: 3, logits: 4, probs: 5 }[id] ?? 0]}
				/>
			{/snippet}
		</Inspector>
	</section>

	<!-- ============================================================ cards -->
	<section id="cards" class="demo">
		<h2>Card · ModelCard · ConceptCard · InputIcon</h2>
		<div class="cards">
			<ModelCard model={MODEL_BY_SLUG.tinycnn} />
			<ModelCard model={MODEL_BY_SLUG['resnet-18']} />
			<ConceptCard concept={CONCEPT_BY_SLUG.tokenization} showSection />
			<ConceptCard concept={CONCEPT_BY_SLUG.lora} showSection />
			<Card href="{base}/atlas/" title="Generic card" status="precomputed" tier="P1" hook="Any link with a status badge." inputs={['audio', 'text']} size="41 MB" />
		</div>
		<p class="row icons">
			{#each ['drawing', 'image', 'text', 'audio', 'series', 'tabular', 'points2d', 'graph', 'game', 'protein', 'ratings', 'video'] as k (k)}
				<span class="pair"><InputIcon kind={k as 'drawing'} size={16} /> {ui(`in_${k}` as 'in_drawing')}</span>
			{/each}
		</p>
	</section>

	<!-- ============================================================ text tools -->
	<section id="text-tools" class="demo">
		<h2>Term · Lang · CodeBlock · FactStrip</h2>
		<p>
			<Lang>
				{#snippet en()}A <Term id="kernel" /> slides over the input; <Term id="relu" /> keeps the positive part; <Term id="softmax" /> turns <Term id="logit">logits</Term> into probabilities.{/snippet}
				{#snippet zh()}<Term id="kernel" />在输入上滑动；<Term id="relu" />保留正的部分；<Term id="softmax" />把 <Term id="logit">logits</Term> 变成概率。{/snippet}
			</Lang>
		</p>
		<FactStrip
			facts={[
				{ label: ui('area'), value: { en: 'Vision', zh: '视觉' } },
				{ label: ui('params'), value: '6,722', mono: true },
				{ label: ui('download'), value: '27.6 KB', mono: true },
				{ label: ui('precision'), value: 'fp32', mono: true }
			]}
			provenance="illustrative"
		/>
		<CodeBlock caption="pipeline/tinycnn/model.py (example)" lang="python" code={`x = self.conv1(x)    # [N, 8, 26, 26]\nx = torch.relu(x)\nx = self.pool(x)     # [N, 8, 13, 13]`} highlight={[1]} />
	</section>

	<!-- ============================================================ predict -->
	<section id="predict" class="demo">
		<h2>PredictFirst · PredictReveal</h2>
		<div class="cols2">
			<PredictFirst q={question} />
			<PredictReveal q={question} />
		</div>
	</section>

	<!-- ============================================================ chrome -->
	<section id="chrome" class="demo">
		<h2>LangToggle · ThemeToggle · page shells</h2>
		<div class="row"><LangToggle /> <ThemeToggle /></div>
		<p class="desc">
			{t({
				en: 'ModelPageShell and ConceptPageShell are the page templates; see them on any planned page, e.g.',
				zh: 'ModelPageShell 与 ConceptPageShell 是页面模板；可在任意计划中的页面查看，例如'
			})}
			<a href="{base}/m/resnet-18/">/m/resnet-18/</a>, <a href="{base}/train/lora/">/train/lora/</a>.
			{t({ en: 'SiteHeader and SiteFooter frame every page.', zh: 'SiteHeader 与 SiteFooter 用于每个页面。' })}
		</p>
	</section>
</div>

<style>
	.kit {
		padding-top: var(--s6);
	}
	.head {
		display: grid;
		gap: 8px;
		margin-bottom: var(--s5);
	}
	.head h1,
	.head p {
		margin: 0;
	}
	.small {
		font-size: var(--fs-sm);
	}
	.toc {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		font-size: var(--fs-sm);
		font-family: var(--font-mono);
	}
	.demo {
		padding: var(--s5) 0;
		border-top: 1px solid var(--rule);
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 14px;
		scroll-margin-top: calc(var(--header-h) + 12px);
		min-width: 0;
	}
	.demo h2 {
		font-family: var(--font-mono);
		font-size: var(--fs-lg);
		font-weight: 600;
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 10px;
	}
	.desc {
		margin: 0;
		color: var(--ink-2);
		max-width: var(--measure);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 10px 16px;
		margin: 0;
	}
	.row.top {
		align-items: flex-start;
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--muted);
		font-family: var(--font-mono);
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.swatches {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 14px;
	}
	.sw {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--fs-xs);
	}
	.chip-c {
		width: 18px;
		height: 18px;
		border-radius: 4px;
	}
	.pair {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--fs-sm);
	}
	.dot {
		width: 12px;
		height: 12px;
		border-radius: 50%;
	}
	.frame {
		width: 16px;
		height: 16px;
		border: 2.5px solid;
		border-radius: 3px;
	}
	.cols2 {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr));
		gap: var(--s4);
		align-items: start;
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 240px), 1fr));
		gap: 12px;
	}
	.usage {
		font-size: var(--fs-xs);
		margin: 0;
	}
	.att {
		max-width: 560px;
	}
	.temp {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
		margin-top: 6px;
	}
	.icons {
		font-size: var(--fs-sm);
	}
</style>
