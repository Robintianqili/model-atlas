<script lang="ts">
	import Term from '$lib/kit/Term.svelte';
	import Tex from '$lib/kit/Tex.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import ShowMe from './ShowMe.svelte';
	import { sym } from '$lib/kit/tex';
	import type { CnnStore } from '../store.svelte';

	let { S, onshow }: { S: CnnStore; onshow: () => void } = $props();
</script>

<h3>1. 一幅画就是一格格数字</h3>
<p>
	模型从来看不到线条或形状。它得到的是 28 × 28 = 784 个数：没有墨的地方是 0，墨最浓的地方是 1。在此之前，你的笔画会按训练图的方式取景：
	先按外接框裁剪，再缩放到长边 26 像素、居中，并按数据集的笔画粗细重新绘制。所以“模型看到的”那张图很重要：那就是全部输入，别无其他。
</p>

<h3>2. 卷积：一个小小的模式探测器，到处滑动</h3>
<p>
	<Term id="kernel">卷积核</Term>是一块 3 × 3 的、学出来的<Term id="weight">权重</Term>。要算出一个输出格，把卷积核放在输入的一个 3 × 3
	窗口上，九对数两两相乘，再全部加起来，最后加上<Term id="bias" />：
</p>
<div class="math">
	<Tex display tex={`${sym('out', 'y_{ij}')} = \\sum_{u=0}^{2}\\sum_{v=0}^{2} ${sym('param', 'K_{uv}')}\\, ${sym('in', 'x_{i+u,\\,j+v}')} + ${sym('param', 'b')}`} />
</div>
<p>
	然后把窗口挪一个像素，再算一次，每个位置都这样做。结果是一张新的网格，叫<Term id="feature-map" />：窗口里的内容越像卷积核，那里的值就越大。
	TinyCNN 不使用<Term id="padding" />，所以 3 × 3 的卷积核在 28 个像素里只能放 26 个位置：每做一次<Term id="convolution" />，四边各少一个像素（28
	→ 26 → 24）。<ShowMe {S} {onshow} layer="conv1_1" label="看卷积 1-1 滑动" />
</p>

<h3>3. 卷积核、通道和特征图</h3>
<p>
	一个卷积核只找一种模式，所以一层里有好几个：卷积 1-1 有 8 个卷积核，输出 8 张图，每张是一个<Term id="channel" />。在检查器里看看它的卷积核：橙色权重奖励墨迹，蓝色权重惩罚墨迹；一边橙一边蓝的卷积核回答的是“这里有没有一条边？”。
</p>
<p>
	下一层同时读取全部 8 张图。它的卷积核是 3 × 3 × 8：每个输入通道一块 3 × 3 的切片。一个输出格等于 8 个小和（每个通道一个）再加偏置，一共 72
	次乘法。后面的层就是这样组合简单模式的：“这里有一条横边，<em>并且</em>那里有一条斜线”。<ShowMe {S} {onshow} layer="conv2_1" label="看 8 个通道如何相加" />
</p>

<h3>4. ReLU：只保留正面的证据</h3>
<p>
	每次卷积之后是 <Term id="relu" />：<Tex tex={'\\max(0, x)'} />。负数变成 0，正数原样通过。可以把每个格子看作在用它的数值投票：“我看到了我的模式”。ReLU
	丢掉的是“我肯定没看到”的那些票。它在数学上也很关键：连续两次卷积只相当于一次更大的卷积；中间加上这个“折弯”，叠起来的层才能表达多得多的东西。
	<ShowMe {S} {onshow} layer="relu1_1" label="比较之前与之后" />
</p>

<h3>5. 最大池化：缩小，并保留最强的</h3>
<p>
	<Term id="pooling">最大池化</Term>把每张图切成 2 × 2 的小块，每块只保留最大的那个值，宽和高都减半（24 → 12，8 → 4）。它没有权重。它记住的是“这个模式出现在这一块里”，忘掉具体在哪个位置，所以笔画往左挪一个像素，对后面的影响很小。
	<ShowMe {S} {onshow} layer="pool1" label="看 2×2 窗口" />
</p>

<h3>6. 感受野：越深的格子看得越多</h3>
<p>
	卷积 1-1 的一个格子取决于你画中 3 × 3 的一小块。卷积 1-2 的格子组合了这些格子的 3 × 3 窗口，所以取决于 5 × 5 个像素。池化让相邻格子之间的步长加倍，因此之后每个 3 × 3
	的卷积核都覆盖更大的区域。<Term id="receptive-field" />就这样逐层变大：
</p>
<table class="rf" use:keyScroll>
	<thead><tr><th>层</th><th>卷积 1-1</th><th>卷积 1-2</th><th>最大池化 1</th><th>卷积 2-1</th><th>卷积 2-2</th><th>最大池化 2</th></tr></thead>
	<tbody><tr><th>能看到</th><td>3×3</td><td>5×5</td><td>6×6</td><td>10×10</td><td>14×14</td><td>16×16</td></tr></tbody>
</table>
<p>所以早期的特征图对笔画的片段有反应，最后几层则对物体的部件有反应：车轮、屋顶、像花瓣一样的光芒。可以在检查器的“这个通道对什么有反应”里验证：最能激活卷积 2-2 通道 2 的测试画片段大多是自行车轮子，通道 6 是房顶和墙角，通道 14 是太阳的光芒。放大视图会框出每个格子能看到的区域。</p>

<h3>7. 展平、全连接与 softmax：从特征图到决定</h3>
<p>
	第二次池化之后剩下 16 张 4 × 4 的图。<Term id="flatten" />把这 256 个数排成一行（逐通道、逐行）。接着<Term id="dense-layer">全连接层</Term>给每个类别各配 256
	个权重和一个偏置，为每个类别算出一个分数，叫 <Term id="logit" />：
</p>
<div class="math">
	<Tex display tex={`${sym('out', 'z_k')} = \\sum_{i=0}^{255} ${sym('param', 'W_{k,i}')}\\,${sym('in', 'x_i')} + ${sym('param', 'b_k')}`} />
</div>
<p>正的乘积把类别 <Tex tex="k" /> 往上推，负的往下压；放大视图列出了两边最强的几项。最后 <Term id="softmax" /> 把 10 个分数变成加起来等于 1 的概率：</p>
<div class="math">
	<Tex display tex={`${sym('prob', 'p_k')} = \\frac{e^{${sym('out', 'z_k')}}}{\\sum_{j} e^{${sym('out', 'z_j')}}}`} />
</div>
<p>
	只有 logit 之间的差值才重要，而且概率永远是在这 10 个已知类别之间分配 100%。没有“都不是”这个选项，所以一幅梯子可能被判成 99.8% 的房子。
	<ShowMe {S} {onshow} layer="dense" label="看什么把一个类别推高" />
</p>

<h3>8. 为什么参数这么少？</h3>
<p>
	整个模型只有 6,722 个<Term id="parameter">参数</Term>（26.9 KB）。有两个想法让它保持小巧：每个卷积核只看一个小窗口（局部连接），而且同一个卷积核在每个位置重复使用（权重共享）。卷积
	1-1 有 8 × 9 个权重加 8 个偏置，共 80 个参数；如果换成把全部 784 个像素连到它的 5,408 个输出的全连接层，就需要 400 多万个参数。最大的一层是最后的全连接层（2,570
	个），因为只有在那里，每个输入都连到每个输出。
</p>

<h3>9. 这个模型做不到什么</h3>
<ul>
	<li>它只认识 10 个类别。其他任何东西都会被硬塞进其中一类（试试梯子）。</li>
	<li>它只看到取景后的 28 × 28 笔画：颜色、你屏幕上的线条粗细和画的大小，在它开始计算之前就已经没有了。</li>
	<li>它是从大约 12 万幅随手涂鸦里学的。一幅认真、写实的画可能反而不像训练数据，结果更差。</li>
	<li>96.81% 的测试准确率是整个测试集的平均值，并不是对你这幅画的保证。</li>
</ul>

<style>
	.math {
		overflow-x: auto;
	}
	.rf {
		border-collapse: collapse;
		font-size: var(--fs-sm);
		margin: 0 0 var(--s4);
		display: block;
		overflow-x: auto;
	}
	.rf th,
	.rf td {
		padding: 3px 10px;
		border-bottom: 1px solid var(--rule);
		text-align: center;
		white-space: nowrap;
	}
	.rf td {
		font-family: var(--font-mono);
	}
	.rf thead th {
		font-weight: 600;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.rf tbody th {
		color: var(--muted);
		font-weight: 500;
	}
</style>
