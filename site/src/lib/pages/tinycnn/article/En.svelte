<script lang="ts">
	import Term from '$lib/kit/Term.svelte';
	import Tex from '$lib/kit/Tex.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import ShowMe from './ShowMe.svelte';
	import { sym } from '$lib/kit/tex';
	import type { CnnStore } from '../store.svelte';

	let { S }: { S: CnnStore } = $props();
</script>

<h3>1. A drawing is a grid of numbers</h3>
<p>
	The model never sees lines or shapes. It gets 28 × 28 = 784 numbers: 0 where there is no ink, 1 where the ink is
	strongest. Before that, your strokes are framed the way the training drawings were: cropped to their bounding box,
	scaled so the long side spans 26 pixels, centered, and redrawn at the dataset's stroke width. That is why the
	"What the model sees" view matters: it is the whole input, nothing else.
</p>

<h3>2. Convolution: one small pattern detector, slid everywhere</h3>
<p>
	A <Term id="kernel">kernel</Term> is a 3 × 3 grid of learned <Term id="weight">weights</Term>. To compute one output cell, lay the
	kernel on a 3 × 3 window of the input, multiply the nine pairs, add them up, and add a <Term id="bias" />:
</p>
<div class="math">
	<Tex display tex={`${sym('out', 'y_{ij}')} = \\sum_{u=0}^{2}\\sum_{v=0}^{2} ${sym('param', 'K_{uv}')}\\, ${sym('in', 'x_{i+u,\\,j+v}')} + ${sym('param', 'b')}`} />
</div>
<p>
	Then slide the window one pixel and do it again, at every position. The result is a new grid, a
	<Term id="feature-map" />, which is large wherever the window looked like the kernel. TinyCNN uses no
	<Term id="padding" />, so a 3 × 3 kernel fits 26 times across 28 pixels: each <Term id="convolution" /> trims one pixel
	from every border (28 → 26 → 24). <ShowMe {S} layer="conv1_1" label="Watch Conv 1-1 slide" />
</p>

<h3>3. Kernels, channels and feature maps</h3>
<p>
	One kernel finds one pattern, so a layer has several: Conv 1-1 has 8 kernels and outputs 8 maps, one
	<Term id="channel" /> each. Look at its kernels in the inspector: orange weights reward ink, blue weights punish it, and
	a kernel that is orange on one side and blue on the other answers "is there an edge here?".
</p>
<p>
	The next layer reads all 8 maps at once. Its kernels are 3 × 3 × 8: one 3 × 3 slice per input channel. One output cell
	is the sum of 8 small sums (one per channel) plus a bias, 72 products in total. That is how later layers combine
	simple patterns: "a horizontal edge here <em>and</em> a diagonal there". <ShowMe {S} layer="conv2_1" label="See 8 channels add up" />
</p>

<h3>4. ReLU: keep the positive evidence</h3>
<p>
	After each convolution comes <Term id="relu" />: <Tex tex={'\\max(0, x)'} />. Negative values become 0; positive ones
	pass unchanged. Think of a cell as voting "I saw my pattern" with its value: ReLU throws away the "I definitely did not"
	votes. It also matters mathematically. Two convolutions in a row are just one bigger convolution; with a bend between
	them, the stack can express much more. <ShowMe {S} layer="relu1_1" label="Compare before and after" />
</p>

<h3>5. Max pooling: shrink and keep the strongest</h3>
<p>
	<Term id="pooling">Max pooling</Term> cuts each map into 2 × 2 blocks and keeps only the largest value of each block,
	halving width and height (24 → 12, 8 → 4). It has no weights. It keeps "the pattern was somewhere in this block" and
	forgets exactly where, so a stroke drawn one pixel further left changes little downstream.
	<ShowMe {S} layer="pool1" label="Watch the 2×2 window" />
</p>

<h3>6. Receptive field: deeper cells see more</h3>
<p>
	A cell of Conv 1-1 depends on a 3 × 3 patch of your drawing. A cell of Conv 1-2 combines a 3 × 3 window of those, so it
	depends on 5 × 5 pixels. Pooling doubles the step between neighboring cells, so after it every 3 × 3 kernel covers a
	larger region. The <Term id="receptive-field" /> grows layer by layer:
</p>
<table class="rf" use:keyScroll>
	<thead><tr><th>layer</th><th>Conv 1-1</th><th>Conv 1-2</th><th>Max pool 1</th><th>Conv 2-1</th><th>Conv 2-2</th><th>Max pool 2</th></tr></thead>
	<tbody><tr><th>sees</th><td>3×3</td><td>5×5</td><td>6×6</td><td>10×10</td><td>14×14</td><td>16×16</td></tr></tbody>
</table>
<p>
	So early maps respond to bits of stroke, while the last maps respond to parts of objects: a wheel, a roof, a petal-like
	ray. You can check this in the inspector's "What this channel responds to": the test patches that excite Conv 2-2
	channel 2 most are mostly bicycle wheels, channel 6 house roofs and corners, channel 14 sun rays. The zoom view outlines
	the patch each cell can see.
</p>

<h3>7. Flatten, dense and softmax: from maps to a decision</h3>
<p>
	After the second pooling there are 16 maps of 4 × 4. <Term id="flatten">Flatten</Term> lays these 256 numbers in one
	line (channel by channel, row by row). The <Term id="dense-layer">dense layer</Term> then gives every class its own 256 weights and a
	bias, and computes one score per class, a <Term id="logit" />:
</p>
<div class="math">
	<Tex display tex={`${sym('out', 'z_k')} = \\sum_{i=0}^{255} ${sym('param', 'W_{k,i}')}\\,${sym('in', 'x_i')} + ${sym('param', 'b_k')}`} />
</div>
<p>
	Positive products push class <Tex tex="k" /> up, negative ones push it down; the zoom view lists the strongest of both.
	Finally <Term id="softmax" /> turns the 10 scores into probabilities that add up to 1:
</p>
<div class="math">
	<Tex display tex={`${sym('prob', 'p_k')} = \\frac{e^{${sym('out', 'z_k')}}}{\\sum_{j} e^{${sym('out', 'z_j')}}}`} />
</div>
<p>
	Only differences between logits matter, and the probabilities always share 100% among the 10 known classes. There is no
	"none of these" option, which is why a ladder can come out as 99.8% house.
	<ShowMe {S} layer="dense" label="See what pushes a class up" />
</p>

<h3>8. Why so few parameters?</h3>
<p>
	The whole model has 6,722 <Term id="parameter">parameters</Term> (26.9 KB). Two ideas keep it small. Each kernel looks
	only at a small window (local connections), and the same kernel is reused at every position (weight sharing). Conv 1-1
	has 8 × 9 weights + 8 biases = 80 parameters; a dense layer from all 784 pixels to its 5,408 outputs would need over
	4 million. The largest single layer is the final dense one (2,570), because it is the only place where every input
	connects to every output.
</p>

<h3>9. What this model cannot do</h3>
<ul>
	<li>It knows exactly 10 classes. Anything else is forced into one of them (try the ladder).</li>
	<li>It only sees 28 × 28 framed strokes: color, line thickness on your screen and the drawing's size are gone before it starts.</li>
	<li>
		It learned from about 120,000 quick doodles. A careful, realistic drawing can look less like the training data than a
		hasty one, and do worse.
	</li>
	<li>Its 96.81% test accuracy is an average over the test set, not a promise about your drawing.</li>
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
