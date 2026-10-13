<script lang="ts">
	import { base } from '$app/paths';
	import Term from '$lib/kit/Term.svelte';
	import Tex from '$lib/kit/Tex.svelte';
</script>

<h3 id="a-tokens">1. 词元：把文字变成数字</h3>
<p>
	语言模型读不了字母。<Term id="tokenizer" />先把文本切成一个个<Term id="token" />，并给每个词元一个 id。TinyGPT 使用字节级
	<Term id="bpe" />，<Term id="vocabulary" />里有 1024 个词元：256 个可能的字节、从儿童故事里学到的 767 条合并规则，以及一个特殊词元
	<code>&lt;|endoftext|&gt;</code>。像 <code>␣the</code>、<code>␣girl</code> 这样常见的片段各是一个词元；不常见的词会被拆成几块。
	一个词通常带着它前面的空格：<code>␣cat</code> 是一个词元，而不带空格的 <code>cat</code> 会被拆成 <code>c</code> + <code>at</code>。
</p>
<p>
	因为分词器以字节为单位，任何文本都能编码，哪怕训练中从没出现过。一个汉字在 UTF-8 中占 3 个字节，而分词器没有为汉字学过合并规则，所以
	“我爱自然语言处理”会变成 24 个字节词元。TinyGPT 从没见过这些字节，它在这些字节后面的预测毫无意义。模型一次最多读 64 个词元，这就是它的
	<Term id="context-window" />。合并规则是怎样学出来的，请看<a href="{base}/train/tokenization/">分词</a>一页。
</p>

<h3 id="a-emb">2. 嵌入：每个词元、每个位置各有一个向量</h3>
<p>
	每个词元 id 从一张学到的表 <Tex tex="E" />（1024 × 76）里取出一行。这一行就是该词元的<Term id="embedding" />：76 个数，训练过程把它们调整到让用法相近的词元拥有相近的向量。
	另一张表 <Tex tex="P" />（64 × 76）为每个位置存了一个学到的向量，两者相加：<Tex tex={'x_t = E[\\text{id}_t] + P[t]'} />。需要位置向量，是因为注意力本身并不知道哪个词元在前（<Term id="positional-encoding" />）。
	对一段有 <Tex tex="T" /> 个词元的文本，结果是一个 <Tex tex={'T \\times 76'} /> 的矩阵：每个词元一行。你可以在课程和“探索”里跟踪这些行穿过整个模型。
</p>

<h3 id="a-attn">3. 注意力：查询、键和值</h3>
<p>
	<Term id="attention" />让一个词元从前面的词元那里收集信息。模型用三个学到的矩阵，从每个词元（归一化后的）向量生成三个新向量：<strong>查询</strong>（这个词元在找什么）、
	<strong>键</strong>（它能提供什么）和<strong>值</strong>（被关注时要交出去的内容），见<Term id="query-key-value" />。词元 <Tex tex="i" /> 与词元 <Tex tex="j" /> 的匹配程度是
	<Tex tex="q_i" /> 与 <Tex tex="k_j" /> 的点积，再除以 <Tex tex={'\\sqrt{19}'} />，免得分数随向量长度变大。<Term id="softmax" /> 把每一行分数变成为正、且总和为 1 的权重。
	词元 <Tex tex="i" /> 的输出是值向量的加权和：
</p>
<div class="eq"><Tex display tex={'a_{ij} = \\operatorname{softmax}_j\\!\\left(\\frac{q_i \\cdot k_j}{\\sqrt{19}}\\right), \\qquad o_i = \\sum_{j \\le i} a_{ij}\\, v_j'} /></div>
<p>
	于是，一个“需要”名字的词元可以把大部分权重放在前面的某个名字上，把那个名字的值复制过来。你可以在“手算点积”面板里用实时数字重算一个分数。
</p>

<h3 id="a-heads">4. 多个头同时工作</h3>
<p>
	一种注意力模式是不够的：一个词元可能为了语法需要前一个词，又为了意思需要很久以前出现的名字。所以每个块并排运行 4 个<Term id="attention-head" />。
	Q、K、V 的 76 列被分成 4 组，每组 19 列；每个头在自己那一组上各自计算分数、softmax 和加权和。4 个输出（各 19 个数）重新拼接成 76 个数，再由输出投影混合。
	在 4 × 4 的小图网格里，你可以寻找有明显“习惯”的头，比如总看前一个词元或第一个词元。在这么小的模型里，这些习惯并不清晰。
</p>

<h3 id="a-mask">5. 因果掩码</h3>
<p>
	TinyGPT 的训练目标是在每个位置同时预测下一个词元（<Term id="next-token-prediction" />）。如果第 5 个词元能看到第 6 个，它就看到了答案。
	<Term id="causal-mask" />阻止了这一点：在 softmax 之前，所有指向后面词元的分数都被设为 <Tex tex={'-\\infty'} />，于是它们的权重恰好为 0。
	这就是每张注意力图右上方的阴影三角。也正因为如此，模型可以一次一个词元地写出文本。
</p>

<h3 id="a-mlp">6. MLP：每个词元单独“思考”</h3>
<p>
	注意力之后，每个词元单独经过一个小的两层网络——<Term id="mlp" />：76 个数先扩展到 304 个，经过 GELU（ReLU 的平滑版本，会把负输入压向 0），再投影回 76 个。
	这里词元之间不交换信息。一个好用的理解方式是：注意力在词元之间搬运信息；MLP 加工每个词元当前持有的信息。TinyGPT 一半以上的参数都在 MLP 里。
</p>

<h3 id="a-resid">7. 残差流与层归一化</h3>
<p>
	块从不替换自己的输入，而是算出一个更新量再加上去：<Tex tex={"x' = x + \\mathrm{Attn}(\\mathrm{LN}(x))"} />，然后
	<Tex tex={"x'' = x' + \\mathrm{MLP}(\\mathrm{LN}(x'))"} />。这条不断累加的“和”叫作<strong>残差流</strong>（<Term id="residual-connection" />）。
	每个块都从它读取、再写回它，所以词元的嵌入可以穿过全部 4 个块保留下来；训练时梯度也有一条很短的路径回到每一层。
</p>
<p>
	在每个子层之前，<Term id="layer-norm" />把每个词元的向量缩放到均值 0、标准差 1，再乘以学到的增益、加上偏置。它对每个词元单独进行，作用在它自己的 76 个数上，
	让数值保持在稳定的范围内，使训练容易得多。最后一个块之后还有一次层归一化。由于每个块都只是往残差流里“加”，我们可以用同样的最终层归一化和输出头提前读取残差流：
	这就是 <strong>logit lens</strong>，它能显示模型的猜测是在什么时候出现的。
</p>

<h3 id="a-out">8. Logits、softmax 与采样</h3>
<p>
	最后，把最后一个词元的向量与嵌入表 <Tex tex="E" /> 的每一行做比较（输出头与输入表“共享权重”，不需要额外参数），得到 1024 个分数，称为
	<Term id="logit" />。softmax 把它们变成概率。为了写出文本，模型从这个分布中选一个词元，把它接到末尾，再在更长的文本上运行一次（<Term id="autoregressive" />）。
</p>
<p>
	怎么选很重要。温度为 0 时总是选最可能的词元（贪心）。否则，先把 logits 除以<Term id="temperature" />再做 softmax：小于 1 分布更尖锐，大于 1 更平坦。
	<Term id="sampling" />中的 top-k 只保留最可能的 k 个词元；top-p 保留概率之和达到 p 的最小集合。然后用一个 0 到 1 之间的随机数 <Tex tex="u" /> 选出一个词元。
	本页的随机数来自带种子的生成器，所以同样的提示词、种子和设置总会得到同样的故事。
</p>

<h3 id="a-scale">TinyGPT 有多小？</h3>
<p>
	TinyGPT 有 364,040 个参数：d = 76，4 个块，4 个头，64 个词元的上下文，1024 个词元的词表。它在 TinyStories 中约 3250 万个词元的数据上训练了 30 遍，
	验证集损失为每词元 1.92 nats（困惑度 6.8：平均而言，它的不确定程度相当于在约 7 个等可能的词元中做选择）。
	<a href="https://poloclub.github.io/transformer-explainer/" rel="noopener">Transformer Explainer</a> 里的 GPT-2 small 有 1.24 亿个参数：d = 768，12 个块，12 个头，
	1024 个词元的上下文，50,257 个词元的词表——但每一步都和这里一样。<a href="https://bbycroft.net/llm" rel="noopener">LLM Visualization</a> 用 3D 展示了同样的结构，细到每一个权重。
	与某些讲解工具不同，本页上的每一个向量都是真实的：在你的浏览器里用训练好的权重算出，并与 PyTorch 核对，误差在 1e-4 以内。
</p>
<p>
	下一步：在<a href="{base}/train/next-token-prediction/">下一个词元预测</a>一页里，看看模型是怎样学会这一切的，那里会回放 TinyGPT 的训练检查点。
</p>

<style>
	.eq {
		overflow-x: auto;
	}
</style>
