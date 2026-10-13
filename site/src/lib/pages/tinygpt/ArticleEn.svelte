<script lang="ts">
	import { base } from '$app/paths';
	import Term from '$lib/kit/Term.svelte';
	import Tex from '$lib/kit/Tex.svelte';
</script>

<h3 id="a-tokens">1. Tokens: text becomes numbers</h3>
<p>
	A language model cannot read letters. A <Term id="tokenizer" /> first cuts the text into <Term id="token">tokens</Term> and
	gives each one an id. TinyGPT uses byte-level <Term id="bpe" /> with a <Term id="vocabulary" /> of 1,024 tokens: the 256
	possible bytes, 767 merges learned from children's stories, and one special token, <code>&lt;|endoftext|&gt;</code>.
	Common pieces such as <code>␣the</code> or <code>␣girl</code> are single tokens; rarer words are split into pieces. A word
	usually carries its leading space: <code>␣cat</code> is one token, but <code>cat</code> without the space is split into
	<code>c</code> + <code>at</code>.
</p>
<p>
	Because the tokenizer works on bytes, any text can be encoded, even if it was never seen in training. A Chinese character
	is 3 bytes in UTF-8, and no merges were learned for them, so 我爱自然语言处理 becomes 24 byte tokens. TinyGPT has never
	seen those bytes, so what it predicts after them means nothing. The model also reads at most 64 tokens at once: its
	<Term id="context-window" />. The page on <a href="{base}/train/tokenization/">tokenization</a> shows how the merges are learned.
</p>

<h3 id="a-emb">2. Embeddings: a vector for each token and each position</h3>
<p>
	Each token id picks one row of a learned table <Tex tex="E" /> of size 1,024 × 76. That row is the token's
	<Term id="embedding" />: 76 numbers that the training process tuned so that tokens used in similar ways get similar
	vectors. A second table <Tex tex="P" /> (64 × 76) holds one learned vector per position, and the two are added:
	<Tex tex={'x_t = E[\\text{id}_t] + P[t]'} />. The position vector is needed because attention, on its own, does not know
	which token came first (<Term id="positional-encoding" />). For a text of <Tex tex="T" /> tokens, the result is a
	<Tex tex={'T \\times 76'} /> matrix: one row per token. You can follow these rows through the whole model in the lesson and in Explore.
</p>

<h3 id="a-attn">3. Attention: queries, keys and values</h3>
<p>
	<Term id="attention" /> lets a token gather information from earlier tokens. From each token's (normalized) vector the model
	makes three new vectors with three learned matrices: a <strong>query</strong> (what this token is looking for), a
	<strong>key</strong> (what it can offer) and a <strong>value</strong> (what it will hand over) — see
	<Term id="query-key-value" />. The match between token <Tex tex="i" /> and token <Tex tex="j" /> is the dot product of
	<Tex tex="q_i" /> and <Tex tex="k_j" />, divided by <Tex tex={'\\sqrt{19}'} /> so that the scores do not grow with the vector
	length. A <Term id="softmax" /> turns each row of scores into weights that are positive and sum to 1. The output for token
	<Tex tex="i" /> is the weighted sum of the value vectors:
</p>
<div class="eq"><Tex display tex={'a_{ij} = \\operatorname{softmax}_j\\!\\left(\\frac{q_i \\cdot k_j}{\\sqrt{19}}\\right), \\qquad o_i = \\sum_{j \\le i} a_{ij}\\, v_j'} /></div>
<p>
	So a token that "needs" a name can put most of its weight on an earlier name and copy that name's value. Use the dot-product
	panel to redo one score by hand with the live numbers.
</p>

<h3 id="a-heads">4. Many heads at once</h3>
<p>
	One attention pattern is not enough: a token may need the previous word for grammar and a name from far back for meaning.
	So each block runs 4 <Term id="attention-head">attention heads</Term> side by side. The 76 columns of Q, K and V are split
	into 4 groups of 19; each head does its own scores, softmax and weighted sum on its group. The 4 outputs (19 numbers each)
	are joined back into 76 numbers and mixed by an output projection. In the grid of 4 × 4 small maps you can look for heads with
	a clear habit, such as attending to the previous token or to the first token. In a model this small the habits are fuzzy.
</p>

<h3 id="a-mask">5. The causal mask</h3>
<p>
	TinyGPT is trained to predict the next token at every position at once (<Term id="next-token-prediction" />). If token 5
	could look at token 6, it would see the answer. The <Term id="causal-mask" /> prevents this: before the softmax, every score
	for a later key is set to <Tex tex={'-\\infty'} />, so its weight becomes exactly 0. That is the hatched upper triangle in
	every attention map. It is also why the model can write text one token at a time.
</p>

<h3 id="a-mlp">6. The MLP: thinking about each token alone</h3>
<p>
	After attention, each token goes through a small two-layer network, the <Term id="mlp" />, on its own: 76 numbers are
	expanded to 304, passed through GELU (a smooth version of ReLU that squashes negative inputs toward 0), and projected back
	to 76. No information moves between tokens here. A useful picture: attention moves information between tokens; the MLP
	processes what each token now holds. The MLP layers hold more than half of TinyGPT's parameters.
</p>

<h3 id="a-resid">7. The residual stream and LayerNorm</h3>
<p>
	A block never replaces its input. It computes an update and adds it: <Tex tex={"x' = x + \\mathrm{Attn}(\\mathrm{LN}(x))"} />,
	then <Tex tex={"x'' = x' + \\mathrm{MLP}(\\mathrm{LN}(x'))"} />. This running sum is called the <strong>residual stream</strong>
	(<Term id="residual-connection" />). Every block reads from it and writes back into it, so the embedding of a token can
	survive all 4 blocks, and during training the gradient has a short path back to every layer.
</p>
<p>
	Before each sub-layer, <Term id="layer-norm">LayerNorm</Term> rescales every token's vector to mean 0 and spread 1, then
	applies a learned gain and bias. It works on each token separately, across its 76 numbers. It keeps the numbers in a steady
	range, which makes training much easier. One more LayerNorm comes after the last block. Because every block only adds to the
	stream, we can read the stream early with the same final LayerNorm and output head: that is the <strong>logit lens</strong>,
	which shows when the model's guess appears.
</p>

<h3 id="a-out">8. Logits, softmax and sampling</h3>
<p>
	At the end, the vector of the last token is compared with every row of the embedding table <Tex tex="E" /> (the output head
	is "tied" to the input table, so it costs no extra parameters). This gives 1,024 scores called <Term id="logit">logits</Term>.
	A softmax turns them into probabilities. To write text, the model picks one token from this distribution, appends it, and runs
	again on the longer text (<Term id="autoregressive" />).
</p>
<p>
	How it picks matters. Temperature 0 always takes the most likely token (greedy). Otherwise the logits are divided by the
	<Term id="temperature" /> before the softmax: below 1 the distribution gets sharper, above 1 flatter. <Term id="sampling">Top-k</Term>
	keeps only the k most likely tokens; top-p keeps the smallest set whose probabilities add up to p. Then a random number
	<Tex tex="u" /> between 0 and 1 chooses a token. On this page the random numbers come from a seeded generator, so the same
	prompt, seed and settings always give the same story.
</p>

<h3 id="a-scale">How small is TinyGPT?</h3>
<p>
	TinyGPT has 364,040 parameters: d = 76, 4 blocks, 4 heads, a 64-token context and 1,024 tokens. It was trained for 30 passes over
	a 32.5-million-token slice of TinyStories and reaches a validation loss of 1.92 nats per token (perplexity 6.8: on average it is
	about as unsure as choosing among 7 equally likely tokens). GPT-2 small, the model in
	<a href="https://poloclub.github.io/transformer-explainer/" rel="noopener">Transformer Explainer</a>, has 124 million
	parameters: d = 768, 12 blocks, 12 heads, a 1,024-token context and 50,257 tokens — but every step is the same as here.
	<a href="https://bbycroft.net/llm" rel="noopener">LLM Visualization</a> shows the same architecture in 3D, down to every
	weight. Unlike some explainers, every vector on this page is real: computed in your browser from the trained weights and
	checked against PyTorch to within 1e-4.
</p>
<p>
	Next: see how the model learned all this in <a href="{base}/train/next-token-prediction/">next-token prediction</a>, which
	replays TinyGPT's training checkpoints.
</p>

<style>
	.eq {
		overflow-x: auto;
	}
</style>
