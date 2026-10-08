/**
 * The PyTorch reference the live numbers are checked against: GPT.trace() from
 * pipeline/tinygpt/model.py (our own code). It computes the same thing as the fast forward() step by
 * step and records every tensor under the ids the page uses; pipeline/parity/tinygpt.parity.ts compares
 * all of them with the browser's tinygpt.ts (max |diff| within 1e-4).
 */
export const TRACE_CODE = "def trace(self, ids: list[int]):\n    \"\"\"Run one sequence step by step and return (order, {layer id: tensor}) in fp32.\n\n    Shapes: sequences [T, D]; attention scores/masked/weights [n_head, T, T];\n    L{i}.attn.scores = q k^T / sqrt(d_head); masked = scores with -inf above the diagonal.\n    \"\"\"\n    cfg = self.cfg\n    T, D, H = len(ids), cfg.d_model, cfg.n_head\n    hd = D // H\n    assert T <= cfg.context\n    idx = torch.tensor(ids, dtype=torch.long)\n    t, order = {}, []\n\n    def rec(name, val):\n        order.append(name)\n        t[name] = val.detach().float().clone()\n        return val\n\n    tok = rec(\"tok_emb\", self.tok_emb.weight[idx])\n    pos = rec(\"pos_emb\", self.pos_emb.weight[:T])\n    x = rec(\"embed\", tok + pos)\n    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)\n    for i, b in enumerate(self.blocks):\n        p = f\"L{i}.\"\n        h = rec(p + \"ln1\", F.layer_norm(x, (D,), b.ln1.weight, b.ln1.bias, cfg.ln_eps))\n        Wq, Wk, Wv = b.qkv.weight.split(D, 0)\n        bq, bk, bv = b.qkv.bias.split(D, 0)\n        q = rec(p + \"attn.q\", h @ Wq.t() + bq)\n        k = rec(p + \"attn.k\", h @ Wk.t() + bk)\n        v = rec(p + \"attn.v\", h @ Wv.t() + bv)\n        qh = q.view(T, H, hd).transpose(0, 1)  # [H, T, hd]\n        kh = k.view(T, H, hd).transpose(0, 1)\n        vh = v.view(T, H, hd).transpose(0, 1)\n        s = rec(p + \"attn.scores\", qh @ kh.transpose(1, 2) / math.sqrt(hd))\n        sm = rec(p + \"attn.masked\", s.masked_fill(mask, float(\"-inf\")))\n        w = rec(p + \"attn.weights\", torch.softmax(sm, dim=-1))\n        heads = rec(p + \"attn.heads\", (w @ vh).transpose(0, 1).reshape(T, D))\n        a = rec(p + \"attn.proj\", heads @ b.attn_proj.weight.t() + b.attn_proj.bias)\n        x = rec(p + \"resid1\", x + a)\n        h2 = rec(p + \"ln2\", F.layer_norm(x, (D,), b.ln2.weight, b.ln2.bias, cfg.ln_eps))\n        f = rec(p + \"mlp.fc\", h2 @ b.fc.weight.t() + b.fc.bias)\n        g = rec(p + \"mlp.gelu\", gelu_tanh(f))\n        m = rec(p + \"mlp.proj\", g @ b.mlp_proj.weight.t() + b.mlp_proj.bias)\n        x = rec(p + \"resid2\", x + m)\n    xf = rec(\"ln_f\", F.layer_norm(x, (D,), self.ln_f.weight, self.ln_f.bias, cfg.ln_eps))\n    logits = rec(\"logits\", xf @ self.tok_emb.weight.t())\n    rec(\"probs\", torch.softmax(logits, dim=-1))\n    return order, t\n";

const NL = String.fromCharCode(10);

/** 1-based line numbers in TRACE_CODE that compute a given trace id (L2.attn.q -> the attn.q line). */
export function codeLines(id: string): number[] {
	const name = id.replace(/^L\d\./, '');
	const block = /^L\d\./.test(id);
	const needle = block ? `p + "${name}"` : `rec("${name}"`;
	const out: number[] = [];
	TRACE_CODE.split(NL).forEach((l, i) => {
		if (l.includes(needle)) out.push(i + 1);
	});
	if (block && name.startsWith('attn.') && ['attn.q', 'attn.k', 'attn.v'].includes(name)) {
		// the weight split lines feed q, k, v
		TRACE_CODE.split(NL).forEach((l, i) => {
			if (l.includes('qkv.weight.split') || l.includes('qkv.bias.split')) out.push(i + 1);
		});
	}
	return out.sort((a, b) => a - b);
}

/** The JavaScript that runs in your browser for the same step (src/lib/models/tinygpt.ts), one line each. */
export const JS_LINES: Record<string, string> = {
	tok_emb: 'tok.set(tokW.subarray(ids[i] * D, ids[i] * D + D), i * D);   // copy row ids[i] of E',
	pos_emb: 'pos.set(posW.subarray(i * D, i * D + D), i * D);             // copy row i of P',
	embed: "x = add(tok, pos);",
	ln1: "h = layerNorm(x, T, D, w(p + 'ln1.weight'), w(p + 'ln1.bias'), eps);",
	'attn.q': "q = linear(h, T, D, w(p + 'attn.q.weight'), w(p + 'attn.q.bias'), D);",
	'attn.k': "k = linear(h, T, D, w(p + 'attn.k.weight'), w(p + 'attn.k.bias'), D);",
	'attn.v': "v = linear(h, T, D, w(p + 'attn.v.weight'), w(p + 'attn.v.bias'), D);",
	'attn.scores': 'for (c < hd) s += q[i*D + hh*hd + c] * k[j*D + hh*hd + c];  scores[o] = s * (1 / Math.sqrt(hd));',
	'attn.masked': 'masked[o] = j > i ? -Infinity : scores[o];',
	'attn.weights': 'att = softmaxRows(masked, H * T, T);',
	'attn.heads': 'for (j <= i) s += att[(hh*T + i)*T + j] * v[j*D + hh*hd + c];  heads[i*D + hh*hd + c] = s;',
	'attn.proj': "proj = linear(heads, T, D, w(p + 'attn.proj.weight'), w(p + 'attn.proj.bias'), D);",
	resid1: 'x = add(x, proj);',
	ln2: "h2 = layerNorm(x, T, D, w(p + 'ln2.weight'), w(p + 'ln2.bias'), eps);",
	'mlp.fc': "fc = linear(h2, T, D, w(p + 'mlp.fc.weight'), w(p + 'mlp.fc.bias'), 4 * D);",
	'mlp.gelu': 'g = gelu(fc);   // 0.5 v (1 + tanh(√(2/π) (v + 0.044715 v³)))',
	'mlp.proj': "m = linear(g, T, 4 * D, w(p + 'mlp.proj.weight'), w(p + 'mlp.proj.bias'), D);",
	resid2: 'x = add(x, m);',
	ln_f: "xf = layerNorm(x, T, D, w('ln_f.weight'), w('ln_f.bias'), eps);",
	logits: 'logits = linear(xf, T, D, tokW, null, V);   // tied head: multiply by E',
	probs: 'probs = softmaxRows(logits, T, V);'
};

export function jsLine(id: string): string {
	return JS_LINES[id.replace(/^L\d\./, '')] ?? '';
}

/** The fast training path (pipeline/tinygpt/model.py): GELU, one Block, and GPT.forward(). */
export const BLOCK_CODE = "def gelu_tanh(x):\n    # GELU, tanh approximation (as in GPT-2): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))\n    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))\n\n\nclass Block(nn.Module):\n    def __init__(self, cfg: GPTConfig):\n        super().__init__()\n        d = cfg.d_model\n        self.n_head = cfg.n_head\n        self.ln1 = nn.LayerNorm(d, eps=cfg.ln_eps)\n        self.qkv = nn.Linear(d, 3 * d)  # q, k, v stacked; split on export\n        self.attn_proj = nn.Linear(d, d)\n        self.ln2 = nn.LayerNorm(d, eps=cfg.ln_eps)\n        self.fc = nn.Linear(d, 4 * d)\n        self.mlp_proj = nn.Linear(4 * d, d)\n\n    def forward(self, x):\n        B, T, D = x.shape\n        H = self.n_head\n        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)\n        q = q.view(B, T, H, D // H).transpose(1, 2)\n        k = k.view(B, T, H, D // H).transpose(1, 2)\n        v = v.view(B, T, H, D // H).transpose(1, 2)\n        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)\n        y = y.transpose(1, 2).contiguous().view(B, T, D)\n        x = x + self.attn_proj(y)\n        x = x + self.mlp_proj(gelu_tanh(self.fc(self.ln2(x))))\n        return x\n\n\n\nclass GPT(nn.Module):\n    ...\n    def forward(self, idx, targets=None):\n        B, T = idx.shape\n        assert T <= self.cfg.context\n        pos = torch.arange(T, device=idx.device)\n        x = self.tok_emb(idx) + self.pos_emb(pos)[None]\n        for b in self.blocks:\n            x = b(x)\n        x = self.ln_f(x)\n        logits = x @ self.tok_emb.weight.t()  # tied output head\n        loss = None\n        if targets is not None:\n            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))\n        return logits, loss\n\n    # ------------------------------------------------------------------------------------------";

/** How this page runs the model (src/lib/models/tinygpt.ts + bpe.ts; in a Web Worker). */
export const USAGE_CODE = `import { loadTokenizer } from '$lib/models/bpe';
import { load } from '$lib/models/tinygpt';

const tok = await loadTokenizer(fetch, base + '/models/tinygpt');
const model = await load(fetch, base + '/models/tinygpt');     // spec.json + weights.bin (fp32)

const ids = tok.encode('Once upon a time, there was a little');
const trace = model.forward(ids);        // 66 tensors: tok_emb, …, L0.attn.weights, …, probs
trace.t['L2.attn.weights'].shape;        // [4, T, T]  (heads, queries, keys)

const out = model.generate(ids, { maxNew: 24, temperature: 0.8, topK: 0, topP: 1, seed: 42 });
tok.decode(out.newIds);                  // same text in every browser for the same seed`;
