// tinymamba: the guided lesson (10 steps). Every number is a {placeholder}: structure from the spec (dims()), live values
// from the trace of the learner's row ('fwd') and from the scan process ('scan', float64), measured ones from facts.json
// ({fact:...}). Claims that are not computed live are tested in claims.test.ts.
//   1 the task → 2 tokens become vectors → 3 a state updated token by token → 4 Δ decides what to keep (predict) →
//   5 B writes, C reads → 6 one scan step with real numbers → 7 the state over the whole row → 8 the output →
//   9 a fixed state versus attention → 10 limits: a fixed state can forget (task)
import type { I18n, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { answers, dec, deltaAt, dims, firstLetter, idsOf, pickCh, T, tok, worked, writeRead, type RT } from './tm.ts';
import { memory } from './views/memory/meta.ts';
import { ASK, isLetter } from './task.ts';

const L = (en: string, zh: string): I18n => ({ en, zh });
const pl = (n: number, w: string) => `${n} ${w}${n === 1 ? '' : 's'}`;

/** What the row holds: counts, the first letter and the first question mark. */
export function row(rt: RT) {
	const ids = idsOf(rt);
	const k = ids.filter(isLetter).length;
	const q = ids.filter((i) => i === ASK).length;
	const a = answers(rt);
	return { ids, n: ids.length, k, q, dots: ids.filter((i) => i === 0).length, a, t1: firstLetter(rt), tq: ids.indexOf(ASK), first: a.find((x) => x.want >= 0) };
}

/** Block 1, followed channel: the largest Δ and the largest change of its state at dots and at letters. */
export function follow(rt: RT) {
	const { D, N } = dims(rt);
	const ch = pickCh(rt, 1);
	const h = T(rt, 'B1.scan#h');
	const r = row(rt);
	const out = { ch, dotDelta: 0, letDelta: 0, dotMove: 0, letMove: 0 };
	r.ids.forEach((id, t) => {
		let move = 0;
		for (let n = 0; n < N && h; n++) move = Math.max(move, Math.abs(h.data[(t * D + ch) * N + n] - (t ? h.data[((t - 1) * D + ch) * N + n] : 0)));
		if (id === 0) (out.dotDelta = Math.max(out.dotDelta, deltaAt(rt, 1, t, ch)), out.dotMove = Math.max(out.dotMove, move));
		if (isLetter(id)) (out.letDelta = Math.max(out.letDelta, deltaAt(rt, 1, t, ch)), out.letMove = Math.max(out.letMove, move));
	});
	return out;
}

/** Step 5's stage: the two cells the narration works out, outlined, and their numbers in the caption. */
function wrProps(rt: RT) {
	const w = writeRead(rt);
	const ok = row(rt).k > 0 && w.tq > w.t;
	return { rows: ['B', 'C'], ch: w.ch, marks: ok ? [{ key: 'B', row: w.slot, col: w.t }, { key: 'C', row: w.slot, col: w.tq }] : [], note: ok ? { en: `Outlined: slot ${w.slot} at “${w.let}” (token ${w.t}), B = ${dec(w.B)}, and at the first ? (token ${w.tq}), C = ${dec(w.C)}. In channel ${w.ch}, Δ = ${dec(w.d)} and u = ${dec(w.u)} at “${w.let}”.`, zh: `加框的是槽 ${w.slot}：在“${w.let}”（第 ${w.t} 个词元）处 B = ${dec(w.B)}，在第一个 ?（第 ${w.tq} 个词元）处 C = ${dec(w.C)}。通道 ${w.ch} 在“${w.let}”处 Δ = ${dec(w.d)}，u = ${dec(w.u)}。` } : undefined };
}

/** Step 6: the scan step of the frame as worked() gives it, its formula (negative numbers in brackets) and its values. */
type M = Record<string, number>;
export function stepOf(m: M) {
	const w = worked({ delta: m.delta, A: m.A, B: m.B, u: m.u, hOld: m.hOld });
	const s = (k: keyof typeof w) => (w[k] < 0 ? String.raw`\left({{${k}}}\right)` : `{{${k}}}`);
	const tex = String.raw`\begin{aligned} h_{\text{new}} &= e^{\Delta A}\, h_{\text{old}} + \Delta\, B\, u \\ &= e^{ {{delta}} \times ${s('A')} } \times ${s('hOld')} \\ &\quad + {{delta}} \times ${s('B')} \times ${s('u')} \\ &= {{decay}} \times ${s('hOld')} + ${s('write')} \\ &= ${s('kept')} + ${s('write')} \\ &= {{hNew}} \end{aligned}`;
	const values = Object.fromEntries(Object.entries(w).map(([k, v]) => [k, dec(v, k === 'hNew' ? 3 : 4)]));
	return { w, tex, values };
}

export const TRIES = [
	{ label: L('Six letters', '六个字母'), text: 'a.c.e.g.b.d??????' },
	{ label: L('Eight letters', '八个字母'), text: 'h.a.d.b.g.c.f.e????????' }
];

const said = (rt: RT): TmplVal => {
	const a = row(rt).a;
	return a.length ? a.map((x) => (x.top >= 0 ? tok(x.top) : '…')).join(' ') : L('nothing', '什么也没有');
};

/** "3 of 3 right", or why nothing can be checked. */
function score(rt: RT): I18n {
	const r = row(rt);
	const a = r.a.filter((x) => x.want >= 0);
	const right = a.filter((x) => x.ok).length;
	if (a.length) return { en: `${right} of ${a.length} right`, zh: `${a.length} 个里对了 ${right} 个` };
	return r.q ? L('no ? asks for a letter yet (none has a letter before it)', '还没有哪个 ? 要字母（它们前面都没有字母）') : L('your row has no ?, so nothing is asked', '你的这一行没有 ?，所以什么也没问');
}

const lesson: LessonDef = {
	title: L('Tiny Mamba: a scan that decides what to remember', '小型 Mamba：一个决定记住什么的扫描'),
	summary: L(
		'Give a tiny selective state-space model a row of dots and letters and follow its fixed-size state token by token: the step size Δ keeps the state at the dots and writes the letters in, B writes, C reads, and at each question mark a letter comes back; then compare its memory with attention’s and find what it cannot hold.',
		'给一个小小的选择性状态空间模型一行点和字母，逐个词元跟踪它大小固定的状态：步长 Δ 在点处保持状态、在字母处写入，B 负责写，C 负责读，每个问号处找回一个字母；再把它的记忆和注意力的比一比，看看它装不下什么。'
	),
	steps: [
		{ id: 'task', source: 'fwd', view: { key: 'tinymamba/scan', props: () => ({ rows: ['want'] }) }, provenance: 'live', title: L('The task: remember the letters', '任务：记住字母'),
			narration: {
				en: 'Your row has {parts}. At each question mark the model must give back one letter, in the order the letters came: the first ? asks for the first letter, the second ? for the second, and so on. {ask} It reads the row only once, from left to right, one token at a time, so it has to carry the letters along until the question marks come.',
				zh: '你的这一行有 {parts}。模型要在每个问号处按字母出现的顺序找回一个字母：第一个 ? 要第一个字母，第二个 ? 要第二个字母，依此类推。{ask}它只从左到右读一遍，一次读一个词元，所以得一路记着这些字母，直到问号出现。',
				vars: (rt) => {
					const r = row(rt);
					return {
						parts: { en: `${pl(r.n, 'token')}: ${pl(r.k, 'letter')}, ${pl(r.dots, 'dot')} and ${pl(r.q, 'question mark')}`, zh: `${r.n} 个词元：${r.k} 个字母、${r.dots} 个点和 ${r.q} 个问号` },
						ask: !r.a.length ? L('Your row has no ? yet, so nothing is asked.', '你的这一行还没有 ?，所以什么也没问。') : r.a[0].want >= 0 ? { en: `Here the first ? asks for “${tok(r.a[0].want)}”.`, zh: `这里第一个 ? 要的是“${tok(r.a[0].want)}”。` } : L('Here the first ? comes before any letter, so it has nothing to give back.', '这里第一个 ? 出现在所有字母之前，所以它没有字母可找回。')
					};
				}
			},
			look: L('Each arc joins a letter to the ? that asks for it, and the line below shows the letter each ? should give back. Change the row in the box: every step follows it.', '每条弧线把一个字母连到要它的那个 ?，下面一行是每个 ? 应当找回的字母。在输入框里改这一行，每一步都会跟着变。'), terms: ['token'] },
		{ id: 'vectors', source: 'fwd', at: { layer: 'emb' }, select: (rt) => ({ layer: 'emb', unit: [firstLetter(rt), 0] }), view: 'zoom.lookup', provenance: 'live', title: L('Tokens become vectors', '词元变成向量'),
			narration: {
				en: 'First every token becomes a vector: its row in a learned table with {V} rows, one per token of the alphabet, {d} numbers each, so every dot gets the same row. There is no table of positions: the model reads the tokens in order, and its state carries the order. Then come {L} blocks, one after the other. Each one rescales the vector to mean 0 and spread 1 (layer normalization), then a learned weighted sum and SiLU (a smooth activation function) turn it into {D} channels, called u.',
				zh: '首先每个词元变成一个向量：在一张学来的表里查出自己那一行。表有 {V} 行，字母表里每个词元一行，每行 {d} 个数，所以每个点得到的都是同一行。这里没有位置表：模型按顺序读词元，顺序由它的状态带着。接下来是 {L} 个块，一个接一个。每个块先把这个向量重新缩放成平均值为 0、标准差为 1（层归一化），再由一个学来的加权和加上 SiLU（一种平滑的激活函数）把它变成 {D} 个通道，记作 u。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'The highlighted row is the one token {t} “{x}” copies (counting from 0).', zh: '高亮的一行就是第 {t} 个词元“{x}”复制的那一行（从 0 数起）。', vars: (rt) => ({ t: firstLetter(rt), x: tok(idsOf(rt)[firstLetter(rt)]) }) }, terms: ['embedding', 'layer-norm', 'activation-function'] },
		{ id: 'state', source: 'scan', at: 'last', play: { from: 'first', to: 'last', interval: 500, unit: L('token', '词元') }, view: { key: 'tinymamba/scan', props: () => ({ rows: ['h'], full: true }) }, provenance: 'live', title: L('A state, updated token by token', '一份逐个词元更新的状态'),
			narration: {
				en: 'Block 1 keeps a state: {D} channels with {N} slots each, one number per slot, {size} numbers in all. Before the first token they are all 0. At every token each channel updates its {N} slots from their old values and the token, and the block’s output at that token is read from the new state. It never looks back at earlier tokens: all it knows about them is in these {size} numbers.',
				zh: '第 1 块保留一份状态：{D} 个通道，每个通道 {N} 个槽，每个槽一个数，一共 {size} 个数。读第一个词元之前它们全是 0。每读一个词元，每个通道都根据旧值和这个词元更新自己的 {N} 个槽，这个块在该词元处的输出就从新状态里读出。它从不回头看前面的词元：关于它们，它知道的一切都在这 {size} 个数里。',
				vars: (rt) => ({ ...dims(rt), size: dims(rt).D * dims(rt).N })
			},
			look: { en: 'Press play and watch the rows under the tokens light up, one token at a time (faint cells are tokens not read yet): they follow channel {ch} (counting from 0), one row per slot. The small grid is the whole state after the token in the frame.', zh: '点播放，看词元下面的几行一个词元一个词元地亮起来（淡的格子是还没读到的词元）：它们跟踪通道 {ch}（从 0 数起），每个槽一行。小方格是框住的那个词元之后的全部状态。', vars: (rt) => ({ ch: pickCh(rt, 1) }) }, terms: ['state-space-model'] },
		{ id: 'delta', source: 'fwd', at: { layer: 'B1.scan' }, view: { key: 'tinymamba/scan', props: () => ({ rows: ['delta'] }) }, provenance: 'live', title: L('Δ decides how much to keep', 'Δ 决定保留多少'),
			narration: {
				en: 'Each channel computes a step size Δ from the token: a learned weighted sum of the channels u, put through softplus, a smooth curve that stays above 0. The old state is multiplied by e to the power Δ·A, where A is a learned negative number (one per slot), and Δ·B·u is added (B comes next). With Δ near 0, e to the power Δ·A is near e⁰ = 1: the state stays and almost nothing is written. With a large Δ the old state fades and the token is written in.',
				zh: '每个通道根据词元算出一个步长 Δ：先对通道 u 做一个学来的加权和，再经过 softplus，也就是一条始终大于 0 的平滑曲线。旧状态乘以 e 的 Δ·A 次方，其中 A 是一个学来的负数（每个槽一个），再加上 Δ·B·u（B 下一步讲）。Δ 接近 0 时，e 的 Δ·A 次方接近 e⁰ = 1：状态保持不变，几乎什么都不写入。Δ 很大时，旧状态淡去，这个词元被写进去。'
			},
			look: L('One row per channel of block 1; the stronger the purple, the larger Δ (cells that blend into the background are near 0). Compare the columns of the letters with those of the dots.', '第 1 块每个通道一行；紫色越浓，Δ 越大（和背景几乎分不出的格子接近 0）。比较字母那几列和点那几列。'),
			predict: {
				id: 'tinymamba-delta',
				question: L('Where will Δ be large: at the letters or at the dots?', 'Δ 会在哪里变大：字母处还是点处？'),
				options: [L('At the letters, in some of the channels', '在字母处，而且只在部分通道里'), L('At the dots, since there are more of them', '在点处，因为点更多'), L('About the same at every token', '每个词元处都差不多')],
				answer: 0,
				explain: L('At the dots Δ stays near 0 in almost every channel, so each dot leaves the state almost exactly as it was. At a letter Δ jumps up in a few channels (not in all of them), and the letter goes into the state. Training found this on its own: nobody told the model which tokens matter.', '在点处，几乎每个通道的 Δ 都接近 0，所以每个点几乎原样留下状态。在字母处，有几个通道（不是全部）的 Δ 跳了上去，字母进入状态。这是训练自己找到的：没有人告诉模型哪些词元重要。')
			}, terms: ['step-size'] },
		{ id: 'write-read', source: 'fwd', at: { layer: 'B1.scan' }, view: { key: 'tinymamba/scan', props: (rt) => wrProps(rt) }, provenance: 'live', title: L('B writes, C reads', 'B 负责写，C 负责读'),
			narration: {
				en: 'Two more sets of {N} numbers come from each token, one number per slot, shared by all channels. B says how strongly the token is written into each slot: the write is Δ·B·u. C says how strongly each slot is read: a channel’s output is y = C·h added up over the slots, plus D·u, a learned shortcut for the token itself. {example}',
				zh: '每个词元还会算出另外两组数，每组 {N} 个，每个槽一个，所有通道共用。B 表示这个词元写进每个槽的力度：写入的量是 Δ·B·u。C 表示每个槽被读出的力度：一个通道的输出 y 等于各个槽的 C·h 之和，再加上 D·u，也就是让词元本身直接通过的一条学来的捷径。{example}',
				vars: (rt) => {
					const w = writeRead(rt);
					const r = row(rt);
					const wr = dec(worked({ delta: w.d, A: 0, B: w.B, u: w.u, hOld: 0 }).write);
					const par = (v: number) => (v < 0 ? `(${dec(v)})` : dec(v));
					const prod = `${dec(w.d)} × ${par(w.B)} × ${par(w.u)} = ${wr}`;
					return {
						N: w.N,
						example: pick(r.k > 0 && r.tq > r.t1, { en: `In channel ${w.ch}, slot ${w.slot} (counting from 0), the letter “${w.let}” has B = ${dec(w.B)} and writes Δ·B·u = ${prod}; the first ? reads that slot with C = ${dec(w.C)}.`, zh: `在通道 ${w.ch} 的槽 ${w.slot}（从 0 数起），字母“${w.let}”的 B = ${dec(w.B)}，写入 Δ·B·u = ${prod}；第一个 ? 用 C = ${dec(w.C)} 读这个槽。` }, L('Put a letter before a ? in your row to see a write and a read with real numbers.', '在你的这一行里，让一个字母出现在 ? 前面，就能看到用真实数字算出的一次写入和一次读出。'))
					};
				}
			},
			look: L('Rows are slots: orange is positive, blue negative. Find the letters’ columns in B and the question marks’ columns in C.', '每一行是一个槽：橙色为正，蓝色为负。在 B 里找字母那几列，在 C 里找问号那几列。'), terms: ['step-size'] },
		{ id: 'step', source: 'scan', at: (rt) => ({ phase: 'token', step: firstLetter(rt) }), view: { key: 'zoom.formula', props: (_rt, at) => { const s = stepOf((at.frame?.metrics ?? {}) as M); return { tex: s.tex, values: s.values }; } }, provenance: 'live', title: L('One scan step, with real numbers', '一步扫描，用真实的数'),
			narration: {
				en: 'Here is the update at {which} “{x}” (token {t}, counting from 0), for channel {ch}, slot {slot}. Δ = {dl} and A = {a}, so the old state is multiplied by e to the power Δ·A, which is {dc}. The new part is Δ·B·u = {wr}. So h goes from {ho} to {dc} × {hoP} {pw} = {kp} {pw} = {hn}: the kept part plus the new part, the same two-part update that all {size} numbers of the state make at every token.',
				zh: '这是在{which}“{x}”（第 {t} 个词元，从 0 数起）处，通道 {ch}、槽 {slot} 的更新。Δ = {dl}，A = {a}，所以旧状态乘以 e 的 Δ·A 次方，也就是 {dc}。新加的部分是 Δ·B·u = {wr}。于是 h 从 {ho} 变成 {dc} × {hoP} {pw} = {kp} {pw} = {hn}：保留下来的部分加上新写入的部分；状态的全部 {size} 个数在每个词元处都这样做两步更新。',
				vars: (rt, at) => {
					const m = (at.frame?.metrics ?? {}) as M;
					const f0 = at.tape?.frame(0).metrics ?? {};
					const v = stepOf(m).values;
					const w = stepOf(m).w;
					return { t: m.t ?? NaN, dl: v.delta, a: v.A, dc: v.decay, wr: v.write, ho: v.hOld, hoP: w.hOld < 0 ? `(${v.hOld})` : v.hOld, kp: v.kept, pw: `${w.write < 0 ? '−' : '+'} ${dec(Math.abs(w.write))}`, hn: v.hNew, ch: f0.ch ?? NaN, slot: f0.slot ?? NaN, x: tok(idsOf(rt)[m.t ?? 0]), which: pick(m.letter === 1, L('the first letter', '第一个字母'), L('the first token', '第一个词元')), size: dims(rt).D * dims(rt).N };
				}
			},
			look: L('Δ, A, B, u and h_old are this step’s own numbers, rounded to 4 digits; every line after them is worked out from the line before, so a calculator gives the same. The last line is the new state of that one slot.', 'Δ、A、B、u 和 h_旧 都是这一步自己的数，取 4 位有效数字；之后的每一行都由上一行算出，所以用计算器算也一样。最后一行就是这个槽的新状态。'), terms: ['step-size'] },
		{ id: 'whole', source: 'scan', at: 'last', play: { from: 'first', to: 'last', interval: 450, unit: L('token', '词元') }, view: { key: 'tinymamba/scan', props: () => ({ rows: ['dch', 'h'], full: true }) }, provenance: 'live', title: L('The state over the whole row', '整行读下来的状态'),
			narration: {
				en: 'Follow channel {ch} (counting from 0) along your row. {dots} {lets} {end}',
				zh: '沿着你的这一行跟踪通道 {ch}（从 0 数起）。{dots}{lets}{end}',
				vars: (rt) => {
					const f = follow(rt);
					const r = row(rt);
					const kept = r.dots > 0 && r.k > 0 && r.tq > r.t1;
					return {
						ch: f.ch,
						dots: pick(r.dots > 0, { en: `Block 1 sees only the token itself, so every dot gets the same Δ, here ${dec(f.dotDelta)}, and no dot changes any of its slots by more than ${dec(f.dotMove)}.`, zh: `第 1 块只看词元本身，所以每个点的 Δ 都一样，这里是 ${dec(f.dotDelta)}；任何一个点让它的槽改变都不超过 ${dec(f.dotMove)}。` }, L('Your row has no dots.', '你的这一行没有点。')),
						lets: pick(r.k > 0, { en: `At the letters Δ reaches ${dec(f.letDelta)}, and a letter moves a slot by up to ${dec(f.letMove)}.`, zh: `在字母处，Δ 达到 ${dec(f.letDelta)}，一个字母让一个槽最多改变 ${dec(f.letMove)}。` }, L('Your row has no letters, so nothing is written.', '你的这一行没有字母，所以什么也没写入。')),
						end: pick(kept, L('So the dots wear the state down only slowly, and by the first question mark it still holds what the letters wrote.', '所以点只会让状态慢慢减弱，到第一个问号时，状态里仍然保存着字母写进去的内容。'), L('', ''))
					};
				}
			},
			look: L('Press play: the top row is Δ of this channel, the rows below are its slots, lit up as each token is read; the columns of the dots barely change colour.', '点播放：最上面一行是这个通道的 Δ，下面几行是它的槽，读到哪个词元就亮到哪里；点那几列的颜色几乎不变。'), terms: ['state-space-model', 'step-size'] },
		{ id: 'output', source: 'fwd', at: { layer: 'probs' }, view: { key: 'tinymamba/scan', props: () => ({ rows: ['want', 'got'], bars: true }) }, provenance: 'live', title: L('The output: the letters come back', '输出：字母回来了'),
			narration: {
				en: 'Each block turns its outputs y back into {d} numbers and adds them to its input (a residual connection). {how} A last weighted sum and softmax give each position chances for the {V} tokens. At each ? the model says “{said}”: {score}.',
				zh: '每个块把输出 y 变回 {d} 个数，再加到自己的输入上（残差连接）。{how}最后一个加权和与 softmax 给每个位置算出 {V} 个词元的概率。在各个 ? 处，模型说的是“{said}”：{score}。',
				vars: (rt) => {
					const r = row(rt);
					return {
						...dims(rt),
						said: said(rt),
						score: score(rt),
						how: pick(r.q > 1, L('Every ? is the same token, so block 1 has the same Δ, B and C at each; only its state changes, so its output differs, and block 2 reads its own state with a different C at each ?.', '每个 ? 都是同一个词元，所以第 1 块在每个 ? 处的 Δ、B、C 都一样；变的只有它的状态，所以它的输出各不相同，第 2 块也在每个 ? 处用不同的 C 读自己的状态。'), L('Block 2 runs a scan of its own on the result.', '第 2 块在结果上做它自己的扫描。'))
					};
				}
			},
			look: L('Under each ? : the letter it should give back, the model’s answer and its chance (✓ right, ✗ wrong).', '每个 ? 下面：应当找回的字母、模型的回答和它的概率（✓ 对，✗ 错）。'), terms: ['residual-connection', 'softmax'] },
		{ id: 'memory', source: 'fwd', at: { layer: 'B1.scan' }, view: { key: 'tinymamba/memory', props: (rt) => ({ ch: pickCh(rt, 1) }) }, provenance: 'live', title: L('A fixed state versus looking back', '固定的状态与回头看'),
			narration: {
				en: 'Attention, as in TinyGPT, keeps a key and a value of every token to look back at: at this model’s size ({d} numbers per token, {L} layers), {kv} numbers for your {toks} and {kv1000} for {big|int} tokens. This model keeps {state} numbers ({perBlock} per block), however long the row. Because each step only scales the state and adds to it, its output is a weighted sum of the u read so far: the matrix shows these look-back weights for channel {ch}. They are never stored; only the state is.',
				zh: '注意力（比如 TinyGPT 里的）为每个词元保存一个键和一个值，好回头去看：按这个模型的大小（每个词元 {d} 个数、{L} 层），你的 {toks}要 {kv} 个数，{big|int} 个词元要 {kv1000} 个。这个模型不管这一行多长，都只保留 {state} 个数（每块 {perBlock} 个）。因为每一步只是把状态缩放、再加上新的部分，它的输出就是到目前为止读过的 u 的加权和：矩阵显示的就是通道 {ch} 的这些回看权重。这些权重从不存下来，存下来的只有状态。',
				vars: (rt) => {
					const m = memory(rt);
					return { d: m.d, L: m.L, kv: m.kv, toks: plural(m.n, { one: '{n} token', many: '{n} tokens', zh: '{n} 个词元' }), kv1000: m.kv1000, big: 1000, state: m.state, perBlock: m.D * m.N, ch: pickCh(rt, 1) };
				}
			},
			look: L('Rows are outputs, columns are the tokens read so far; the empty upper half is the future, which no token can read. See which columns light up in the question marks’ rows.', '行是输出，列是到目前为止读过的词元；空着的右上半是未来，没有词元能读到。看看问号那几行里哪些列亮了。'), terms: ['attention', 'kv-cache'] },
		{ id: 'limits', source: 'fwd', at: { layer: 'probs' }, view: { key: 'tinymamba/scan', props: () => ({ rows: ['want', 'got'], tries: TRIES }) }, provenance: 'live', kindOf: 'limits', title: L('Limits: a fixed state can forget', '极限：固定的状态会遗忘'),
			narration: {
				en: 'Everything the model remembers must fit into {state} numbers, and its training rows had at most {fact:kMax} letters. On new rows with four letters it gives back {k4} of them; with five only {fact:accK5|pct}, with six {fact:accK6|pct}, with eight {fact:accK8|pct}. These numbers cannot separate the fixed size from training that never asked for more, but a fixed state never grows with the row, while attention keeps every letter to look back at. On your row: {score}.',
				zh: '模型记住的一切都必须装进 {state} 个数里，而它训练时见到的每一行最多只有 {fact:kMax} 个字母。在有四个字母的新行上，它能找回 {k4}；五个字母时只有 {fact:accK5|pct}，六个时 {fact:accK6|pct}，八个时 {fact:accK8|pct}。单凭这些数，分不清其中多少是因为状态大小固定、多少是因为训练从没要求过更多；但固定的状态不会随这一行变长而变大，注意力却为每个字母都留着记录，随时可以回头看。在你的这一行上：{score}。',
				vars: (rt) => ({ state: memory(rt).state, score: score(rt), k4: `${(100 * rt.num('fact:accK4')).toFixed(2)}%` })
			},
			look: L('Each ✗ marks a ? where the model gave back the wrong letter; the percentage under it is how sure the model was.', '每个 ✗ 标出一个模型找回了错误字母的 ?；下面的百分比是模型有多确定。'),
			task: {
				prompt: L('Make it forget: put more letters in your row than training ever had, with one ? per letter, until an answer turns ✗. The buttons under the row put in six or eight letters for you.', '让它遗忘：在你的这一行里放进比训练时更多的字母，每个字母配一个 ?，直到出现 ✗。这一行下面的按钮可以直接放进六个或八个字母。'),
				done: (rt) => row(rt).a.some((x) => x.want >= 0 && !x.ok),
				waiting: L('Not yet: every answer is still right. Add letters (and as many ?).', '还没有：每个回答都还是对的。再加字母（和同样多的 ?）。')
			},
			linger: 12000, terms: ['state-space-model'] }
	],
	fixtures: [
		{ id: 'late', label: L('a ? before the letters', '问号在字母前面'), raw: '?..b.c..??' },
		{ id: 'letters', label: L('letters only, no dots', '只有字母，没有点'), raw: 'abcd????' }
	]
};

export default lesson;
