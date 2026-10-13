// View 'positional-encodings/rope': RoPE on the first 2-D pair (dims 0 and 1) of head 0 of TinyGPT's layer-0 query (last
// token) and key (first token), without positions (process 'pe', phase 'rope'): each arrow before (dashed) and after it
// is turned to its position (query p + k, key p; pair 0 turns by 1 radian per position), with the pair's dot product and
// the whole head's.
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { arr, demoStage, stateOf } from '../demo.ts';

export function ropeOf(p: Pick<StageProps, 'at'>) {
	const s = stateOf(p);
	const m = p.at?.frame?.metrics ?? {};
	const [q, kv, qr, kr] = [arr(s.q), arr(s.kv), arr(s.qr), arr(s.kr)];
	return {
		ok: q.length >= 2,
		q: [q[0], q[1]],
		k: [kv[0], kv[1]],
		qr: [qr[0], qr[1]],
		kr: [kr[0], kr[1]],
		m: m.m ?? 0,
		n: m.n ?? 0,
		pairRaw: m.pairRaw ?? NaN,
		pair: m.pair ?? NaN,
		score: m.score ?? NaN,
		raw: m.raw ?? NaN
	};
}

export function describeRope(o: ReturnType<typeof ropeOf>): I18n {
	if (!o.ok) return { en: 'No query and key yet.', zh: '还没有查询和键。' };
	const v = (a: number[]) => `(${fmt3(a[0])}, ${fmt3(a[1])})`;
	return {
		en: `Pair 0 of head 0: the query ${v(o.q)} turned by ${o.m} radians to ${v(o.qr)} (position ${o.m}); the key ${v(o.k)} turned by ${o.n} radians to ${v(o.kr)} (position ${o.n}). The pair’s dot product goes from ${fmt3(o.pairRaw)} to ${fmt3(o.pair)}; over the whole head, from ${fmt3(o.raw)} to ${fmt3(o.score)}.`,
		zh: `头 0 的第 0 对：查询 ${v(o.q)} 转了 ${o.m} 弧度，变成 ${v(o.qr)}（位置 ${o.m}）；键 ${v(o.k)} 转了 ${o.n} 弧度，变成 ${v(o.kr)}（位置 ${o.n}）。这一对的点积从 ${fmt3(o.pairRaw)} 变成 ${fmt3(o.pair)}；整个头的点积从 ${fmt3(o.raw)} 变成 ${fmt3(o.score)}。`
	};
}

const meta = {
	key: 'positional-encodings/rope',
	title: { en: 'RoPE turns a query and a key', zh: 'RoPE 旋转查询和键' },
	kinds: ['process'],
	describe: (p: StageProps) => describeRope(ropeOf(p)),
	demo: () => demoStage('rope'),
	minHeight: 300
} satisfies ViewMeta;

export default meta;
