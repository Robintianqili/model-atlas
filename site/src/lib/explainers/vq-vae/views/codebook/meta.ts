// View 'vq-vae/codebook': the K codes, each drawn as the block it makes among empty patches (vq.ts tilesOf), with
// props.mode 'gallery' (default; the codes your drawing uses are outlined, with how many patches use each) or 'usage'
// (how many of the test patches snapped to each code, usage.json, most used first, next to your drawing's counts).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { codesOf, usageOf } from '../../vq.ts';

export function countsOf(codes: number[], K: number): number[] {
	const n = new Array(K).fill(0);
	for (const k of codes) n[k]++;
	return n;
}

export function describeCodebook(p: Omit<StageProps, 'ctx'>): I18n {
	const codes = codesOf(p.rt), u = usageOf(p.rt), K = u?.counts.length ?? 0;
	const mine = countsOf(codes, K).map((n, k) => (n ? `${k} ×${n}` : '')).filter(Boolean).join(', ');
	const yours = { en: codes.length ? ` Your drawing uses: ${mine}.` : '', zh: codes.length ? ` 你的画用到：${mine}。` : '' };
	if (p.props.mode === 'usage' && u) {
		const top = u.counts.map((n, k) => ({ k, s: n / u.patches })).sort((a, b) => b.s - a.s);
		const list = top.map((r) => `${r.k}: ${fmtPct(r.s)}`).join(', ');
		return { en: `Share of the test patches that snapped to each code, most used first: ${list}.${yours.en}`, zh: `测试图块吸附到每个码字的比例，从多到少：${list}。${yours.zh}` };
	}
	return { en: `The codebook: ${K} codes, each shown as the block it draws among empty patches.${yours.en}`, zh: `码本：${K} 个码字，每个都画成它放在空白图块中间时画出的小块。${yours.zh}` };
}

const meta = {
	key: 'vq-vae/codebook',
	title: { en: 'The codebook and how it is used', zh: '码本及其使用情况' },
	kinds: ['trace'],
	describe: describeCodebook,
	async demo() {
		// made-up data: 4 patches using codes 0, 2 and 2 again, and made-up test counts for 4 codes (no model: no pictures)
		const trace = { order: ['zq'], t: { 'zq#idx': { shape: [2, 2], data: Float32Array.from([0, 2, 2, 3]) } } };
		const rt = Object.assign(fakeRuntime({ slug: 'vq-vae', trace }), { assets: { 'usage.json': { counts: [50, 10, 30, 10], patches: 100, blank: 0 } } });
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'usage' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
