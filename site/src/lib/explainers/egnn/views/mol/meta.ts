// egnn view 'mol': the molecule on the plane (atoms coloured by kind, numbered from 0); the drawing is View.svelte.
//   atoms    the atoms and a faint line for every pair (the full graph), with the first table rows
//   push     layer 1's push on the followed atom: one thin arrow per other atom, the thick arrow is their sum
//   forces   the network's force on every atom (solid) and the toy formula's (dashed), the two energies
//   spin     the molecule of the current 'spin' frame, turned, with the table of the turns so far
//   turn     the task: a slider turns your molecule (raw.turn); the force on the followed atom and the energy
//   mirror   your molecule and its mirror image, each with the network's forces and energy
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { energy, molOf, presetRaw } from '../../mol.ts';
import { focus, mol, variant } from '../../look.ts';

type P = Omit<StageProps, 'ctx'>;

export function describeMol(p: P): I18n {
	const m = mol(p.rt);
	if (!m) return { en: 'No atoms yet.', zh: '还没有原子。' };
	const mode = String(p.props.mode ?? 'atoms');
	const head = { en: `${m.n} atoms. The network's energy is ${fmt3(m.E)}; the formula gives ${fmt3(m.trueE)}.`, zh: `${m.n} 个原子。网络给出的能量是 ${fmt3(m.E)}；公式给出 ${fmt3(m.trueE)}。` };
	const i = focus(p.rt);
	let more = { en: '', zh: '' };
	if (mode === 'push' || mode === 'turn' || mode === 'forces') more = { en: ` The force on atom ${i} is (${fmt3(m.f[2 * i])}, ${fmt3(m.f[2 * i + 1])}).`, zh: ` 原子 ${i} 受的力是 (${fmt3(m.f[2 * i])}, ${fmt3(m.f[2 * i + 1])})。` };
	if (mode === 'mirror') {
		const v = variant(p.rt, 'mirror');
		if (v) more = { en: ` Mirrored, the energy is ${fmt3(v.E)}.`, zh: ` 镜像之后，能量是 ${fmt3(v.E)}。` };
	}
	if (mode === 'spin' && p.at.frame?.metrics) {
		const f = p.at.frame.metrics;
		more = { en: ` Turned by ${f.deg} degrees: energy ${fmt3(f.E)}.`, zh: ` 转了 ${f.deg} 度：能量 ${fmt3(f.E)}。` };
	}
	return { en: head.en + more.en, zh: head.zh + more.zh };
}

const meta = {
	key: 'egnn/mol',
	title: { en: 'The molecule on the plane', zh: '平面上的分子' },
	kinds: ['trace', 'process'],
	describe: describeMol,
	async demo() {
		const m = molOf(presetRaw('bent').raw);
		const pos = Float32Array.from(m.pos), e = energy(pos, m.kind);
		const kind = new Float32Array(3 * m.n);
		m.kind.forEach((k, i) => (kind[3 * i + k] = 1));
		const t = { pos: { shape: [m.n, 2], data: pos }, kind: { shape: [m.n, 3], data: kind }, energy: { shape: [1], data: Float32Array.of(e.E) }, force: { shape: [m.n, 2], data: Float32Array.from(e.F) } };
		const rt = fakeRuntime({ slug: 'egnn', input: presetRaw('bent').raw, trace: { order: Object.keys(t), t } });
		return { rt, source: 'fwd', at: atFor(rt, 'fwd', 0), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'forces' } };
	},
	minHeight: 380
} satisfies ViewMeta;

export default meta;
