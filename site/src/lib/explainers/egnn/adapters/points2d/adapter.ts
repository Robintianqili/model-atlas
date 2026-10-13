// egnn: the kit's points2d adapter (its Capture.svelte is used as it is, with three classes: the atom kinds) with the
// page's own molecules (mol.ts) and the model's two inputs: pos [n, 2] (the painted points × SCALE, turned by raw.turn
// degrees and mirrored when raw.flip, float32) and kind [n, 3] (one-hot atom kinds). The raw value may carry turn and
// flip (the stage's turn slider and mirror switch), which the Capture keeps when it paints or erases.
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import kit from '../../../../atlas/adapters/points2d/adapter.ts';
import { K, MAX_ATOMS, PRESET_IDS, molOf, presetRaw, type MolRaw } from '../../mol.ts';

const adapter: InputAdapter<MolRaw> = {
	...kit,
	preprocess(raw): Preprocessed {
		const m = molOf(raw);
		const notes = m.n ? [] : [{ en: 'There are no atoms yet. Click to place some (the class is the atom kind), or pick a molecule.', zh: '还没有原子。点击放下几个原子（类别就是原子的种类），或者选一个分子。' }];
		if (m.more) notes.push({ en: `Only the first ${MAX_ATOMS} atoms are used.`, zh: `只用前 ${MAX_ATOMS} 个原子。` });
		const label = { en: 'Your atoms: x and y', zh: '你的原子：x 和 y' };
		if (!m.n) return { tensors: { xy: { shape: [0, 2], data: new Float32Array(0) } }, steps: [{ id: 'pos', label }], empty: true, notes };
		const kind = new Float32Array(m.n * K);
		m.kind.forEach((k, i) => (kind[i * K + k] = 1));
		return {
			tensors: { pos: { shape: [m.n, 2], data: Float32Array.from(m.pos) }, kind: { shape: [m.n, K], data: kind } },
			steps: [{ id: 'pos', label }, { id: 'kind', label: { en: 'Atom kinds, one-hot', zh: '原子种类（独热编码）' } }],
			empty: false,
			notes
		};
	},
	presets: () => PRESET_IDS.map((id) => ({ id, ...presetRaw(id) })),
	edgeFixtures: () => [
		{ id: 'empty', label: { en: 'No atoms', zh: '没有原子' }, raw: { points: [] } },
		{ id: 'one-atom', label: { en: 'One atom', zh: '一个原子' }, raw: { points: [{ x: 0.1, y: -0.1, c: 2 }] } },
		{ id: 'crowded', label: { en: 'Thirteen atoms in a tight spiral', zh: '十三个原子挤成一团螺旋' }, raw: { points: Array.from({ length: 13 }, (_, i) => ({ x: Math.round(750 * Math.cos(i * 2.4) * Math.sqrt((i + 0.5) / 13)) / 1000, y: Math.round(750 * Math.sin(i * 2.4) * Math.sqrt((i + 0.5) / 13)) / 1000, c: i % 3 })) } }
	],
	layer0: {
		view: 'egnn/mol',
		narration: {
			what: { en: 'Each atom is two numbers, x and y, so your molecule is a table of {shape} numbers; a second table says which kind each atom is.', zh: '每个原子是两个数 x 和 y，所以你的分子是一张 {shape} 的数表；另一张表说明每个原子是哪一类。' },
			how: { en: 'The points you place are scaled to molecule units; the stage may also turn or mirror them.', zh: '你放下的点会换算成分子单位；画面也可能把它们转动或镜像。' }
		}
	}
};

export default adapter;
