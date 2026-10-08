// cart-pole: local input adapter 'cart-pole' (InputKind 'game'). The learner's raw input is a CartRaw (sim.ts): the start
// state [x, ẋ, θ, θ̇] of the episode the lesson follows. The model reads it as a [4] tensor named 'input' (the state
// vector the policy network gets; the live-model steps trace it).
import type { I18n, InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { startOf, type CartRaw, type State } from '../../sim.ts';

const P = (id: string, en: string, zh: string, start: State) => ({ id, label: { en, zh } as I18n, raw: { start } as CartRaw });
/** Generated presets (no presets file): the default start first. */
export const PRESETS = [
	P('lean', 'Leaning a little right', '稍微向右倾斜', [0, 0, 0.05, 0]),
	P('upright', 'Upright and still', '竖直且静止', [0, 0, 0, 0]),
	P('falling', 'Leaning left and falling', '向左倾斜并在倒下', [0, 0, -0.1, -0.5]),
	P('edge', 'Near the right end, moving right', '靠近右端，正往右走', [1.5, 0.5, 0, 0])
];

const adapter: InputAdapter<CartRaw> = {
	id: 'cart-pole',
	kind: 'game',
	preprocess(raw): Preprocessed {
		const t = { shape: [4], data: Float32Array.from(startOf(raw)) };
		return { tensors: { input: t }, steps: [{ id: 'state', label: { en: 'Your start state [x, ẋ, θ, θ̇]', zh: '你的起始状态 [x, ẋ, θ, θ̇]' }, tensor: t }], empty: false };
	},
	presets: () => PRESETS,
	edgeFixtures: () => [
		P('steep', 'the steepest start, falling fast: over in one step', '最陡的起点，正在快速倒下：一步就结束', [0, 0, 0.2, 1]),
		P('far-left', 'the cart at the left end, moving left', '小车在最左端，正往左走', [-2, -1, 0, 0])
	],
	layer0: { view: 'layer.tensor', narration: { what: { en: 'Your start state is {n|int} numbers: x, ẋ, θ and θ̇.', zh: '你的起始状态是 {n|int} 个数：x、ẋ、θ 和 θ̇。' } } }
};

export default adapter;
