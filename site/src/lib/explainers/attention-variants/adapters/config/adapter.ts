// attention-variants: local input adapter 'config'. The learner's input is one attention configuration for the KV-cache
// calculator ({ preset, layers, heads, kvHeads, dHead, tokens, bytes, latent }; data.ts configOf clamps it), set with
// preset buttons and number fields. The page reads it as a [7] tensor named 'input'. Pure TS (Capture.svelte is the widget).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { configOf, PRESETS, presetConfig, type Config } from '../../data.ts';

const adapter: InputAdapter<Config> = {
	id: 'config',
	kind: 'tabular',
	preprocess(raw): Preprocessed {
		const c = configOf(raw);
		const t = { shape: [7], data: Float64Array.of(c.layers, c.heads, c.kvHeads, c.dHead, c.tokens, c.bytes, c.latent) };
		return {
			tensors: { input: t },
			steps: [{ id: 'config', label: { en: 'Your configuration: layers, heads, KV heads, head width, tokens, bytes, latent', zh: '你的配置：层数、头数、KV 头数、头宽、词元数、字节数、潜向量宽度' }, tensor: t }],
			empty: false
		};
	},
	presets: () => PRESETS.map((p) => ({ id: p.id, label: p.label, raw: presetConfig(p) })),
	edgeFixtures: () => [
		{ id: 'one-token', label: { en: 'one token, one layer, one head', zh: '一个词元、一层、一个头' }, raw: { preset: 'custom', layers: 1, heads: 1, kvHeads: 1, dHead: 1, tokens: 1, bytes: 1, latent: 0 } },
		{ id: 'odd-kv', label: { en: '12 heads, 5 KV heads asked (snaps to 4)', zh: '12 个头，要求 5 个 KV 头（取最近的 4）' }, raw: { preset: 'custom', layers: 24, heads: 12, kvHeads: 5, dHead: 64, tokens: 100000, bytes: 1, latent: 0 } },
		{ id: 'huge', label: { en: 'every field at its largest', zh: '每一项都取最大值' }, raw: { preset: 'custom', layers: 1e9, heads: 1e9, kvHeads: 1e9, dHead: 1e9, tokens: 1e12, bytes: 4, latent: 0 } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'Your input is {n|int} numbers: layers, query heads, key/value heads, head width, tokens, bytes per number and the latent width (0 unless MLA).',
				zh: '你的输入是 {n|int} 个数：层数、查询头数、键值头数、头宽、词元数、每个数的字节数和潜向量宽度（不是 MLA 时为 0）。'
			}
		}
	}
};

export default adapter;
