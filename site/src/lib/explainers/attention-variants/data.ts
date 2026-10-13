// attention-variants: the KV-cache arithmetic (analytic) and the published model configurations of the calculator.
// A configuration: layers L, query heads H, key/value heads K (a divisor of H), head width d, tokens T, bytes per number b,
// and for multi-head latent attention (MLA) the cached latent width (0 otherwise). Per token and block the cache holds
//   MHA/GQA/MQA: one key and one value vector of d numbers per KV head = 2·K·d numbers;
//   MLA:         one latent vector plus one shared RoPE key = latent numbers.
// The whole cache: L · (that) · T · b bytes (one sequence). The presets are published configurations (credits in
// explainer.ts); TinyGPT's comes from its spec.json (static/models/tinygpt).
import type { I18n } from '../../atlas/types/index.ts';

export type Config = { preset: string; layers: number; heads: number; kvHeads: number; dHead: number; tokens: number; bytes: number; latent: number };
export type Preset = { id: string; label: I18n; cfg: Omit<Config, 'preset' | 'tokens'>; tokens: number; note: I18n };

/** Published configurations (layers, query heads, key/value heads, head width). Bytes: TinyGPT is served in fp32 on this
 *  site; the others at 2 bytes (fp16 / bf16), how they are usually run. */
export const PRESETS: Preset[] = [
	{ id: 'tinygpt', label: { en: 'TinyGPT', zh: 'TinyGPT' }, cfg: { layers: 4, heads: 4, kvHeads: 4, dHead: 19, bytes: 4, latent: 0 }, tokens: 64, note: { en: 'this site’s model, MHA', zh: '本站的模型，MHA' } },
	{ id: 'gpt2-small', label: { en: 'GPT-2 small', zh: 'GPT-2 small' }, cfg: { layers: 12, heads: 12, kvHeads: 12, dHead: 64, bytes: 2, latent: 0 }, tokens: 1024, note: { en: 'MHA (2019)', zh: 'MHA（2019）' } },
	{ id: 'llama2-7b', label: { en: 'Llama 2 7B', zh: 'Llama 2 7B' }, cfg: { layers: 32, heads: 32, kvHeads: 32, dHead: 128, bytes: 2, latent: 0 }, tokens: 4096, note: { en: 'MHA (2023)', zh: 'MHA（2023）' } },
	{ id: 'llama3-8b', label: { en: 'Llama 3 8B', zh: 'Llama 3 8B' }, cfg: { layers: 32, heads: 32, kvHeads: 8, dHead: 128, bytes: 2, latent: 0 }, tokens: 4096, note: { en: 'GQA, 8 KV heads (2024)', zh: 'GQA，8 个 KV 头（2024）' } },
	{ id: 'llama2-70b', label: { en: 'Llama 2 70B', zh: 'Llama 2 70B' }, cfg: { layers: 80, heads: 64, kvHeads: 8, dHead: 128, bytes: 2, latent: 0 }, tokens: 4096, note: { en: 'GQA, 8 KV heads (2023)', zh: 'GQA，8 个 KV 头（2023）' } },
	{ id: 'falcon-7b', label: { en: 'Falcon-7B', zh: 'Falcon-7B' }, cfg: { layers: 32, heads: 71, kvHeads: 1, dHead: 64, bytes: 2, latent: 0 }, tokens: 2048, note: { en: 'MQA (2023)', zh: 'MQA（2023）' } },
	{ id: 'deepseek-v2', label: { en: 'DeepSeek-V2', zh: 'DeepSeek-V2' }, cfg: { layers: 60, heads: 128, kvHeads: 128, dHead: 128, bytes: 2, latent: 576 }, tokens: 4096, note: { en: 'MLA: latent 512 + RoPE key 64 (2024)', zh: 'MLA：潜向量 512 + RoPE 键 64（2024）' } }
];
/** DeepSeek-V2's MLA widths: the compressed key/value latent d_c and the decoupled RoPE key d_h^R (one per token). */
export const MLA = { dc: 512, dr: 64 };

export const LIMITS = { layers: [1, 200], heads: [1, 256], dHead: [1, 512], tokens: [1, 1_048_576], latent: [0, 65_536] } as const;
export const BYTES = [4, 2, 1];
export const START: Config = { preset: 'tinygpt', ...PRESETS[0].cfg, tokens: PRESETS[0].tokens };

const int = (v: unknown, lo: number, hi: number, d: number) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, Math.round(v))) : d);
/** The divisors of n, ascending: the KV-head counts that split n query heads into equal groups. */
export const divisors = (n: number) => Array.from({ length: n }, (_, i) => i + 1).filter((k) => n % k === 0);
/** The divisor of n nearest to k (ties to the smaller). */
export const nearestDivisor = (n: number, k: number) => divisors(n).reduce((a, b) => (Math.abs(b - k) < Math.abs(a - k) ? b : a), 1);

/** The learner's raw input, clamped: whole numbers in range, K a divisor of H, b one of 4, 2, 1. */
export function configOf(raw: unknown): Config {
	const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<Config>;
	const heads = int(r.heads, LIMITS.heads[0], LIMITS.heads[1], START.heads);
	const bytes = BYTES.includes(r.bytes as number) ? (r.bytes as number) : START.bytes;
	return {
		preset: typeof r.preset === 'string' && PRESETS.some((p) => p.id === r.preset) ? r.preset : 'custom',
		layers: int(r.layers, LIMITS.layers[0], LIMITS.layers[1], START.layers),
		heads,
		kvHeads: nearestDivisor(heads, int(r.kvHeads, 1, heads, heads)),
		dHead: int(r.dHead, LIMITS.dHead[0], LIMITS.dHead[1], START.dHead),
		tokens: int(r.tokens, LIMITS.tokens[0], LIMITS.tokens[1], START.tokens),
		bytes,
		latent: int(r.latent, LIMITS.latent[0], LIMITS.latent[1], 0)
	};
}
export const presetConfig = (p: Preset): Config => ({ preset: p.id, ...p.cfg, tokens: p.tokens });

/** 0 MHA (one KV head per query head), 1 GQA, 2 MQA (one KV head), 3 MLA (a cached latent). */
export const kindOf = (c: Config) => (c.latent > 0 ? 3 : c.kvHeads === c.heads ? 0 : c.kvHeads === 1 ? 2 : 1);
export const KIND: I18n[] = [
	{ en: 'multi-head attention (MHA)', zh: '多头注意力（MHA）' },
	{ en: 'grouped-query attention (GQA)', zh: '分组查询注意力（GQA）' },
	{ en: 'multi-query attention (MQA)', zh: '多查询注意力（MQA）' },
	{ en: 'multi-head latent attention (MLA)', zh: '多头潜在注意力（MLA）' }
];
export const KIND_SHORT = ['MHA', 'GQA', 'MQA', 'MLA'];

/** Numbers cached per token and block. */
export const perBlock = (c: Config) => (c.latent > 0 ? c.latent : 2 * c.kvHeads * c.dHead);
/** Bytes cached per token (all blocks), and for the whole sequence. */
export const perToken = (c: Config) => c.layers * perBlock(c) * c.bytes;
export const total = (c: Config) => perToken(c) * c.tokens;
/** The same configuration with K key/value heads (and no latent). */
export const withKv = (c: Config, kvHeads: number): Config => ({ ...c, kvHeads, latent: 0 });
/** Multiply-adds of the attention scores for one new token in one block: every query head against every cached key. */
export const scoreMacs = (c: Config) => c.heads * c.tokens * c.dHead;

/** Bytes as text with binary prefixes (B = bytes, 1 KiB = 1,024 bytes), 3 significant digits: "608 B", "2.38 KiB", "2 GiB". */
export function bytesText(b: number): string {
	if (!Number.isFinite(b)) return '…';
	const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
	let u = 0;
	let v = b;
	while (v >= 1024 && u < units.length - 1) {
		v /= 1024;
		u++;
	}
	const s = u === 0 || Number.isInteger(v) ? String(Math.round(v * 1000) / 1000) : String(Number(v.toPrecision(3)));
	return `${s.replace(/\B(?=(\d{3})+(?!\d))/g, ',')} ${units[u]}`;
}
