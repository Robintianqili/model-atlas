/**
 * The real tokenizers shown side by side. Files are self-hosted in static/pages/tokenization/tok/
 * (made by pipeline/tokenizers/export.py); sizes are the exact file sizes in bytes.
 * Colors are categorical slots, the same in every view of the page (always shown with the name).
 */
import type { I18n } from '../../i18n/types.ts';

export type RealId = 'gpt2' | 'bert' | 'cl100k' | 'qwen';
export type RowId = 'yours' | 'tinygpt' | RealId;

export type RealSpec = {
	id: RealId;
	name: string;
	usedBy: I18n;
	algo: 'bpe' | 'wordpiece';
	file: string;
	bytes: number;
	vocab: number;
	pattern?: 'gpt2' | 'cl100k' | 'qwen';
	nfc?: boolean;
	lowercase?: boolean;
	license: string;
	source: string;
	note: I18n;
};

export const REAL: RealSpec[] = [
	{
		id: 'gpt2',
		name: 'GPT-2',
		usedBy: { en: 'GPT-2 (2019), also GPT-3', zh: 'GPT-2（2019），GPT-3 也用它' },
		algo: 'bpe',
		file: 'gpt2.tokb',
		bytes: 371092,
		vocab: 50257,
		pattern: 'gpt2',
		license: 'MIT',
		source: 'https://huggingface.co/openai-community/gpt2',
		note: {
			en: 'Byte-level BPE, 50,000 merges trained on English web text (WebText).',
			zh: '字节级 BPE，在英文网页文本（WebText）上学到 50,000 次合并。'
		}
	},
	{
		id: 'bert',
		name: 'BERT (uncased)',
		usedBy: { en: 'bert-base-uncased (2018)', zh: 'bert-base-uncased（2018）' },
		algo: 'wordpiece',
		file: 'bert-base-uncased.txt',
		bytes: 231508,
		vocab: 30522,
		lowercase: true,
		license: 'Apache-2.0',
		source: 'https://huggingface.co/google-bert/bert-base-uncased',
		note: {
			en: 'WordPiece: lowercases, strips accents, splits on spaces and punctuation, then takes the longest piece in the vocabulary; later pieces start with ##. A word it cannot spell becomes [UNK].',
			zh: 'WordPiece：先转小写、去掉重音符号，按空格和标点切开，再每次取词表里最长的片段；非开头的片段以 ## 开头。拼不出来的词变成 [UNK]。'
		}
	},
	{
		id: 'cl100k',
		name: 'GPT-4 (cl100k_base)',
		usedBy: { en: 'GPT-4, GPT-3.5 (2023)', zh: 'GPT-4、GPT-3.5（2023）' },
		algo: 'bpe',
		file: 'cl100k_base.tokb',
		bytes: 744094,
		vocab: 100277,
		pattern: 'cl100k',
		license: 'MIT (tiktoken)',
		source: 'https://github.com/openai/tiktoken',
		note: {
			en: 'Byte-level BPE with about 100,000 tokens; digits are grouped in threes.',
			zh: '字节级 BPE，约 100,000 个词元；数字按三位一组切分。'
		}
	},
	{
		id: 'qwen',
		name: 'Qwen2.5',
		usedBy: { en: 'Qwen2 and Qwen2.5 (2024)', zh: 'Qwen2 与 Qwen2.5（2024）' },
		algo: 'bpe',
		file: 'qwen2.5.tokb',
		bytes: 1127620,
		vocab: 151665,
		pattern: 'qwen',
		nfc: true,
		license: 'Apache-2.0',
		source: 'https://huggingface.co/Qwen/Qwen2.5-0.5B',
		note: {
			en: 'Byte-level BPE with about 151,000 tokens and many Chinese words; every digit is its own token.',
			zh: '字节级 BPE，约 151,000 个词元，包含大量中文词；每个数字单独成一个词元。'
		}
	}
];

export const REAL_BY_ID = Object.fromEntries(REAL.map((r) => [r.id, r])) as Record<RealId, RealSpec>;
export const REAL_TOTAL_BYTES = REAL.reduce((s, r) => s + r.bytes, 0);

/** Categorical color slot of each row (the same in chips, bars, chart and table). */
export const ROW_COLOR: Record<RowId, number> = { yours: 0, tinygpt: 1, gpt2: 2, bert: 3, cl100k: 4, qwen: 7 };

export const ROW_NAME: Record<RowId, I18n> = {
	yours: { en: 'Your BPE', zh: '你的 BPE' },
	tinygpt: { en: 'TinyGPT', zh: 'TinyGPT' },
	gpt2: { en: 'GPT-2', zh: 'GPT-2' },
	bert: { en: 'BERT (uncased)', zh: 'BERT（uncased）' },
	cl100k: { en: 'GPT-4 (cl100k)', zh: 'GPT-4（cl100k）' },
	qwen: { en: 'Qwen2.5', zh: 'Qwen2.5' }
};

/** Parity result (pipeline/tokenizers/parity.ts), shown on the badges. */
export const PARITY = {
	strings: 578,
	note: {
		en: 'Token ids identical to the reference implementations (tiktoken 0.14 for GPT-2 and cl100k_base; Hugging Face tokenizers 0.23 for BERT and Qwen2.5) on 578 of 578 test strings, including Chinese, emoji and numbers.',
		zh: '在 578 条测试文本（含中文、表情符号和数字）上，词元 ID 与参考实现（GPT-2 和 cl100k_base 用 tiktoken 0.14；BERT 和 Qwen2.5 用 Hugging Face tokenizers 0.23）全部一致（578/578）。'
	} as I18n
};
