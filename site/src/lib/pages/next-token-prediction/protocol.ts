/**
 * Types shared by the page, the client, the engine and the worker (no runtime code except keyOf),
 * so importing them never pulls the model code into the page bundle.
 */

/** 'final' = weights.bin (fp32); a number = the fp16 checkpoint saved at that training step. */
export type Which = 'final' | number;

export type TopTok = { id: number; p: number };

export type ScoreResult = {
	which: Which;
	ids: number[];
	/** p[i] = probability the model gave token i, given tokens 0..i-1 (NaN for i = 0) */
	p: number[];
	/** rank of token i among all 1,024 predictions at position i (1 = top guess; 0 for i = 0) */
	rank: number[];
	/** the model's top guesses for position i (empty for i = 0) */
	top: TopTok[][];
	/** entropy of the predicted distribution at position i, in bits (NaN for i = 0) */
	entropy: number[];
	meanNats: number;
	meanBits: number;
	perplexity: number;
	scored: number;
	windows: number;
	ms: number;
};

export type NextResult = { which: Which; logits: Float32Array; top: TopTok[]; contextStart: number; ms: number };

export type GenOptions = {
	maxNew?: number;
	temperature?: number;
	topK?: number;
	topP?: number;
	seed?: number;
	stopAtEot?: boolean;
	topN?: number;
};

export type GenStep = {
	index: number;
	token: number;
	/** probability in the distribution it was drawn from (after temperature / top-k / top-p) */
	p: number;
	/** probability under the plain model */
	pModel: number;
	top: TopTok[];
	u: number | null;
	contextStart: number;
};

export type GenResult = { which: Which; newIds: number[]; steps: GenStep[]; ms: number };

export type LoadStatus = {
	which: Which;
	state: 'loading' | 'ready' | 'error';
	loaded: number;
	total: number;
	message?: string;
};

export const keyOf = (w: Which) => (w === 'final' ? 'final' : `ckpt:${w}`);


// message protocol (worker <-> page)

export type Request =
	| { type: 'init'; baseUrl: string }
	| { type: 'load'; req: number; channel: string; which: Which }
	| { type: 'score'; req: number; channel: string; which: Which; ids: number[]; topN?: number }
	| { type: 'next'; req: number; channel: string; which: Which; ids: number[]; topN?: number }
	| { type: 'generate'; req: number; channel: string; which: Which; ids: number[]; opts: GenOptions }
	| { type: 'cancel'; channel: string };

export type Reply =
	| { type: 'status'; status: LoadStatus }
	| { type: 'result'; req: number; data: unknown }
	| { type: 'step'; req: number; data: GenStep }
	| { type: 'cancelled'; req: number }
	| { type: 'error'; req: number; message: string };
