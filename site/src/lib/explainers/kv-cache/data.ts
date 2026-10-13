// kv-cache: the learner's two prompts and what the page does with them. Raw input: { text, ids, text2, ids2 }, the two
// texts and their token ids from TinyGPT's tokenizer (the Capture encodes them; the presets carry their ids). Prompt A is
// continued by TinyGPT; prompt B is the second request that may share A's beginning.
import type { I18n } from '../../atlas/types/index.ts';
import { EOT } from './gpt.ts';

export type PromptsRaw = { text?: string; ids?: number[]; text2?: string; ids2?: number[] } | null | undefined;

/** The page keeps the first PROMPT_MAX tokens of each prompt, so that NEW generated tokens fit TinyGPT's 64-token window. */
export const PROMPT_MAX = 40;
export const NEW = 24;
/** The break-it task: prompt B differs from A in its first token but has at least this many later tokens in A's places. */
export const TASK_SAME = 3;

const idsOf = (v: unknown): number[] => (Array.isArray(v) ? v.map(Number).filter((x) => Number.isInteger(x) && x >= 0 && x <= EOT) : []);

/** A prompt as the page runs it: its first PROMPT_MAX ids; an empty prompt starts from the end-of-text token. */
export function promptOf(ids: unknown): number[] {
	const a = idsOf(ids).slice(0, PROMPT_MAX);
	return a.length ? a : [EOT];
}
export const promptA = (raw: PromptsRaw) => promptOf(raw?.ids);
export const promptB = (raw: PromptsRaw) => promptOf(raw?.ids2);
/** How many ids the learner gave (before the cut to PROMPT_MAX). */
export const givenA = (raw: PromptsRaw) => idsOf(raw?.ids).length;
export const givenB = (raw: PromptsRaw) => idsOf(raw?.ids2).length;

/** The number of leading tokens two prompts share. */
export function sharedPrefix(a: number[], b: number[]): number {
	let s = 0;
	while (s < a.length && s < b.length && a[s] === b[s]) s++;
	return s;
}

/** Positions after the first difference where both prompts have the same token. */
export function samePlaces(a: number[], b: number[]): number[] {
	const s = sharedPrefix(a, b);
	const out: number[] = [];
	for (let j = s + 1; j < Math.min(a.length, b.length); j++) if (a[j] === b[j]) out.push(j);
	return out;
}

/** The break-it task is done: the first tokens differ and at least TASK_SAME later tokens sit in the same places. */
export const taskDone = (raw: PromptsRaw) => {
	const a = promptA(raw);
	const b = promptB(raw);
	return sharedPrefix(a, b) === 0 && samePlaces(a, b).length >= TASK_SAME;
};

type Preset = { id: string; label: I18n; raw: { text: string; ids: number[]; text2: string; ids2: number[] } };
/** Pairs of prompts in the style of TinyStories (written for this page; ids from TinyGPT's tokenizer, checked in claims). */
export const PRESETS: Preset[] = [
	{
		id: 'lily',
		label: { en: 'Lily (13 shared tokens)', zh: 'Lily（共享 13 个词元）' },
		raw: {
			text: 'Once upon a time, there was a little girl named Lily. She loved to',
			ids: [428, 437, 258, 397, 44, 400, 282, 258, 390, 474, 401, 404, 46, 338, 501, 265],
			text2: 'Once upon a time, there was a little girl named Lily. One day, she',
			ids2: [428, 437, 258, 397, 44, 400, 282, 258, 390, 474, 401, 404, 46, 526, 326, 44, 391]
		}
	},
	{
		id: 'beach',
		label: { en: 'The beach (16 shared tokens)', zh: '海滩（共享 16 个词元）' },
		raw: {
			text: 'The sun was hot. Mia and her mom went to the beach. Mia wanted to',
			ids: [409, 627, 282, 267, 299, 46, 643, 266, 335, 387, 427, 265, 262, 321, 589, 46, 643, 406, 265],
			text2: 'The sun was hot. Mia and her mom went to the beach. They saw a',
			ids2: [409, 627, 282, 267, 299, 46, 643, 266, 335, 387, 427, 265, 262, 321, 589, 46, 311, 381, 258]
		}
	},
	{
		id: 'tom',
		label: { en: 'Tom’s ball (7 shared tokens)', zh: 'Tom 的球（共享 7 个词元）' },
		raw: {
			text: 'Tom had a big red ball. He liked to',
			ids: [735, 363, 258, 345, 673, 477, 46, 315, 528, 265],
			text2: 'Tom had a big red ball. His dog',
			ids2: [735, 363, 258, 345, 673, 477, 46, 804, 463]
		}
	}
];
export const START = PRESETS[0].raw;
/** Prompt B of the start pair with its first word changed (Once → One): the break-it task done. */
export const ONE_UPON = { ...START, text2: 'One upon a time, there was a little girl named Lily. One day, she', ids2: [382, ...START.ids2.slice(1)] };
