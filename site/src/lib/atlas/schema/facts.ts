// Hand-written validator for facts.json (design §4.2 `atlas-facts/1`), read by narration as {fact:<key>}.
//   { format: 'atlas-facts/1', facts: { <key>: { value, unit?, source, measured } } }
//   value: a finite number, a string, or bilingual { en, zh }; source: who measured it (script path);
//   measured: when (ISO date) or on what (a split, a run id).
//   validateFacts(raw) -> SchemaIssue[];  readFacts(raw) -> Record<key, value> (throws on errors)
import type { I18n } from '../types/index.ts';
import { Issues, checkI18n, errorsOf, formatIssues, isObj, isStr, type SchemaIssue } from './spec.ts';

const KEY = /^[A-Za-z][\w.-]*$/;

export function validateFacts(raw: unknown): SchemaIssue[] {
	const is = new Issues();
	if (!isObj(raw)) {
		is.err('', 'facts.json must be a JSON object');
		return is.list;
	}
	if (raw.format !== 'atlas-facts/1') is.err('format', "must be 'atlas-facts/1'");
	if (!isObj(raw.facts)) {
		is.err('facts', 'must map keys to { value, unit?, source, measured }');
		return is.list;
	}
	for (const [k, f] of Object.entries(raw.facts)) {
		const p = `facts.${k}`;
		if (!KEY.test(k)) is.err(p, 'keys start with a letter and use letters, digits, _ . -');
		if (!isObj(f)) {
			is.err(p, 'must be { value, unit?, source, measured }');
			continue;
		}
		const v = f.value;
		if (typeof v === 'number') {
			if (!Number.isFinite(v)) is.err(`${p}.value`, 'must be finite');
		} else if (isObj(v)) checkI18n(is, `${p}.value`, v);
		else if (!isStr(v)) is.err(`${p}.value`, 'must be a number, a string or { en, zh }');
		if (f.unit !== undefined && !isStr(f.unit)) is.err(`${p}.unit`, 'must be a string');
		if (!isStr(f.source) || !f.source) is.err(`${p}.source`, 'must say where the number comes from (script path)');
		if (!isStr(f.measured) || !f.measured) is.err(`${p}.measured`, 'must say when or on what it was measured');
	}
	return is.list;
}

/** The fact values by key; throws when the file is invalid. */
export function readFacts(raw: unknown): Record<string, number | string | I18n> {
	const errs = errorsOf(validateFacts(raw));
	if (errs.length) throw new Error(`facts.json is invalid:\n${formatIssues(errs)}`);
	const out: Record<string, number | string | I18n> = {};
	for (const [k, f] of Object.entries((raw as { facts: Record<string, { value: number | string | I18n }> }).facts)) out[k] = f.value;
	return out;
}
