// The tokens kit.tokens shows (pure TS; meta.ts uses it for describe()).
import { stageModel, stageTrace, tokenIds, tokenTexts, type Props } from '../kit.parts/stage.ts';

export interface Toks {
	items: { text: string; id?: number }[];
	hasIds: boolean;
	selected: number | null;
	marked: number[];
}

export function tokensOf(p: Props): Toks | null {
	const ids = tokenIds(stageTrace(p), stageModel(p));
	const fromProps = Array.isArray(p.props.tokens) ? p.props.tokens.length : 0;
	const preTokens = (p.rt.pre as { tokens?: unknown } | null)?.tokens;
	const fromPre = Array.isArray(preTokens) ? preTokens.length : 0;
	const n = ids?.length ?? (fromProps || fromPre);
	if (!n) return null;
	const texts = tokenTexts(p, n, ids);
	const items = texts.map((text, i) => ({ text, id: ids ? ids[i] : undefined }));
	const u = p.at.unit;
	const selected = typeof p.props.selected === 'number' ? p.props.selected : typeof p.sel.t === 'number' ? p.sel.t : u && u.unit.length === 2 ? u.unit[0] : null;
	const marked: number[] = [];
	for (const f of p.focus) if (Array.isArray(f.index) && typeof f.index[0] === 'number' && (f.field === 'tokens' || f.field === 'input')) marked.push(f.index[0] as number);
	return { items, hasIds: !!ids, selected: selected !== null && selected < n ? selected : null, marked };
}
