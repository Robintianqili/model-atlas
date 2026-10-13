// Shared, pure-TS helpers for the kit views (W1-D2). Not a view: this folder has no meta.ts, so the registry and the
// view contract test skip it. Node-safe (type stripping, relative .ts imports), so meta.ts files can use it too.
//
// What a view gets is StageProps (types/view.ts). These helpers turn it into the things every zoom view needs:
//   stageTrace / stageModel / stageSpec   the trace, model and spec behind the stage's source (scan sources carry both)
//   stageLayerId                          the layer the stage is about: at.unit.layer, props.layer, at.layer, sel.layer
//   resolveUnit                           the UnitExplanation: at.unit, or model.explain() at props.unit / sel.unit /
//                                         the op's own pick (argmax, strongestChannel, …), or at a unit the view chose
//   unitWhere                             the unit's position in words, matching the op narration's numbering
//   scanDriven                            true when the host scrubs a scan tape (the view must not slide on its own)
import type { AtlasLayer, AtlasSpec, Axis, Dim, GraphModel, I18n, ResolvedOp, SourceState, Tensor, Trace, UnitExplanation, UnitPick } from '../../../types/index.ts';
import type { StageProps } from '../../../types/view.ts';
import { fmt3, fmtInt, visibleToken } from '../../../core/format.ts';
import { traceInputs } from '../../../graph/executor.ts';
import { mergeParams } from '../../../ops/registry.ts';

/** StageProps with an optional ctx (what ViewMeta.demo() returns, and what the gallery passes). */
export type Props = Omit<StageProps, 'ctx'> & { ctx?: StageProps['ctx'] };

export const parentId = (id: string): string => id.split('#')[0];
export const subOf = (id: string): string | null => (id.includes('#') ? id.slice(id.indexOf('#') + 1) : null);

export function sourceState(p: Props): SourceState | undefined {
	return p.rt.sources[p.source];
}

/** The trace of the stage's source; else props.trace (a source name); else the first trace-like source. */
export function stageTrace(p: Props): Trace | null {
	const own = sourceState(p);
	if (own?.trace) return own.trace;
	const named = typeof p.props.trace === 'string' ? p.rt.sources[p.props.trace]?.trace : null;
	if (named) return named;
	for (const s of Object.values(p.rt.sources)) if ((s.kind === 'trace' || s.kind === 'live-model' || s.kind === 'scan') && s.trace) return s.trace;
	return null;
}

/** The model that made the stage's trace (a live-model source has its own), else the runtime's. */
export function stageModel(p: Props): GraphModel | null {
	const own = sourceState(p);
	if (own?.model) return own.model;
	if (p.rt.model) return p.rt.model;
	for (const s of Object.values(p.rt.sources)) if (s.model) return s.model;
	return null;
}

export function stageSpec(p: Props): AtlasSpec | null {
	return stageModel(p)?.spec ?? p.rt.spec ?? null;
}

/** The layer (or sub-step) id the stage shows. */
export function stageLayerId(p: Props): string | undefined {
	const fromProps = typeof p.props.layer === 'string' ? p.props.layer : undefined;
	return p.at.unit?.layer ?? fromProps ?? p.at.layer ?? p.sel.layer;
}

export function layerOf(model: GraphModel | null, spec: AtlasSpec | null, id: string | undefined): AtlasLayer | undefined {
	if (!id) return undefined;
	const pid = parentId(id);
	try {
		if (model) return model.layer(pid);
	} catch {
		/* not a layer of this model: try the spec */
	}
	return spec?.layers.find((l) => l.id === pid);
}

export function opOf(model: GraphModel | null, layer: AtlasLayer | undefined): ResolvedOp | null {
	if (!model || !layer) return null;
	try {
		return model.ops.get(layer.op);
	} catch {
		return null;
	}
}

/** Axes of a layer output: declared, else guessed from the rank (3: c,y,x; 2: t,d; 1: f). */
export function axesOf(layer: AtlasLayer | undefined, shape: readonly Dim[]): Axis[] {
	if (layer?.axes && layer.axes.length === shape.length) return layer.axes.slice();
	if (shape.length === 3) return ['c', 'y', 'x'];
	if (shape.length === 2) return ['t', 'd'];
	if (shape.length === 1) return ['f'];
	return [];
}

/** Axes of a tensor id of the trace (a layer, a sub-step, or a declared input). */
export function axesOfId(model: GraphModel | null, id: string, shape: readonly Dim[]): Axis[] {
	const decl = model?.spec.inputs?.find((d) => d.name === id);
	if (decl?.axes && decl.axes.length === shape.length) return decl.axes.slice();
	const sub = subOf(id);
	if (sub && model) {
		try {
			const l = model.layer(parentId(id));
			const s = model.ops.get(l.op).def.subs?.find((x) => x.name === sub);
			if (s?.axes && s.axes.length === shape.length) return s.axes.slice();
		} catch {
			/* fall back to the rank */
		}
	}
	return axesOf(layerOf(model, null, id), shape);
}

/** True when the stage's source is a scan tape: the host's scrubber moves the unit, so the view must not slide itself. */
export function scanDriven(p: Props): boolean {
	return sourceState(p)?.kind === 'scan';
}

function clampUnit(shape: readonly number[], u: readonly number[]): number[] | null {
	if (u.length !== shape.length) return null;
	return u.map((v, i) => Math.max(0, Math.min(shape[i] - 1, Math.floor(v))));
}

export function unflat(shape: readonly number[], f: number): number[] {
	const idx = new Array<number>(shape.length);
	for (let k = shape.length - 1; k >= 0; k--) {
		idx[k] = f % shape[k];
		f = Math.floor(f / shape[k]);
	}
	return idx;
}

export function flat(shape: readonly number[], idx: readonly number[]): number {
	let f = 0;
	for (let k = 0; k < shape.length; k++) f = f * shape[k] + (idx[k] ?? 0);
	return f;
}

/** The unit an op's UnitPick starts on (ch: restrict a [C,H,W] pick to one channel). Rows [T, X]: the last row.
 *  Attention tables [H, Tq, Tk]: the last query's row of head `ch` (default the first), where the next-token
 *  prediction is made (the global argmax of a weights table is the trivial 1.0 of the first query). */
export function pickUnit(t: Tensor, pick: UnitPick | undefined, ch?: number, axes?: readonly Axis[]): number[] {
	const shape = t.shape;
	const d = t.data;
	if (pick && typeof pick === 'object') return clampUnit(shape, pick.unit) ?? unflat(shape, 0);
	let from = 0;
	let to = d.length;
	if (axes && axes.join() === 'h,tq,tk' && (pick === undefined || pick === 'argmax' || pick === 'argmaxAbs')) {
		const [H, Tq, Tk] = shape;
		const h = Math.max(0, Math.min(H - 1, ch ?? 0));
		from = (h * Tq + Tq - 1) * Tk;
		to = from + Tk;
	} else if (ch !== undefined && shape.length === 3) {
		const per = d.length / shape[0];
		from = Math.max(0, Math.min(shape[0] - 1, ch)) * per;
		to = from + per;
	} else if (shape.length === 2 && (pick === 'argmax' || pick === 'argmaxAbs' || pick === undefined)) {
		from = d.length - shape[1]; // the last row: the next-token prediction
	}
	const best = (score: (v: number) => number) => {
		let b = from;
		for (let i = from + 1; i < to; i++) if (score(d[i]) > score(d[b])) b = i;
		return unflat(shape, b);
	};
	switch (pick ?? 'argmax') {
		case 'first':
			return unflat(shape, from);
		case 'last':
			return unflat(shape, to - 1);
		case 'center':
			return shape.map((n, i) => (i === 0 && ch !== undefined && shape.length === 3 ? ch : Math.floor((n - 1) / 2)));
		case 'argmin':
			return best((v) => (Number.isFinite(v) ? -v : -Infinity));
		case 'argmaxAbs':
			return best((v) => (Number.isFinite(v) ? Math.abs(v) : -1));
		case 'strongestChannel': {
			if (shape.length !== 3 || ch !== undefined) return best((v) => v);
			const [C, H, W] = shape;
			let bc = 0;
			let bs = -Infinity;
			for (let c = 0; c < C; c++) {
				let s = 0;
				for (let i = 0; i < H * W; i++) s += d[c * H * W + i];
				if (s > bs) (bs = s), (bc = c);
			}
			from = bc * H * W;
			to = from + H * W;
			return best((v) => v);
		}
		default:
			return best((v) => (Number.isFinite(v) ? v : -Infinity));
	}
}

export interface UnitResult {
	e: UnitExplanation | null;
	/** the layer (parent of a sub-step id) */
	layer?: AtlasLayer;
	op: ResolvedOp | null;
	trace: Trace | null;
	model: GraphModel | null;
	/** the id that was explained (may be a sub-step 'lstm1#c') */
	id?: string;
	error?: string;
}

/** The UnitExplanation the stage shows. `override` is a unit the view itself picked (a hover on the output map). */
export function resolveUnit(p: Props, override?: readonly number[] | null): UnitResult {
	const trace = stageTrace(p);
	const model = stageModel(p);
	const id = stageLayerId(p);
	const layer = layerOf(model, stageSpec(p), id);
	const op = opOf(model, layer);
	const base: UnitResult = { e: null, layer, op, trace, model, id };
	if (!override && p.at.unit && (!id || p.at.unit.layer === id)) return { ...base, e: p.at.unit };
	if (!id || !trace || !model) return { ...base, e: p.at.unit ?? null, error: !id ? 'no layer selected' : 'no trace or model' };
	const out = trace.t[id];
	if (!out) return { ...base, error: `the trace has no '${id}'` };
	let unit: number[] | null = null;
	if (override) unit = clampUnit(out.shape, override);
	const asked = Array.isArray(p.props.unit) ? (p.props.unit as number[]) : p.sel.layer === id && Array.isArray(p.sel.unit) ? p.sel.unit : null;
	if (!unit && asked) unit = clampUnit(out.shape, asked);
	if (!unit) {
		const ch = typeof p.sel.ch === 'number' ? p.sel.ch : typeof p.sel.head === 'number' ? p.sel.head : undefined;
		unit = pickUnit(out, op?.def.unit?.pick, ch, axesOfId(model, id, out.shape));
	}
	try {
		return { ...base, e: model.explain(trace, id, unit) };
	} catch (err) {
		return { ...base, error: err instanceof Error ? err.message : String(err) };
	}
}

/** A tensor of the trace by id, or a declared input the executor kept aside (token ids are not in the trace). */
export function tensorOf(trace: Trace | null, id: string): Tensor | null {
	if (!trace) return null;
	const t = trace.t[id];
	if (t) return t;
	const ins = traceInputs(trace)?.[id];
	return ins ? { shape: ins.shape.slice(), data: ins.data instanceof Int32Array ? Float64Array.from(ins.data) : ins.data } : null;
}

/** Token ids of the model's i32 input, if it has one. */
export function tokenIds(trace: Trace | null, model: GraphModel | null): ArrayLike<number> | null {
	const decl = model?.spec.inputs?.find((d) => d.dtype === 'i32');
	if (!decl || !trace) return null;
	return tensorOf(trace, decl.name)?.data ?? null;
}

/** The text of one display name: a string, an I18n (in `lang`) or a { text } token. null for anything else. */
export function nameText(x: unknown, lang: 'en' | 'zh' = 'en'): string | null {
	if (typeof x === 'string') return x;
	if (x && typeof x === 'object') {
		const o = x as { text?: unknown; en?: unknown; zh?: unknown };
		if (typeof o.text === 'string') return o.text;
		if (typeof o.en === 'string') return lang === 'zh' && typeof o.zh === 'string' ? o.zh : o.en;
	}
	return null;
}

function nameList(v: unknown, lang: 'en' | 'zh'): string[] | null {
	if (!Array.isArray(v)) return null;
	const s = v.map((x) => nameText(x, lang));
	return s.every((x) => x !== null) ? (s as string[]) : null;
}

/** Display texts of the T tokens: props.tokens (strings, I18n or { text }; at least T of them), then rt.pre.names
 *  (bilingual names of the input positions, K4) or rt.pre.tokens (an adapter extra), then the spec's rowNames (an export
 *  extra for models without text: a ViT's [CLS], p0 … p15; K0), else '#id' (or 0..T-1). K4: the runtime's lists name the
 *  rows only of a layer just as long (TinyViT's 17 rows no longer took the text's tokens; a decoder layer no longer takes
 *  the encoder input's characters); `lang` picks the language of I18n names (views pass the page's language). */
export function tokenTexts(p: Props, T: number, ids?: ArrayLike<number> | null, lang: 'en' | 'zh' = 'en'): string[] {
	const pre = p.rt.pre as { tokens?: unknown; names?: unknown } | null;
	const own = nameList(p.props.tokens, lang);
	const list = own && own.length >= T ? own : [nameList(pre?.names, lang), nameList(pre?.tokens, lang), nameList((p.rt.spec as { rowNames?: unknown } | null)?.rowNames, lang)].find((l) => l && l.length === T);
	if (list) return list.slice(0, T).map(visibleToken);
	return Array.from({ length: T }, (_, i) => (ids && i < ids.length ? `#${ids[i]}` : `${i}`));
}

/** K4: the names of an attention layer's keys: props.keys (a second sequence: an encoder-decoder's cross-attention reads
 *  the encoder's positions), else the tokens (tokenTexts). */
export function keyTexts(p: Props, Tk: number, ids?: ArrayLike<number> | null, lang: 'en' | 'zh' = 'en'): string[] {
	const own = nameList(p.props.keys, lang);
	if (own && own.length >= Tk) return own.slice(0, Tk).map(visibleToken);
	return tokenTexts(p, Tk, ids, lang);
}

/** K4: words for axes the kit has no name for, from the step's props: axisWords ({ k: { en: 'mode', zh: '模式' } }),
 *  axisItems (a name per index: { xy: [{ en: 'real part', zh: '实部' }, { en: 'imaginary part', zh: '虚部' }] }) and keys
 *  (the key names of a cross-attention layer). unitWhere and the zoom headers read them. */
export interface AxisNames {
	words?: Record<string, I18n>;
	items?: Record<string, readonly (string | I18n)[]>;
	keys?: readonly (string | I18n)[];
}
export function axisNames(p: Pick<Props, 'props'>): AxisNames {
	const o: AxisNames = {};
	const w = p.props.axisWords;
	if (w && typeof w === 'object') o.words = w as Record<string, I18n>;
	const it = p.props.axisItems;
	if (it && typeof it === 'object') o.items = it as Record<string, (string | I18n)[]>;
	if (Array.isArray(p.props.keys)) o.keys = p.props.keys as (string | I18n)[];
	return o;
}

/** Text for a vocabulary id: props.vocab, else rt.pre.vocab (an adapter extra; string[] or (id) => string), else '#id'. */
export function vocabText(p: Props, id: number): string {
	const v = p.props.vocab ?? (p.rt.pre as { vocab?: unknown } | null)?.vocab;
	if (typeof v === 'function') {
		const s = (v as (i: number) => unknown)(id);
		if (typeof s === 'string') return s;
	}
	if (Array.isArray(v) && typeof v[id] === 'string') return v[id] as string;
	return `#${id}`;
}

const L = (en: string, zh: string): I18n => ({ en, zh });

/** Labels for class / vocab outputs: spec.classes (strings, { en, zh } or { id, en, zh }), else props.vocab, else none. */
export function outLabels(p: Props, model: GraphModel | null): ((i: number) => I18n | string) | undefined {
	const list = (model?.spec.classes ?? model?.spec.labels ?? p.rt.spec?.classes) as unknown;
	if (Array.isArray(list) && list.length) {
		return (i: number) => {
			const v = list[i];
			if (typeof v === 'string') return v;
			if (v && typeof v === 'object' && typeof (v as I18n).en === 'string') return { en: (v as I18n).en, zh: (v as I18n).zh };
			if (v && typeof v === 'object' && typeof (v as { id?: unknown }).id === 'string') return (v as { id: string }).id;
			return `#${i}`;
		};
	}
	if (p.props.vocab || (p.rt.pre as { vocab?: unknown } | null)?.vocab) return (i: number) => vocabText(p, i);
	return undefined;
}

/** The unit's position in words. One index base everywhere (R0): channels, heads, rows, columns, positions, queries,
 *  keys and features count from 0, as PyTorch, the kit TensorGrid readout ("Row 3, column 4") and the TileStack labels
 *  do, and the phrase ends with "(counting from 0)" when it shows an index. K4: `names` (axisNames(props)) gives words for
 *  axes the kit cannot name (`k`: a mode, an expert, a point), a name per index (`xy`: the real and the imaginary part)
 *  and the key names of a cross-attention layer (an `xy` axis means coordinates in egnn and a real/imaginary pair in fno,
 *  so it has no default names). */
export function unitWhere(unit: readonly number[], axes: readonly Axis[], tokens?: readonly string[], labels?: (i: number) => I18n | string, names: AxisNames = {}): I18n {
	const en: string[] = [];
	const zh: string[] = [];
	let indexed = false;
	axes.forEach((a, i) => {
		const v = unit[i];
		if (v === undefined) return;
		const item = names.items?.[a]?.[v];
		if (item !== undefined) {
			en.push(nameText(item, 'en') ?? String(v));
			zh.push(nameText(item, 'zh') ?? String(v));
			return;
		}
		const word = names.words?.[a];
		if (word) {
			en.push(`${word.en} ${v}`);
			zh.push(`${word.zh ?? word.en} ${v}`);
			indexed = true;
			return;
		}
		switch (a) {
			case 'c':
				en.push(`channel ${v}`);
				zh.push(`通道 ${v}`);
				break;
			case 'y':
				en.push(`row ${v}`);
				zh.push(`第 ${v} 行`);
				break;
			case 'x':
				en.push(`column ${v}`);
				zh.push(`第 ${v} 列`);
				break;
			case 'h':
				en.push(`head ${v}`);
				zh.push(`头 ${v}`);
				break;
			case 't':
			case 'tq': {
				const tok = tokens?.[v];
				const w = a === 'tq' ? ['query', '查询'] : ['position', '位置'];
				en.push(`${w[0]} ${v}${tok ? ` “${tok}”` : ''}`);
				zh.push(`${w[1]} ${v}${tok ? `“${tok}”` : ''}`);
				break;
			}
			case 'tk': {
				const ke = names.keys ? nameText(names.keys[v], 'en') : tokens?.[v];
				const kz = names.keys ? nameText(names.keys[v], 'zh') : tokens?.[v];
				en.push(`key ${v}${ke ? ` “${ke}”` : ''}`);
				zh.push(`键 ${v}${kz ? `“${kz}”` : ''}`);
				break;
			}
			case 'd':
			case 'f':
				en.push(`feature ${v}`);
				zh.push(`特征 ${v}`);
				break;
			// K1: graph nodes and batch items
			case 'n':
			case 'nk':
				en.push(`node ${v}`);
				zh.push(`节点 ${v}`);
				break;
			case 'b':
				en.push(`item ${v}`);
				zh.push(`第 ${v} 项`);
				break;
			case 'p':
				en.push(`patch ${v}`);
				zh.push(`图块 ${v}`);
				break;
			case 'vocab':
			case 'class': {
				const lab = labels?.(v);
				const le = lab === undefined ? String(v) : typeof lab === 'string' ? `“${lab}”` : `“${lab.en}”`;
				const lz = lab === undefined ? String(v) : typeof lab === 'string' ? `“${lab}”` : `“${lab.zh ?? lab.en}”`;
				en.push(`${a === 'vocab' ? 'token' : 'class'} ${le}`);
				zh.push(`${a === 'vocab' ? '词元' : '类别'}${lz}`);
				break;
			}
			default:
				en.push(`index ${v}`);
				zh.push(`索引 ${v}`);
		}
		if (a !== 'class' && a !== 'vocab') indexed = true;
	});
	const note = indexed ? L(' (counting from 0)', '（从 0 数起）') : L('', '');
	return en.length ? L(en.join(', ') + note.en, zh.join('，') + note.zh!) : L(`unit [${unit.join(', ')}] (counting from 0)`, `单元 [${unit.join(', ')}]（从 0 数起）`);
}

/** K4: rows of the most square grid that holds exactly n cells (12 -> 3 x 4, 3 -> 1 x 3, 7 -> 1 x 7): no empty cell. */
export function gridRows(n: number): number {
	if (!(n > 0)) return 0;
	for (let r = Math.floor(Math.sqrt(n)); r > 1; r--) if (n % r === 0) return r;
	return 1;
}

/** "Recomputed here … stored by the forward pass …": the self-check every zoom view prints. */
export function checkText(e: UnitExplanation): I18n {
	const same = e.result === e.traced || (Number.isNaN(e.result) && Number.isNaN(e.traced));
	const diff = same ? '0' : fmt3(Math.abs(e.result - e.traced));
	return L(
		`Recomputed here: ${fmt3(e.result)}. Stored by the forward pass: ${fmt3(e.traced)} (difference ${diff}).`,
		`这里重新算出：${fmt3(e.result)}。前向传播存下的值：${fmt3(e.traced)}（差 ${diff}）。`
	);
}

/** Largest finite |v| over arrays; `floor` when there is none. */
export function maxAbs(floor: number, ...arrays: (ArrayLike<number> | null | undefined)[]): number {
	let m = 0;
	for (const a of arrays) {
		if (!a) continue;
		for (let i = 0; i < a.length; i++) {
			const v = Math.abs(a[i]);
			if (Number.isFinite(v) && v > m) m = v;
		}
	}
	return m || floor;
}

/** Diverging [-m, m] or sequential [min(0, lo), hi] over finite values (−∞ masks are ignored). */
export function domainOf(kind: 'diverging' | 'sequential', ...arrays: (ArrayLike<number> | null | undefined)[]): [number, number] {
	if (kind === 'diverging') {
		const m = maxAbs(1, ...arrays);
		return [-m, m];
	}
	let lo = 0;
	let hi = 0;
	for (const a of arrays) {
		if (!a) continue;
		for (let i = 0; i < a.length; i++) {
			const v = a[i];
			if (!Number.isFinite(v)) continue;
			if (v < lo) lo = v;
			if (v > hi) hi = v;
		}
	}
	return [lo, hi > lo ? hi : lo + 1];
}

/** A label for a layer id: the spec label, or the id itself. */
export function layerLabel(model: GraphModel | null, id: string): I18n {
	const l = layerOf(model, null, id);
	const sub = subOf(id);
	if (!l) {
		const decl = model?.spec.inputs?.find((d) => d.name === id);
		return decl ? L(`input ${id}`, `输入 ${id}`) : L(id, id);
	}
	if (!sub) return l.label;
	return L(`${l.label.en} · ${sub}`, `${l.label.zh ?? l.label.en} · ${sub}`);
}

export { fmt3, fmtInt };
export const text = L;

/** The parameters a layer's op runs with (attr defaults, alias params, layer params). */
export function opParams(r: { op: ResolvedOp | null; layer?: AtlasLayer }): Record<string, unknown> {
	if (!r.op) return {};
	return mergeParams(r.op.def, r.op.params, r.layer?.params);
}

/** y = f(x) on a grid of n points from x0 to x1, computed by the layer's own op forward() (elementwise ops only). */
export function opCurve(r: { op: ResolvedOp | null; layer?: AtlasLayer }, x0: number, x1: number, n = 161): { x: Float64Array; y: Float64Array } | null {
	if (!r.op || !r.layer) return null;
	const x = new Float64Array(n);
	for (let i = 0; i < n; i++) x[i] = x0 + ((x1 - x0) * i) / (n - 1);
	const layer = r.layer;
	try {
		const out = r.op.def.forward([{ shape: [n], data: x }], opParams(r), {
			layer,
			dims: {},
			w: (slot: string) => {
				const e = new Error(`curve: ${layer.op} has no weights (${slot})`);
				e.name = 'MissingWeightError';
				throw e;
			},
			emit: () => {}
		});
		if (out.data.length !== n) return null;
		return { x, y: Float64Array.from(out.data) };
	} catch {
		return null;
	}
}

/** The slice of a tensor a unit belongs to: its channel ([C,H,W]), its row ([T,X]) or everything ([N]).
 *  Returns the flat offset, the length and the unit of slice element i. */
export function unitSlice(shape: readonly number[], unit: readonly number[]): { from: number; n: number; unitAt: (i: number) => number[]; at: number } {
	if (shape.length >= 2) {
		const per = shape.slice(1).reduce((a, b) => a * b, 1);
		const lead = unit[0] ?? 0;
		const rest = shape.slice(1);
		return { from: lead * per, n: per, unitAt: (i) => [lead, ...unflat(rest, i)], at: flat(rest, unit.slice(1)) };
	}
	return { from: 0, n: shape[0] ?? 0, unitAt: (i) => [i], at: unit[0] ?? 0 };
}
