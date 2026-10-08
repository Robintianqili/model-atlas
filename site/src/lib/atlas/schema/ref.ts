// Hand-written validator for reference.json (design §4.2 `atlas-ref-1`; the two v1 layouts are accepted).
//   validateRef(raw, { spec? }) -> SchemaIssue[]
// atlas-ref-1: { format: 'atlas-ref-1', model, version, against, note, cases: [{ id, label, edge, inputs: { <name>: { shape,
//   dtype: 'float32'|'int32', b64 } }, order, tensors: { <id or id#sub>: { shape, dtype: 'float32', b64 } } }] }
// With a spec: every tensor id is a layer id or `${layer}#${sub}` of an existing layer, and every input is declared.
import type { AtlasSpec } from '../types/index.ts';
import { Issues, b64Bytes, isInt, isObj, isStr, type SchemaIssue } from './spec.ts';
import { parseReference } from '../graph/refs.ts';

function checkB64Tensor(is: Issues, path: string, t: unknown, dtypes: string[]): void {
	if (!isObj(t)) return is.err(path, 'must be { shape, dtype, b64 }');
	if (!Array.isArray(t.shape) || !t.shape.every((d) => isInt(d) && d >= 0)) return is.err(`${path}.shape`, 'must be non-negative integers');
	const dtype = t.dtype ?? 'float32';
	if (!isStr(dtype) || !dtypes.includes(dtype)) return is.err(`${path}.dtype`, `must be ${dtypes.join(' or ')}`);
	if (!isStr(t.b64)) return is.err(`${path}.b64`, 'must be a base64 string');
	const bytes = b64Bytes(t.b64);
	const n = (t.shape as number[]).reduce((a, b) => a * b, 1);
	const size = dtype === 'float64' ? 8 : 4;
	if (bytes < 0) is.err(`${path}.b64`, 'is not valid base64');
	else if (bytes !== n * size) is.err(`${path}.b64`, `holds ${bytes} bytes, shape [${t.shape}] needs ${n * size}`);
}

export function validateRef(raw: unknown, o: { spec?: AtlasSpec } = {}): SchemaIssue[] {
	const is = new Issues();
	if (!isObj(raw)) {
		is.err('', 'reference must be a JSON object');
		return is.list;
	}
	const layerIds = new Set(o.spec?.layers.map((l) => l.id) ?? []);
	const inputNames = new Set(o.spec?.inputs?.map((i) => i.name) ?? []);
	const idOk = (id: string) => {
		if (!o.spec) return true;
		const [layer, sub, extra] = id.split('#');
		return extra === undefined && layerIds.has(layer) && (sub === undefined || sub.length > 0);
	};
	if (raw.format !== 'atlas-ref-1') {
		// v1 layouts: structural check by decoding
		try {
			const cases = parseReference(raw);
			if (!cases.length) is.err('', 'reference has no cases');
			cases.forEach((c, i) => {
				for (const id of c.order) if (!idOk(id)) is.err(`cases[${i}].order`, `'${id}' is not a layer of the spec`);
			});
		} catch (e) {
			is.err('', (e as Error).message);
		}
		return is.list;
	}
	for (const k of ['model', 'version', 'against'] as const) if (!isStr(raw[k])) is.err(k, 'must be a string');
	if (!Array.isArray(raw.cases) || raw.cases.length === 0) {
		is.err('cases', 'must be a non-empty array');
		return is.list;
	}
	const ids = new Set<string>();
	raw.cases.forEach((c, i) => {
		const p = `cases[${i}]`;
		if (!isObj(c)) return is.err(p, 'must be an object');
		if (!isStr(c.id) || !c.id) is.err(`${p}.id`, 'must be a non-empty string');
		else if (ids.has(c.id)) is.err(`${p}.id`, `duplicate case id '${c.id}'`);
		else ids.add(c.id);
		if (c.label !== undefined && !isStr(c.label)) is.err(`${p}.label`, 'must be a string');
		if (c.edge !== undefined && typeof c.edge !== 'boolean') is.err(`${p}.edge`, 'must be a boolean');
		if (!isObj(c.inputs) || !Object.keys(c.inputs).length) is.err(`${p}.inputs`, 'must map input names to tensors');
		else
			for (const [k, v] of Object.entries(c.inputs)) {
				checkB64Tensor(is, `${p}.inputs.${k}`, v, ['float32', 'int32']);
				if (o.spec && inputNames.size && !inputNames.has(k)) is.err(`${p}.inputs.${k}`, `'${k}' is not a declared input of the spec`);
			}
		if (!isObj(c.tensors) || !Object.keys(c.tensors).length) return is.err(`${p}.tensors`, 'must map ids to tensors');
		for (const [k, v] of Object.entries(c.tensors)) {
			checkB64Tensor(is, `${p}.tensors.${k}`, v, ['float32']);
			if (!idOk(k)) is.err(`${p}.tensors.${k}`, `'${k}' is not a layer id or '<layer>#<sub>' of the spec`);
		}
		if (c.order !== undefined) {
			if (!Array.isArray(c.order) || !c.order.every(isStr)) is.err(`${p}.order`, 'must be an array of ids');
			else for (const id of c.order) if (!(id in c.tensors)) is.err(`${p}.order`, `'${id}' has no tensor`);
		}
	});
	return is.list;
}
