// Coded CLI errors (design §4.12). Keep the code table in sync with pipeline/atlas_kit/errors.py.
//
//   AK201  parity: layer "B1.attn.weights" max |diff| 3.2e-02 > 1e-4
//     where   case "house#2", head 3, row 5, col 2: js 0.412, torch 0.380
//     likely  PyTorch applied a mask the spec does not know (spec says causal=false)
//     fix     set causal at pipeline/tinyvit/export.py:41, or teach the op
//     docs    site/DEVKIT.md#ak201

export const CODES = {
	AK001: 'Node or npm',
	AK002: 'Python or Python packages',
	AK003: 'port busy',
	AK004: 'git refuses the repo',
	AK101: 'graph differs between cases',
	AK102: 'CUDA or TF32 during export',
	AK103: 'describe() differs from module(x)',
	AK104: 'recurrent gate reference differs from the module',
	AK105: 'budget at export',
	AK106: 'op attributes differ from ops.json',
	AK201: 'layer over the parity bar (kind A)',
	AK202: 'explain() differs from the trace',
	AK203: 'twin mismatch (kind B)',
	AK204: 'reference mismatch (kind C, first divergence)',
	AK205: 'near-tie divergence (warning)',
	AK301: 'schema',
	AK302: 'inferShape differs from outShape at a dim extreme',
	AK303: 'size budget',
	AK401: 'placeholder unresolved or non-finite',
	AK402: 'literal multi-digit number in narration (warning)',
	AK403: 'en/zh placeholder sets differ',
	AK404: 'cursor not found on a fixture',
	AK405: 'view not registered',
	AK406: 'missing zh',
	AK407: 'lesson shape',
	AK408: 'claim test failed',
	AK409: 'reviewed lesson made only of auto steps',
	AK410: 'two different numbers print the same (warning)',
	AK411: 'step term in no glossary',
	AK412: 'predict narration gives the answer away (warning)',
	AK501: 'TODO markers in the manifest',
	AK502: 'step without provenance',
	AK503: 'dependency rule or Math.random',
	AK504: 'empty describe()',
	AK507: 'view meta.ts imports a Svelte module or another package',
	AK601: 'console error',
	AK602: 'horizontal scroll at 360 px',
	AK603: 'long task',
	AK604: 'accessibility violation (axe: serious or critical)',
	AK605: 'lesson layout mode (app or flow) wrong for the window',
	AK606: 'loaded JS over budget (warning)',
	AK701: 'slug not in catalog',
	AK702: 'change outside owned paths',
	AK703: 'sync refused',
	AK704: 'credits missing'
} as const;
export type AkCode = keyof typeof CODES;

export interface AtlasErrorInfo {
	where?: string;
	likely?: string;
	fix?: string;
	docs?: string;
}

export class AtlasError extends Error {
	readonly code: AkCode;
	readonly info: AtlasErrorInfo;
	constructor(code: AkCode, message: string, info: AtlasErrorInfo = {}) {
		super(message);
		this.name = 'AtlasError';
		this.code = code;
		this.info = info;
	}
	format(): string {
		return formatIssue(this.code, this.message, this.info);
	}
}

export function formatIssue(code: string, message: string, info: AtlasErrorInfo = {}): string {
	const lines = [`${code}  ${message}`];
	const docs = info.docs ?? `site/DEVKIT.md#${code.toLowerCase()}`;
	for (const [label, value] of [
		['where', info.where],
		['likely', info.likely],
		['fix', info.fix],
		['docs', docs]
	] as const) {
		if (value) lines.push(`  ${label.padEnd(7)} ${value}`);
	}
	return lines.join('\n');
}

/** A Python AtlasError block printed on stderr ("AK102  export: ...") -> its code, or null. */
export function codeInText(text: string): AkCode | null {
	const m = /^(AK\d{3})\s{2}/m.exec(text);
	return m && m[1] in CODES ? (m[1] as AkCode) : null;
}
