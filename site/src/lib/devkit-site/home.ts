// The DevKit landing page (/devkit/): the three kinds with a live example lesson each, the workflow, the afternoon test
// and the quick start. Pure (Node-safe). Everything that can change with the kit is read from its sources at build time:
// the example lessons from the catalog, the commands from DEVKIT.md's generated command table, the quick start from the
// code blocks of DEVKIT.md section 4, the afternoon test from research/devkit/afternoon-test.md.
import type { I18nText } from './tutorial.ts';
import type { Guide } from './guide.ts';
import { generatedTable, subsection } from './guide.ts';
import { stripLocalOnly } from './local.ts';

export interface KindDef {
	kind: 'A' | 'B' | 'C';
	name: I18nText;
	what: I18nText;
	youWrite: I18nText;
	/** the live example lesson (catalog slug) */
	example: string;
}

export const KINDS: KindDef[] = [
	{
		kind: 'A',
		name: { en: 'A trained model', zh: '训练好的模型' },
		what: {
			en: 'A model trained in PyTorch runs on the learner’s own input, layer by layer, in the browser. Every layer matches PyTorch within the parity bar.',
			zh: '在 PyTorch 里训练好的模型，在浏览器里对学习者自己的输入逐层运行。每一层都与 PyTorch 一致，误差在阈值之内。'
		},
		youWrite: { en: 'model.py, train.py and export.py (PyTorch), then the lesson', zh: 'model.py、train.py 和 export.py（PyTorch），然后写课程' },
		example: 'tinyvit'
	},
	{
		kind: 'B',
		name: { en: 'Live training', zh: '实时训练' },
		what: {
			en: 'A small network learns in the browser while the learner watches: the loss, the gradients and one update with real numbers.',
			zh: '一个小网络在浏览器里学习，学习者在一旁看着：损失、梯度，以及一次带真实数字的更新。'
		},
		youWrite: { en: 'trainer.ts (the kit’s autograd), then the lesson', zh: 'trainer.ts（用工具包的自动求导），然后写课程' },
		example: 'mlp-playground'
	},
	{
		kind: 'C',
		name: { en: 'An algorithm, step by step', zh: '逐步运行的算法' },
		what: {
			en: 'A process runs one event at a time (a pull, an update, a merge), and every frame is recorded so the lesson can stop on any of them.',
			zh: '一个过程一次运行一个事件（拉一次、更新一次、合并一次），每一帧都被记录下来，课程可以停在任意一帧。'
		},
		youWrite: { en: 'process.ts (a TypeScript generator), then the lesson', zh: 'process.ts（一个 TypeScript 生成器），然后写课程' },
		example: 'gridworld'
	}
];

export interface WorkflowStep {
	id: string;
	name: I18nText;
	/** the atlas commands of this step (each must be in DEVKIT.md's command table) */
	cmds: string[];
	who: 'you' | 'maintainers' | 'reviewer';
	what: I18nText;
}

export const WORKFLOW: WorkflowStep[] = [
	{ id: 'scaffold', name: { en: 'Scaffold', zh: '脚手架' }, cmds: ['new'], who: 'you', what: { en: 'a working package of every kind you ask for', zh: '为你要的每种类型生成一个能运行的包' } },
	{ id: 'export', name: { en: 'Export or run', zh: '导出或运行' }, cmds: ['export', 'run'], who: 'you', what: { en: 'kind A: the model’s files and parity; kinds B and C: the tape in Node', zh: 'A 类：模型文件和一致性检查；B、C 类：在 Node 里打印磁带' } },
	{ id: 'lesson', name: { en: 'Lesson', zh: '课程' }, cmds: ['lesson'], who: 'you', what: { en: 'steps with live numbers, linted in both languages', zh: '带实时数字的步骤，两种语言都经过 lint' } },
	{ id: 'check', name: { en: 'Check', zh: '检查' }, cmds: ['check', 'e2e'], who: 'you', what: { en: 'every gate, a build and the browser test', zh: '所有关卡、一次构建和浏览器测试' } },
	{ id: 'review', name: { en: 'Review', zh: '评审' }, cmds: ['review'], who: 'reviewer', what: { en: 'a teaching review against a checklist', zh: '按检查清单做教学评审' } },
	{ id: 'promote', name: { en: 'Promote', zh: '上线' }, cmds: ['merge', 'promote'], who: 'maintainers', what: { en: 'credits and the catalog patch; the page goes live', zh: '致谢与目录补丁；页面上线' } }
];

/** The CLI table of DEVKIT.md section 12 (generated from the CLI by docs.ts): command -> { usage, summary }. */
export function commandTable(g: Guide): Record<string, { usage: string; summary: string }> {
	const sec = g.sections.find((s) => s.md.includes('<!-- atlas:generated commands -->'));
	const out: Record<string, { usage: string; summary: string }> = {};
	for (const r of sec ? generatedTable(sec.md, 'commands') : []) {
		const name = /`([^`]+)`/.exec(r[0])?.[1];
		if (name) out[name] = { usage: (/`([^`]+)`/.exec(r[1])?.[1] ?? '').replace(/\\\|/g, '|'), summary: r[2] ?? '' };
	}
	return out;
}

/** The port table of DEVKIT.md section 11: package or lane -> port. */
export function portTable(g: Guide): Record<string, number> {
	const sec = g.sections.find((s) => s.md.includes('<!-- atlas:generated ports -->'));
	const out: Record<string, number> = {};
	for (const r of sec ? generatedTable(sec.md, 'ports') : []) {
		const name = /`([^`]+)`/.exec(r[0])?.[1];
		const port = Number(r[1]);
		if (name && Number.isInteger(port)) out[name] = port;
	}
	return out;
}

/** The quick start: for each subsection of section 4, the commands it gives, in order: the lines of its first shell code
 *  block (later blocks are alternatives, such as `new model --like`, which the guide explains) and the `npm run ...`
 *  commands written inline in its text (such as kind A's `npm run atlas -- py .../train.py`). Lane commands
 *  (`atlas worktree`) are left out: lanes are the maintainers' tool (CONTRIBUTING.md). */
export function quickStart(g: Guide): { title: string; lines: string[] }[] {
	const s4 = g.sections.find((s) => s.n === 4);
	if (!s4) return [];
	const out: { title: string; lines: string[] }[] = [];
	for (const sub of s4.subs) {
		const md = subsection(s4, sub.text.split(' ')[0]);
		if (md === null) continue;
		const lines = [...new Set(sectionCommands(md))].filter((l) => !/^npm run atlas -- worktree\b/.test(l));
		if (lines.length) out.push({ title: sub.text, lines });
	}
	return out;
}

/** The commands of a piece of Markdown, in text order: the `npm run` lines of its first shell code block (with their
 *  comments) and every `npm run ...` code span of its text. */
export function sectionCommands(md: string): string[] {
	const out: string[] = [];
	let blocks = 0;
	const lines = md.replace(/\r\n/g, '\n').split('\n');
	for (let i = 0; i < lines.length; i++) {
		const f = /^\s*```\s*([\w-]*)/.exec(lines[i]);
		if (f) {
			const body: string[] = [];
			for (i++; i < lines.length && !/^\s*```/.test(lines[i]); i++) body.push(lines[i].trim());
			const cmds = body.filter((l) => l.startsWith('npm run'));
			if (cmds.length && (f[1] === '' || f[1] === 'sh') && blocks++ === 0) out.push(...cmds);
			continue;
		}
		for (const m of lines[i].matchAll(/`(npm run [^`]+)`/g)) out.push(m[1]);
	}
	return out;
}

/** From CONTRIBUTING.md: the status note (its first block quote, as Markdown without the "> "; null when there is none,
 *  e.g. once the repository is public) and the numbered list of its "## Setup" part. */
export function contributingStart(text: string | null): { status: string | null; setup: string | null } {
	if (!text) return { status: null, setup: null };
	const lines = stripLocalOnly(text).text.split('\n');
	const q: string[] = [];
	const at = lines.findIndex((l) => /^>\s?/.test(l));
	for (let i = at; at >= 0 && i < lines.length && /^>/.test(lines[i]); i++) q.push(lines[i].replace(/^>\s?/, ''));
	const from = lines.findIndex((l) => /^## Setup\b/.test(l));
	const to = from < 0 ? -1 : lines.findIndex((l, i) => i > from && /^## /.test(l));
	const setup = from < 0 ? '' : lines.slice(from + 1, to < 0 ? undefined : to).join('\n').trim();
	return { status: q.length ? q.join('\n').trim() : null, setup: setup || null };
}

export interface AfternoonResult {
	/** wall clock from the start of the reading to the draft's green gates */
	seconds: number;
	/** the numbered doc gaps the test found ("Where DEVKIT.md was unclear, wrong or missing") */
	gaps: number;
	/** what was built */
	built: string;
}

/** The afternoon test's result from its public record, research/devkit/afternoon-test.md (null when the file or its
 *  summary line "... to draft gates green: **<m> min <s> s** ..." is missing). The record keeps durations only, no clock
 *  times. */
export function afternoonResult(text: string | null): AfternoonResult | null {
	if (!text) return null;
	const m = /to draft gates green: \*\*(\d+) min (\d{1,2}) s\*\*/.exec(text);
	if (!m) return null;
	const gapsPart = /### Where DEVKIT\.md was unclear[^\n]*\n([\s\S]*?)(\n### |\n## |$)/.exec(text)?.[1] ?? '';
	const gaps = (gapsPart.match(/^\d+\.\s/gm) ?? []).length;
	const built = /^# Afternoon test: ([^\s(]+)/m.exec(text)?.[1] ?? '';
	return { seconds: Number(m[1]) * 60 + Number(m[2]), gaps, built };
}
