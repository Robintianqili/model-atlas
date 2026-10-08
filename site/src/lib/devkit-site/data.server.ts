// Build-time data of the DevKit pages (/devkit/...), for src/routes/devkit/[...path]/+page.server.ts. Server only: the
// pages are prerendered, so this runs once per build (and on every request of the dev server, which re-reads the files).
//
// Every page is generated from the kit's own sources (sources.ts): the guide from site/DEVKIT.md, the reference from
// schema/ops.json, the op definitions and the view metas, the tutorial from the hello-xor files, the contribute page from
// CONTRIBUTING.md, the design's checklist and the error list. Before anything is rendered, check.ts compares the few
// hand-made parts (Chinese summaries, code meanings, op demos, tutorial anchors) with those sources and throws when one
// is stale, which fails the build.
//
// Notes about the maintainers' own computer never reach a page: DEVKIT.md wraps them in <!-- local-only:start --> ...
// <!-- local-only:end --> (the renderer drops them, local.ts), and every page's data is checked for local-machine strings
// (local.ts: its generic patterns and this computer's own, machine.ts) before it is returned: a hit fails the build,
// naming the page and the string.
import katex from 'katex';
import { dev } from '$app/environment';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import type { I18n, ViewMeta } from '../atlas/types/index.ts';
import { CORE_OPS } from '../atlas/ops/index.ts';
import { formatValue, PLACEHOLDER_RE, type Lang } from '../atlas/lesson/narrate.ts';
import { record } from '../atlas/process/record.ts';
import { trainingProcess } from '../atlas/train/trainer.ts';
import { fmtBytes } from '../atlas/core/format.ts';
import { MODEL_BY_SLUG } from '../catalog/models.ts';
import helloManifest from '../explainers/hello-xor/explainer.ts';
import helloLesson from '../explainers/hello-xor/lesson.ts';
import * as helloTrainers from '../explainers/hello-xor/trainer.ts';
import * as helloProcesses from '../explainers/hello-xor/process.ts';
import { BASE, codeBlock, render } from './md.ts';
import { devkitPagePaths, firstTable, guideUrl, parseGuide, renderIntro, renderSection, subsection, type Guide, type GuideSection } from './guide.ts';
import { CONTRIBUTING_STATUS_ZH, CONTRIBUTING_ZH, SECTION_ZH } from './summaries.ts';
import { CODE_GROUPS, CODE_ZH, groupOf } from './codes.ts';
import { OP_DEMOS, opLayer, zoomKeyOf } from './opdemos.ts';
import { HELLO, TUTORIAL, excerpt, stepCommand, type TutorialStepDef } from './tutorial.ts';
import { KINDS, WORKFLOW, afternoonResult, commandTable, contributingStart, portTable, quickStart } from './home.ts';
import { readSources, repoFile, type Sources } from './sources.ts';
import { staleness } from './check.ts';
import { findLocalStrings, LOCAL_END, LOCAL_START } from './local.ts';
import type { ContributeData, GuideIndexData, HomeData, OpData, OpsData, PageData, Produce, ReferenceData, SectionData, SectionRef, T2, TutorialData, TutorialStep, ViewData, ViewsData } from './types.ts';

const kitMetas = import.meta.glob<{ default: ViewMeta }>('../atlas/ui/views/*/meta.ts', { eager: true });
const pkgMetas = import.meta.glob<{ default: ViewMeta }>('../explainers/*/views/*/meta.ts', { eager: true });

const t2 = (x: I18n | undefined | null): T2 => ({ en: x?.en ?? '', zh: x?.zh ?? x?.en ?? '' });

interface Ctx {
	s: Sources;
	g: Guide;
}
let memo: Ctx | null = null;

function ctx(): Ctx {
	if (memo && !dev) return memo;
	const s = readSources(process.cwd());
	const g = parseGuide(s.devkit);
	const problems = staleness(s, g);
	if (problems.length)
		throw new Error(`The DevKit pages (/devkit/) are out of date with their sources (src/lib/devkit-site/check.ts):\n${problems.map((p) => `  - ${p}`).join('\n')}`);
	memo = { s, g };
	return memo;
}

/** Every path under /devkit/ (the route's prerender entries). */
export function devkitPaths(): string[] {
	const { g } = ctx();
	return devkitPagePaths(g);
}

/** The data of one page, or null for an unknown path. Throws when the page would publish a note about the maintainers'
 *  own computer (local.ts). */
export async function devkitPage(path: string): Promise<PageData | null> {
	const p = path.replace(/^\/+|\/+$/g, '');
	const page = await pageData(ctx(), p);
	const hits = page ? findLocalStrings(JSON.stringify(page)) : [];
	if (hits.length)
		throw new Error(
			`The DevKit page /devkit/${p}${p ? '/' : ''} would publish notes about the maintainers' own computer (src/lib/devkit-site/local.ts):\n${hits.map((h) => `  - ${h}`).join('\n')}\n` +
				`Wrap those lines of their source (site/DEVKIT.md, CONTRIBUTING.md) in ${LOCAL_START} ... ${LOCAL_END}; DEVKIT.md's notes listed in the git-ignored src/lib/devkit-site/machine.local.json are wrapped by node src/lib/devkit-site/mark-local.ts DEVKIT.md from site/.`
		);
	return page;
}

async function pageData(c: Ctx, p: string): Promise<PageData | null> {
	if (p === '') return home(c);
	if (p === 'guide') return guideIndex(c);
	const m = /^guide\/([\w-]+)$/.exec(p);
	if (m) return section(c, m[1]);
	if (p === 'reference') return reference(c);
	if (p === 'reference/ops') return ops(c);
	if (p === 'reference/views') return views(c);
	if (p === 'tutorial') return tutorial(c);
	if (p === 'contribute') return contribute(c);
	return null;
}

// ------------------------------------------------------------------ shared
const ref = (x: GuideSection): SectionRef => ({ n: x.n, slug: x.slug, title: x.title, zh: SECTION_ZH[x.slug]?.title ?? x.title });

/** In rendered text (outside code, links and headings): "section 4" and "section 4.2" become links to the guide, and
 *  error codes ("AK502") links to their explanation: `codes` gives the link of a code ('contribute': the contribute
 *  page; 'here': this page's own #ak502 anchor; default: the guide's error section). */
function linkSections(html: string, g: Guide, here: string | null, codes: 'guide' | 'contribute' | 'here' = 'guide', sections = true): string {
	const errSec = g.sections.find((x) => x.md.includes('<!-- atlas:generated codes -->'));
	const codeHref = (c: string) =>
		codes === 'here' || (codes === 'guide' && errSec?.slug === here) ? `#${c}` : codes === 'contribute' ? `${BASE}/devkit/contribute/#${c}` : guideUrl(errSec?.slug, c);
	return html
		.split(/(<pre[\s\S]*?<\/pre>|<code>[\s\S]*?<\/code>|<a\b[\s\S]*?<\/a>|<h\d[\s\S]*?<\/h\d>|<[^>]+>)/)
		.map((part, i) => {
			if (i % 2 === 1) return part;
			return part
				.replace(/\bsection (\d+)(?:\.(\d+))?\b/g, (all, a: string, b?: string) => {
					if (!sections) return all;
					const sec = g.sections.find((x) => x.n === Number(a));
					if (!sec) return all;
					const sub = b ? sec.subs.find((x) => x.text.startsWith(`${a}.${b} `)) : null;
					if (b && !sub) return all;
					if (sec.slug === here && !sub) return all;
					const href = sec.slug === here ? `#${sub!.id}` : guideUrl(sec.slug, sub?.id);
					return `<a href="${href}">${all}</a>`;
				})
				.replace(/\bAK\d{3}\b/g, (c) => (errSec && g.anchors.has(c.toLowerCase()) ? `<a href="${codeHref(c.toLowerCase())}">${c}</a>` : c));
		})
		.join('');
}

/** A narration with {name|fmt} placeholders filled from build-time values, with the kit's number formats. */
function fill(text: string, vars: Record<string, unknown>, lang: Lang): string {
	return text.replace(PLACEHOLDER_RE, (all, name: string, fmt?: string) => {
		const v = vars[name];
		if (v === undefined) throw new Error(`devkit-site: the tutorial narration uses {${name}}, which the build does not compute`);
		return formatValue(v, fmt, lang) ?? all;
	});
}
const fill2 = (t: T2, vars: Record<string, unknown>): T2 => ({ en: fill(t.en, vars, 'en'), zh: fill(t.zh, vars, 'zh') });

// ------------------------------------------------------------------ /devkit/
function home({ s, g }: Ctx): HomeData {
	const cmds = commandTable(g);
	const qs = quickStart(g);
	const quick = qs.map((q) => `# ${q.title}\n${q.lines.join('\n')}`).join('\n\n');
	const start = contributingStart(s.contributing);
	const contribLink = (href: string) => (/^(?:site\/)?DEVKIT\.md$/.test(href) ? { href: guideUrl() } : /^https?:/.test(href) ? { href, external: true } : null);
	// the live model lessons, and which of them are DevKit packages (the others are older hand-built pages)
	const live = Object.values(MODEL_BY_SLUG).filter((m) => m.status === 'live');
	const kitBuilt = live.filter((m) => existsSync(join(s.site, 'src', 'lib', 'explainers', m.slug, 'explainer.ts')));
	return {
		kind: 'home',
		counts: { sections: g.sections.length, ops: s.ops.length, views: s.kitViews.length, commands: Object.keys(cmds).length, codes: Object.keys(s.codes).length },
		built: { live: live.length, kit: kitBuilt.map((m) => ({ slug: m.slug, name: t2(m.name) })) },
		start: {
			statusHtml: start.status ? linkSections(render(start.status, { link: contribLink }), g, null, 'contribute') : null,
			statusZh: CONTRIBUTING_STATUS_ZH,
			setupHtml: start.setup ? linkSections(render(start.setup, { link: contribLink, copy: true }), g, null, 'contribute') : null
		},
		kinds: KINDS.map((k) => {
			const m = MODEL_BY_SLUG[k.example];
			return { ...k, example: { slug: k.example, name: t2(m?.name), steps: m?.lesson?.steps ?? null, summary: m?.lesson ? t2(m.lesson.summary) : null } };
		}),
		workflow: WORKFLOW.map((w) => ({ id: w.id, name: w.name, what: w.what, who: w.who, cmds: w.cmds.map((name) => ({ name, summary: cmds[name]?.summary ?? '' })) })),
		afternoon: afternoonResult(s.afternoon)!,
		quick: { title: g.sections.find((x) => x.n === 4)?.title ?? '', html: codeBlock(quick, 'sh', { copy: true }) },
		sections: g.sections.map(ref)
	};
}

// ------------------------------------------------------------------ /devkit/guide/
function guideIndex({ g }: Ctx): GuideIndexData {
	return {
		kind: 'guide',
		title: g.title,
		introHtml: linkSections(renderIntro(g), g, null),
		sections: g.sections.map((x) => ({ ...ref(x), summary: SECTION_ZH[x.slug]?.summary ?? '', subs: x.subs.length }))
	};
}

function section({ g }: Ctx, slug: string): SectionData | null {
	const i = g.sections.findIndex((x) => x.slug === slug);
	if (i < 0) return null;
	const sec = g.sections[i];
	const { html } = renderSection(g, sec);
	return {
		kind: 'section',
		section: { ...ref(sec), html: linkSections(html, g, sec.slug), subs: sec.subs, summary: SECTION_ZH[sec.slug]?.summary ?? '', omitted: sec.omitted },
		prev: i > 0 ? ref(g.sections[i - 1]) : null,
		next: i + 1 < g.sections.length ? ref(g.sections[i + 1]) : null,
		sections: g.sections.map(ref)
	};
}

// ------------------------------------------------------------------ /devkit/reference/
function reference({ s }: Ctx): ReferenceData {
	return { kind: 'reference', ops: s.ops.length, views: s.kitViews.length, packageViews: Object.keys(pkgMetas).length };
}

const CATEGORY: Record<string, T2> = {
	io: { en: 'Input and output', zh: '输入与输出' },
	linear: { en: 'Linear', zh: '线性' },
	conv: { en: 'Convolution', zh: '卷积' },
	activation: { en: 'Activation', zh: '激活函数' },
	pool: { en: 'Pooling', zh: '池化' },
	norm: { en: 'Normalization', zh: '归一化' },
	attention: { en: 'Attention', zh: '注意力' },
	embedding: { en: 'Embedding', zh: '嵌入' },
	shape: { en: 'Shape', zh: '形状' },
	merge: { en: 'Merge', zh: '合并' },
	recurrent: { en: 'Recurrent', zh: '循环' },
	graph: { en: 'Graph', zh: '图' },
	output: { en: 'Output', zh: '输出' }
};

/** An op formula as MathML (no KaTeX stylesheet or font on the page): the {{name}} slots that the zoom view fills with
 *  live numbers read ⟨name⟩. */
function formulaMathml(tex: string): string {
	const t = tex.replace(/\{\{\s*([\w.-]+)\s*\}\}/g, (_, n: string) => `\\langle\\text{${n}}\\rangle`);
	const html = katex.renderToString(t, { output: 'mathml', displayMode: true, throwOnError: false, strict: 'ignore', trust: (c) => c.command === '\\htmlClass' });
	return /<math[\s\S]*<\/math>/.exec(html)?.[0] ?? '';
}

const NARRATION_KEYS = ['what', 'how', 'why', 'look', 'unit', 'repeat'] as const;

function ops({ s, g }: Ctx): OpsData {
	const list: OpData[] = s.ops.map((o) => {
		const def = CORE_OPS.find((d) => d.op === o.op);
		const l = opLayer(o.op);
		let pytorch: string | null = null;
		try {
			pytorch = def?.pytorch && l ? def.pytorch(l.layer) : null;
		} catch {
			pytorch = null;
		}
		const d = OP_DEMOS[o.op];
		return {
			op: o.op,
			label: t2(o.label),
			category: o.category,
			view: zoomKeyOf(o.zoom),
			formula: def?.formula ? formulaMathml(def.formula) : null,
			narration: NARRATION_KEYS.filter((k) => def?.narration[k]).map((k) => ({ key: k, en: def!.narration[k]!.en, zh: def!.narration[k]!.zh ?? '' })),
			attrs: Object.entries(o.attrs).map(([name, a]) => ({ name, type: a.type, def: a.default !== undefined ? JSON.stringify(a.default) : null, required: !!a.required, doc: a.doc ?? null })),
			weights: o.weights.map((w) => `${w.slot}${w.optional ? '?' : ''}`),
			subs: o.subs.map((x) => (typeof x === 'string' ? x : x.name)),
			aliases: Object.entries(o.aliases).map(([k, p]) => (Object.keys(p).length ? `${k} (${Object.entries(p).map(([a, v]) => `${a}=${JSON.stringify(v)}`).join(', ')})` : k)),
			parityBar: o.parityBar,
			pytorch,
			demo: { model: d.model, layer: d.layer, unit: d.unit ?? null }
		};
	});
	const cats = [...new Set(s.ops.map((o) => o.category))];
	return { kind: 'ops', ops: list, categories: cats.map((id) => ({ id, label: CATEGORY[id] ?? { en: id, zh: id } })), guide: guideUrl(g.sections.find((x) => x.md.includes('<!-- atlas:generated ops -->'))?.slug) };
}

async function views({ s, g }: Ctx): Promise<ViewsData> {
	const out: ViewData[] = [];
	const ctxProps = { shown: true, reduced: true, playing: false, done: () => {} };
	const add = async (key: string, meta: ViewMeta, pkg: string | null) => {
		let describe: T2 | null = null;
		try {
			// the text a screen reader hears for the view's own demo (demo() runs the real view code's data in Node)
			const st = await meta.demo();
			describe = t2(meta.describe({ ...st, ctx: ctxProps } as never));
		} catch {
			describe = null;
		}
		out.push({ key, title: t2(meta.title), kinds: [...meta.kinds], minHeight: meta.minHeight ?? null, describe, pkg, ops: s.ops.filter((o) => zoomKeyOf(o.zoom) === key).map((o) => o.op) });
	};
	// the key a lesson step names is the meta's own key (the folder name when a meta has none)
	for (const [path, mod] of Object.entries(kitMetas)) await add(mod.default.key || /\/views\/([^/]+)\/meta\.ts$/.exec(path)![1], mod.default, null);
	for (const [path, mod] of Object.entries(pkgMetas)) {
		const m = /explainers\/([^/]+)\/views\/([^/]+)\/meta\.ts$/.exec(path)!;
		await add(mod.default.key || `${m[1]}/${m[2]}`, mod.default, m[1]);
	}
	out.sort((a, b) => (a.pkg ? 1 : 0) - (b.pkg ? 1 : 0) || a.key.localeCompare(b.key));
	return { kind: 'views', views: out, guide: guideUrl(g.sections.find((x) => x.md.includes('<!-- atlas:generated views -->'))?.slug) };
}

// ------------------------------------------------------------------ /devkit/tutorial/
const num = (v: unknown) => (typeof v === 'number' ? formatValue(v, undefined, 'en') ?? String(v) : String(v ?? ''));
/** a parity bar as the op reference prints it: 1e−4 */
const barText = (v: number) => v.toExponential().replace(/\.?0+e/, 'e').replace('e-', 'e−').replace('e+', 'e');

function produces(st: TutorialStepDef, { s, g }: Ctx, run: TutorialRuns): Produce {
	const static_ = join(s.site, 'static', 'models', HELLO);
	switch (st.produces) {
		case 'doctor':
			return { caption: { en: 'What doctor checks (from the command table)', zh: 'doctor 检查什么（来自命令表）' }, text: commandTable(g).doctor?.summary ?? '' };
		case 'scaffold': {
			const tj = JSON.parse(readFileSync(join(s.site, 'scripts', 'atlas', 'templates', 'base', 'template.json'), 'utf8')) as { files: { to: string; when?: string }[] };
			const where = (to: string) => (to === 'notes' ? 'notes/my-xor.md' : to.startsWith('package/') ? `site/src/lib/explainers/my-xor/${to.slice(8)}` : `pipeline/my-xor/${to.slice(9)}`);
			return {
				caption: { en: 'The files it writes (from the scaffold template)', zh: '它写出的文件（来自脚手架模板）' },
				list: tj.files.map((f) => where(f.to)),
				note: { en: 'Each kind gets a working toy to replace. hello-xor shows what the finished files look like.', zh: '每种类型都有一个能运行的小例子，等你替换。hello-xor 展示了完成后的文件。' }
			};
		}
		case 'params': {
			const spec = run.spec;
			return {
				caption: { en: 'Its weights, as spec.json lists them', zh: '它的权重（spec.json 中的列表）' },
				table: { head: [{ en: 'Tensor', zh: '张量' }, { en: 'Shape', zh: '形状' }, { en: 'Numbers', zh: '数的个数' }], rows: spec.tensors.map((x) => [x.name, `[${x.shape.join(', ')}]`, String(x.length)]), num: [false, true, true] },
				note: { en: `${spec.params} numbers in all, stored as float32 in weights.bin.`, zh: `一共 ${spec.params} 个数，以 float32 存在 weights.bin 里。` }
			};
		}
		case 'metrics': {
			const p = `pipeline/${HELLO}/train_metrics.json`;
			const text = repoFile(s, p);
			return { caption: { en: 'What train.py wrote', zh: 'train.py 写出的文件' }, codeHtml: text ? codeBlock(text, 'json', { caption: p }) : undefined, text: text ? undefined : p };
		}
		case 'export': {
			const spec = run.spec;
			const files = existsSync(static_) ? readdirSync(static_).filter((f) => statSync(join(static_, f)).isFile()) : [];
			const parity = spec.parity?.max_abs_diff;
			return {
				caption: { en: 'spec.json: one layer per builder call', zh: 'spec.json：每个构建器调用一层' },
				table: {
					head: [{ en: 'Layer id', zh: '层 id' }, { en: 'Op', zh: '算子' }, { en: 'Output shape', zh: '输出形状' }, { en: 'Label', zh: '标签' }],
					rows: spec.layers.map((l) => [l.id, l.op, `[${l.outShape.join(', ')}]`, l.label.en])
				},
				list: files.map((f) => `site/static/models/${HELLO}/${f} · ${fmtBytes(statSync(join(static_, f)).size)}`),
				note: {
					en: `Parity: every layer within ${parity == null ? '…' : num(parity)} of PyTorch fp32 (bar ${barText(spec.parity?.bar ?? 1e-4)}), written into spec.json by atlas parity.`,
					zh: `一致性：每一层与 PyTorch fp32 的差都不超过 ${parity == null ? '…' : num(parity)}（阈值 ${barText(spec.parity?.bar ?? 1e-4)}），由 atlas parity 写进 spec.json。`
				}
			};
		}
		case 'trainRun': {
			const r = run.train;
			const pick = [...new Set([0, 1, 2, 3, 4, 5, r.frames.length - 2, r.frames.length - 1])].filter((i) => i >= 0 && i < r.frames.length);
			const rows: string[][] = [];
			pick.forEach((i, k) => {
				if (k > 0 && i - pick[k - 1] > 1) rows.push(['…', '', '', '', '']);
				const f = r.frames[i];
				rows.push([String(i), f.phase, f.step === undefined ? '' : String(f.step), num(f.metrics?.loss), f.metrics?.correct === undefined ? '' : String(f.metrics.correct)]);
			});
			return {
				caption: { en: `The tape of source '${r.source}' (trainer '${r.trainer}', seed ${r.seed})`, zh: `来源 '${r.source}' 的磁带（训练器 '${r.trainer}'，随机种子 ${r.seed}）` },
				provenance: 'precomputed',
				table: { head: [{ en: 'Frame', zh: '帧' }, { en: 'Phase', zh: '阶段' }, { en: 'Step', zh: '步' }, { en: 'Loss', zh: '损失' }, { en: 'Right (of 4)', zh: '答对（共 4 个）' }], rows, num: [true, false, true, true, true] },
				note: { en: `Run in Node when this page was built, with the kit's recorder: ${r.frames.length} frames.`, zh: `在构建本页时用工具包的记录器在 Node 里运行：共 ${r.frames.length} 帧。` }
			};
		}
		case 'stepRun': {
			const r = run.step;
			return {
				caption: { en: `The frames of process '${r.process}' (seed ${r.seed})`, zh: `过程 '${r.process}' 的各帧（随机种子 ${r.seed}）` },
				provenance: 'precomputed',
				table: {
					head: [{ en: 'Frame', zh: '帧' }, { en: 'Phase', zh: '阶段' }, { en: 'What the phase narration says', zh: '这一阶段的旁白' }],
					rows: r.frames.map((f, i) => [String(i), `${f.phase}${f.sub !== undefined ? ` ${f.sub}` : ''}`, f.text])
				},
				note: { en: 'Run in Node when this page was built; the narration is the phase’s own template filled with the frame’s metrics.', zh: '在构建本页时在 Node 里运行；旁白是该阶段自己的模板，用这一帧的 metrics 填好。' }
			};
		}
		case 'sources':
			return {
				caption: { en: 'The sources hello-xor declares', zh: 'hello-xor 声明的来源' },
				table: {
					head: [{ en: 'Source', zh: '来源' }, { en: 'Kind', zh: '类型' }, { en: 'Runs', zh: '运行的是' }],
					rows: Object.entries(helloManifest.sources ?? {}).map(([name, d]) => {
						const x = d as { kind: string; trainer?: string; process?: string; of?: string; opt?: unknown };
						return [name, x.kind, x.trainer ? `trainer ${x.trainer}${x.opt ? ` ${JSON.stringify(x.opt)}` : ''}` : x.process ? `process ${x.process}` : x.of ? `the weights of '${x.of}'` : 'spec.json + weights.bin'];
					})
				}
			};
		case 'lessonSteps':
			return {
				caption: { en: `The lesson: ${helloLesson.steps.length} steps`, zh: `这节课：${helloLesson.steps.length} 步` },
				table: {
					head: [{ en: '#', zh: '#' }, { en: 'Step id', zh: '步骤 id' }, { en: 'Title', zh: '标题' }, { en: 'Kind of step', zh: '步骤类型' }],
					rows: helloLesson.steps.map((x, i) => {
						const a = x as { id?: string; auto?: string; autoAll?: unknown; title?: I18n; override?: { title?: I18n }; predict?: unknown; task?: unknown; kindOf?: string };
						const title = a.title ?? a.override?.title;
						const kinds = [a.auto ? `auto ${a.auto}` : 'author', a.predict ? 'predict' : '', a.task ? 'task' : '', a.kindOf ?? ''].filter(Boolean).join(' · ');
						return [String(i + 1), a.id ?? a.auto ?? '', title ? t2(title) : '', kinds];
					}),
					num: [true, false, false, false]
				}
			};
		case 'claims': {
			const text = repoFile(s, `site/src/lib/explainers/${HELLO}/claims.test.ts`) ?? '';
			return { caption: { en: 'The claims hello-xor tests', zh: 'hello-xor 用测试验证的论断' }, list: [...text.matchAll(/^test\('((?:[^'\\]|\\.)*)'/gm)].map((m) => m[1].replace(/\\'/g, "'")) };
		}
		case 'lab': {
			const port = portTable(g)[HELLO];
			return {
				caption: { en: 'Where to look', zh: '去哪里看' },
				list: ['http://localhost:<your port>/dev/atlas/lab/my-xor/', `http://localhost:${port}/dev/atlas/lab/${HELLO}/`],
				note: {
					en: `Your package's port is the one atlas new printed (notes/my-xor.md); hello-xor's row of the port table is ${port}.`,
					zh: `你的包用 atlas new 打印出的端口（写在 notes/my-xor.md 里）；hello-xor 在端口表中的端口是 ${port}。`
				}
			};
		}
		case 'gates': {
			const sec = g.sections.find((x) => /^checks/i.test(x.title))!;
			return { caption: { en: 'The gates check runs', zh: 'check 运行的关卡' }, codeHtml: linkSections(render(firstTable(sec.md) ?? ''), g, null, 'contribute') };
		}
	}
}

interface TutorialRuns {
	spec: { params: number; tensors: { name: string; shape: number[]; length: number }[]; layers: { id: string; op: string; outShape: (number | string)[]; label: I18n }[]; parity?: { max_abs_diff?: number | null; bar?: number } };
	train: { source: string; trainer: string; seed: number; frames: { phase: string; step?: number; metrics?: Record<string, number> }[] };
	step: { process: string; seed: number; frames: { phase: string; sub?: number; text: T2 }[]; last: Record<string, number> };
}

/** Run hello-xor's train source and its slowed-down step in Node, as `atlas run` does. */
function runs(s: Sources): TutorialRuns {
	const spec = JSON.parse(readFileSync(join(s.site, 'static', 'models', HELLO, 'spec.json'), 'utf8')) as TutorialRuns['spec'];
	const sources = (helloManifest.sources ?? {}) as Record<string, { kind: string; trainer?: string; process?: string; opt?: Record<string, unknown>; seed?: number }>;
	const [tName, tDef] = Object.entries(sources).find(([, d]) => d.kind === 'train')!;
	const trainer = Object.values(helloTrainers).find((x) => (x as { id?: string })?.id === tDef.trainer) as Parameters<typeof trainingProcess>[0];
	const seed = typeof tDef.opt?.seed === 'number' ? tDef.opt.seed : 1;
	const input = helloManifest.input?.start ?? null;
	const tape = record(trainingProcess(trainer), input, tDef.opt ?? {}, { seed, detail: 'coarse' });
	const trainFrames = Array.from({ length: tape.length }, (_, i) => {
		const f = tape.frame(i);
		return { phase: f.phase, step: f.step, metrics: f.metrics as Record<string, number> | undefined };
	});
	const [, pDef] = Object.entries(sources).find(([, d]) => d.kind === 'process')!;
	const proc = Object.values(helloProcesses).find((x) => (x as { id?: string })?.id === pDef.process) as typeof helloProcesses.sgdStep;
	const pseed = pDef.seed ?? 1;
	const ptape = record(proc, input, {}, { seed: pseed });
	const stepFrames = Array.from({ length: ptape.length }, (_, i) => {
		const f = ptape.frame(i);
		const what = proc.phases[f.phase]?.narration?.what;
		const vars = Object.fromEntries(Object.entries(f.metrics ?? {}).map(([k, v]) => [`frame.${k}`, v]));
		const say = (lang: Lang) => (what ? (lang === 'zh' ? (what.zh ?? what.en) : what.en).replace(PLACEHOLDER_RE, (all, n: string, fmt?: string) => (n in vars ? (formatValue(vars[n], fmt, lang) ?? all) : all)) : '');
		return { phase: f.phase, sub: f.sub, text: { en: say('en'), zh: say('zh') } };
	});
	const lastMetrics = (ptape.length ? (ptape.frame(ptape.length - 1).metrics ?? {}) : {}) as Record<string, number>;
	return { spec, train: { source: tName, trainer: tDef.trainer!, seed, frames: trainFrames }, step: { process: pDef.process!, seed: pseed, frames: stepFrames, last: lastMetrics } };
}

function tutorial(c: Ctx): TutorialData {
	const { s, g } = c;
	const run = runs(s);
	const s4 = g.sections.find((x) => x.n === 4)!;
	const hidden = run.spec.layers.find((l) => l.op === 'tanh')?.outShape[0];
	const last = run.train.frames[run.train.frames.length - 1];
	const firstLoss = run.train.frames.find((f) => typeof f.metrics?.loss === 'number')?.metrics?.loss;
	const vars: Record<string, unknown> = {
		params: run.spec.params,
		hidden,
		parity: formatValue(run.spec.parity?.max_abs_diff ?? NaN, undefined, 'en') ?? '…',
		first: firstLoss,
		last: last.metrics?.loss,
		steps: last.step,
		frames: run.step.frames.length,
		loss: run.step.last.loss,
		lossAfter: run.step.last.lossAfter
	};
	const steps: TutorialStep[] = TUTORIAL.map((st) => {
		const text = st.file ? repoFile(s, st.file.path) : null;
		const ex = st.file && text ? excerpt(text, st.file) : null;
		const cmd = st.command ? stepCommand(subsection(s4, st.command.section4), st.command) : null;
		return {
			id: st.id,
			title: st.title,
			narration: fill2(st.narration, vars),
			look: st.look ?? null,
			file: st.file?.path ?? null,
			fileHtml: ex ? codeBlock(ex.code, st.file!.lang, { caption: `${st.file!.path} · ${ex.start}–${ex.start + ex.code.split('\n').length - 1}`, copy: true }) : null,
			commandHtml: cmd ? codeBlock(cmd, 'sh', { copy: true }) : null,
			produces: produces(st, c, run)
		};
	});
	const spec = JSON.parse(readFileSync(join(s.site, 'static', 'models', HELLO, 'spec.json'), 'utf8')) as unknown;
	return { kind: 'tutorial', steps, hello: { slug: HELLO, spec, port: portTable(g)[HELLO] ?? null }, guide: guideUrl(s4.slug) };
}

// ------------------------------------------------------------------ /devkit/contribute/
function contribute({ s, g }: Ctx): ContributeData {
	const errSec = g.sections.find((x) => x.md.includes('<!-- atlas:generated codes -->'))!;
	const gates = g.sections.find((x) => /^checks/i.test(x.title))!;
	const design = s.design ?? '';
	const checklist = /^### 6\.2 [^\n]*\n([\s\S]*?)(?=^### |^## )/m.exec(design)?.[1] ?? '';
	const quality = /\*\*Quality levels\*\*[^\n]*(\n[^\n]+)*/.exec(gates.md)?.[0] ?? '';
	const link = (href: string) => (href === 'site/DEVKIT.md' || href === 'DEVKIT.md' ? { href: guideUrl() } : /^https?:/.test(href) ? { href, external: true } : null);
	return {
		kind: 'contribute',
		// CONTRIBUTING.md without its title; its ## parts sit under the page's own h2, so they become h3
		contributingZh: CONTRIBUTING_ZH.summary,
		// its "site/DEVKIT.md section 4" links to the guide's section 4, as in the guide's own pages
		contributingHtml: linkSections(render((s.contributing ?? '').replace(/^# [^\n]*\n/, ''), { link, copy: true, shift: 1 }), g, null, 'here'),
		// the design's own section numbers are not the guide's: link only the error codes
		checklistHtml: linkSections(render(checklist, { link }), g, null, 'here', false),
		gatesHtml: linkSections(render(firstTable(gates.md) ?? ''), g, null, 'here'),
		qualityHtml: linkSections(render(quality), g, null, 'here'),
		codes: Object.entries(s.codes).map(([code, en]) => ({
			code,
			en,
			zh: CODE_ZH[code]?.zh ?? en,
			group: groupOf(code),
			html: linkSections(render(subsection(errSec, code) ?? '', { link: (h) => (h.startsWith('#') ? { href: guideUrl(errSec.slug, h.slice(1)) } : null) }), g, null, 'here')
		})),
		groups: Object.entries(CODE_GROUPS).map(([id, label]) => ({ id, label })),
		errorsGuide: guideUrl(errSec.slug),
		gatesGuide: guideUrl(gates.slug)
	};
}

