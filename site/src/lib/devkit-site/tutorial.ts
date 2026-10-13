// "Build hello-xor step by step" (/devkit/tutorial/): the steps of DEVKIT.md section 4, played as a guided Walkthrough
// over the REAL hello-xor package. Each step names a file (an excerpt cut out of the real file at build time, between two
// anchors below) or a command, and what it produces (computed or read at build time by the route's server load).
// Pure data and functions (Node-safe): devkit-site.test.ts checks that every excerpt still finds its anchors, so a change
// to hello-xor that breaks a step fails the build instead of showing a stale or empty excerpt.
//
// Narration placeholders ({name}) are filled at build time from values the server load computes (vars below), with
// the kit's number formats; no number is typed into the text.

export interface I18nText {
	en: string;
	zh: string;
}

/** A piece of a real file: from the first line that contains `from` through the first line after it that contains `to`
 *  (or up to the line before the one that contains `before`). Paths are relative to the repository root. */
export interface Excerpt {
	path: string;
	from: string;
	to?: string;
	before?: string;
	lang: string;
}

export type ProduceKind = 'doctor' | 'scaffold' | 'params' | 'metrics' | 'export' | 'trainRun' | 'stepRun' | 'sources' | 'lessonSteps' | 'claims' | 'lab' | 'gates';

export interface TutorialStepDef {
	id: string;
	title: I18nText;
	narration: I18nText;
	look?: I18nText;
	file?: Excerpt;
	/** a command from DEVKIT.md subsection 4.n: the first command line (in a code block or inline code) whose subcommand
	 *  (the text after "npm run atlas -- ") matches the regular expression `re`; `block` shows its whole code block */
	command?: { section4: string; re: string; block?: boolean };
	produces: ProduceKind;
}

export const HELLO = 'hello-xor';
const PKG = `site/src/lib/explainers/${HELLO}`;
const PY = `pipeline/${HELLO}`;

export const TUTORIAL: TutorialStepDef[] = [
	{
		id: 'doctor',
		title: { en: 'Check your machine', zh: '检查你的电脑' },
		narration: {
			en: 'Install the site’s packages once (npm install in the site folder; guide section 1). From then on everything runs from the site folder through one command line. doctor checks the tools the kit needs and prints the exact fix for anything that is missing. Kinds B and C need no Python at all.',
			zh: '先在 site 文件夹里安装一次网站的依赖包（npm install，见指南第 1 节）。之后所有操作都在 site 文件夹里，通过同一个命令行完成。doctor 检查工具包需要的工具，并为缺少的每一项打印确切的修复方法。B 类和 C 类根本不需要 Python。'
		},
		command: { section4: '4.1', re: '^doctor' },
		produces: 'doctor'
	},
	{
		id: 'scaffold',
		title: { en: 'Scaffold the package', zh: '生成脚手架' },
		narration: {
			en: 'atlas new writes a working toy for each kind you ask for, so the package runs before you change anything. It never overwrites a file. hello-xor is the finished answer to this scaffold: the next steps show its real files.',
			zh: 'atlas new 为你要的每种类型写好一个能运行的小例子，所以包在你改动之前就能跑。它从不覆盖已有文件。hello-xor 就是这个脚手架的完成版：后面几步展示的都是它的真实文件。'
		},
		command: { section4: '4.1', re: '^new model [^ ]+ --dev --kind' },
		produces: 'scaffold'
	},
	{
		id: 'model',
		title: { en: 'Write the model in PyTorch', zh: '用 PyTorch 写模型' },
		narration: {
			en: 'Kind A starts with a plain PyTorch module. hello-xor is the smallest layer-by-layer network that can learn XOR: two inputs, {hidden|int} hidden units with tanh, one output score, {params|int} weights and biases in all.',
			zh: 'A 类从一个普通的 PyTorch 模块开始。hello-xor 是能学会 XOR 的最小的逐层网络：两个输入，{hidden|int} 个带 tanh 的隐藏单元，一个输出分数，一共 {params|int} 个权重和偏置。'
		},
		look: { en: 'The submodule names fc1 and fc2: they become the layer ids of the page.', zh: '子模块的名字 fc1 和 fc2：它们会成为页面上各层的 id。' },
		file: { path: `${PY}/model.py`, from: 'X = torch.tensor', to: 'return self.fc2', lang: 'python' },
		produces: 'params'
	},
	{
		id: 'train',
		title: { en: 'Train it with a fixed seed', zh: '用固定的随机种子训练' },
		narration: {
			en: 'Train with a fixed seed so anyone can get the same weights again. Run your scripts through atlas py: it uses the Python the kit is configured with. This run wrote the checkpoint and the metrics file shown here.',
			zh: '用固定的随机种子训练，这样任何人都能得到同样的权重。用 atlas py 运行你的脚本：它使用工具包配置好的 Python。这次运行写出了检查点和这里显示的指标文件。'
		},
		file: { path: `${PY}/train.py`, from: 'def train(', to: 'return m.eval()', lang: 'python' },
		command: { section4: '4.2', re: '^py ' },
		produces: 'metrics'
	},
	{
		id: 'export',
		title: { en: 'Describe the graph once, then export', zh: '把计算图描述一遍，然后导出' },
		narration: {
			en: 'describe() makes one builder call per step of forward(); each call computes the PyTorch value and records a layer. export checks that describe() equals the module, writes the files the page loads, then compares every layer of the kit’s own forward pass with PyTorch: the largest difference is {parity}.',
			zh: 'describe() 为 forward() 的每一步写一个构建器调用；每次调用都算出 PyTorch 的值并记录一层。export 先确认 describe() 与模块一致，写出页面要加载的文件，再把工具包自己的前向计算与 PyTorch 逐层比较：最大差异是 {parity}。'
		},
		look: { en: 'The layer table: one row per builder call.', zh: '层的表格：每个构建器调用一行。' },
		file: { path: `${PY}/export.py`, from: 'def describe(', to: 'b.output(', lang: 'python' },
		command: { section4: '4.2', re: '^export ' },
		produces: 'export'
	},
	{
		id: 'trainer',
		title: { en: 'Kind B: train it live in the browser', zh: 'B 类：在浏览器里实时训练' },
		narration: {
			en: 'A TrainerDef says how to build the model, which batch to use and what the loss is; the kit runs the loop. atlas run prints the tape in Node before any page exists. With the lesson’s seed the loss falls from {first} to {last} in {steps|int} steps.',
			zh: 'TrainerDef 说明怎样建模型、用哪个批次、损失是什么；训练循环由工具包运行。atlas run 在还没有任何页面之前，就在 Node 里打印出磁带。用课程的随机种子，损失在 {steps|int} 步里从 {first} 降到 {last}。'
		},
		file: { path: `${PKG}/trainer.ts`, from: 'export const xor', to: '};', lang: 'ts' },
		command: { section4: '4.3', re: '^run ' },
		produces: 'trainRun'
	},
	{
		id: 'process',
		title: { en: 'Kind C: slow one step down', zh: 'C 类：把一步放慢' },
		narration: {
			en: 'A ProcessDef is a generator that yields one frame per event, with every number the narration needs in the frame’s metrics. hello-xor’s process does one gradient-descent step in {frames|int} frames; the loss goes from {loss|n4} to {lossAfter|n4}.',
			zh: 'ProcessDef 是一个生成器，每个事件 yield 一帧，旁白要用的数都放进这一帧的 metrics。hello-xor 的过程用 {frames|int} 帧做完一步梯度下降；损失从 {loss|n4} 变成 {lossAfter|n4}。'
		},
		file: { path: `${PKG}/process.ts`, from: '// update: w <- w - lr', to: "yield { phase: 'update'", lang: 'ts' },
		command: { section4: '4.4', re: '^run ' },
		produces: 'stepRun'
	},
	{
		id: 'manifest',
		title: { en: 'Declare the sources', zh: '声明来源' },
		narration: {
			en: 'The manifest, explainer.ts, names the sources a lesson step can play: the exported model, the live trainer, the slowed-down step, and the live weights traced on the learner’s point. It also holds the goals, the first question and the credits.',
			zh: '清单 explainer.ts 列出课程每一步可以播放的来源：导出的模型、实时训练器、放慢的那一步，以及用实时权重计算学习者的点。它还写着学习目标、开篇问题和致谢。'
		},
		file: { path: `${PKG}/explainer.ts`, from: 'sources: {', before: 'overview:', lang: 'ts' },
		produces: 'sources'
	},
	{
		id: 'lesson',
		title: { en: 'Write the lesson', zh: '写课程' },
		narration: {
			en: 'A lesson step points at a source, a frame and a view, and its narration binds every number to a placeholder. Start from the auto steps and replace them with your own. Lint enters every step on every input fixture in both languages.',
			zh: '课程的一步指向一个来源、一帧和一个视图，旁白里的每个数都绑定到占位符。先用自动步骤，再换成你自己写的。lint 会在每个测试输入、两种语言下进入每一步。'
		},
		look: { en: 'The update step: the formula in the narration is the arithmetic that ran.', zh: '更新这一步：旁白里的公式就是实际运行的计算。' },
		file: { path: `${PKG}/lesson.ts`, from: "id: 'sgd-update'", to: 'vars:', lang: 'ts' },
		command: { section4: '4.5', re: '^lesson [^ ]+ lint' },
		produces: 'lessonSteps'
	},
	{
		id: 'claims',
		title: { en: 'Back every claim with a test', zh: '为每个论断写测试' },
		narration: {
			en: 'A sentence that is not computed live, such as “more steps do not help”, needs a test in claims.test.ts. check runs these tests; a failing claim stops the package.',
			zh: '不是实时算出的句子，比如“再多训练也没用”，要在 claims.test.ts 里有测试。check 会运行这些测试；论断不成立，包就通不过。'
		},
		file: { path: `${PKG}/claims.test.ts`, from: "test('step \"stuck\"", to: 'more steps do not help', lang: 'ts' },
		produces: 'claims'
	},
	{
		id: 'look',
		title: { en: 'Look at it in the Lesson Lab', zh: '在课程实验室里看' },
		narration: {
			en: 'Run the dev server on the package’s own port and open the Lesson Lab: every step, a switch for inputs and languages, and the lint list. Edits reload the page. The lab is not published, so the finished hello-xor lesson is embedded at the end of this page.',
			zh: '在包自己的端口上启动开发服务器，打开课程实验室：可以看到每一步、切换输入和语言的开关，以及 lint 列表。改了文件，页面会自动刷新。实验室不会发布到网站上，所以完成的 hello-xor 课程嵌在本页末尾。'
		},
		command: { section4: '4.6', re: '^npm run dev' },
		produces: 'lab'
	},
	{
		id: 'check',
		title: { en: 'Check, then write it up', zh: '检查，然后写笔记' },
		narration: {
			en: 'While you iterate, check --quick runs every gate except the build and the browser test. The full check runs once, at the end: it adds a build of your pages and the one browser test, two passes with an accessibility scan and a sheet of every step. A draft is done when check is green and the notes file has its five sections. Then the teaching review, which reads those sheets, and the maintainers promote it.',
			zh: '反复修改时，check --quick 运行除构建和浏览器测试以外的所有关卡。完整的 check 只在最后运行一次：它再构建本包的页面，并运行唯一一次浏览器测试，即两轮检查、一次无障碍扫描，外加一张列出每一步的总览图。check 全部通过、笔记的五个部分写好，草稿就完成了。接下来是教学评审（评审者直接读这些总览图），然后由维护者让它上线。'
		},
		command: { section4: '4.7', re: '^check ', block: true },
		produces: 'gates'
	}
];

/** Cut an excerpt out of a file's text; null when an anchor is missing. Line numbers count from 1. */
export function excerpt(text: string, e: Pick<Excerpt, 'from' | 'to' | 'before'>): { code: string; start: number } | null {
	const lines = text.replace(/\r\n/g, '\n').split('\n');
	const a = lines.findIndex((l) => l.includes(e.from));
	if (a < 0) return null;
	let b = lines.length - 1;
	const end = e.to ?? e.before;
	if (end !== undefined) {
		const k = lines.findIndex((l, i) => i > a && l.includes(end));
		if (k < 0) return null;
		b = e.to !== undefined ? k : k - 1;
	}
	// dedent by the excerpt's common indentation (tabs shown as two spaces, so excerpts fit a phone)
	const part = lines.slice(a, b + 1);
	const ind = Math.min(...part.filter((l) => l.trim()).map((l) => /^[\t ]*/.exec(l)![0].length));
	return { code: part.map((l) => l.slice(ind).replace(/^\t+/, (t) => '  '.repeat(t.length))).join('\n'), start: a + 1 };
}

/** Command lines of a Markdown text, in order: the lines of code blocks and the inline code spans that start with
 *  "npm run"; `block` is the code block a line came from (the line itself for inline code). */
export function commandLines(md: string): { line: string; block: string }[] {
	const out: { line: string; block: string }[] = [];
	const lines = md.replace(/\r\n/g, '\n').split('\n');
	for (let i = 0; i < lines.length; i++) {
		if (/^\s*```/.test(lines[i])) {
			const body: string[] = [];
			for (i++; i < lines.length && !/^\s*```/.test(lines[i]); i++) body.push(lines[i].trim());
			const block = body.join('\n');
			for (const l of body) if (l.startsWith('npm run')) out.push({ line: l, block });
			continue;
		}
		for (const m of lines[i].matchAll(/`(npm run [^`]+)`/g)) out.push({ line: m[1], block: m[1] });
	}
	return out;
}

/** A step's command, from the text of its DEVKIT.md subsection (null when it is not there any more). */
export function stepCommand(sub: string | null, c: NonNullable<TutorialStepDef['command']>): string | null {
	if (sub === null) return null;
	const re = new RegExp(c.re);
	const hit = commandLines(sub).find((x) => re.test(x.line.replace(/^npm run atlas -- /, '')));
	return hit ? (c.block ? hit.block : hit.line) : null;
}

/** The tutorial's files, by repository path. */
export const tutorialFiles = (): string[] => [...new Set(TUTORIAL.flatMap((s) => (s.file ? [s.file.path] : [])))];
