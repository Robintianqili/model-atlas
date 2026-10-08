// Kit-side UI strings for the atlas UI (design §6.2 item 3: atlas has its own strings; the shared i18n/strings.ts is not
// touched). say(key, vars) is reactive like ui(): it re-renders when the language changes.
import { t } from '$lib/i18n/i18n.svelte';
import type { I18n } from '$lib/i18n/types';

export const STR = {
	loadingModel: { en: 'Loading the model…', zh: '正在加载模型…' },
	computing: { en: 'Computing…', zh: '正在计算…' },
	computingOf: { en: 'Computing… {k} / {n}', zh: '正在计算… {k} / {n}' },
	computingFrames: { en: 'Computing… {k} frames so far', zh: '正在计算… 已有 {k} 帧' },
	trainingFirst: { en: 'Training the network first…', zh: '先训练网络…' },
	trainingFirstOf: { en: 'Training the network first… step {k} / {n}', zh: '先训练网络… 第 {k} / {n} 步' },
	stageError: { en: 'This step could not be shown.', zh: '这一步无法显示。' },
	emptyInput: { en: 'There is no input yet. Give the model something to work on and this step fills with real numbers.', zh: '还没有输入。给模型一点东西，这一步就会填上真实的数。' },
	clamped: {
		en: 'This moment did not happen for your input, so the nearest recorded one is shown.',
		zh: '你的输入没有出现这一刻，所以显示最接近的一刻。'
	},
	viewMissing: { en: 'View “{key}” is not registered in this build: a generic view is shown.', zh: '这个构建里没有注册视图“{key}”：显示的是通用视图。' },
	showAsText: { en: 'Show as text', zh: '以文字显示' },
	predictFirst: { en: 'Predict first', zh: '先猜一猜' },
	predictHint: { en: 'Pick an answer to reveal this step.', zh: '先选一个答案，再看这一步。' },
	right: { en: 'Right.', zh: '对了。' },
	notQuite: { en: 'Not quite.', zh: '不太对。' },
	yourTurn: { en: 'Your turn', zh: '轮到你了' },
	// K1: neutral (it used to say "well spotted", which fits a predict, not a task); a task sets task.doneText for its own
	taskDone: { en: 'Done.', zh: '完成了。' },
	taskWaiting: { en: 'Not yet: do what the task says.', zh: '还没有：照任务说的做。' },
	// Play waits for the learner (LessonPlayer: WalkStep.waitText)
	waitAnswer: { en: 'Waiting for your answer…', zh: '等你回答…' },
	waitTask: { en: 'Waiting for you to do the task…', zh: '等你完成任务…' },
	hint: { en: 'Hint', zh: '提示' },
	termsLabel: { en: 'Terms', zh: '术语' },
	frameOf: { en: 'Frame {i} of {n}', zh: '第 {i} / {n} 帧' },
	// a scan's counter: which output number of the layer (R0: "Position i of n" read wrong on a scan over features)
	positionOf: { en: 'Output {i} of {n}', zh: '输出 {i} / {n}' },
	// a scan with play.unit (K1): "Patch 3 of 16"
	unitOf: { en: '{unit} {i} of {n}', zh: '{unit} {i} / {n}' },
	jumpTo: { en: 'Jump to', zh: '跳到' },
	phase: { en: 'Phase', zh: '阶段' },
	yourInput: { en: 'Your input', zh: '你的输入' },
	// the answer line under "Your input" (K1)
	ansTop: { en: 'Its answer', zh: '模型的回答' },
	ansNext: { en: 'Most likely next', zh: '最可能的下一个' },
	whatModelSees: { en: 'What the model sees', zh: '模型看到的' },
	presets: { en: 'Presets', zh: '预设' },
	redrawHere: { en: 'Change it here', zh: '在这里修改' },
	hideInput: { en: 'Hide', zh: '收起' },
	inputUpdates: { en: 'Change your input at any time: the step you are on updates at once.', zh: '随时可以改输入：你所在的这一步会立刻更新。' },
	lessonEnd: { en: 'That is the whole lesson.', zh: '课程到此结束。' },
	lessonEndMore: { en: 'Change the input and play it again, or explore every layer below.', zh: '换个输入再播放一遍，或者在下面查看每一层。' },
	// K1: a page without layers (kinds B and C, no spec)
	lessonEndMoreFree: { en: 'Change the input and play it again, or explore the page below.', zh: '换个输入再播放一遍，或者看看下面的内容。' },
	exploreFreely: { en: 'Explore freely', zh: '自由探索' },
	lessonFoot: {
		en: 'Every number in this lesson is computed from your input unless its badge says otherwise. The lesson keeps its own selection, so the views below stay where you left them.',
		zh: '除非徽章另有说明，课程里的每个数都由你的输入算出。课程使用自己的选中状态，所以下面的视图会停在你离开时的位置。'
	},
	unitTitle: { en: 'The arithmetic behind one number', zh: '一个数背后的计算' },
	unitPick: { en: 'Click or tap a cell of the output to explain that number instead.', zh: '点击输出里的一个格子，改为解释那个数。' },
	layer: { en: 'Layer', zh: '层' },
	output: { en: 'Output', zh: '输出' },
	value: { en: 'Value', zh: '数值' },
	sum: { en: 'Sum', zh: '求和' },
	bias: { en: 'Bias', zh: '偏置' },
	result: { en: 'Result', zh: '结果' },
	terms: { en: '{n} products', zh: '{n} 个乘积' },
	moreTerms: { en: '… and {n} more', zh: '……还有 {n} 个' },
	inputCol: { en: 'input', zh: '输入' },
	weightCol: { en: 'weight', zh: '权重' },
	productCol: { en: 'product', zh: '乘积' },
	readsFrom: { en: 'Reads from {t}', zh: '读取自 {t}' },
	noTensor: { en: 'Nothing to show for this frame.', zh: '这一帧没有可显示的内容。' },
	metrics: { en: 'Numbers in this frame', zh: '这一帧的数' },
	topClasses: { en: 'Top answers', zh: '排名靠前的回答' },
	training: { en: 'Training', zh: '训练' },
	// the auto Training section (K1: bilingual; K-R3)
	trDataset: { en: 'Data', zh: '数据' },
	trSteps: { en: 'Training steps', zh: '训练步数' },
	trParams: { en: 'Parameters', zh: '参数' },
	trAfter: { en: 'After {n} training steps', zh: '训练 {n} 步之后' },
	trFinal: { en: 'At the end of training', zh: '训练结束时' },
	trLossNow: { en: 'Loss', zh: '损失' },
	trNotRun: { en: 'Press Train to run it on your input.', zh: '按“开始训练”，在你的输入上运行。' },
	lossOverSteps: { en: 'Loss over the training steps', zh: '训练过程中的损失' },
	trainSteps: { en: '{n} steps', zh: '{n} 步' },
	runTraining: { en: 'Train', zh: '开始训练' },
	pause: { en: 'Pause', zh: '暂停' },
	quizTitle: { en: 'Check yourself', zh: '自我检查' },
	showAnswer: { en: 'Show the answer', zh: '显示答案' },
	yourGuess: { en: 'Your guess', zh: '你的猜测' },
	check: { en: 'Check', zh: '核对' },
	measured: { en: 'Measured on your input: {v}', zh: '在你的输入上测得：{v}' },
	tryPredict: { en: 'Predict first', zh: '先预测' },
	tryLook: { en: 'Then look', zh: '然后看' },
	trySetUp: { en: 'Set it up', zh: '设置好' },
	tryGoto: { en: 'Open the step', zh: '打开这一步' },
	trySetDone: { en: 'Done: the input is set.', zh: '好了：输入已设置。' },
	codeFor: { en: 'PyTorch for this layer', zh: '这一层的 PyTorch 写法' },
	codeWhere: { en: 'Defined at {file}:{line}', zh: '定义在 {file}:{line}' },
	noCode: { en: 'No code is recorded for this layer.', zh: '这一层没有记录代码。' },
	creditsTitle: { en: 'Credits', zh: '致谢' },
	groupOf: { en: '{n} layers', zh: '{n} 层' },
	residual: { en: 'The residual line carries each block’s input around it and adds it back.', zh: '残差线把每个块的输入绕过它，再加回来。' },
	misconception: { en: 'A common mix-up', zh: '一个常见的误解' },
	truth: { en: 'What actually happens', zh: '实际情况' },
	modelError: { en: 'The model could not be loaded: {e}', zh: '模型加载失败：{e}' },
	noPackage: { en: 'No explainer package: the page is generated from spec.json alone.', zh: '没有讲解包：页面完全由 spec.json 生成。' },
	// Lesson Lab
	labTitle: { en: 'Lesson Lab', zh: '课程实验室' },
	labIntro: {
		en: 'The auto page and lesson of one package, with every fixture, both languages and the lint report. Draft pages are only built here and in the dev server.',
		zh: '一个讲解包的自动页面和课程，带上所有测试输入、两种语言和检查报告。草稿页面只在这里和开发服务器里构建。'
	},
	labFixture: { en: 'Input fixture', zh: '测试输入' },
	labCurrent: { en: '(current input)', zh: '（当前输入）' },
	labLang: { en: 'Language', zh: '语言' },
	labCopy: { en: 'Copy deep link', zh: '复制深层链接' },
	labCopied: { en: 'Copied', zh: '已复制' },
	labCopyFail: { en: 'Copy failed: select the link below.', zh: '复制失败：请手动选择下面的链接。' },
	labLint: { en: 'Lint', zh: '检查' },
	labLinting: { en: 'Linting {k} / {n} fixtures…', zh: '正在检查 {k} / {n} 个测试输入…' },
	labLintAgain: { en: 'Run again', zh: '重新检查' },
	labLintSummary: { en: '{e} errors, {w} warnings over {f} fixtures × {s} steps × en/zh', zh: '{e} 个错误、{w} 个警告（{f} 个测试输入 × {s} 步 × 中英文）' },
	labClean: { en: 'No issues.', zh: '没有问题。' },
	labParity: { en: 'Per-layer parity', zh: '逐层一致性' },
	labNoPerLayer: {
		en: 'No per-layer parity recorded yet (atlas parity writes spec.parity.perLayer). Overall max |diff| {d} against {a}, bar {b}.',
		zh: '还没有逐层一致性记录（atlas parity 会写入 spec.parity.perLayer）。整体最大 |差| {d}，对照 {a}，门槛 {b}。'
	},
	labNoParity: { en: 'No parity recorded yet: run atlas parity.', zh: '还没有一致性记录：请运行 atlas parity。' },
	labSpec: { en: 'Assets', zh: '资源' },
	labSteps: { en: 'Lesson steps', zh: '课程步骤' },
	labWarnings: { en: 'Runtime warnings', zh: '运行时警告' },
	labUnknown: { en: 'No package and no spec for “{slug}”. Pass ?spec=models/<folder>.', zh: '没有“{slug}”的讲解包，也没有 spec。请加上 ?spec=models/<文件夹>。' },
	labDisabled: { en: 'The Lesson Lab is only built in the dev server or with ATLAS_INCLUDE_DRAFTS=1.', zh: '课程实验室只在开发服务器里或设置 ATLAS_INCLUDE_DRAFTS=1 时构建。' }
} satisfies Record<string, I18n>;

export type StrKey = keyof typeof STR;

/** The string for `key` in the current language, with {placeholders} filled. Reactive (reads i18n.lang). */
export function say(key: StrKey, vars?: Record<string, string | number>): string {
	let s = t(STR[key]);
	if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, String(v));
	return s;
}

/** The I18n of `key` with {placeholders} filled in both languages (for props that take Text). */
export function sayI18n(key: StrKey, vars?: Record<string, string | number>): I18n {
	const fill = (s: string) => {
		if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, String(v));
		return s;
	};
	const v: I18n = STR[key];
	return { en: fill(v.en), zh: v.zh === undefined ? undefined : fill(v.zh) };
}
