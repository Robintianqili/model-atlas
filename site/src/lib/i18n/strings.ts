/**
 * UI string dictionary. Every key has English and Chinese. Use `ui('key')` in components.
 * Placeholders: `{name}` is replaced from the vars object: ui('stepOf', { i: 3, n: 10 }).
 * Page-specific prose does not belong here: write it inline with t({ en, zh }) or as an article.
 *
 * Three dictionaries (redesign wave F): UI is the site-wide vocabulary that every page carries; LESSON_UI (the lesson
 * app: views, Next, Start, Skip, feedback, cover, end card) and CATALOG_UI (cards, filters, paths, the atlas) are
 * separate exports, so a page bundles them only when it uses lui() / cui() (i18n.svelte.ts).
 */
export const UI = {
	// site chrome
	siteTagline: { en: 'See every layer of a model work on your own input.', zh: '用你自己的输入，看模型每一层如何工作。' },
	skipToContent: { en: 'Skip to content', zh: '跳到正文' },
	navAtlas: { en: 'Atlas', zh: '图鉴' },
	navTrain: { en: 'Training concepts', zh: '训练概念' },
	navGlossary: { en: 'Glossary', zh: '术语表' },
	navAbout: { en: 'About', zh: '关于' },
	navDevkit: { en: 'Build your own', zh: '自己动手' },
	navKit: { en: 'Component kit', zh: '组件库' },
	navLessons: { en: 'Lessons', zh: '课程' },
	navPaths: { en: 'Paths', zh: '学习路径' },
	navMain: { en: 'Site', zh: '网站导航' },
	searchLabel: { en: 'Search lessons and terms', zh: '搜索课程与术语' },
	searchOpen: { en: 'Search', zh: '搜索' },
	footerLearn: { en: 'Learn', zh: '学习' },
	footerBuild: { en: 'Build', zh: '构建' },
	footerAllLessons: { en: 'All lessons', zh: '全部课程' },
	footerPaths: { en: 'Learning paths', zh: '学习路径' },
	footerDevkit: { en: 'DevKit: build a lesson', zh: 'DevKit：自己做一节课' },
	footerSource: { en: 'Source code', zh: '源代码' },
	footerPrivacyLink: { en: 'Privacy', zh: '隐私' },
	footerNumbers: {
		en: 'Every number on this site is computed from your input in your browser, or recorded from a real run and marked so. Nothing you draw or type leaves your device.',
		zh: '本站的每个数，要么是在你的浏览器里根据你的输入算出的，要么来自一次真实运行并标明。你画的、写的内容不会离开你的设备。'
	},
	footerLegal: {
		en: 'Code: MIT. Lesson text, articles and figures: CC BY 4.0. Datasets, vocabularies and fonts keep their own licences.',
		zh: '代码采用 MIT 许可证；课程文字、文章和插图采用 CC BY 4.0；数据集、词表和字体保留各自的许可证。'
	},
	footerTech: {
		en: 'A static site: every model runs or is replayed in your browser. Fonts: Atkinson Hyperlegible Next and JetBrains Mono (SIL OFL). Math: KaTeX (MIT).',
		zh: '纯静态网站：所有模型都在你的浏览器中运行或回放。字体：Atkinson Hyperlegible Next 和 JetBrains Mono（SIL OFL）。公式：KaTeX（MIT）。'
	},
	menu: { en: 'Menu', zh: '菜单' },
	language: { en: 'Language', zh: '语言' },
	theme: { en: 'Theme', zh: '主题' },
	themeLight: { en: 'Light', zh: '浅色' },
	themeDark: { en: 'Dark', zh: '深色' },
	themeSystem: { en: 'System', zh: '跟随系统' },
	themeToggleLabel: { en: 'Color theme: {mode}. Click to change.', zh: '配色：{mode}。点击切换。' },
	footerLicenses: { en: 'Licenses and credits', zh: '许可与致谢' },
	footerHowNumbers: { en: 'How numbers are made', zh: '数字从哪里来' },
	onThisPage: { en: 'On this page', zh: '本页内容' },
	open: { en: 'Open', zh: '打开' },
	close: { en: 'Close', zh: '关闭' },
	show: { en: 'Show', zh: '显示' },
	hide: { en: 'Hide', zh: '隐藏' },
	reset: { en: 'Reset', zh: '重置' },
	notTranslatedYet: { en: 'Not translated yet.', zh: '中文版尚未完成，以下为英文原文。' },

	// provenance + status
	provLive: { en: 'Live', zh: '实时' },
	provPrecomputed: { en: 'Precomputed', zh: '预计算' },
	provIllustrative: { en: 'Illustrative', zh: '示意' },
	provLiveHelp: {
		en: 'Computed in your browser right now, from the real model weights.',
		zh: '此刻在你的浏览器中用真实模型权重计算。'
	},
	provPrecomputedHelp: {
		en: 'Computed offline from the same model and saved with the page.',
		zh: '用同一个模型离线计算后随页面保存。'
	},
	provIllustrativeHelp: {
		en: 'Made-up example values that show the idea. Not output of a real model.',
		zh: '为说明概念而编造的示例数值，并非真实模型的输出。'
	},
	provPrecision: { en: 'Precision', zh: '精度' },
	provTolerance: { en: 'Max |diff| vs {against}', zh: '与 {against} 的最大绝对误差' },
	provApproximate: { en: 'approximate (quantized)', zh: '近似（量化）' },
	provLiveNote: { en: 'computed from your input, now', zh: '根据你的输入现场计算' },
	provPrecomputedNote: { en: 'recorded from a real run', zh: '来自一次真实运行的记录' },
	provIllustrativeNote: { en: 'a drawing, not model output', zh: '示意图，不是模型输出' },
	provLegend: { en: 'Where the numbers come from', zh: '数字的来源' },
	minutes: { en: '≈ {m} min', zh: '约 {m} 分钟' },
	statusLive: { en: 'Live', zh: '可玩' },
	statusPrecomputed: { en: 'Precomputed', zh: '预计算' },
	statusPlanned: { en: 'Planned', zh: '计划中' },
	statusPlannedHelp: { en: 'This page is planned and not built yet.', zh: '此页面在计划中，尚未完成。' },
	tier: { en: 'Tier', zh: '阶段' },
	tierP0: { en: 'P0 · first version', zh: 'P0 · 首个版本' },
	tierV1: { en: 'V1 · flagship of its area', zh: 'V1 · 本领域旗舰' },
	tierP1: { en: 'P1 · next after v1', zh: 'P1 · v1 之后' },
	tierP2: { en: 'P2 · later', zh: 'P2 · 更晚' },

	// catalog vocabulary
	area: { en: 'Area', zh: '领域' },
	areas: { en: 'Areas', zh: '领域' },
	family: { en: 'Family', zh: '模型族' },
	model: { en: 'Model', zh: '模型' },
	models: { en: 'Models', zh: '模型' },
	concept: { en: 'Concept', zh: '概念' },
	concepts: { en: 'Concepts', zh: '概念' },
	input: { en: 'Input', zh: '输入' },
	inputs: { en: 'Inputs', zh: '输入' },
	download: { en: 'Download', zh: '下载大小' },
	params: { en: 'Parameters', zh: '参数量' },
	runtime: { en: 'Runtime', zh: '运行方式' },
	precision: { en: 'Precision', zh: '数值精度' },
	prerequisites: { en: 'Before this, read', zh: '建议先读' },
	whatItWillShow: { en: 'What this page will show', zh: '这个页面将展示什么' },
	howYouPlay: { en: 'How you will play with it', zh: '你将如何互动' },
	engine: { en: 'Engine', zh: '实现方式' },
	linksToModels: { en: 'Models used', zh: '用到的模型' },
	seeAlsoConcepts: { en: 'Concepts that use this model', zh: '用到此模型的概念' },
	plannedPageTitle: { en: 'Planned page', zh: '计划中的页面' },
	plannedPageBody: {
		en: 'This page is not built yet. It is listed so you can see where the atlas is going. Each planned page is built once the shared component kit covers most of it.',
		zh: '此页面尚未完成。列在这里是为了展示图鉴的规划。共享组件库覆盖大部分需求后，计划中的页面会逐一完成。'
	},
	countModels: { en: '{n} models', zh: '{n} 个模型' },
	countConcepts: { en: '{n} concepts', zh: '{n} 个概念' },
	countLive: { en: '{n} live', zh: '{n} 个可玩' },
	countPlanned: { en: '{n} planned', zh: '{n} 个计划中' },
	livePages: { en: 'Live now', zh: '现在可玩' },
	plannedPages: { en: 'Planned', zh: '计划中' },
	browseAtlas: { en: 'Browse the atlas', zh: '浏览图鉴' },
	section: { en: 'Section', zh: '章节' },
	inV1: { en: 'v1 page', zh: 'v1 页面' },

	// input kinds
	in_drawing: { en: 'Drawing', zh: '手绘' },
	in_image: { en: 'Image', zh: '图片' },
	in_webcam: { en: 'Webcam', zh: '摄像头' },
	in_text: { en: 'Text', zh: '文本' },
	in_audio: { en: 'Audio', zh: '音频' },
	in_series: { en: 'Time series', zh: '时间序列' },
	in_tabular: { en: 'Table row', zh: '表格数据' },
	in_points2d: { en: '2D points', zh: '二维点集' },
	in_graph: { en: 'Graph', zh: '图' },
	in_game: { en: 'Game state', zh: '游戏状态' },
	in_protein: { en: 'Protein sequence', zh: '蛋白质序列' },
	in_ratings: { en: 'Ratings', zh: '评分' },
	in_conversation: { en: 'Conversation', zh: '对话' },
	in_video: { en: 'Video', zh: '视频' },
	in_pointcloud: { en: 'Point cloud', zh: '点云' },
	in_tokens: { en: 'Token sequence', zh: '词元序列' },
	in_none: { en: 'No input (analytic)', zh: '无需输入（解析）' },

	// tensors and numbers
	shape: { en: 'Shape', zh: '形状' },
	value: { en: 'Value', zh: '数值' },
	row: { en: 'Row', zh: '行' },
	col: { en: 'Col', zh: '列' },
	index: { en: 'Index', zh: '索引' },
	channel: { en: 'Channel', zh: '通道' },
	channels: { en: 'Channels', zh: '通道' },
	min: { en: 'Min', zh: '最小值' },
	max: { en: 'Max', zh: '最大值' },
	mean: { en: 'Mean', zh: '均值' },
	std: { en: 'Std', zh: '标准差' },
	sparsity: { en: 'Zeros', zh: '零值占比' },
	count: { en: 'Count', zh: '元素数' },
	histogram: { en: 'Histogram', zh: '直方图' },
	colorScope: { en: 'Color scale range', zh: '色阶范围' },
	scopeUnit: { en: 'This map', zh: '本图' },
	scopeLayer: { en: 'Whole layer', zh: '整层' },
	scopeGlobal: { en: 'All layers', zh: '所有层' },
	scopeHelp: {
		en: 'Which values set the darkest color: only this map, the whole layer, or every layer.',
		zh: '决定最深颜色对应的数值范围：仅本图、整层或所有层。'
	},
	negative: { en: 'negative', zh: '负' },
	positive: { en: 'positive', zh: '正' },
	zero: { en: 'zero', zh: '零' },
	low: { en: 'low', zh: '低' },
	high: { en: 'high', zh: '高' },
	elided: { en: '{n} more', zh: '另有 {n} 个' },
	cellAt: { en: 'Row {r}, column {c}: {v}', zh: '第 {r} 行，第 {c} 列：{v}' },
	itemAt: { en: 'Index {i}: {v}', zh: '索引 {i}：{v}' },
	gridKeysHelp: { en: 'Use the arrow keys to move between cells.', zh: '用方向键在单元格之间移动。' },

	// roles
	roleActivation: { en: 'Activation', zh: '激活值' },
	roleParameter: { en: 'Parameter (weight)', zh: '参数（权重）' },
	roleGradient: { en: 'Gradient', zh: '梯度' },
	roleCache: { en: 'Cached tensor', zh: '缓存张量' },
	roleOptimizer: { en: 'Optimizer state', zh: '优化器状态' },
	frozen: { en: 'Frozen', zh: '冻结' },
	trainable: { en: 'Trainable', zh: '可训练' },

	// player
	play: { en: 'Play', zh: '播放' },
	pause: { en: 'Pause', zh: '暂停' },
	stepBack: { en: 'Previous step', zh: '上一步' },
	stepForward: { en: 'Next step', zh: '下一步' },
	restart: { en: 'Back to start', zh: '回到开头' },
	speed: { en: 'Speed', zh: '速度' },
	stepOf: { en: 'Step {i} of {n}', zh: '第 {i} / {n} 步' },
	scrub: { en: 'Scrub through the steps', zh: '拖动以浏览步骤' },
	reducedMotionNote: {
		en: 'Reduced motion is on, so steps advance only when you press Next.',
		zh: '已开启减少动态效果，只有点击“下一步”时才前进。'
	},

	// guided lesson (Walkthrough)
	secLesson: { en: 'Step by step', zh: '分步课程' },
	lessonStart: { en: 'Start the step-by-step lesson', zh: '开始分步课程' },
	lessonHomeTitle: { en: 'Start a step-by-step lesson', zh: '开始一节分步课程' },
	lessonSteps: { en: '{n} steps', zh: '{n} 步' },
	lessonPlay: { en: 'Play lesson', zh: '自动播放' },
	lessonPause: { en: 'Pause', zh: '暂停' },
	lessonPrev: { en: 'Back', zh: '上一步' },
	lessonNext: { en: 'Next', zh: '下一步' },
	lessonRestart: { en: 'Restart lesson', zh: '重新开始' },
	lessonAllSteps: { en: 'All steps', zh: '全部步骤' },
	lessonLook: { en: 'Look at', zh: '看这里' },
	lessonProgress: { en: 'Lesson progress', zh: '课程进度' },
	lessonGoTo: { en: 'Step {i}: {title}', zh: '第 {i} 步：{title}' },
	lessonKeys: { en: 'Keys: ← and → move between steps; Home and End jump to the first and last.', zh: '按键：← 和 → 切换步骤；Home 和 End 跳到第一步和最后一步。' },
	lessonWaiting: { en: 'Playing this step’s animation…', zh: '正在播放这一步的动画…' },
	lessonReduced: {
		en: 'Reduced motion is on: the lesson moves only when you press Next.',
		zh: '已开启减少动态效果：只有点击“下一步”时课程才会前进。'
	},
	lessonEnd: { en: 'End of the lesson', zh: '课程结束' },
	exploreFreely: { en: 'Explore freely', zh: '自由探索' },
	exploreFreelyIntro: {
		en: 'Everything from the lesson, all at once. Click any layer, map or number.',
		zh: '课程里的一切都摆在这里。点任意一层、一张图或一个数。'
	},
	notFoundTitle: { en: 'This page is not here', zh: '找不到这个页面' },
	notFoundText: {
		en: 'The link may be old, or the page may not exist yet. Most models in the atlas are still planned.',
		zh: '链接可能已过期，或者这个页面还不存在。图鉴里的大多数模型仍在计划中。'
	},
	notFoundHome: { en: 'Go to the home page', zh: '回到首页' },

	// draw pad
	drawHere: { en: 'Draw here', zh: '在这里画' },
	drawPadLabel: {
		en: 'Drawing pad. Draw with a mouse, finger or pen. Keyboard users can pick a preset instead.',
		zh: '画板。可用鼠标、手指或触控笔绘画。键盘用户可以选择预设图案。'
	},
	clear: { en: 'Clear', zh: '清空' },
	undo: { en: 'Undo', zh: '撤销' },
	brush: { en: 'Brush', zh: '笔刷' },
	presets: { en: 'Presets', zh: '预设' },
	strokes: { en: '{n} strokes', zh: '{n} 笔' },
	emptyCanvas: { en: 'The pad is empty. Draw something or pick a preset.', zh: '画板是空的。画点什么，或选择一个预设。' },

	// text input
	typeHere: { en: 'Type here', zh: '在这里输入' },
	charCount: { en: '{n} / {max} characters', zh: '{n} / {max} 个字符' },
	limitReached: { en: 'Length limit reached', zh: '已达长度上限' },
	staysOnDevice: { en: 'Stays on your device', zh: '仅在你的设备上处理' },
	run: { en: 'Run', zh: '运行' },

	// inspector
	inspector: { en: 'Layer inspector', zh: '层检查器' },
	tabValues: { en: 'Values', zh: '数值' },
	tabStats: { en: 'Stats', zh: '统计' },
	tabMath: { en: 'Math', zh: '数学' },
	tabCode: { en: 'Code', zh: '代码' },
	prevLayer: { en: 'Previous layer', zh: '上一层' },
	nextLayer: { en: 'Next layer', zh: '下一层' },
	layerKeys: { en: 'Press [ and ] to move between layers.', zh: '按 [ 和 ] 在各层之间切换。' },
	layer: { en: 'Layer', zh: '层' },
	layers: { en: 'Layers', zh: '层' },
	overview: { en: 'Model overview', zh: '模型总览' },
	noDataYet: { en: 'Nothing to show yet.', zh: '暂无数据。' },

	// outputs
	probability: { en: 'Probability', zh: '概率' },
	chosen: { en: 'chosen', zh: '选中' },
	topK: { en: 'Top {k}', zh: '前 {k} 个' },
	token: { en: 'Token', zh: '词元' },
	tokens: { en: 'Tokens', zh: '词元' },
	tokenId: { en: 'ID', zh: 'ID' },
	bytes: { en: 'Bytes', zh: '字节' },
	position: { en: 'Position', zh: '位置' },
	special: { en: 'special token', zh: '特殊词元' },
	query: { en: 'Query', zh: '查询' },
	key: { en: 'Key', zh: '键' },
	masked: { en: 'masked (future)', zh: '已遮蔽（未来位置）' },
	attentionLabel: {
		en: 'Attention weights: each row is a query token, each column a key token.',
		zh: '注意力权重：每行是一个查询词元，每列是一个键词元。'
	},
	formulaWhere: { en: 'where', zh: '其中' },

	// page templates (§4.2 and §4.4)
	secGoals: { en: 'Goals', zh: '学习目标' },
	secPredict: { en: 'Predict first', zh: '先预测' },
	secFacts: { en: 'Facts', zh: '概况' },
	secInput: { en: 'Your input', zh: '你的输入' },
	secOverview: { en: 'Overview', zh: '总览' },
	secInspector: { en: 'Inside a layer', zh: '层的内部' },
	secUnit: { en: 'The arithmetic', zh: '具体计算' },
	secOutput: { en: 'Output', zh: '输出' },
	secTryThis: { en: 'Try this', zh: '试一试' },
	secTraining: { en: 'Training', zh: '训练' },
	secArticle: { en: 'How it works', zh: '原理' },
	secCode: { en: 'Code', zh: '代码' },
	secLinks: { en: 'Concepts, comparisons, next', zh: '相关概念、对比与下一步' },
	secCheckpoint: { en: 'Checkpoint', zh: '检查点' },
	secHook: { en: 'The question', zh: '问题' },
	secToy: { en: 'Toy', zh: '玩具实验' },
	secReal: { en: 'On a real model', zh: '在真实模型上' },
	secCompare: { en: 'Compare', zh: '对比' },
	secPitfalls: { en: 'Pitfalls and misconceptions', zh: '常见误区' },
	secPost: { en: 'Back to your prediction', zh: '回到你的预测' },
	secFurther: { en: 'Links and further reading', zh: '链接与延伸阅读' },
	credits: { en: 'Credits and licenses', zh: '致谢与许可' },
	yourPrediction: { en: 'Your prediction', zh: '你的预测' },
	noPredictionYet: { en: 'You have not made a prediction yet.', zh: '你还没有做出预测。' },
	revealAnswer: { en: 'Show the answer', zh: '显示答案' },
	answer: { en: 'Answer', zh: '答案' },
	correct: { en: 'Correct', zh: '正确' },
	notQuite: { en: 'Not quite', zh: '不太对' },
	savePrediction: { en: 'Lock in my prediction', zh: '确定我的预测' },
	changePrediction: { en: 'Change', zh: '修改' },
	predictionSaved: { en: 'Saved. You will see it again at the end of the page.', zh: '已保存。你会在页面末尾再次看到它。' },
	glossaryLink: { en: 'Open in glossary', zh: '在术语表中查看' },
	launcherDraw: { en: 'Draw', zh: '画画' },
	launcherType: { en: 'Type', zh: '打字' },
	launcherTitle: { en: 'Start from your own input', zh: '从你自己的输入开始' },
	launcherAccepts: { en: 'Models that accept this input', zh: '接受这种输入的模型' },
	launcherNothingLoaded: {
		en: 'Nothing is downloaded until you open a model.',
		zh: '在你打开某个模型之前，不会下载任何东西。'
	}
} as const;

export type UIKey = keyof typeof UI;

/**
 * The lesson app (redesign spec §4.4, §5): view tabs, the action bar, feedback, the cover and the end card. Read with
 * lui('key', vars) from i18n.svelte.ts.
 */
export const LESSON_UI = {
	// views (hash-addressed panels of a lesson route)
	views: { en: 'Views', zh: '视图' },
	viewLesson: { en: 'Lesson', zh: '课程' },
	viewExplore: { en: 'Explore', zh: '探索' },
	viewArticle: { en: 'How it works', zh: '原理' },
	viewCode: { en: 'Code', zh: '代码' },
	viewTraining: { en: 'Training', zh: '训练' },
	viewTrainingLong: { en: 'Training and data', zh: '训练与数据' },
	viewPitfalls: { en: 'Pitfalls', zh: '常见误区' },
	viewSources: { en: 'Sources', zh: '资料来源' },
	viewTryThis: { en: 'Try this', zh: '试一试' },
	viewCheck: { en: 'Check yourself', zh: '自测' },
	backToLesson: { en: 'Back to the lesson · step {k} of {n}', zh: '回到课程 · 第 {k} / {n} 步' },
	// landmarks and labels
	stage: { en: 'Stage', zh: '演示区' },
	steps: { en: 'Steps', zh: '步骤' },
	controls: { en: 'Lesson controls', zh: '课程控制' },
	more: { en: 'More', zh: '更多' },
	aboutModel: { en: 'About this model', zh: '关于这个模型' },
	aboutLesson: { en: 'About this lesson', zh: '关于这节课' },
	crumbAtlas: { en: 'Atlas', zh: '图鉴' },
	// eyebrow and progress
	stepOf: { en: 'Step {k} of {n}', zh: '第 {k} / {n} 步' },
	stepOfIn: { en: '{chapter} · step {k} of {n}', zh: '{chapter} · 第 {k} / {n} 步' },
	goToStep: { en: 'Step {k}: {title}', zh: '第 {k} 步：{title}' },
	animated: { en: 'animated step', zh: '带动画的步骤' },
	done: { en: 'done', zh: '已完成' },
	// the action bar and the Next button
	back: { en: 'Back', zh: '上一步' },
	play: { en: 'Play', zh: '播放' },
	pause: { en: 'Pause', zh: '暂停' },
	next: { en: 'Next', zh: '下一步' },
	nextStep: { en: 'Next · step {k}', zh: '下一步 · 第 {k} 步' },
	start: { en: 'Start', zh: '开始' },
	startStep: { en: 'Start · step {k}', zh: '开始 · 第 {k} 步' },
	resumeAt: { en: 'Resume at step {k}', zh: '从第 {k} 步继续' },
	startOver: { en: 'Start over', zh: '从头开始' },
	resumed: { en: 'Resumed at step {k}', zh: '已回到第 {k} 步' },
	continue: { en: 'Continue', zh: '继续' },
	continueStep: { en: 'Continue · step {k}', zh: '继续 · 第 {k} 步' },
	skip: { en: 'Skip', zh: '跳过' },
	skipStep: { en: 'Skip · step {k}', zh: '跳过 · 第 {k} 步' },
	finish: { en: 'Finish', zh: '完成' },
	restart: { en: 'Restart the lesson', zh: '重新开始这节课' },
	// narration parts
	look: { en: 'Look', zh: '看这里' },
	why: { en: 'Why?', zh: '为什么？' },
	yourTurn: { en: 'Your turn', zh: '轮到你了' },
	predict: { en: 'Predict', zh: '先预测' },
	showMe: { en: 'Show me', zh: '直接看答案' },
	answerFirst: { en: 'Answer first.', zh: '先作答。' },
	answerFirstBody: { en: 'The stage shows its numbers once you commit.', zh: '你作答之后，演示区才会显示数字。' },
	right: { en: 'Right.', zh: '答对了。' },
	notQuite: { en: 'Not quite.', zh: '不太对。' },
	whyItMatters: { en: 'Why it matters', zh: '为什么重要' },
	showAsText: { en: 'Show as text', zh: '以文字显示' },
	showNumbers: { en: 'Show the {n} numbers', zh: '显示这 {n} 个数' },
	// the input chip and sheet
	yourInput: { en: 'Your input', zh: '你的输入' },
	yourDrawing: { en: 'Your drawing', zh: '你的画' },
	yourText: { en: 'Your text', zh: '你的文字' },
	edit: { en: 'Edit', zh: '修改' },
	redraw: { en: 'Redraw', zh: '重画' },
	// the cover
	coverMeta: { en: 'Lesson · {n} steps · ≈ {m} min · in your browser', zh: '课程 · {n} 步 · 约 {m} 分钟 · 在你的浏览器中运行' },
	byTheEnd: { en: 'By the end you can', zh: '学完后你能' },
	showsAtStep: { en: 'The lesson shows you at step {k}.', zh: '课程会在第 {k} 步告诉你答案。' },
	// the end card
	recap: { en: 'What you saw', zh: '你看到了什么' },
	predictions: { en: 'Your predictions: {k} of {n} right', zh: '你的预测：{n} 题答对 {k} 题' },
	upNext: { en: 'Up next', zh: '下一课' },
	goDeeper: { en: 'Go deeper', zh: '深入了解' },
	exploreNote: { en: 'Every layer and number, freely', zh: '自由查看每一层、每个数' },
	articleNote: { en: 'The idea in one article', zh: '一篇文章讲清原理' },
	codeNote: { en: 'The code, and how it was checked', zh: '代码，以及如何核对' },
	trainingNote: { en: 'How it learned, and from what', zh: '它怎么学的，用了什么数据' },
	// the About sheet
	parity: { en: 'Checked against {against} on {n} inputs: every layer within {tol}.', zh: '在 {n} 个输入上与 {against} 对照：每一层的误差都在 {tol} 以内。' },
	learnedNumbers: { en: '{n} learned numbers', zh: '{n} 个学到的参数' },
	keys: { en: 'Keys: ← → steps · Enter next · 1-4 answer · Esc close', zh: '按键：← → 切换步骤 · Enter 下一步 · 1-4 作答 · Esc 关闭' }
} as const;
export type LessonUIKey = keyof typeof LESSON_UI;

/**
 * Catalog pages (redesign spec §4.5, §6): cards, rows, filters, paths, areas. Read with cui('key', vars) from
 * i18n.svelte.ts.
 */
export const CATALOG_UI = {
	lessonsCount: { en: '{n} step-by-step lessons', zh: '{n} 节分步课程' },
	lessonsN: { en: '{n} lessons', zh: '{n} 节课' },
	methodsCount: { en: '{n} training methods', zh: '{n} 种训练方法' },
	liveAndPlanned: { en: '{n} live · {m} planned', zh: '{n} 节可学 · {m} 节计划中' },
	liveN: { en: '{n} live', zh: '{n} 节可学' },
	plannedN: { en: 'Planned · {n}', zh: '计划中 · {n}' },
	notLiveYet: { en: 'Not live yet', zh: '尚未上线' },
	planned: { en: 'Planned', zh: '计划中' },
	coming: { en: 'coming', zh: '即将上线' },
	comingN: { en: '{n} coming', zh: '{n} 节即将上线' },
	allN: { en: 'All {n}', zh: '全部 {n}' },
	andMore: { en: 'and {n} more', zh: '另外 {n} 个' },
	stepsMin: { en: '{n} steps · ≈ {m} min', zh: '{n} 步 · 约 {m} 分钟' },
	pathMeta: { en: '{n} lessons · ≈ {m} min', zh: '{n} 节课 · 约 {m} 分钟' },
	startWith: { en: 'Start with {name}', zh: '从 {name} 开始' },
	browseAll: { en: 'Browse all {n} lessons', zh: '浏览全部 {n} 节课' },
	byInput: { en: 'Or start from your kind of input', zh: '或者从你的输入类型开始' },
	startHere: { en: 'Start here', zh: '从这里开始' },
	paths: { en: 'Learning paths', zh: '学习路径' },
	byArea: { en: 'Browse by area', zh: '按领域浏览' },
	methods: { en: 'Training methods', zh: '训练方法' },
	modelsByArea: { en: 'Models by area', zh: '各领域的模型' },
	allLessons: { en: 'All lessons', zh: '全部课程' },
	followIt: { en: 'Follow it step by step', zh: '一步一步跟着看' },
	kind: { en: 'Kind', zh: '类型' },
	kindAll: { en: 'All', zh: '全部' },
	kindModels: { en: 'Models', zh: '模型' },
	kindMethods: { en: 'Training methods', zh: '训练方法' },
	layout: { en: 'Layout', zh: '布局' },
	rows: { en: 'Rows', zh: '列表' },
	pictures: { en: 'Pictures', zh: '图卡' },
	filters: { en: 'Filters', zh: '筛选' },
	jumpTo: { en: 'Jump to', zh: '跳到' },
	noMatch: { en: 'Nothing matches “{q}”.', zh: '没有与“{q}”匹配的内容。' },
	clearFilters: { en: 'Clear filters', zh: '清除筛选' },
	usedIn: { en: 'Used in', zh: '用在' },
	related: { en: 'Related terms', zh: '相关术语' },
	willShow: { en: 'What this lesson will show', zh: '这节课将展示什么' },
	liveFirst: { en: 'Before this, try', zh: '可以先学' },
	notFoundBody: { en: 'This page is not here. Search, or start from the lessons.', zh: '找不到这个页面。可以搜索，或者从课程开始。' }
} as const;
export type CatalogUIKey = keyof typeof CATALOG_UI;
