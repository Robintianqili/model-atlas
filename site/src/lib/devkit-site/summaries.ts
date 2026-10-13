// Chinese titles and short Chinese summaries of the sections of site/DEVKIT.md, shown at the top of each guide page
// (the long body stays English). Keyed by the section's URL slug (guide.ts).
//
// `hash` is sectionHash() of the English section the summary was written for (generated tables left out). When a
// section changes, the build fails (check.ts, "stale summary") until someone re-reads the section, updates the summary
// if needed and sets the new hash that the error message prints. A new or renamed section fails the same way.
//   node src/lib/devkit-site/check.ts --hashes      prints every section's current hash
// Do not put numbers that the kit can change (op or view counts) in a summary: the pages print those from the code.
// The hash is the one of the section's public text (local.ts): DEVKIT.md's local-only blocks (notes about the
// maintainers' own computer) are not on the website, so a summary never mentions them.

export interface SectionZh {
	title: string;
	summary: string;
	hash: string;
}

export const SECTION_ZH: Record<string, SectionZh> = {
	'set-up': {
		title: '准备环境',
		summary:
			'需要 Node 24、npm 和 git；只有类型 A（导出训练好的模型）和可选的 Python 对照才需要 Python 3.10 以上及 torch、numpy、networkx。npm install 同时装上浏览器测试要用的 playwright-core 和做无障碍扫描的 @axe-core/playwright。先运行 npm run atlas -- doctor：它检查所有依赖，并为每个问题给出修复命令。命令行按 ATLAS_PYTHON 环境变量、本机配置文件（不提交）、公共配置文件的顺序选择 Python，并设好 PYTHONPATH、ATLAS_SITE 和 ATLAS_ROOT。你自己的训练和数据脚本也用 atlas py 运行，不要直接用 python。',
		hash: '87cf5bf2b27d'
	},
	'the-three-kinds-of-illustration': {
		title: '三种图解',
		summary:
			'每节课都播放一条“磁带”，也就是一串记录下来的帧；生成帧的方式就是图解的类型。A 类：训练好的模型在学习者的输入上逐层推理，每层都与 PyTorch 对比。B 类：一个小网络在浏览器里实时训练，可选用 PyTorch float64 对照。C 类：一个算法一次一个事件地运行，可选用独立的 Python 实现对照。一个包可以混合几种类型；“来源”（trace、scan、train、process、tape、live-model）把它们接到课程的每一步。',
		hash: '72419829dae7'
	},
	'what-a-package-is': {
		title: '什么是一个包',
		summary:
			'每个模型一个文件夹，工具包按名字找到它；加一个模型不需要修改任何共享文件。必需的是 explainer.ts（清单）和 lesson.ts（课程），另有可选的 trainer.ts、process.ts、claims.test.ts、本地算子、视图和适配器，以及 Python 端和笔记。TypeScript 文件只能从 atlas 工具包和自己的包导入，不能用 Math.random，所有用户能看到的文字都要中英双语。各种 id 只需在包内唯一，预测题的 id 除外。概念页（/train/<slug>/）也是这样一个包，多一个 concept.ts（第 18 节）。',
		hash: '1c49953d8392'
	},
	'walk-through-your-first-package': {
		title: '动手做第一个包',
		summary:
			'以工具包自带的 hello-xor 为标准答案，从头走一遍：用 atlas new 生成脚手架；A 类在 PyTorch 里训练并导出；B 类写训练器；C 类写过程；填写清单和课程；在课程实验室里查看。反复修改时只用 check --quick（不构建、不开浏览器，一致性结果来自缓存）；完成时只运行一次完整的 check，其中包含唯一一次浏览器测试（两轮、无障碍扫描、每一步的总览图和旁白数字清单），哪个关卡没通过就只重跑哪个，并写好笔记；目录条目应写的内容放进清单的 catalog 字段，用 promote --verify 检查。check 和 e2e 会复用正在运行的开发服务器。本站的“教程”页按同样的顺序，用 hello-xor 的真实文件逐步演示。',
		hash: 'e7b6fe31290b'
	},
	'kind-a-export-a-trained-model': {
		title: 'A 类：导出训练好的模型',
		summary:
			'Python 的 SpecBuilder 每调用一次就算出 PyTorch 的值并记录一层；export_model 证明 describe() 与 module(x) 一致，然后写出 spec.json、weights.bin、reference.json、facts.json 和 presets.json。atlas parity 用工具包的 TypeScript 前向计算逐层对比 PyTorch，并检查每个算子的 explain() 与轨迹一致。本节还讲输入适配器（工具包自带的 drawing28、text-bpe、text-char、points2d、series、graph 各有哪些选项：复制一个之前，先看选项是否已经够用；presetsFromAdapter 让适配器自己生成预设）、类别名、答案行（output.row，例如带掩码的模型读 [MASK] 那一行）、侧栏里的答案何时显示、轴名和行名，以及没有合适的核心算子时如何写本地算子。',
		hash: 'f1494da638ea'
	},
	'kind-b-train-live-in-the-browser': {
		title: 'B 类：在浏览器里实时训练',
		summary:
			'在 trainer.ts 里导出一个 TrainerDef：怎么建模型、怎么取一个批次（批次函数也能拿到正在训练的模型）、损失是什么；训练循环由工具包负责。磁带有粗、细两种：粗的每步一帧，细的每步分成前向、损失、反向、更新四帧。学习者换了输入，训练就从头开始；输入为空时训练来源不运行（runOnEmpty: true 例外）。live-model 来源用刚训练出的权重（at: \'first\' 时用起始权重）计算学习者的输入，模型需要 toSpec()。每步更新之后还可以用 project() 施加约束（例如把向量缩放到单位长度），细帧会注明更新被投影过；自动求导另有 clamp、minimum、maximum。可选的 twin.py 用 PyTorch float64 重做同样的训练，每组训练器和选项各比一次，换来更强的徽章；原样包装训练器的过程也能得到这个徽章，对照结果有缓存。',
		hash: 'fa4baf4d2229'
	},
	'kind-c-a-process-an-algorithm-step-by-step': {
		title: 'C 类：一个过程（逐步运行的算法）',
		summary:
			'在 process.ts 里导出一个 ProcessDef：一个生成器，每个事件 yield 一帧，并给每个阶段写标签、旁白和公式（公式里的值也可以是字符串，例如按设定原样显示的学习率）。旁白和公式用到的数都放进那一帧的 metrics。同样的输入、选项和随机种子必须得到同样的帧，随机数只能来自 ctx.rng。过程默认在包的 Web Worker 里运行；输入里有 Worker 收不到的东西（例如 rt.model）时，用 run: \'main\' 留在主线程，每一帧都要远少于 50 毫秒。after 让过程等另一个来源跑完再开始（例如读取训练好的权重）。输入为空、过程不产生帧时，它的步骤显示空输入提示。可选的 reference.py 用 Python 独立地算出同样的帧，atlas parity 逐帧比较；reference: false 表示这个来源没有 Python 参考。',
		hash: 'b26cb321b4ed'
	},
	'the-lesson': {
		title: '课程',
		summary:
			'lesson.ts 是一串步骤：每一步指向一个来源里的一帧和一个视图，带有中英文旁白和出处徽章。可以先用自动规划的步骤，再逐步换成自己写的。预测题和任务在等学习者作答时不会打断播放；下一步的实时来源会提前开始计算。旁白里的每个数都是绑定到计算结果的 {占位符}，不能手写；本节列出格式（包括两位小数的百分比 pct2 和科学记数法 sci）、内置路径和格式化工具，以及怎样叙述很小的变化。预测题的选项、答案和解释可以用 predictAt 从学习者自己的输入算出。输入为空而被跳过的步骤只留下不含数字的句子，所以要把这一步的道理写在不带数字的句子里。舞台在桌面上位于旁白左边、在手机上位于旁白下方，所以旁白说“舞台里”，不说“上面”或“左边”。不是实时算出的事实句要在 claims.test.ts 里有测试；lint 在每个输入和两种语言下检查每一步，相同的运行只记录一次。',
		hash: 'c2c01a50dbc6'
	},
	'views-sections-and-other-ways-to-customize': {
		title: '视图、页面分区和其他定制方式',
		summary:
			'视图负责画出一步的舞台，工具包自带的视图覆盖大多数需要。写视图之前先复用：现成的工具包视图、带参数的工具包视图、用 ViewSlot 和 kit.parts 组合的本地视图，最后才写新的绘图代码。图表只用工具包的两种数值色阶（有正有负的数用发散色阶，比例和概率用一种紫色的顺序色阶），类别用经过色盲检查的类别色并在颜色旁写名字，坐标轴要有标签，数值按比例画。视图要能用键盘操作：滚动框用 keyScroll，文字用 -text 颜色令牌，不用透明度淡化文字。定制分五级：从只写 Python（L0）到完全自定义的页面（L4）；常规做法是 L1，即清单、课程和论断测试。本地视图由 meta.ts 和 View.svelte 组成，describe() 是给读屏软件的文字，不能为空。本节还列出几个工具包视图的选项（行、键和坐标轴的名字，包括交叉注意力的键；公式的有效数字；窄屏下长公式换行；在作答前隐藏答案的 atlas-answer 类；修改输入时保留滑块焦点的 refocus），以及自动页面各部分显示什么；page.trainingCurve 设定训练曲线，page.sections 修改分区的标题、介绍，或隐藏分区。',
		hash: 'ad757f4765c3'
	},
	'checks-and-gates': {
		title: '检查与关卡',
		summary:
			'atlas check <slug> 有三种用法。修改时用 --quick：运行格式、一致性、算子测试、课程、确定性、诚实性、体积预算、类型检查、笔记和所有权，不构建也不开浏览器，一致性和算子测试的结果在输入没变时来自缓存。完成时运行一次完整的 check，再加上只含本包页面的构建和浏览器测试。--fast 重跑除类型检查以外的快速关卡。几个包可以共用一次构建和一轮浏览器测试。检查从简，通过的关卡不重跑：构建者只在最后的完整检查里跑一次浏览器测试；评审者不跑浏览器测试，只读评审材料（总览图、旁白数字清单和无障碍结果），材料缺失或过期时才重跑；合并时运行一次不带包名的 check --all，信任各车道的结果；改了工具包或布局时，挑约 6 个不同类型的包跑一次默认两轮的浏览器测试；只有公开发布前才给每个页面截一张图，做全站检查；主线程长任务（AK603）只警告，从不阻断。浏览器测试默认两轮（1280 像素英文浅色；390 像素中文深色，并在 360 像素检查横向滚动），每一步都做无障碍扫描（AK604），并写出总览图和旁白数字清单。check 和 e2e 会复用这个网站正在运行的开发服务器，或者启动一个并保留下来，空闲 20 分钟后它自动停止。算子测试关卡还会对包自己的视图做视图契约检查；测试文件可以用 // atlas-timeout 放宽时间上限，超时会明确报告；--only 只重跑选中的关卡，其余沿用上次的结果；页面代码行数不计清单里的 catalog 数据。笔记文件 notes/<slug>.md 缺失时只给出警告，因为公开仓库里没有 notes/ 文件夹。质量分三级：draft、reviewed 和 flagship。',
		hash: '4325583d943d'
	},
	'lanes-ports-and-merging': {
		title: '并行工作区（车道）、端口与合并',
		summary:
			'多人并行时，每人在一个“车道”里工作：车道是网站的一份副本（默认在 site/.work/<名字>/site），有自己的开发服务器端口。车道文件夹可以用 ATLAS_LANES_DIR 环境变量或配置文件里的 lanesDir 移到别处，比如另一块磁盘；主站点之外的车道共用一份依赖包（lanes --deps 安装），并从 lane.json 读出仓库根目录和主站点。用 --from-git HEAD 从最近一次提交建车道，主树正在被别人修改也不受影响。merge 只把车道拥有的路径拷回主树，改了别的文件就会被拒绝（AK702）。车道从不修改工具包：遇到工具包的问题，就在包内绕过并写进笔记。车道拥有评审材料 notes/review-<slug>.md；仓库之外的车道把浏览器测试的截图写在车道自己的文件夹里；lanes --clean 删除构建产物和缓存；e2e --as-live 让未合并的包在测试时按上线后的目录条目显示。promote 从清单的 catalog 字段生成目录补丁，--verify 在目录的副本上检查，--apply 在主树中写入并保留每个文件的换行符；只有别的页面也用到的术语才进入网站术语表，atlas glossary 列出只有一个包用到的术语。本节还给出合并几个车道的步骤：逐个试合并、合并、上线、检查、运行一次不带包名的 check --all（各车道的完整检查已经跑过浏览器测试；改了工具包或布局时才带上约 6 个不同类型的包），最后用 lanes --rm 删掉已合并的车道。端口表由代码生成。',
		hash: '3b0c21337e6f'
	},
	'cli-reference': {
		title: '命令行参考',
		summary:
			'所有命令的形式都是 npm run atlas -- <命令>：从检查环境的 doctor、生成脚手架的 new，到导出、对比、运行、课程、检查、构建、浏览器测试、车道、合并、评审和上线。命令表由代码生成，表下的说明补充每个命令的细节，例如浏览器测试的各轮怎样命名、写出哪些文件，以及 lanes 怎样列出车道、安装车道共用的依赖包、删除已合并的车道或清理缓存。',
		hash: '60b4e344767f'
	},
	'error-codes': {
		title: '错误代码',
		summary:
			'每个问题都用同样的格式打印：代码、位置、可能的原因、修复方法和文档链接。代码按百位分组：0xx 环境，1xx 导出，2xx 一致性，3xx 格式与预算，4xx 课程，5xx 诚实性与清单，6xx 浏览器测试，7xx 目录与车道。表格之后，每个代码都有一段说明和修复方法；AK604 列出工具包对常见无障碍问题的现成修法；AK501 说明缺少笔记文件只是警告；AK401 只把单独出现的“…”当作缺失的数值；AK507 要求视图的 meta.ts 不导入 Svelte 模块或其他包的代码；AK605 检查课程应用在不同窗口下的布局模式（960 × 600 及以上为应用模式，页面恰好一屏高）；AK603 统计主线程长任务，只警告，从不阻断检查，也不必为它重跑；AK606 在构建好的站点上统计课程加载的 JS（目前只警告）。“参与贡献”页把它们列成了中英对照表。',
		hash: 'c007de27d398'
	},
	'op-reference': {
		title: '算子参考',
		summary:
			'工具包的核心算子及其 Python 构建器调用，表格由 schema/ops.json 和算子定义生成：标签、类别、放大视图、属性、权重、子步骤，以及自动步骤的第一句话。导出时，构建器调用的属性要与这张表一致（AK106）。“参考 · 算子”页为每个算子给出公式、两种语言的旁白模板和一个演示。',
		hash: 'd84eafe127ea'
	},
	'view-reference': {
		title: '视图参考',
		summary:
			'工具包里每个视图的键、标题、适用的图解类型和最小高度，由各视图的 meta.ts 生成。把键写进一步的 view 字段就能使用；包自己的视图（键为 <slug>/<key>）不列在这张表里。“参考 · 视图”页为每个视图给出演示。',
		hash: 'dcfadd28576f'
	},
	'file-formats': {
		title: '文件格式',
		summary:
			'工具包读写的文件格式：spec.json（1.1 版）、reference.json（atlas-ref-1）、facts.json、atlas-tape/1 磁带（一个 JSON 头加一个二进制文件），以及训练对照用的 train_fixture。',
		hash: '14d80e267280'
	},
	'troubleshooting-and-machine-notes': {
		title: '疑难排解与机器注意事项',
		summary:
			'常见问题和解决办法：文件在 Node 中无法加载（AK503）、占位符显示“…”、两个训练来源的起点不同、过程的 run() 通不过类型检查、Python 太旧、车道的开发服务器看不到修改、主站点之外的车道找不到 vite 或 svelte（运行 lanes --deps）、live-model 的数字与截图不同、深链接失败、来源的输入无法发送给 Worker、run: \'main\' 的过程在主线程上运行、过程步骤显示“先训练网络……”、输入为空时训练步骤只显示提示、reference.py 读不了很长的 --input、e2e 之后开发服务器还在运行、--quick 通过而完整检查失败、工具包组件报 AK604、截图里出现 Vite 错误提示、浏览器测试打开了别的车道的页面、目录补丁无法用脚本应用。还重申：工具包从不下载数据，新的数据集要先得到同意。',
		hash: '1d0632265be4'
	},
	'concept-packages': {
		title: '概念包',
		summary:
			'概念页（/train/<slug>/）讲一个概念而不是一个模型，例如损失与梯度下降。它和模型包是同一种文件夹，通过同样的关卡，不同的只是课程周围的页面。explainer.ts 写 type: \'concept\'，slug 来自概念目录；concept.ts 给出开场问题、玩具、真实模型、对比、常见误解和延伸阅读，其中的舞台和课程用同样的步骤格式。用 atlas new concept 生成（--like 可以复制一个概念包，或把模型包改成概念包），check 照常运行（最后一次完整检查里的浏览器测试打开 /train/<slug>/），promote 输出概念目录的补丁（--verify 和 --apply 同样适用）。',
		hash: '2d6c96f3171f'
	}
};

/** The Chinese summary of CONTRIBUTING.md on /devkit/contribute/ (the file itself is shown in English below it). `hash`
 *  is sectionHash() of the whole file; check.ts fails the build when the file changes, as for the guide's sections. */
export const CONTRIBUTING_ZH: SectionZh = {
	title: '怎样贡献',
	summary:
		'要点：为目录里已有的 slug 做一个模型包或概念包，在自己的 git 分支上工作；只改动自己包的路径（包括评审材料），目录、许可清单和工具包由维护者修改，目录条目应写的内容放进清单的 catalog 字段并用 promote --verify 检查；每个数字都要真实，每个舞台都有出处徽章，中英双语，无障碍扫描不能报严重问题。修改时用 check --quick，最后运行一次完整的 check（其中包含唯一一次浏览器测试，哪个关卡没通过就只重跑哪个），全部通过后再提交拉取请求，并附上评审材料和总览图；改动工具包时，挑约六个不同类型的包跑浏览器测试即可。用 Claude Code 的贡献者可以用仓库里的项目技能 model-atlas-builder，它简要地给出同一套流程和规则。仓库公开在 github.com/Robintianqili/model-atlas；代码采用 MIT 许可证，课程文字、文章和插图采用 CC BY 4.0。每次推送到 main，GitHub Actions 会构建网站并部署到 GitHub Pages；拉取请求的持续集成还在计划中。',
	hash: '0dcc3882ec4f'
};

/** The Chinese of CONTRIBUTING.md's status note (its block quote), shown on /devkit/ in the Chinese page next to the
 *  start (home.ts contributingStart). It is covered by CONTRIBUTING_ZH.hash: when the note changes, the build fails until
 *  this follows; when the file has no status note any more (the repository is public), set it to '' (check.ts). */
export const CONTRIBUTING_STATUS_ZH: string =
	'仓库公开在 github.com/Robintianqili/model-atlas，网站地址是 robintianqili.github.io/model-atlas。代码采用 MIT 许可证（LICENSE），课程文字、文章和插图采用 CC BY 4.0（LICENSE-CONTENT.md）；数据集、词表、字体和模型保留各自的许可证（LICENSING.md、LICENSES/ 中的许可证全文，以及 site/static/licenses.json）。';
