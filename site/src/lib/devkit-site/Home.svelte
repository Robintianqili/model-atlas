<!--
  /devkit/: what the DevKit is and who it is for, the three kinds with a live example lesson each, the workflow, the
  afternoon test and the quick start. The counts, examples, commands, quick start and test result are read from the
  kit's sources at build time (home.ts).
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import Html from './Html.svelte';
	import type { HomeData, T2 } from './types.ts';

	let { page }: { page: HomeData } = $props();

	const who: Record<string, T2> = {
		you: { en: 'by you', zh: '由你完成' },
		reviewer: { en: 'by a teaching reviewer', zh: '由教学评审者完成' },
		maintainers: { en: 'by the maintainers', zh: '由维护者完成' }
	};
	const yours = $derived(page.workflow.filter((w) => w.who === 'you').length);
	const mins = $derived(Math.floor(page.afternoon.seconds / 60));
	const secs = $derived(page.afternoon.seconds % 60);
	const guideSec = (n: number) => page.sections.find((s) => s.n === n);
	const counts = $derived([
		{ n: page.counts.sections, label: { en: 'guide sections', zh: '个指南章节' }, href: '/devkit/guide/' },
		{ n: page.counts.ops, label: { en: 'core ops', zh: '个核心算子' }, href: '/devkit/reference/ops/' },
		{ n: page.counts.views, label: { en: 'kit views', zh: '个工具包视图' }, href: '/devkit/reference/views/' },
		{ n: page.counts.commands, label: { en: 'CLI commands', zh: '条命令' }, href: `/devkit/guide/${guideSec(12)?.slug ?? ''}/` },
		{ n: page.counts.codes, label: { en: 'error codes', zh: '个错误代码' }, href: '/devkit/contribute/#error-codes' }
	]);
</script>

<div class="container home">
	<header class="hero">
		<p class="eyebrow">DevKit · {t({ en: 'Build your own', zh: '自己动手' })}</p>
		<h1>{t({ en: 'Build your own illustration', zh: '自己动手做一个模型图解' })}</h1>
		<p class="lede">
			{t({
				en: 'The DevKit is the toolkit for building this site’s guided lessons: a command line, a kit of views and a lesson format. If you know PyTorch, you can add a model. You write a few files of plain data and functions; the kit makes the page, the step-by-step lesson with real numbers and the checks. You do not need to know Svelte.',
				zh: 'DevKit 是制作本站分步课程的工具包：一个命令行、一套视图和一种课程格式。只要你会 PyTorch，就能加一个模型。你只写几个文件，内容是普通的数据和函数；页面、带真实数字的分步课程和各项检查都由工具包生成。你不需要会 Svelte。'
			})}
		</p>
		<div class="cta">
			<a class="btn primary" href="{base}/devkit/tutorial/">{t({ en: 'Start the tutorial', zh: '从教程开始' })}</a>
			<a class="btn" href="{base}/devkit/guide/">{t({ en: 'Read the guide', zh: '阅读指南' })}</a>
		</div>
		{#if page.start.statusHtml}
			<div class="status" role="note">
				{#if i18n.lang === 'zh' && page.start.statusZh}
					<p lang="zh-CN">{page.start.statusZh}</p>
				{:else}
					<Html html={page.start.statusHtml} />
				{/if}
				<p class="small muted">{t({ en: 'From CONTRIBUTING.md:', zh: '来自 CONTRIBUTING.md：' })} <a href="{base}/devkit/contribute/#how">{t({ en: 'the contributor path', zh: '贡献者的路径' })}</a></p>
			</div>
		{/if}
		<ul class="counts" aria-label={t({ en: 'The kit today', zh: '工具包现状' })}>
			{#each counts as c (c.href)}
				<li><a href={`${base}${c.href}`}><span class="n num">{c.n}</span> <span class="l">{t(c.label)}</span></a></li>
			{/each}
		</ul>
		{#if page.built.live}
			<p class="built small">
				{t({
					en: `${page.built.kit.length} of the ${page.built.live} live model lessons are built with it so far${page.built.kit.length ? ':' : '.'}`,
					zh: `本站已上线的 ${page.built.live} 个模型课程中，目前有 ${page.built.kit.length} 个是用它做的${page.built.kit.length ? '：' : '。'}`
				})}
				{#each page.built.kit as k, i (k.slug)}{i ? ' · ' : ' '}<a href="{base}/m/{k.slug}/">{t(k.name)}</a>{/each}{page.built.kit.length ? t({ en: '.', zh: '。' }) : ''}
				{t({ en: 'The others were made by hand, without the kit.', zh: '其余的是不用工具包、手工做的页面。' })}
			</p>
		{/if}
		<p class="gen muted">
			{t({ en: 'These numbers, the guide, the reference and the tutorial’s files and results are read from the kit’s own files when the site is built, so the pages grow with the kit.', zh: '这些数字、指南、参考，以及教程里的文件和结果，都在网站构建时从工具包自己的文件读出，所以页面会随工具包一起增长。' })}
		</p>
	</header>

	<section aria-labelledby="who-h">
		<h2 id="who-h">{t({ en: 'Who it is for', zh: '适合谁' })}</h2>
		<div class="who">
			<div class="panel">
				<h3>{t({ en: 'You know PyTorch', zh: '你会 PyTorch' })}</h3>
				<p>{t({ en: 'Train a small model as you always do, describe its forward pass once in Python, and the kit checks every layer of its browser copy against PyTorch. Then you write the lesson as typed data.', zh: '像平时一样训练一个小模型，用 Python 把它的前向计算描述一遍，工具包会把浏览器里的版本逐层与 PyTorch 对比。然后你用带类型的数据写课程。' })}</p>
			</div>
			<div class="panel">
				<h3>{t({ en: 'You write JavaScript or teach', zh: '你写 JavaScript，或者你是老师' })}</h3>
				<p>{t({ en: 'Live training and step-by-step algorithms need Node only: a trainer or a process in TypeScript, then the lesson. A Python twin is optional and earns a stronger badge.', zh: '实时训练和逐步运行的算法只需要 Node：用 TypeScript 写一个训练器或过程，再写课程。Python 对照是可选的，能换来更强的徽章。' })}</p>
			</div>
			<div class="panel">
				<h3>{t({ en: 'You care about honest numbers', zh: '你在意数字是否诚实' })}</h3>
				<p>{t({ en: 'Every number in a lesson is bound to a computed value, every stage carries a provenance badge, and every claim that is not computed live has a test. The checks enforce it, not good intentions.', zh: '课程里的每个数都绑定到计算结果，每个舞台都有出处徽章，每个不是实时算出的论断都有测试。靠的是自动检查，而不是自觉。' })}</p>
			</div>
		</div>
	</section>

	<section aria-labelledby="kinds-h">
		<h2 id="kinds-h">{t({ en: 'Three kinds of illustration', zh: '三种图解' })}</h2>
		<p class="intro">
			{t({ en: 'Every lesson plays a tape of recorded frames; what makes the frames is the kind. One package can mix kinds. Each kind below links to a live lesson built with the kit.', zh: '每节课都播放一条记录好的“磁带”；生成帧的方式就是图解的类型。一个包可以混合几种类型。下面每种类型都链接到一节用工具包做成、已经上线的课程。' })}
			<a href="{base}/devkit/guide/{guideSec(2)?.slug ?? ''}/">{t({ en: 'Guide section 2', zh: '指南第 2 节' })}</a>
		</p>
		<div class="kinds">
			{#each page.kinds as k (k.kind)}
				<article class="kind panel">
					<p class="badge"><span class="letter" aria-hidden="true">{k.kind}</span> {t({ en: `Kind ${k.kind}`, zh: `${k.kind} 类` })}</p>
					<h3>{t(k.name)}</h3>
					<p>{t(k.what)}</p>
					<p class="small"><strong>{t({ en: 'You write:', zh: '你要写：' })}</strong> {t(k.youWrite)}</p>
					<div class="ex">
						<p class="small muted">{t({ en: 'Live example', zh: '已上线的示例' })}</p>
						<a class="exlink" href="{base}/m/{k.example.slug}/#lesson">
							<strong>{t(k.example.name)}</strong>
							{#if k.example.steps}<span class="chip">{t({ en: `${k.example.steps} steps`, zh: `${k.example.steps} 步` })}</span>{/if}
						</a>
						{#if k.example.summary}<p class="small">{t(k.example.summary)}</p>{/if}
					</div>
				</article>
			{/each}
		</div>
	</section>

	<section aria-labelledby="flow-h">
		<h2 id="flow-h">{t({ en: 'From an idea to a live page', zh: '从想法到上线的页面' })}</h2>
		<p class="intro">{t({ en: `${page.workflow.length} steps, each with its commands. The first ${yours} are yours; a teaching reviewer and the maintainers do the rest.`, zh: `${page.workflow.length} 个步骤，每一步都有对应的命令。前 ${yours} 步由你完成，其余由教学评审者和维护者完成。` })}</p>
		<ol class="flow">
			{#each page.workflow as w, i (w.id)}
				<li class="node" class:theirs={w.who !== 'you'}>
					<span class="step num" aria-hidden="true">{i + 1}</span>
					<span class="name">{t(w.name)}</span>
					<span class="cmds">{#each w.cmds as c (c.name)}<code title={c.summary}>atlas {c.name}</code>{/each}</span>
					<span class="what">{t(w.what)}</span>
					<span class="by">{t(who[w.who])}</span>
				</li>
			{/each}
		</ol>
	</section>

	<section aria-labelledby="test-h" class="test">
		<h2 id="test-h">{t({ en: 'The afternoon test', zh: '“一个下午”测试' })}</h2>
		<div class="testcard panel">
			<p class="big num">{t({ en: `${mins} min ${secs} s`, zh: `${mins} 分 ${secs} 秒` })}</p>
			<div>
				<p>
					{t({
						en: `Before the guide was published, a fresh agent that had only the guide built ${page.afternoon.built}, an XOR network with a sigmoid output, from the scaffold to a draft that passes every automated gate, in ${mins} min ${secs} s of wall-clock time.`,
						zh: `指南发布之前，一个只拿到指南的全新智能体从脚手架开始，做出了 ${page.afternoon.built}（一个输出层用 sigmoid 的 XOR 网络），直到草稿通过所有自动关卡，实际用时 ${mins} 分 ${secs} 秒。`
					})}
				</p>
				<p>
					{t({
						en: `It also listed ${page.afternoon.gaps} places where the guide was unclear or wrong; the guide on this site is the version after those fixes. A person needs longer than an agent: the reading, the training and the lesson writing take most of an afternoon, which is the test's name and its target.`,
						zh: `它还列出了指南中 ${page.afternoon.gaps} 处不清楚或有错的地方；本站的指南是修正之后的版本。人比智能体需要更多时间：阅读、训练和写课程会占去一个下午的大部分，这正是测试名字的由来，也是它的目标。`
					})}
				</p>
				<p class="small muted">{t({ en: 'The full record: research/devkit/afternoon-test.md in the repository.', zh: '完整记录：仓库中的 research/devkit/afternoon-test.md。' })}</p>
			</div>
		</div>
	</section>

	<section aria-labelledby="quick-h">
		<h2 id="quick-h">{t({ en: 'Quick start', zh: '快速开始' })}</h2>
		{#if page.start.setupHtml}
			<h3 class="qh"><span class="num">1</span> {t({ en: 'Set up once', zh: '准备一次环境' })}</h3>
			<p class="intro">
				{t({ en: 'The setup steps of CONTRIBUTING.md, for your own computer. Guide section 1 has the details.', zh: 'CONTRIBUTING.md 里的准备步骤（英文原文），适用于你自己的电脑。细节见指南第 1 节。' })}
				<a href="{base}/devkit/guide/{guideSec(1)?.slug ?? ''}/">{t({ en: 'Guide section 1', zh: '指南第 1 节' })}</a>
			</p>
			<Html html={page.start.setupHtml} class="setup" />
			<h3 class="qh"><span class="num">2</span> {t({ en: 'Build a package', zh: '做一个包' })}</h3>
		{/if}
		<p class="intro">
			{t({ en: 'The commands of the walk-through, in order, from the site folder. They are taken from the guide when the site is built; read the guide’s text for each before you run it.', zh: '走一遍流程要用到的命令，按顺序排列，都在 site 文件夹里运行。它们在网站构建时取自指南；运行每条命令之前，先读指南里对应的说明。' })}
			<a href="{base}/devkit/guide/{guideSec(4)?.slug ?? ''}/">{t({ en: 'Guide section 4', zh: '指南第 4 节' })}</a>
		</p>
		<Html html={page.quick.html} />
		<p class="intro">
			{t({ en: 'Building with Claude Code? The repository’s project skill', zh: '用 Claude Code 来做？仓库里的项目技能' })}
			<code>.claude/skills/model-atlas-builder/</code>
			{t({
				en: 'gives the agent this routine in short form: the quick checks while you iterate and one full check at the end, reuse before you write, the teaching bar, the chart and accessibility rules, and the review.',
				zh: '把这套流程简要地交给智能体：修改时用快速检查，最后只做一次完整检查；先复用再动手写；教学标准、图表和无障碍规则，以及评审。'
			})}
			<a href="{base}/devkit/contribute/#how">{t({ en: 'More in CONTRIBUTING.md', zh: '详见 CONTRIBUTING.md' })}</a>
		</p>
	</section>

	<section aria-labelledby="next-h">
		<h2 id="next-h">{t({ en: 'Where next', zh: '接下来' })}</h2>
		<div class="next">
			<a class="panel go" href="{base}/devkit/tutorial/"><strong>{t({ en: 'Tutorial', zh: '教程' })}</strong><span>{t({ en: 'Build hello-xor step by step, with its real files.', zh: '用真实文件一步一步做出 hello-xor。' })}</span></a>
			<a class="panel go" href="{base}/devkit/guide/"><strong>{t({ en: 'Guide', zh: '指南' })}</strong><span>{t({ en: 'The full DevKit guide, section by section.', zh: '完整的 DevKit 指南，按章节阅读。' })}</span></a>
			<a class="panel go" href="{base}/devkit/reference/ops/"><strong>{t({ en: 'Op reference', zh: '算子参考' })}</strong><span>{t({ en: 'Every core op with its formula, narration and a demo.', zh: '每个核心算子的公式、旁白和演示。' })}</span></a>
			<a class="panel go" href="{base}/devkit/reference/views/"><strong>{t({ en: 'View reference', zh: '视图参考' })}</strong><span>{t({ en: 'Every view a lesson step can show, live.', zh: '课程每一步可以显示的视图，实时演示。' })}</span></a>
			<a class="panel go" href="{base}/devkit/contribute/"><strong>{t({ en: 'Contribute', zh: '参与贡献' })}</strong><span>{t({ en: 'The rules, the checklist, the gates and every error code.', zh: '规则、检查清单、关卡和所有错误代码。' })}</span></a>
		</div>
	</section>
</div>

<style>
	.home {
		padding-top: var(--s6);
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s6);
	}
	.hero h1 {
		margin-bottom: var(--s3);
	}
	.cta {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
		margin: var(--s4) 0 var(--s5);
	}
	.counts {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.counts a {
		display: inline-flex;
		align-items: baseline;
		gap: 6px;
		padding: 6px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r-pill);
		background: var(--surface);
		color: var(--ink-2);
		text-decoration: none;
		font-size: var(--fs-sm);
	}
	.counts a:hover {
		border-color: var(--accent);
		color: var(--ink);
	}
	.counts .n {
		font-weight: 700;
		color: var(--ink);
		font-size: var(--fs-md);
	}
	.gen {
		margin: 10px 0 0;
		font-size: var(--fs-xs);
		max-width: var(--measure);
	}
	.status {
		max-width: calc(var(--measure) + 40px);
		margin: 0 0 var(--s4);
		padding: 10px 14px;
		border-left: 3px solid var(--pre);
		border-radius: 0 var(--r2) var(--r2) 0;
		background: var(--surface-2);
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.status p {
		margin: 0 0 6px;
	}
	.status p:last-child {
		margin: 0;
	}
	.status :global(.dk-md p) {
		margin-bottom: 6px;
	}
	.built {
		margin: 12px 0 0;
		max-width: var(--measure);
		color: var(--ink-2);
	}
	.qh {
		display: flex;
		align-items: center;
		gap: 10px;
		margin: var(--s4) 0 var(--s2);
		font-size: var(--fs-lg);
	}
	.qh .num {
		display: inline-grid;
		place-items: center;
		width: 26px;
		height: 26px;
		border-radius: 50%;
		background: var(--accent-soft);
		color: var(--accent-strong);
		font-size: var(--fs-sm);
		font-weight: 700;
	}
	section h2 {
		font-size: var(--fs-xl);
		margin-bottom: var(--s3);
	}
	.intro {
		color: var(--ink-2);
		max-width: var(--measure);
	}
	.who,
	.kinds {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 270px), 1fr));
		gap: var(--s4);
	}
	.panel h3 {
		font-size: var(--fs-lg);
		margin-bottom: 8px;
	}
	.panel p {
		margin-bottom: 10px;
	}
	.panel p:last-child {
		margin-bottom: 0;
	}
	.small {
		font-size: var(--fs-sm);
	}
	.kind {
		display: flex;
		flex-direction: column;
	}
	.badge {
		display: flex;
		align-items: center;
		gap: 8px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--muted);
	}
	.letter {
		display: inline-grid;
		place-items: center;
		width: 28px;
		height: 28px;
		border-radius: 8px;
		background: var(--accent);
		color: var(--accent-ink);
		font-weight: 700;
		font-size: var(--fs-md);
	}
	.ex {
		margin-top: auto;
		padding-top: 10px;
		border-top: 1px solid var(--rule);
	}
	.ex p {
		margin-bottom: 4px;
	}
	.exlink {
		display: inline-flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 8px;
		font-size: var(--fs-lg);
		margin-bottom: 4px;
	}
	.flow {
		display: grid;
		grid-template-columns: repeat(6, minmax(0, 1fr));
		gap: 14px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.node {
		position: relative;
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding: 12px;
		border: 1px solid var(--rule);
		border-top: 3px solid var(--accent);
		border-radius: var(--r2);
		background: var(--surface);
		min-width: 0;
	}
	.node.theirs {
		border-top-color: var(--pre);
		background: var(--surface-2);
	}
	.node:not(:last-child)::after {
		content: '→';
		position: absolute;
		right: -13px;
		top: 18px;
		color: var(--muted);
		font-size: 14px;
		line-height: 1;
	}
	.step {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.name {
		font-family: var(--font-display);
		font-weight: 650;
		font-size: var(--fs-lg);
		line-height: 1.2;
	}
	.cmds {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.cmds code {
		padding: 1px 6px;
		border-radius: 4px;
		background: var(--surface-2);
		font-size: 0.74rem;
		white-space: nowrap;
	}
	.theirs .cmds code {
		background: var(--surface);
	}
	.what {
		font-size: var(--fs-sm);
		color: var(--ink-2);
		line-height: 1.45;
	}
	.by {
		margin-top: auto;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	@media (max-width: 1060px) {
		.flow {
			grid-template-columns: repeat(3, minmax(0, 1fr));
		}
		.node:nth-child(3)::after {
			content: none;
		}
	}
	@media (max-width: 640px) {
		.flow {
			grid-template-columns: minmax(0, 1fr);
			gap: 18px;
		}
		.node:not(:last-child)::after {
			content: '↓';
			right: auto;
			left: 50%;
			top: auto;
			bottom: -17px;
		}
		.node:nth-child(3)::after {
			content: '↓';
		}
	}
	.testcard {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr);
		gap: var(--s5);
		align-items: start;
	}
	.big {
		margin: 0;
		font-size: var(--fs-2xl);
		font-weight: 700;
		color: var(--accent-strong);
		white-space: nowrap;
	}
	@media (max-width: 640px) {
		.testcard {
			grid-template-columns: minmax(0, 1fr);
			gap: var(--s3);
		}
	}
	.next {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 210px), 1fr));
		gap: var(--s3);
	}
	.go {
		display: flex;
		flex-direction: column;
		gap: 4px;
		color: var(--ink);
		text-decoration: none;
	}
	.go strong {
		color: var(--accent-strong);
	}
	.go span {
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.go:hover {
		border-color: var(--accent);
	}
</style>
