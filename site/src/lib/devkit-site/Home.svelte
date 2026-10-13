<!--
  /devkit/: what the DevKit is and who it is for, the kit today (its counts, the lessons built with it, the afternoon
  test), the three kinds with a live example lesson each, the workflow as a route line, and the quick start in two tabs.
  The counts, examples, commands, quick start and test result are read from the kit's sources at build time (home.ts).
  One page at a laptop's height budget (redesign spec §12, W3: at most 3 screens at 1280 x 800).
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import Glyph from './Glyph.svelte';
	import { arrow, code, flask, quiz } from './icons.ts';
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
	const share = $derived(page.built.live ? page.built.kit.length / page.built.live : 0);
	const personas: { icon: string; title: T2; text: T2 }[] = [
		{
			icon: code,
			title: { en: 'You know PyTorch', zh: '你会 PyTorch' },
			text: {
				en: 'Train a small model as you always do and describe its forward pass once in Python; the kit checks every layer of its browser copy against PyTorch. Then you write the lesson as typed data. You do not need Svelte.',
				zh: '像平时一样训练一个小模型，用 Python 把它的前向计算描述一遍，工具包会把浏览器里的版本逐层与 PyTorch 对比。然后你用带类型的数据写课程。你不需要会 Svelte。'
			}
		},
		{
			icon: flask,
			title: { en: 'You write JavaScript or teach', zh: '你写 JavaScript，或者你是老师' },
			text: {
				en: 'Live training and step-by-step algorithms need Node only: a trainer or a process in TypeScript, then the lesson. A Python twin is optional and earns a stronger badge.',
				zh: '实时训练和逐步运行的算法只需要 Node：用 TypeScript 写一个训练器或过程，再写课程。Python 对照是可选的，能换来更强的徽章。'
			}
		},
		{
			icon: quiz,
			title: { en: 'You care about honest numbers', zh: '你在意数字是否诚实' },
			text: {
				en: 'Every number in a lesson is bound to a computed value, every stage carries a provenance badge, and every claim that is not computed live has a test. The checks enforce it, not good intentions.',
				zh: '课程里的每个数都绑定到计算结果，每个舞台都有出处徽章，每个不是实时算出的论断都有测试。靠的是自动检查，而不是自觉。'
			}
		}
	];

	// the quick start: two tabs (set up once, build a package); the long command list opens in place
	let tab = $state<'setup' | 'build'>('setup');
	let allLines = $state(false);
	const quickLines = $derived((page.quick.html.match(/\n/g)?.length ?? 0) + 1);
	const CLIP = 12;
	function tabKey(e: KeyboardEvent) {
		if (!page.start.setupHtml || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
		e.preventDefault();
		tab = e.key === 'Home' ? 'setup' : e.key === 'End' ? 'build' : tab === 'setup' ? 'build' : 'setup';
		document.getElementById(`qs-tab-${tab}`)?.focus();
	}
</script>

<div class="home">
	<header class="hero">
		<div class="intro">
			<p class="eyebrow">DevKit · {t({ en: 'for builders', zh: '给制作者' })}</p>
			<h1>{t({ en: 'Build your own lesson', zh: '自己动手做一节课' })}</h1>
			<p class="lede">
				{t({
					en: 'The toolkit behind this site’s guided lessons: a command line, a kit of views and a lesson format. You write a few files of plain data and functions; the kit makes the page, the step-by-step lesson with real numbers, and the checks.',
					zh: '本站分步课程背后的工具包：一个命令行、一套视图和一种课程格式。你只写几个文件，内容是普通的数据和函数；页面、带真实数字的分步课程和各项检查都由工具包生成。'
				})}
			</p>
			<div class="cta">
				<a class="btn btn-act" href="{base}/devkit/tutorial/">{t({ en: 'Start the tutorial', zh: '从教程开始' })}<Glyph d={arrow} size={18} /></a>
				<a class="btn" href="{base}/devkit/guide/">{t({ en: 'Read the guide', zh: '阅读指南' })}</a>
			</div>
			<ul class="who" aria-label={t({ en: 'Who it is for', zh: '适合谁' })}>
				{#each personas as p (p.title.en)}
					<li>
						<Glyph d={p.icon} size={18} />
						<p><strong>{t(p.title)}{t({ en: '.', zh: '。' })}</strong> {t(p.text)}</p>
					</li>
				{/each}
			</ul>
		</div>

		<aside class="today" aria-labelledby="today-h">
			<h2 id="today-h">{t({ en: 'The kit today', zh: '工具包现状' })}</h2>
			<ul class="counts">
				{#each counts as c (c.href)}
					<li><a href="{base}{c.href}"><span class="n num">{c.n}</span><span class="l">{t(c.label)}</span></a></li>
				{/each}
			</ul>
			{#if page.built.live}
				<div class="built">
					<p>
						{t({
							en: `${page.built.kit.length} of the ${page.built.live} live model lessons are built with it.`,
							zh: `已上线的 ${page.built.live} 个模型课程中，有 ${page.built.kit.length} 个是用它做的。`
						})}
					</p>
					<div class="meter" aria-hidden="true"><span style="width: {(share * 100).toFixed(1)}%"></span></div>
					{#if page.built.kit.length}
						<details>
							<summary>{t({ en: `Show the ${page.built.kit.length}`, zh: `列出这 ${page.built.kit.length} 个` })}</summary>
							<p class="names">
								{#each page.built.kit as k, i (k.slug)}{i ? ' · ' : ''}<a href="{base}/m/{k.slug}/">{t(k.name)}</a>{/each}{t({ en: '.', zh: '。' })}
								{t({ en: 'The others were made by hand, without the kit.', zh: '其余的是不用工具包、手工做的页面。' })}
							</p>
						</details>
					{/if}
				</div>
			{/if}
			<div class="test">
				<p class="tk">{t({ en: 'The afternoon test', zh: '“一个下午”测试' })}</p>
				<p class="tv"><span class="num">{t({ en: `${mins} min ${secs} s`, zh: `${mins} 分 ${secs} 秒` })}</span></p>
				<p class="ts">
					{t({
						en: `A fresh agent with only the guide built ${page.afternoon.built} to green gates, and listed ${page.afternoon.gaps} gaps that are now fixed.`,
						zh: `一个只拿到指南的全新智能体做出了 ${page.afternoon.built} 并通过所有关卡，还列出了 ${page.afternoon.gaps} 处已修正的问题。`
					})}
				</p>
				<details>
					<summary>{t({ en: 'How the test works', zh: '测试是怎么做的' })}</summary>
					<p>
						{t({
							en: `Before the guide was published, a fresh agent that had only the guide built ${page.afternoon.built}, an XOR network with a sigmoid output, from the scaffold to a draft that passes every automated gate, in ${mins} min ${secs} s of wall-clock time. It also listed ${page.afternoon.gaps} places where the guide was unclear or wrong; the guide on this site is the version after those fixes. A person needs longer than an agent: the reading, the training and the lesson writing take most of an afternoon, which is the test's name and its target.`,
							zh: `指南发布之前，一个只拿到指南的全新智能体从脚手架开始，做出了 ${page.afternoon.built}（一个输出层用 sigmoid 的 XOR 网络），直到草稿通过所有自动关卡，实际用时 ${mins} 分 ${secs} 秒。它还列出了指南中 ${page.afternoon.gaps} 处不清楚或有错的地方；本站的指南是修正之后的版本。人比智能体需要更多时间：阅读、训练和写课程会占去一个下午的大部分，这正是测试名字的由来，也是它的目标。`
						})}
					</p>
					<p>{t({ en: 'The full record: research/devkit/afternoon-test.md in the repository.', zh: '完整记录：仓库中的 research/devkit/afternoon-test.md。' })}</p>
				</details>
			</div>
			<p class="gen">{t({ en: 'Read from the kit’s own files when the site is built, so this page grows with the kit.', zh: '这些内容在网站构建时从工具包自己的文件读出，所以页面会随工具包一起增长。' })}</p>
		</aside>
	</header>

	<section aria-labelledby="kinds-h">
		<div class="sh">
			<h2 id="kinds-h">{t({ en: 'Three kinds of illustration', zh: '三种图解' })}</h2>
			<a href="{base}/devkit/guide/{guideSec(2)?.slug ?? ''}/">{t({ en: 'Guide section 2', zh: '指南第 2 节' })}</a>
		</div>
		<p class="sub">
			{t({ en: 'Every lesson plays a tape of recorded frames; the kind is what makes the frames. One package can mix kinds.', zh: '每节课都播放一条记录好的“磁带”；生成帧的方式就是图解的类型。一个包可以混合几种类型。' })}
		</p>
		<div class="kinds">
			{#each page.kinds as k (k.kind)}
				<article class="kind">
					<p class="kh"><span class="letter" aria-hidden="true">{k.kind}</span><span>{t({ en: `Kind ${k.kind}`, zh: `${k.kind} 类` })}</span></p>
					<h3>{t(k.name)}</h3>
					<p class="what">{t(k.what)}</p>
					<p class="write"><span>{t({ en: 'You write', zh: '你要写' })}</span>{t(k.youWrite)}</p>
					<a class="ex" href="{base}/m/{k.example.slug}/#lesson">
						<span class="exk">{t({ en: 'Live example', zh: '已上线的示例' })}</span>
						<span class="exn">{t(k.example.name)}{#if k.example.steps}<span class="steps">{t({ en: ` · ${k.example.steps} steps`, zh: ` · ${k.example.steps} 步` })}</span>{/if}</span>
						<Glyph d={arrow} size={16} />
					</a>
				</article>
			{/each}
		</div>
	</section>

	<section aria-labelledby="flow-h">
		<h2 id="flow-h">{t({ en: 'From an idea to a live page', zh: '从想法到上线的页面' })}</h2>
		<p class="sub">{t({ en: `${page.workflow.length} steps, each with its commands. The first ${yours} are yours; a teaching reviewer and the maintainers do the rest.`, zh: `${page.workflow.length} 个步骤，每一步都有对应的命令。前 ${yours} 步由你完成，其余由教学评审者和维护者完成。` })}</p>
		<ol class="route">
			{#each page.workflow as w, i (w.id)}
				<li class="st" class:theirs={w.who !== 'you'}>
					<span class="dot" aria-hidden="true">{i + 1}</span>
					<span class="body">
						<span class="name">{t(w.name)}</span>
						<span class="cmds">{#each w.cmds as c (c.name)}<code title={c.summary}>atlas {c.name}</code>{/each}</span>
						<span class="what">{t(w.what)}</span>
						<span class="by">{t(who[w.who])}</span>
					</span>
				</li>
			{/each}
		</ol>
	</section>

	<section aria-labelledby="quick-h">
		<div class="sh">
			<h2 id="quick-h">{t({ en: 'Quick start', zh: '快速开始' })}</h2>
			<a href="{base}/devkit/guide/{guideSec(4)?.slug ?? ''}/">{t({ en: 'Guide section 4', zh: '指南第 4 节' })}</a>
		</div>
		{#if page.start.setupHtml}
			<div class="seg tabs" role="tablist" aria-label={t({ en: 'Quick start', zh: '快速开始' })}>
				<button type="button" role="tab" id="qs-tab-setup" aria-selected={tab === 'setup'} aria-controls="qs-setup" tabindex={tab === 'setup' ? 0 : -1} onclick={() => (tab = 'setup')} onkeydown={tabKey}>
					{t({ en: '1 · Set up once', zh: '1 · 准备一次环境' })}
				</button>
				<button type="button" role="tab" id="qs-tab-build" aria-selected={tab === 'build'} aria-controls="qs-build" tabindex={tab === 'build' ? 0 : -1} onclick={() => (tab = 'build')} onkeydown={tabKey}>
					{t({ en: '2 · Build a package', zh: '2 · 做一个包' })}
				</button>
			</div>
			<div class="panel-q" role="tabpanel" id="qs-setup" aria-labelledby="qs-tab-setup" hidden={tab !== 'setup'}>
				<p class="sub">
					{t({ en: 'The setup steps of CONTRIBUTING.md, for your own computer. Guide section 1 has the details.', zh: 'CONTRIBUTING.md 里的准备步骤（英文原文），适用于你自己的电脑。细节见指南第 1 节。' })}
				</p>
				<div class="setup-grid">
				<Html html={page.start.setupHtml} class="setup compact" />
				{#if page.start.statusHtml}
					<div class="status" role="note">
						{#if i18n.lang === 'zh' && page.start.statusZh}
							<p lang="zh-CN">{page.start.statusZh}</p>
						{:else}
							<Html html={page.start.statusHtml} class="compact" />
						{/if}
						<p class="from">{t({ en: 'From CONTRIBUTING.md:', zh: '来自 CONTRIBUTING.md：' })} <a href="{base}/devkit/contribute/#how">{t({ en: 'the contributor path', zh: '贡献者的路径' })}</a></p>
					</div>
				{/if}
				</div>
			</div>
		{/if}
		<div class="panel-q" role={page.start.setupHtml ? 'tabpanel' : undefined} id="qs-build" aria-labelledby={page.start.setupHtml ? 'qs-tab-build' : undefined} hidden={!!page.start.setupHtml && tab !== 'build'}>
			<p class="sub">
				{t({ en: 'The commands of the walk-through, in order, from the site folder. They are taken from the guide when the site is built; read the guide’s text for each before you run it.', zh: '走一遍流程要用到的命令，按顺序排列，都在 site 文件夹里运行。它们在网站构建时取自指南；运行每条命令之前，先读指南里对应的说明。' })}
			</p>
			<div class="clip" class:open={allLines || quickLines <= CLIP} style:--clip="{CLIP * 22 + 24 + 37}px">
				<Html html={page.quick.html} />
			</div>
			{#if quickLines > CLIP}
				<button type="button" class="btn btn-quiet more" aria-expanded={allLines} onclick={() => (allLines = !allLines)}>
					{allLines ? t({ en: 'Show fewer lines', zh: '收起' }) : t({ en: `Show all ${quickLines} lines`, zh: `显示全部 ${quickLines} 行` })}
				</button>
			{/if}
			<p class="note">
				{t({ en: 'Building with Claude Code? The repository’s project skill', zh: '用 Claude Code 来做？仓库里的项目技能' })}
				<code>.claude/skills/model-atlas-builder/</code>
				{t({
					en: 'gives the agent this routine in short form: the quick checks while you iterate and one full check at the end, reuse before you write, the teaching bar, the chart and accessibility rules, and the review.',
					zh: '把这套流程简要地交给智能体：修改时用快速检查，最后只做一次完整检查；先复用再动手写；教学标准、图表和无障碍规则，以及评审。'
				})}
				<a href="{base}/devkit/contribute/#how">{t({ en: 'More in CONTRIBUTING.md', zh: '详见 CONTRIBUTING.md' })}</a>
			</p>
		</div>
	</section>

</div>

<style>
	.home {
		container-type: inline-size;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: calc(var(--s6) + var(--s2));
	}

	/* hero: the words, and beside them the kit today */
	.hero {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s5);
		align-items: start;
	}
	@container (min-width: 820px) {
		.hero {
			grid-template-columns: minmax(0, 1fr) 320px;
			gap: var(--s6);
		}
	}
	.intro .eyebrow {
		margin: 0 0 var(--s2);
	}
	.intro h1 {
		margin-bottom: var(--s3);
	}
	.cta {
		display: flex;
		flex-wrap: wrap;
		gap: var(--s3);
		margin-top: var(--s5);
	}
	.today {
		display: grid;
		gap: var(--s4);
		padding: var(--s4) var(--s5) var(--s5);
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.today h2 {
		margin: 0;
		font-size: var(--t-md);
		line-height: var(--lh-md);
	}
	.counts {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: var(--s1) var(--s3);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.counts a {
		display: flex;
		align-items: baseline;
		gap: var(--s2);
		padding: 2px 0;
		color: var(--ink-2);
		text-decoration: none;
	}
	.counts a:hover .l {
		text-decoration: underline;
	}
	.counts .n {
		min-width: 2ch;
		font-size: var(--t-lg);
		font-weight: var(--w-ui);
		color: var(--ink);
	}
	.built,
	.test {
		display: grid;
		gap: var(--s2);
		padding-top: var(--s3);
		border-top: 1px solid var(--rule);
	}
	.built p,
	.test p {
		margin: 0;
	}
	.meter {
		height: 6px;
		border-radius: var(--r-pill);
		background: var(--rule);
		overflow: hidden;
	}
	.meter span {
		display: block;
		height: 100%;
		background: var(--ink-2);
	}
	details summary {
		padding-block: 6px;
		cursor: pointer;
		color: var(--ink);
		font-weight: var(--w-ui);
	}
	details p {
		margin: var(--s1) 0 var(--s2);
		color: var(--ink-2);
	}
	.names {
		line-height: var(--lh-sm);
	}
	.tk {
		color: var(--muted);
	}
	.tv {
		font-size: var(--t-2xl);
		line-height: var(--lh-2xl);
		font-weight: var(--w-title);
	}
	.ts {
		color: var(--ink-2);
	}
	.gen {
		margin: 0;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}

	/* who it is for: three short lines under the call to action */
	.who {
		display: grid;
		gap: var(--s3);
		margin: var(--s5) 0 0;
		padding: var(--s4) 0 0;
		border-top: 1px solid var(--rule);
		list-style: none;
	}
	.who li {
		display: grid;
		grid-template-columns: 18px minmax(0, 1fr);
		gap: var(--s3);
		align-items: start;
	}
	.who li :global(.icon) {
		margin-top: 1px;
		color: var(--ink-2);
	}
	.who p {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
		max-width: 75ch;
	}
	.who strong {
		color: var(--ink);
		font-weight: var(--w-ui);
	}

	/* section heads */
	.sh {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: var(--s1) var(--s4);
	}
	.sh h2,
	section > h2 {
		margin: 0 0 var(--s1);
	}
	.sh a {
		font-size: var(--t-sm);
	}
	.sub {
		margin: 0 0 var(--s4);
		color: var(--ink-2);
		max-width: 75ch;
	}

	/* the three kinds */
	.kinds {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s3);
	}
	@container (min-width: 700px) {
		.kinds {
			grid-template-columns: repeat(3, minmax(0, 1fr));
		}
	}
	.kind {
		display: flex;
		flex-direction: column;
		gap: var(--s2);
		padding: var(--s4);
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
	}
	.kind p,
	.kind h3 {
		margin: 0;
	}
	.kh {
		display: flex;
		align-items: center;
		gap: var(--s2);
		font-size: var(--t-sm);
		color: var(--muted);
	}
	.letter {
		display: inline-grid;
		place-items: center;
		width: 28px;
		height: 28px;
		border-radius: var(--r2);
		background: var(--act);
		color: var(--act-ink);
		font-weight: var(--w-title);
	}
	.kind h3 {
		font-size: var(--t-md);
		line-height: var(--lh-md);
	}
	.what,
	.write {
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.write span {
		display: block;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	.ex {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		align-items: center;
		gap: 0 var(--s2);
		margin-top: auto;
		padding-top: var(--s3);
		border-top: 1px solid var(--rule);
		color: var(--ink);
		text-decoration: none;
	}
	.exk {
		grid-column: 1;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	.exn {
		grid-column: 1;
		font-weight: var(--w-ui);
		text-decoration: underline;
		text-decoration-thickness: 1px;
		text-underline-offset: 3px;
	}
	.ex:hover .exn {
		text-decoration-thickness: 2px;
	}
	.ex :global(.icon) {
		grid-column: 2;
		grid-row: 1 / span 2;
	}
	.steps {
		font-weight: var(--w-body);
		color: var(--muted);
	}

	/* the workflow: a route line through numbered stations; the stations done by others are outlined, their line dashed */
	.route {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.st {
		position: relative;
		display: grid;
		align-content: start;
		grid-template-columns: 32px minmax(0, 1fr);
		gap: var(--s3);
		padding-bottom: var(--s4);
	}
	.st:not(:last-child)::after {
		content: '';
		position: absolute;
		left: 14.5px;
		top: 32px;
		bottom: 0;
		border-left: 3px solid var(--ink);
	}
	.st:has(+ .theirs)::after,
	.st.theirs:not(:last-child)::after {
		border-left-style: dashed;
		border-left-color: var(--ink-2);
	}
	.dot {
		position: relative;
		z-index: 1;
		display: inline-grid;
		place-items: center;
		width: 32px;
		height: 32px;
		border-radius: 50%;
		background: var(--act);
		color: var(--act-ink);
		font-weight: var(--w-title);
		font-size: var(--t-sm);
		font-variant-numeric: tabular-nums;
	}
	.theirs .dot {
		background: var(--panel);
		color: var(--ink);
		box-shadow: inset 0 0 0 2px var(--ink-2);
	}
	.body {
		display: grid;
		gap: var(--s1);
		min-width: 0;
	}
	.name {
		font-weight: var(--w-title);
		line-height: 32px;
	}
	.cmds {
		display: flex;
		flex-wrap: wrap;
		gap: var(--s1);
	}
	.cmds code {
		padding: 0 6px;
		border: 1px solid var(--rule);
		border-radius: var(--r1);
		background: var(--panel);
		font-size: var(--t-xs);
		line-height: 20px;
		white-space: nowrap;
	}
	.route .what {
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.by {
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	@container (min-width: 820px) {
		.route {
			grid-template-columns: repeat(6, minmax(0, 1fr));
			column-gap: var(--s3);
		}
		.st {
			grid-template-columns: minmax(0, 1fr);
			gap: var(--s2);
			padding-bottom: 0;
		}
		.st:not(:last-child)::after {
			left: 32px;
			right: calc(-1 * var(--s3));
			top: 14.5px;
			bottom: auto;
			border-left: 0;
			border-top: 3px solid var(--ink);
		}
		.st:has(+ .theirs)::after,
		.st.theirs:not(:last-child)::after {
			border-left: 0;
			border-top-style: dashed;
			border-top-color: var(--ink-2);
		}
		.name {
			line-height: var(--lh-md);
		}
	}

	/* quick start */
	.tabs {
		margin-bottom: var(--s4);
	}
	.panel-q[hidden] {
		display: none;
	}
	.setup-grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4) var(--s5);
		align-items: start;
	}
	@container (min-width: 820px) {
		.setup-grid {
			grid-template-columns: minmax(0, 1fr) 300px;
		}
	}
	.status {
		padding: var(--s3) var(--s4);
		border: 1px solid var(--rule);
		border-left: 3px solid var(--line-control);
		border-radius: 0 var(--r3) var(--r3) 0;
		background: var(--panel);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.status p {
		margin: 0 0 var(--s2);
	}
	.status .from {
		margin: var(--s2) 0 0;
		color: var(--muted);
	}
	.clip {
		position: relative;
		max-height: var(--clip);
		overflow: hidden;
	}
	.clip.open {
		max-height: none;
	}
	.clip:not(.open)::after {
		content: '';
		position: absolute;
		inset: auto 1px 1px;
		height: 56px;
		border-radius: 0 0 var(--r3) var(--r3);
		background: linear-gradient(to bottom, transparent, var(--well));
		pointer-events: none;
	}
	.clip :global(.code) {
		margin-bottom: 0;
	}
	.more {
		margin-top: var(--s2);
	}
	.note {
		margin: var(--s4) 0 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
		max-width: 75ch;
	}
</style>
