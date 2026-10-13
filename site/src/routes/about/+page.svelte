<!--
  About (redesign spec §6.7): a sticky in-page nav (left at 1100 px and up, a chip bar on phones) and five sections:
  what this is, how numbers are made (the three provenance kinds with a real example label each, and how to read the
  colours), accessibility and privacy, credits and licences (grouped, collapsed), project and source. The old anchors
  (#numbers, #colors, #privacy, #scope, #source, #licenses) still land on their section.
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Icon from '$lib/kit/Icon.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import RoleChip from '$lib/kit/RoleChip.svelte';
	import LicenceList from '$lib/kit/catalog/LicenceList.svelte';
	import { CATEGORICAL } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { right } from '$lib/kit/catalog/glyphs';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import { ct } from '$lib/kit/catalog/text';
	import { OWN_LICENSES, REPO_URL, repoFile } from '$lib/project';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const n = $derived(data.scope);
	const p = $derived(data.parity);

	const nav = [
		{ id: 'what', label: { en: 'What this is', zh: '这是什么' } },
		{ id: 'numbers', label: { en: 'How numbers are made', zh: '数字从哪里来' } },
		{ id: 'privacy', label: { en: 'Accessibility and privacy', zh: '无障碍与隐私' } },
		{ id: 'licenses', label: { en: 'Credits and licences', zh: '致谢与许可' } },
		{ id: 'source', label: { en: 'Project and source', zh: '项目与源代码' } }
	];
	let current = $state('what');
	$effect(() => {
		const els = nav.map((s) => document.getElementById(s.id)).filter((e): e is HTMLElement => !!e);
		let raf = 0;
		const spy = () => {
			raf = 0;
			const line = Math.max((parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--header-h')) || 64) + 120, innerHeight * 0.35);
			let id = nav[0].id;
			for (const el of els) if (el.getBoundingClientRect().top <= line) id = el.id;
			if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) id = nav[nav.length - 1].id;
			current = id;
		};
		const on = () => {
			if (!raf) raf = requestAnimationFrame(spy);
		};
		spy();
		void document.fonts?.ready.then(on);
		addEventListener('scroll', on, { passive: true });
		return () => removeEventListener('scroll', on);
	});
</script>

<svelte:head><title>{ui('navAbout')} · Model Atlas</title></svelte:head>

<div class="container about">
	<header class="head">
		<h1>{t({ en: 'About Model Atlas', zh: '关于模型图鉴' })}</h1>
		<p class="lede">
			{t({
				en: 'Where the numbers come from, how exact they are, what happens to your data, and whose work this builds on.',
				zh: '数字从哪里来、有多精确、你的数据会怎样，以及我们借鉴了谁的工作。'
			})}
		</p>
	</header>

	<div class="layout">
		<nav class="toc" aria-label={ui('onThisPage')}>
			<ul>
				{#each nav as s (s.id)}
					<li><a href="#{s.id}" aria-current={current === s.id ? 'true' : undefined}>{t(s.label)}</a></li>
				{/each}
			</ul>
		</nav>

		<div class="body">
			<section id="what" aria-labelledby="what-h">
				<h2 id="what-h">{t(nav[0].label)}</h2>
				<p>
					{t({
						en: 'Model Atlas teaches how neural networks and their training work through short, guided lessons. Each step shows one idea on a small model that runs on your own input, with the real numbers, and says where those numbers came from. The models are small on purpose: small enough that every number can be on the screen.',
						zh: '模型图鉴用短小的分步课程讲神经网络和它们的训练怎样工作。每一步只讲一个想法，用一个在你自己的输入上运行的小模型，给出真实的数，并说明这些数从哪里来。模型故意做得小：小到每个数都能放在屏幕上。'
					})}
				</p>
				<p id="scope">
					{t({
						en: `${n.liveModels} models in ${n.liveAreas} of the ${n.areas} areas and ${n.liveConcepts} training methods have a lesson now. The atlas also lists what is planned (${n.models} models and ${n.concepts} training methods in all), each with a page that says what it will show.`,
						zh: `现在有 ${n.liveModels} 个模型（${n.areas} 个领域中的 ${n.liveAreas} 个）和 ${n.liveConcepts} 种训练方法有课程。图鉴也列出了计划中的内容（共 ${n.models} 个模型、${n.concepts} 种训练方法），每个都有页面说明它将展示什么。`
					})}
				</p>
				<p class="links">
					<a href="{base}/atlas/">{ct('allLessons')}<Icon d={right} size={16} /></a>
					<a href="{base}/glossary/">{ui('navGlossary')}<Icon d={right} size={16} /></a>
				</p>
			</section>

			<section id="numbers" aria-labelledby="numbers-h">
				<h2 id="numbers-h">{t(nav[1].label)}</h2>
				<p>
					{t({
						en: 'Every stage that shows model numbers carries one label. Hover, focus or tap a label for its details.',
						zh: '每个展示模型数字的画面都带一个标记。悬停、聚焦或点按标记即可查看详情。'
					})}
				</p>
				<dl class="prov">
					<div>
						<dt><ProvenanceBadge mode="live" precision={p.precision} tolerance={p.diff} against="PyTorch fp32" /></dt>
						<dd>
							{t({
								en: `Computed in your browser right now, from the real weights. A parity test runs the same inputs through PyTorch and through the site's code and compares every layer; a model passes only within 1e-4. TinyCNN, the label shown here, differs by at most ${p.diff !== undefined ? fmt3(p.diff, 2) : '—'} over ${p.inputs ?? '—'} inputs.`,
								zh: `此刻在你的浏览器里用真实权重算出。一致性测试把同样的输入分别送进 PyTorch 和本站的代码，逐层比较；只有误差都在 1e-4 以内才算通过。这里的标记来自 TinyCNN：在 ${p.inputs ?? '—'} 个输入上最大差 ${p.diff !== undefined ? fmt3(p.diff, 2) : '—'}。`
							})}
						</dd>
					</div>
					<div>
						<dt><ProvenanceBadge mode="precomputed" /></dt>
						<dd>
							{t({
								en: 'Computed offline from the same model and saved with the page: models too large for a browser, training runs that take hours, and the TinyCNN preview on the home page. Quantized runs are labelled approximate, never exact.',
								zh: '用同一个模型离线算好、随页面保存：浏览器跑不动的大模型、要训练几个小时的过程，以及首页上的 TinyCNN 预览。量化运行标为“近似”，从不标为“精确”。'
							})}
						</dd>
					</div>
					<div>
						<dt><ProvenanceBadge mode="illustrative" /></dt>
						<dd>
							{t({
								en: 'A drawing that shows an idea, not the output of a model. Used rarely and always labelled; a diagram is never filled with made-up numbers presented as a model’s.',
								zh: '用来说明想法的示意图，不是模型的输出。很少使用，而且总会标明；不会用编造的数字冒充模型的输出。'
							})}
						</dd>
					</div>
				</dl>
				<p class="small">
					{t({
						en: 'Why the numbers differ at all: floating-point addition is not associative, so adding the same products in another order (a GPU kernel or a JavaScript loop) changes the last bits. In fp32 that is about 1e-6 to 1e-5, growing slowly with depth.',
						zh: '数字为什么会有差别：浮点加法不满足结合律，同样的乘积换个顺序相加（GPU 内核或 JavaScript 循环），最后几位就会变。fp32 下大约是 1e-6 到 1e-5，随深度缓慢增大。'
					})}
				</p>
				<h3 id="colors">{t({ en: 'Reading the colours', zh: '如何读颜色' })}</h3>
				<div class="legends">
					<ColorLegend kind="diverging" domain={[-1, 1]} label={{ en: 'Signed values: blue below zero, orange above', zh: '有正负的值：负为蓝，正为橙' }} width={280} />
					<ColorLegend kind="sequential" domain={[0, 1]} label={{ en: 'Shares, probabilities and attention', zh: '占比、概率与注意力' }} width={280} />
				</div>
				<p class="sw">
					{#each CATEGORICAL[themeState.resolved] as c, i (i)}<span class="swatch" style:background={c}></span>{/each}
					<span>{t({ en: 'Classes keep one colour on every view, always next to their name.', zh: '同一类别在所有视图里颜色相同，并总是配着名字。' })}</span>
				</p>
				<p class="roles"><RoleChip role="activation" /> <RoleChip role="param" /> <RoleChip role="grad" /> <RoleChip role="frozen" /> <RoleChip role="trainable" /></p>
				<p class="small">
					{t({
						en: 'The scales are built in a perceptual colour space and checked for common colour blindness; no rainbow maps. Every hover readout also prints the number.',
						zh: '色阶在感知均匀的色彩空间里构建，并针对常见色盲做过检查；不用彩虹色图。每个悬停读数也都会显示数字。'
					})}
				</p>
			</section>

			<section id="privacy" aria-labelledby="privacy-h">
				<h2 id="privacy-h">{t(nav[2].label)}</h2>
				<ul class="bul">
					<li>{t({ en: 'What you draw, type or record is processed on your device and never uploaded.', zh: '你画的、写的、录的内容都在你的设备上处理，从不上传。' })}</li>
					<li>{t({ en: 'The site makes no requests to other hosts: fonts, maths and model files come from this site. No analytics, no cookies; your theme and language stay in your browser’s storage.', zh: '网站不向其他主机发请求：字体、公式和模型文件都由本站提供。没有统计脚本，没有 Cookie；主题和语言只保存在你的浏览器里。' })}</li>
					<li id="accessibility">{t({ en: 'Lessons are built to work with a keyboard and a screen reader: each view has a text description, controls have names, and the arrow keys and Enter move through steps and choices. With reduced motion nothing plays by itself. Text keeps 4.5:1 contrast in both themes, and the site checks every lesson step with axe.', zh: '课程按键盘和读屏软件可用来设计：每个视图都有文字描述，控件都有名字，方向键和回车可以切换步骤和选项。开启“减少动态效果”后，不会自动播放。两种主题下文字对比度都不低于 4.5:1，每一步都用 axe 检查过。' })}</li>
				</ul>
			</section>

			<section id="licenses" aria-labelledby="licenses-h">
				<h2 id="licenses-h">{t(nav[3].label)}</h2>
				<p>
					{t({
						en: `The code is under the ${OWN_LICENSES.code.name} licence; lesson text, articles and figures are under ${OWN_LICENSES.content.name}, so you may share and adapt them if you credit Model Atlas and say what you changed. The ${data.licenceCount} sources below keep their own licences; the full texts are in the repository's LICENSES folder.`,
						zh: `代码采用 ${OWN_LICENSES.code.name} 许可证；课程文字、文章和插图采用 ${OWN_LICENSES.content.name}，注明出自模型图鉴并说明改动即可分享和改编。下面 ${data.licenceCount} 项来源保留各自的许可证，全文在仓库的 LICENSES 文件夹中。`
					})}
				</p>
				<LicenceList groups={data.licences} />
			</section>

			<section id="source" aria-labelledby="source-h">
				<h2 id="source-h">{t(nav[4].label)}</h2>
				<ul class="bul">
					<li><a href={REPO_URL} rel="noopener">{t({ en: 'Source code and issues on GitHub', zh: 'GitHub 上的源代码与问题反馈' })}</a></li>
					<li><a href={repoFile(OWN_LICENSES.code.file)} rel="noopener">{t({ en: 'The MIT licence (code)', zh: 'MIT 许可证（代码）' })}</a> · <a href={OWN_LICENSES.content.url} rel="noopener">{t({ en: 'CC BY 4.0 (lesson text, articles and figures)', zh: 'CC BY 4.0（课程文字、文章和插图）' })}</a></li>
					<li><a href={`${REPO_URL}/tree/main/LICENSES`} rel="noopener">{t({ en: 'Third-party licence texts (LICENSES/)', zh: '第三方许可证全文（LICENSES/）' })}</a></li>
					<li><a href="{base}/devkit/">{ui('footerDevkit')}</a>: {t({ en: 'how the lessons are built, and how to add one.', zh: '课程是怎样做出来的，以及如何添加一节。' })}</li>
				</ul>
			</section>
		</div>
	</div>
</div>

<style>
	.head {
		padding: 32px 0 8px;
	}
	h1 {
		margin: 0 0 6px;
	}
	.lede {
		margin: 0;
	}
	.layout {
		display: grid;
		grid-template-columns: 220px minmax(0, 720px);
		gap: 48px;
		padding-top: 24px;
	}
	.toc {
		position: sticky;
		top: calc(var(--header-h) + 24px);
		align-self: start;
	}
	.toc ul,
	.bul {
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.toc a {
		position: relative;
		display: block;
		padding: 7px 12px;
		border-radius: var(--r2);
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		text-decoration: none;
	}
	.toc a:hover {
		background: var(--well);
		color: var(--ink);
	}
	.toc a[aria-current] {
		background: var(--here-bg);
		color: var(--ink);
		font-weight: var(--w-title);
	}
	.toc a[aria-current]::before {
		content: '';
		position: absolute;
		left: 0;
		top: 8px;
		bottom: 8px;
		width: 3px;
		border-radius: 3px;
		background: var(--here);
	}
	section {
		padding-bottom: 36px;
		scroll-margin-top: calc(var(--header-h) + 16px);
	}
	section + section {
		padding-top: 28px;
		border-top: 1px solid var(--rule);
	}
	section h2 {
		margin: 0 0 12px;
	}
	section h3 {
		margin: 24px 0 12px;
		font-size: var(--t-md);
		line-height: var(--lh-md);
		scroll-margin-top: calc(var(--header-h) + 16px);
	}
	section p {
		max-width: 68ch;
	}
	.small {
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.links {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 24px;
		margin: 0;
	}
	.links a {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-weight: var(--w-ui);
	}
	.prov {
		display: grid;
		gap: 14px;
		margin: 0 0 16px;
	}
	.prov div {
		display: grid;
		grid-template-columns: 210px minmax(0, 1fr);
		gap: 6px 16px;
		align-items: start;
		padding: 12px 14px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--panel);
	}
	.prov dt {
		margin: 0;
	}
	.prov dd {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.legends {
		display: flex;
		flex-wrap: wrap;
		gap: 16px 32px;
		margin-bottom: 12px;
	}
	.sw {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
	}
	.sw span:last-child {
		margin-left: 8px;
		font-size: var(--t-sm);
		color: var(--ink-2);
	}
	.swatch {
		width: 16px;
		height: 16px;
		border-radius: var(--r1);
	}
	/* the kit legend and role chips set 11 px ticks and glyphs; the site's floor is 12 px */
	.legends :global(.ticks span),
	.roles :global(span.g) {
		font-size: var(--t-xs);
	}
	.roles {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.bul {
		display: flex;
		flex-direction: column;
		gap: 10px;
		max-width: 68ch;
	}
	.bul li {
		padding-left: 18px;
		position: relative;
	}
	.bul li::before {
		content: '';
		position: absolute;
		left: 2px;
		top: 0.7em;
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--ink-2);
	}
	@media (max-width: 1099px) {
		.layout {
			grid-template-columns: minmax(0, 1fr);
			gap: 0;
			padding-top: 8px;
		}
		.toc {
			top: var(--header-h);
			z-index: 5;
			margin: 0 calc(-1 * var(--gutter)) 16px;
			padding: 8px var(--gutter);
			background: var(--paper);
			border-bottom: 1px solid var(--rule);
		}
		.toc ul {
			display: flex;
			gap: 8px;
			overflow-x: auto;
			scrollbar-width: none;
		}
		.toc a {
			flex: none;
			white-space: nowrap;
			padding: 8px 14px;
			border: 1px solid var(--line-control);
			border-radius: var(--r-pill);
		}
		.toc a[aria-current]::before {
			content: none;
		}
		.toc a[aria-current] {
			border-color: var(--ink);
		}
		section {
			scroll-margin-top: calc(var(--header-h) + 72px);
		}
	}
	@media (max-width: 759px) {
		.head {
			padding: 24px 0 4px;
		}
		.prov div {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
