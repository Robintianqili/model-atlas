<script lang="ts">
	import { base } from '$app/paths';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import RoleChip from '$lib/kit/RoleChip.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import { CATEGORICAL } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import { OWN_LICENSES, REPO_URL, repoFile } from '$lib/project';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const licenses = $derived(data.licenses);
	const n = $derived(data.scope);
	const isUrl = (u: string) => /^https?:\/\//.test(u);
</script>

<svelte:head><title>{ui('navAbout')} · Model Atlas</title></svelte:head>

<div class="container page prose-wide">
	<header class="head">
		<p class="eyebrow">/about/</p>
		<h1>{t({ en: 'About Model Atlas', zh: '关于模型图鉴' })}</h1>
		<p class="lede">
			{t({
				en: 'Model Atlas is a static website that explains neural network models by letting you run them on your own input and look inside every layer. This page explains where the numbers come from, how exact they are, what happens to your data, and whose work we build on.',
				zh: '模型图鉴是一个静态网站：你可以用自己的输入运行神经网络模型，并查看每一层的内部。本页说明数字从哪里来、有多精确、你的数据会怎样，以及我们借鉴了谁的工作。'
			})}
		</p>
	</header>

	<section id="numbers" aria-labelledby="numbers-h">
		<h2 id="numbers-h">{ui('footerHowNumbers')}</h2>
		<p>
			{t({
				en: 'Every panel that shows model numbers carries one of three badges. Hover, focus or tap a badge to see its details.',
				zh: '每个展示模型数字的面板都带有以下三种标记之一。悬停、聚焦或点按标记即可查看详情。'
			})}
		</p>
		<dl class="prov">
			<div>
				<dt><ProvenanceBadge mode="live" precision="fp32" /></dt>
				<dd>
					<Lang>
						{#snippet en()}Computed in your browser right now. For our own tiny models we wrote the forward pass in plain TypeScript and load the trained weights as raw float32 numbers. A parity test runs the same inputs through PyTorch and through our code and compares every intermediate tensor. A model passes only if no value differs by more than <span class="num">1e-4</span>; the measured maximum difference is printed on its badge (for example <span class="num">Δ≤3.1e-6</span>).{/snippet}
						{#snippet zh()}此刻在你的浏览器中计算。对于我们自己训练的小模型，前向传播用纯 TypeScript 实现，训练好的权重以原始 float32 数值加载。一致性测试会把相同输入分别送入 PyTorch 和我们的代码，并比较每一个中间张量。只有所有数值差异都不超过 <span class="num">1e-4</span> 才算通过；实测的最大差异显示在标记上（例如 <span class="num">Δ≤3.1e-6</span>）。{/snippet}
					</Lang>
				</dd>
			</div>
			<div>
				<dt><ProvenanceBadge mode="precomputed" precision="fp16" /></dt>
				<dd>
					{t({
						en: 'Computed offline from the same model and saved with the page, for models too large to run in a browser or for training runs that take hours. Quantized runs (for example q8 or q4) are labeled approximate, never exact.',
						zh: '用同一个模型离线计算并随页面保存，用于无法在浏览器中运行的大模型或需要数小时的训练过程。量化运行（如 q8、q4）标为“近似”，从不标为“精确”。'
					})}
				</dd>
			</div>
			<div>
				<dt><ProvenanceBadge mode="illustrative" /></dt>
				<dd>
					{t({
						en: 'Made-up example values that show an idea, used rarely and always labeled. We never fill a diagram with random numbers and present them as a model’s output.',
						zh: '为说明概念而编造的示例数值，很少使用且总会标注。我们绝不会用随机数填充图示并把它当作模型的输出。'
					})}
				</dd>
			</div>
		</dl>
		<p>
			{t({
				en: 'Why the numbers can differ at all: floating-point addition is not associative, so summing the same products in a different order (a GPU kernel versus a JavaScript loop) changes the last few bits. For fp32 the differences are usually around 1e-6 to 1e-5 and grow slowly with depth.',
				zh: '为什么数字会有差异：浮点加法不满足结合律，以不同顺序（GPU 内核与 JavaScript 循环）累加相同的乘积，会改变最后几位。对于 fp32，差异通常在 1e-6 到 1e-5 之间，并随网络深度缓慢增大。'
			})}
		</p>
	</section>

	<section id="colors" aria-labelledby="colors-h">
		<h2 id="colors-h">{t({ en: 'Reading the colors', zh: '如何读颜色' })}</h2>
		<div class="legends">
			<ColorLegend kind="diverging" domain={[-1, 1]} label={{ en: 'Signed values: blue below zero, orange above', zh: '有正负的值：负为蓝，正为橙' }} width={300} />
			<ColorLegend kind="sequential" domain={[0, 1]} label={{ en: 'Probabilities and attention: light to dark purple', zh: '概率与注意力：由浅到深的紫色' }} width={300} />
		</div>
		<p class="sw">
			{#each CATEGORICAL[themeState.resolved] as c, i (i)}<span class="chip-c" style="background: {c}" title="class {i}"></span>{/each}
			<span>{t({ en: 'Classes keep the same color on every view, always next to their name.', zh: '同一类别在所有视图中颜色相同，并始终配有名称。' })}</span>
		</p>
		<p class="roles">
			<RoleChip role="activation" /> <RoleChip role="param" /> <RoleChip role="grad" /> <RoleChip role="frozen" /> <RoleChip role="trainable" />
		</p>
		<p>
			{t({
				en: 'The scales were built in a perceptual color space and checked for common forms of color blindness; no rainbow ("jet") maps are used. Every hover readout also prints the number.',
				zh: '色阶在感知均匀的色彩空间中构建，并针对常见色盲类型做过检查；不使用彩虹（jet）色图。每个悬停读数都会同时显示数字。'
			})}
		</p>
	</section>

	<section id="privacy" aria-labelledby="privacy-h">
		<h2 id="privacy-h">{t({ en: 'Privacy', zh: '隐私' })}</h2>
		<ul>
			<li>{t({ en: 'Your drawings, text, images and sound are processed on your device. They are never uploaded.', zh: '你的绘画、文本、图片和声音都在你的设备上处理，从不上传。' })}</li>
			<li>{t({ en: 'The site makes no requests to third-party hosts: fonts, math rendering and model files are served from this site.', zh: '网站不向任何第三方主机发起请求：字体、公式渲染和模型文件都由本站提供。' })}</li>
			<li>{t({ en: 'There are no analytics and no cookies. Your theme and language choices are kept in your browser’s local storage only.', zh: '没有统计脚本，也没有 Cookie。你的主题和语言选择只保存在浏览器的本地存储中。' })}</li>
		</ul>
	</section>

	<section id="scope" aria-labelledby="scope-h">
		<h2 id="scope-h">{t({ en: 'What is here, and what is planned', zh: '已有内容与计划' })}</h2>
		<p>
			{t({
				en: `${n.liveModels} models in ${n.liveAreas} of the ${n.areas} areas and ${n.liveConcepts} concept pages are live, each with a guided, step-by-step lesson that runs on your own input. The atlas also lists every planned model (${n.models} in all) and concept (${n.concepts} in all), each with a page that says what it will show. New lessons are built with the DevKit, which anyone can use to add one.`,
				zh: `目前有 ${n.liveModels} 个模型（分布在 ${n.areas} 个领域中的 ${n.liveAreas} 个）和 ${n.liveConcepts} 个概念页面可以玩，每个都有一节用你自己的输入运行的分步课程。图鉴还列出了所有计划中的模型（共 ${n.models} 个）和概念（共 ${n.concepts} 个），每个都有页面说明它将展示什么。新的课程用 DevKit 制作，任何人都可以用它添加课程。`
			})}
			<a href="{base}/atlas/">{ui('browseAtlas')} →</a>
			<a href="{base}/devkit/">{ui('navDevkit')} →</a>
		</p>
	</section>

	<section id="source" aria-labelledby="source-h">
		<h2 id="source-h">{t({ en: 'Licence and source code', zh: '许可证与源代码' })}</h2>
		<p>
			{t({
				en: `Model Atlas is open source. Its code is under the ${OWN_LICENSES.code.name} licence; its lesson text, articles, narration and figures are under ${OWN_LICENSES.content.name}, so you may share and adapt them if you credit Model Atlas and say what you changed. Datasets, vocabularies, model files derived from them, fonts and libraries keep their own licences: the table below, with the full licence texts in the repository's LICENSES folder.`,
				zh: `模型图鉴是开源项目。代码采用 ${OWN_LICENSES.code.name} 许可证；课程文字、文章、旁白和插图采用 ${OWN_LICENSES.content.name}，只要注明出自模型图鉴并说明你改动了什么，就可以分享和改编。数据集、词表、由它们得到的模型文件、字体和程序库保留各自的许可证：见下表，许可证全文在仓库的 LICENSES 文件夹中。`
			})}
		</p>
		<ul class="links">
			<li><a href={REPO_URL} rel="noopener">{t({ en: 'Source code and issues on GitHub', zh: 'GitHub 上的源代码与问题反馈' })}</a> <span class="num muted">github.com/Robintianqili/model-atlas</span></li>
			<li><a href={repoFile(OWN_LICENSES.code.file)} rel="noopener">{t({ en: 'The MIT licence (code)', zh: 'MIT 许可证（代码）' })}</a></li>
			<li><a href={OWN_LICENSES.content.url} rel="noopener">{t({ en: 'CC BY 4.0 (lesson text, articles and figures)', zh: 'CC BY 4.0（课程文字、文章和插图）' })}</a></li>
			<li><a href={`${REPO_URL}/tree/main/LICENSES`} rel="noopener">{t({ en: 'Third-party licence texts (LICENSES/)', zh: '第三方许可证全文（LICENSES/）' })}</a></li>
		</ul>
	</section>

	<section id="licenses" aria-labelledby="licenses-h">
		<h2 id="licenses-h">{ui('footerLicenses')}</h2>
		<p>
			{t({
				en: 'Every third-party dataset, model, piece of code and font used by this site. This table is generated from static/licenses.json when the site is built.',
				zh: '本站使用的所有第三方数据集、模型、代码与字体。此表在构建网站时由 static/licenses.json 生成。'
			})}
		</p>
		{#if licenses.length}
			<div class="tbl-wrap">
				<table>
					<thead>
						<tr>
							<th scope="col">{t({ en: 'Name', zh: '名称' })}</th>
							<th scope="col">{t({ en: 'License', zh: '许可' })}</th>
							<th scope="col">{t({ en: 'Used for', zh: '用途' })}</th>
							<th scope="col">{t({ en: 'Attribution', zh: '署名' })}</th>
						</tr>
					</thead>
					<tbody>
						{#each licenses as l, i (i)}
							<tr>
								<td>{#if isUrl(l.url)}<a href={l.url} rel="noopener noreferrer">{l.name}</a>{:else}{l.name}{/if}</td>
								<td class="num">{#if l.licenseText}<a href={repoFile(l.licenseText)} rel="noopener" title={t({ en: 'Full licence text', zh: '许可证全文' })}>{l.license}</a>{:else}{l.license}{/if}</td>
								<td>{l.usedFor}</td>
								<td class="att">{l.attribution ?? ''}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<p class="muted">{t({ en: 'licenses.json is empty.', zh: 'licenses.json 为空。' })}</p>
		{/if}
	</section>
</div>

<style>
	.page {
		padding-top: var(--s6);
		max-width: 980px;
	}
	.head {
		display: grid;
		gap: 8px;
	}
	.head h1,
	.head p {
		margin: 0;
	}
	section {
		padding-top: var(--s6);
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	section p,
	section li {
		max-width: var(--measure);
	}
	.prov {
		display: grid;
		gap: 14px;
		margin: 0 0 var(--s4);
	}
	.prov div {
		display: grid;
		grid-template-columns: 230px 1fr;
		gap: 12px;
		align-items: start;
	}
	@media (max-width: 640px) {
		.prov div {
			grid-template-columns: 1fr;
		}
	}
	.prov dd {
		margin: 0;
		color: var(--ink-2);
	}
	.legends {
		display: flex;
		flex-wrap: wrap;
		gap: 16px 32px;
		margin-bottom: var(--s3);
	}
	.sw {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
	}
	.sw span:last-child {
		margin-left: 8px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.chip-c {
		width: 16px;
		height: 16px;
		border-radius: 4px;
	}
	.roles {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.tbl-wrap {
		overflow-x: auto;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: var(--fs-sm);
		background: var(--surface);
	}
	th,
	td {
		text-align: left;
		padding: 8px 12px;
		border-bottom: 1px solid var(--rule);
		vertical-align: top;
	}
	th {
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--muted);
		background: var(--surface-2);
	}
	.att {
		color: var(--ink-2);
		min-width: 200px;
	}
	.links {
		display: grid;
		gap: 4px;
		padding-left: 1.2em;
	}
</style>
