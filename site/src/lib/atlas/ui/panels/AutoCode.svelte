<!--
  AutoCode (redesign spec §5.9): a kit model page's Code view, loaded only when the view opens: the forward pass as
  PyTorch lines (from the spec's ops), how the browser's numbers were checked against it (the parity record) and how to
  run the check again. A package replaces it with explainers/<slug>/sections/Code.svelte (props { store, pkg, spec }).
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import CodeBlock from '$lib/kit/CodeBlock.svelte';
	import Section from '$lib/kit/Section.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { AtlasSpec } from '../../types/spec.ts';
	import type { PackageModules } from '../../types/runtime.ts';
	import type { AtlasStore } from '../runtime.svelte.ts';
	import { loadOverrides, type Overrides } from './overrides.ts';
	import { say } from '../strings.ts';

	interface Props {
		vp: { store: AtlasStore | null; pkg: PackageModules | null; spec: AtlasSpec | null; slug: string };
		section?: string;
	}
	let { vp }: Props = $props();
	const store = $derived(vp.store);
	const spec = $derived(vp.spec);
	let ov = $state.raw<Overrides>({});
	let ready = $state(false);
	onMount(() => {
		loadOverrides(vp.slug, ['Code']).then((o) => {
			ov = o;
			ready = true;
		});
	});
	const C = $derived(ov.Code ?? null);
	const lines = $derived.by(() => {
		const m = store?.r.model;
		if (!spec || !m) return [];
		return spec.layers
			.map((l) => {
				try {
					const d = m.ops.get(l.op).def;
					return d.pytorch ? `${l.id.padEnd(12)} = ${d.pytorch(l)}` : null;
				} catch {
					return null;
				}
			})
			.filter((x): x is string => !!x);
	});
	const parity = $derived(spec?.parity && typeof spec.parity.max_abs_diff === 'number' ? spec.parity : null);
	const sources = $derived(Object.entries(vp.pkg?.parity?.sources ?? {}));
	const slug = $derived(vp.slug ?? vp.pkg?.slug ?? '');
	const exp = (v: number) => (v === 0 ? '0' : v.toExponential(2));
</script>

<!-- a package's Code section takes a store that is never null: wait for the runtime (a #code deep link opens this view
     before the lesson has made its store) -->
{#if !ready || (C && !vp.store)}
	<p class="muted">{t({ en: 'Loading…', zh: '加载中……' })}</p>
{:else if C}
	<C store={vp.store} pkg={vp.pkg} spec={vp.spec} />
{:else if lines.length}
	<Section id="code-forward" title={t({ en: 'The forward pass', zh: '前向计算' })}>
		<CodeBlock code={lines.join('\n')} lang="python" caption={say('codeFor')} />
	</Section>
{/if}
<Section id="code-check" title={t({ en: 'How the numbers were checked', zh: '这些数字如何核对' })}>
	{#if parity}
		<p>
			{t({
				en: `The browser runs this model in ${spec?.precision ?? 'fp32'}. Checked against ${parity.against ?? 'PyTorch'}: every layer within ${exp(parity.max_abs_diff as number)}.`,
				zh: `浏览器用 ${spec?.precision ?? 'fp32'} 运行这个模型。与 ${parity.against ?? 'PyTorch'} 对照：每一层的误差都在 ${exp(parity.max_abs_diff as number)} 以内。`
			})}
		</p>
	{/if}
	{#each sources as [name, r] (name)}
		<p>
			<code>{name}</code>: {r.pass ? t({ en: 'matches', zh: '吻合' }) : t({ en: 'does not match', zh: '不吻合' })}
			{r.against} · max <span class="num">{exp(r.max)}</span>{r.relative ? t({ en: ' (relative)', zh: '（相对）' }) : ''}
		</p>
	{/each}
	{#if !parity && !sources.length}
		<p class="muted">{t({ en: 'The numbers are computed in your browser in JavaScript (f64); this lesson has no separate reference run.', zh: '这些数在你的浏览器里用 JavaScript（f64）计算；这节课没有单独的参考运行。' })}</p>
	{/if}
	{#if slug}
		<p class="muted">{t({ en: 'Run the checks again from the site folder:', zh: '在 site 目录下重新运行检查：' })}</p>
		<CodeBlock code={`npm run atlas -- check ${slug}`} lang="bash" />
	{/if}
</Section>
