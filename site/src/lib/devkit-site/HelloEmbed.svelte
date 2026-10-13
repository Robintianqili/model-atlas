<!--
  The finished hello-xor lesson, embedded on the tutorial page: the kit's own auto page (the one /m/<slug>/ renders),
  loaded through the same lazy door the model pages use (atlas/ui/autoentry.ts), only when the learner asks for it.
  hello-xor is lab-only (devOnly), so the kit lays it out as the Lesson Lab does, without the catalog page shell.
-->
<script lang="ts">
	import type { Component } from 'svelte';
	import { t } from '$lib/i18n/i18n.svelte';

	let { slug, spec }: { slug: string; spec: unknown } = $props();
	let Auto = $state.raw<Component<any> | null>(null);
	let status = $state<'idle' | 'loading' | 'error'>('idle');
	let err = $state('');

	async function load() {
		status = 'loading';
		try {
			Auto = await (await import('$lib/atlas/ui/autoentry.ts')).loadAutoPage();
			status = 'idle';
		} catch (e) {
			err = e instanceof Error ? e.message : String(e);
			status = 'error';
		}
	}
</script>

{#if Auto}
	<div class="frame" role="region" aria-label={t({ en: 'The hello-xor lesson, embedded', zh: '嵌入的 hello-xor 课程' })}>
		<Auto {slug} {spec} />
	</div>
{:else}
	<div class="load">
		<p>{t({ en: 'It loads the kit and the package, and runs its sources in your browser, the live training included.', zh: '这会加载工具包和这个包，并在你的浏览器里运行它的各个来源，包括实时训练。' })}</p>
		<button type="button" class="btn" onclick={load} disabled={status === 'loading'}>
			{status === 'loading' ? t({ en: 'Loading…', zh: '正在加载…' }) : t({ en: 'Load the hello-xor lesson here', zh: '在这里加载 hello-xor 课程' })}
		</button>
		{#if status === 'error'}<p class="bad">{err}</p>{/if}
	</div>
{/if}

<style>
	.frame {
		margin-top: var(--s3);
		border: 1px solid var(--rule-strong);
		border-radius: var(--r4);
		background: var(--paper);
		overflow: hidden;
		min-width: 0;
	}
	.frame :global(.lab-page) {
		padding-top: var(--s4);
		padding-bottom: var(--s4);
	}
	.load {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--s3) var(--s5);
		padding: var(--s4) var(--s5);
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
	}
	.load p {
		flex: 1 1 280px;
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.bad {
		flex-basis: 100%;
		color: var(--div-neg-text);
	}
</style>
