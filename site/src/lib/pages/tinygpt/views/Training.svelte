<!--
  TinyGPT's Training view (redesign spec §5.9), loaded only when the view opens: how the model was trained, once and
  offline (precomputed numbers from that run; the page never trains), then the credits (#credits opens them).
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Section from '$lib/kit/Section.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import Credits from '../Credits.svelte';
	import type { TGPage } from '../page.svelte';

	let { vp }: { vp: { P: TGPage }; section?: string } = $props();
	const metrics = $derived((vp.P.spec?.training as { metrics?: Record<string, number> } | undefined)?.metrics);
</script>

<Section id="training-data" title={ui('secTraining')} provenance="precomputed">
	<div class="train">
		<p>
			{t({
				en: 'TinyGPT was trained once, offline, on a laptop GPU. These numbers come from that run (precomputed); the page does not train.',
				zh: 'TinyGPT 是在一块笔记本 GPU 上离线训练好的。下面的数字来自那次训练（预计算）；本页不会进行训练。'
			})}
		</p>
		<dl class="tfacts">
			<div><dt>{t({ en: 'Data', zh: '数据' })}</dt><dd>TinyStories (Eldan &amp; Li 2023, CDLA-Sharing-1.0) · 32.5M {t({ en: 'tokens', zh: '词元' })}</dd></div>
			<div><dt>{t({ en: 'Training', zh: '训练' })}</dt><dd class="num">30,000 {t({ en: 'steps × 512 × 64 tokens ≈ 30 passes', zh: '步 × 512 × 64 词元 ≈ 30 遍' })}</dd></div>
			<div><dt>{t({ en: 'Validation loss', zh: '验证集损失' })}</dt><dd class="num">{#if metrics}<Num value={metrics.val_loss} digits={4} />{:else}1.919{/if} nats / {t({ en: 'token', zh: '词元' })}</dd></div>
			<div><dt>{t({ en: 'Perplexity', zh: '困惑度' })}</dt><dd class="num">{#if metrics}<Num value={metrics.val_perplexity} digits={3} />{:else}6.81{/if}</dd></div>
			<div><dt>{t({ en: 'Guessing at random', zh: '随机乱猜' })}</dt><dd class="num">ln 1024 = 6.93 nats</dd></div>
		</dl>
		<p>
			<a href="{base}/train/next-token-prediction/">{t({ en: 'Watch TinyGPT learn: replay its 10 training checkpoints on the next-token prediction page →', zh: '看 TinyGPT 如何学会：在“下一个词元预测”页回放它的 10 个训练检查点 →' })}</a>
		</p>
	</div>
</Section>
<Section id="training-credits" title={ui('credits')}>
	<Credits />
</Section>

<style>
	.train {
		max-width: var(--measure);
	}
	.tfacts {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
		gap: 8px 20px;
		margin: 0 0 12px;
	}
	.tfacts dt {
		color: var(--muted);
		font-size: var(--t-xs);
	}
	.tfacts dd {
		margin: 0;
		font-weight: var(--w-ui);
	}
</style>
