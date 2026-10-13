<!--
  Code section: the PyTorch definition the model was trained with (shown, not executed) and the
  TypeScript loop that recomputes a convolution in this page.
-->
<script lang="ts">
	import CodeBlock from '$lib/kit/CodeBlock.svelte';
	import Term from '$lib/kit/Term.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { CODE_LINES, MODEL_SOURCE, MODEL_SOURCE_START } from './data';
	import type { CnnStore } from './store.svelte';

	let { S }: { S: CnnStore } = $props();
	const TS_CONV = `function conv2dValid(
  x: Float32Array, C: number, H: number, W: number,
  w: Float32Array, b: Float32Array, O: number, k: number, stride: number
): Tensor {
  const OH = Math.floor((H - k) / stride) + 1;
  const OW = Math.floor((W - k) / stride) + 1;
  const out = new Float32Array(O * OH * OW);
  for (let o = 0; o < O; o++) {
    for (let oy = 0; oy < OH; oy++) {
      for (let ox = 0; ox < OW; ox++) {
        let s = 0;
        for (let c = 0; c < C; c++) {
          const wBase = (o * C + c) * k * k;
          const xBase = c * H * W;
          for (let ky = 0; ky < k; ky++) {
            const row = xBase + (oy * stride + ky) * W + ox * stride;
            const wRow = wBase + ky * k;
            for (let kx = 0; kx < k; kx++) s += x[row + kx] * w[wRow + kx];
          }
        }
        out[(o * OH + oy) * OW + ox] = s + b[o];
      }
    }
  }
  return { shape: [O, OH, OW], data: out };
}`;
	const tol = $derived(S.spec.parity?.max_abs_diff as number | undefined);
</script>

<div class="code">
	<p class="lead">
		<Lang>
			{#snippet en()}The model as it was defined and trained in PyTorch (<code>pipeline/tinycnn/common.py</code>). The highlighted lines belong to the layer selected in Explore: <strong>{t(S.layer(S.sel.layer).label)}</strong>. This page does not run Python.{/snippet}
			{#snippet zh()}这是模型在 PyTorch 中的定义和训练代码（<code>pipeline/tinycnn/common.py</code>）。高亮的行属于在“探索”里选中的层：<strong>{t(S.layer(S.sel.layer).label)}</strong>。本页并不运行 Python。{/snippet}
		</Lang>
	</p>
	<CodeBlock code={MODEL_SOURCE} lang="python" start={MODEL_SOURCE_START} highlight={CODE_LINES[S.sel.layer] ?? []} caption="pipeline/tinycnn/common.py · PyTorch" />
	<p class="lead">
		<Lang>
			{#snippet en()}In your browser the same layers run as plain TypeScript loops (<code>src/lib/models/tinycnn.ts</code>), reading the trained weights from a 26.9 KB file. This is the whole convolution. The numbers match PyTorch to within {tol !== undefined ? fmt3(tol) : '1e-4'} (a <Term id="parity">parity</Term> check on every layer).{/snippet}
			{#snippet zh()}在你的浏览器里，同样的层以普通的 TypeScript 循环运行（<code>src/lib/models/tinycnn.ts</code>），权重来自一个 26.9 KB 的文件。下面就是完整的卷积。每一层的数值与 PyTorch 相差不超过 {tol !== undefined ? fmt3(tol) : '1e-4'}（逐层<Term id="parity">一致性校验</Term>）。{/snippet}
		</Lang>
	</p>
	<CodeBlock code={TS_CONV} lang="typescript" start={234} caption="src/lib/models/tinycnn.ts · TypeScript" />
</div>

<style>
	.code {
		display: grid;
		gap: 12px;
		grid-template-columns: minmax(0, 1fr);
		max-width: 900px;
	}
	.lead {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
</style>
