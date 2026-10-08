<!-- mlp-playground: the page's Output section on the live process sources (why: notes/mlp-playground.md, Kit issues 1 and 3) -->
<script lang="ts">import Live from '../parts/LiveStage.svelte'; let { store } = $props();</script>
<Live {store} source="run" view="plane2d" note={{ en: 'The answer everywhere on the plane: the share for class 1, with the border where both classes get one half.', zh: '整个平面上的回答：属于第 1 类的比例，以及两类各占一半的边界。' }} />
