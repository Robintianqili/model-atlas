<!-- matrix-factorization: the page's Output section, the filled-in matrix and your top recommendations -->
<script lang="ts">import Explore from '../parts/Explore.svelte'; let { store } = $props();</script>
<Explore {store} view="matrix-factorization/matrix" props={{ fill: true, rank: true }} note={{ en: 'Every empty cell filled with a dot product, and your three best unrated movies numbered. Change your ratings in “Your input” above to see the list change.', zh: '每个空格都填上了一个点积，你没评过的电影里最好的三部标了号。在上面的“你的输入”里改评分，看列表怎么变。' }} />
