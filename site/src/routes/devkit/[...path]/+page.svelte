<!-- /devkit/...: the DevKit pages (src/lib/devkit-site/). The page data is generated at build time (+page.server.ts). -->
<script lang="ts">
	import DevkitPage from '$lib/devkit-site/DevkitPage.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<DevkitPage page={data.page} />
