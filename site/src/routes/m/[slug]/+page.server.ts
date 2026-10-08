import { autoPage } from '$lib/atlas/ui/discover.server';
import type { PageServerLoad } from './$types';

// DevKit (design §5.1): an included explainer package (catalog status 'live', or any draft while ATLAS_INCLUDE_DRAFTS=1;
// the dev server includes drafts) renders its auto page; the spec and the measured auto-load bytes are embedded at build
// time. Every other slug gets `auto: null` and renders the Planned page exactly as before.
export const load: PageServerLoad = ({ params }) => ({ auto: autoPage(params.slug) });
