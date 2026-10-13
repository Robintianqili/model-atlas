// /atlas/ (redesign spec §6.4): the catalog index at build time (catalog pages ship no catalog modules; kit/catalog/lessons.ts).
import { catalogIndex } from '$lib/kit/catalog/lessons';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = () => ({ index: catalogIndex() });
