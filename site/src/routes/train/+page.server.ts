// /train/ (redesign spec §6.5): the atlas filtered to Kind = Training methods, from the same build-time index.
import { catalogIndex } from '$lib/kit/catalog/lessons';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = () => ({ index: catalogIndex() });
