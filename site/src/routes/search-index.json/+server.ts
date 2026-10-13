// The site search index (redesign spec §10.5): every live lesson (name and hook), area and section, glossary term (names
// only) and planned lesson, prerendered as one JSON file that the header's SearchPalette fetches on first use.
import { json } from '@sveltejs/kit';
import { searchIndex } from '$lib/kit/catalog/lessons';
import type { RequestHandler } from './$types';

export const prerender = true;

export const GET: RequestHandler = () => json(searchIndex());
