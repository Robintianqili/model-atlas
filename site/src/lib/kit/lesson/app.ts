/**
 * The lesson app's context (redesign spec §4.4, §5): what a lesson page shell (ModelPageShell, ConceptPageShell) tells
 * the Walkthrough inside its Lesson view: the page's name and crumbs, the views it has, the input chip and sheet, the
 * About sheet, the cover's goals and question, and the end card's links. The Walkthrough writes back where the learner
 * is (step k of N), which the shell's top bar and the "Back to the lesson" pill read in the other views.
 *
 * A Walkthrough without this context (the DevKit tutorial, the kit gallery, the Lesson Lab of a devOnly package) renders
 * as a card in the page: no top bar, no app mode, no cover.
 */
import { getContext, setContext, type Snippet } from 'svelte';
import type { PredictQuestion, Text, WalkContext } from '../types';
import type { LessonKind, ViewId } from './views';

export interface ViewTab {
	id: ViewId;
	label: Text;
	/** one line for the More sheet and the end card */
	note?: Text;
	/** an icon of $lib/kit/icons (inner SVG markup) */
	icon: string;
}

export interface LessonNav {
	view: ViewId;
	section?: string;
}

/** Where the learner is; the Walkthrough writes it. */
export interface LessonPos {
	at: number;
	n: number;
	/** the cover (step 0) is showing */
	cover: boolean;
	title?: Text;
	chapter?: Text;
	/** the lesson's own deep-link prefix */
	id: string;
}

export interface LessonApp {
	slug: string;
	kind: LessonKind;
	/** the page's name (the model or concept) */
	name: Text;
	/** Atlas / Vision (models) or Training methods (concepts) */
	crumbs: { href: string; label: Text }[];
	/** the views with content, Lesson first */
	tabs: ViewTab[];
	/** the page has the site header and footer around it (a hand-built page before its migration): no app mode */
	embedded: boolean;
	nav: LessonNav;
	pos: LessonPos;
	/** the cover's goals, question and promise */
	goals: Text[];
	prequestion?: PredictQuestion;
	summary?: Text;
	minutes?: number;
	/** the input chip of the stage header, and the sheet it opens */
	inputChip?: Snippet<[() => void]>;
	inputSheet?: Snippet;
	/** the cover's stage (the Walkthrough's own `cover` wins) */
	cover?: Snippet<[WalkContext]>;
	/** the About sheet's body */
	about?: Snippet;
	aboutTitle: Text;
	/** the end card's "Up next" */
	upNext?: { href: string; title: Text; note?: Text };
	/** open the input sheet (the chip, #input) */
	input: { open: boolean };
	/** switch the view (a history entry, like a tab click) */
	go: (view: ViewId, section?: string) => void;
}

const KEY = Symbol('atlas-lesson-app');

export function setLessonApp(app: LessonApp): LessonApp {
	return setContext(KEY, app);
}

export function getLessonApp(): LessonApp | undefined {
	return getContext<LessonApp | undefined>(KEY);
}
