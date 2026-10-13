// Page data of the DevKit pages (/devkit/...), built at build time by data.server.ts and rendered by the components in
// this folder. HTML strings come from md.ts (escaped source text); every site URL in them starts with md.ts BASE.

export interface T2 {
	en: string;
	zh: string;
}

export interface SectionRef {
	n: number;
	slug: string;
	title: string;
	zh: string;
}

export interface HomeData {
	kind: 'home';
	counts: { sections: number; ops: number; views: number; commands: number; codes: number };
	/** the live model lessons of the catalog, and those of them that are DevKit packages */
	built: { live: number; kit: { slug: string; name: T2 }[] };
	/** from CONTRIBUTING.md: its status note (null when it has none) with its Chinese, and its setup list */
	start: { statusHtml: string | null; statusZh: string; setupHtml: string | null };
	kinds: { kind: 'A' | 'B' | 'C'; name: T2; what: T2; youWrite: T2; example: { slug: string; name: T2; steps: number | null; summary: T2 | null } }[];
	workflow: { id: string; name: T2; what: T2; who: 'you' | 'maintainers' | 'reviewer'; cmds: { name: string; summary: string }[] }[];
	afternoon: { seconds: number; gaps: number; built: string };
	quick: { title: string; html: string };
	sections: SectionRef[];
}

export interface GuideIndexData {
	kind: 'guide';
	title: string;
	introHtml: string;
	sections: (SectionRef & { subs: number })[];
}

export interface SectionData {
	kind: 'section';
	/** omitted: local-only blocks of DEVKIT.md (notes about the maintainers' own computer) left out of this page */
	section: SectionRef & { html: string; subs: { id: string; text: string }[]; summary: string; omitted: number };
	prev: SectionRef | null;
	next: SectionRef | null;
	sections: SectionRef[];
}

export interface OpData {
	op: string;
	label: T2;
	category: string;
	view: string;
	formula: string | null;
	narration: { key: string; en: string; zh: string }[];
	attrs: { name: string; type: string; def: string | null; required: boolean; doc: string | null }[];
	weights: string[];
	subs: string[];
	aliases: string[];
	parityBar: number;
	pytorch: string | null;
	demo: { model: string; layer: string; unit: number[] | null };
}

export interface OpsData {
	kind: 'ops';
	ops: OpData[];
	categories: { id: string; label: T2 }[];
	guide: string;
}

export interface ViewData {
	key: string;
	title: T2;
	kinds: string[];
	minHeight: number | null;
	describe: T2 | null;
	pkg: string | null;
	/** the package's live lesson (BASE-prefixed href), where the view is seen at work; null for kit views */
	lesson: { href: string; name: T2 } | null;
	ops: string[];
}

export interface ViewsData {
	kind: 'views';
	views: ViewData[];
	guide: string;
}

export interface ReferenceData {
	kind: 'reference';
	ops: number;
	views: number;
	packageViews: number;
}

export interface Produce {
	caption: T2;
	note?: T2;
	provenance?: 'precomputed';
	/** cells are language-neutral text or bilingual */
	table?: { head: T2[]; rows: (string | T2)[][]; num?: boolean[] };
	codeHtml?: string;
	list?: string[];
	text?: string;
}

export interface TutorialStep {
	id: string;
	title: T2;
	narration: T2;
	look: T2 | null;
	fileHtml: string | null;
	file: string | null;
	commandHtml: string | null;
	produces: Produce;
}

export interface TutorialData {
	kind: 'tutorial';
	steps: TutorialStep[];
	hello: { slug: string; spec: unknown; port: number | null };
	guide: string;
}

export interface CodeRow {
	code: string;
	en: string;
	zh: string;
	group: string;
	html: string;
}

export interface ContributeData {
	kind: 'contribute';
	contributingHtml: string;
	contributingZh: string;
	checklistHtml: string;
	gatesHtml: string;
	qualityHtml: string;
	codes: CodeRow[];
	groups: { id: string; label: T2 }[];
	errorsGuide: string;
	gatesGuide: string;
}

export type PageData = HomeData | GuideIndexData | SectionData | OpsData | ViewsData | ReferenceData | TutorialData | ContributeData;
