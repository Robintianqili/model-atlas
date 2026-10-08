// Where Model Atlas lives and under which licences (the About page and the site footer show them). Pure data.

/** The public repository: source code, issues, the full licence texts (LICENSES/). */
export const REPO_URL = 'https://github.com/Robintianqili/model-atlas';
/** The published site. */
export const SITE_URL = 'https://robintianqili.github.io/model-atlas/';

/** A file of the repository on GitHub (its main branch), e.g. repoFile('LICENSES/Apache-2.0.txt'). */
export const repoFile = (path: string): string => `${REPO_URL}/blob/main/${path.replace(/^\/+/, '')}`;

/** The project's own licences. Third-party material keeps its own (static/licenses.json; texts in LICENSES/). */
export const OWN_LICENSES = {
	/** everything that runs, builds or tests the site and the models */
	code: { name: 'MIT', file: 'LICENSE' },
	/** lesson text, articles, narration, figures and diagrams */
	content: { name: 'CC BY 4.0', file: 'LICENSE-CONTENT.md', url: 'https://creativecommons.org/licenses/by/4.0/' }
} as const;
