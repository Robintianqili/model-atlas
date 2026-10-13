// The static dev-server port table (design §6.4, §7; DEVKIT.md prints it). One port per package, so parallel
// lanes never fight over 5173. Run: npm run dev -- --port <port>
// A package without a row gets a port when `atlas new` scaffolds it: the first free one from 5210 up that no table row
// and no other package's notes (notes/<slug>.md, "npm run dev -- --port N") already use; the notes record it.
// `atlas e2e` and `atlas check` start their own dev server: --port n, else the package's port (table row or notes) when
// it is free, else the first free port from 5210 up. They print the port they use.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { join } from 'node:path';
export const PORTS: Record<string, number> = {
	// wave 2 models (design §7)
	tinyvit: 5191,
	'mlp-playground': 5192,
	gridworld: 5193,
	'char-rnn': 5194,
	gcn: 5195,
	'flow-2d': 5196,
	tinyclip: 5197,
	tinyforecaster: 5198,
	pinn: 5199,
	// wave 1 kit lanes (5201 upward, design §8.1)
	'w1-a': 5201,
	'w1-b1': 5202,
	'w1-c': 5203,
	'w1-e1': 5204,
	// wave 1b lanes (the ports they actually ran on; integration 1b)
	'w1-b2': 5211,
	'w1-d1': 5212,
	'w1-d2': 5213,
	'w1-e2': 5214,
	// dev-only kit packages (Lesson Lab only; hello-xor was built in the w1-e2 lane)
	'hello-xor': 5214
};

export const DEFAULT_DEV_PORT = 5173;

export const FIRST_FREE_PORT = 5210;

/** The port recorded in notes/<slug>.md by `atlas new` ("npm run dev -- --port N"), or null. */
export function notesPort(root: string, slug: string): number | null {
	const f = join(root, 'notes', `${slug}.md`);
	if (!existsSync(f)) return null;
	const m = /npm run dev -- --port (\d{4,5})/.exec(readFileSync(f, 'utf8'));
	return m ? Number(m[1]) : null;
}

/** Every port the table and the notes of packages already claim (except `slug`'s own). */
export function claimedPorts(root: string | null, except?: string): Set<number> {
	const used = new Set(Object.entries(PORTS).filter(([k]) => k !== except).map(([, p]) => p));
	if (root && existsSync(join(root, 'notes')))
		for (const f of readdirSync(join(root, 'notes'))) {
			if (!f.endsWith('.md') || f === `${except}.md`) continue;
			const p = notesPort(root, f.slice(0, -3));
			if (p !== null) used.add(p);
		}
	return used;
}

/** The port of a package: its table row, else the port its notes record, else null. */
export function packagePort(slug: string, root: string | null): number | null {
	return PORTS[slug] ?? (root ? notesPort(root, slug) : null);
}

/** A port for a package without a row, without checking what is running: 5210 upward, skipping the table (and, with
 *  `root`, every port another package's notes claim). `atlas new` and `atlas e2e` use pickPort(), which also skips busy
 *  ports. */
export function suggestPort(slug: string, root: string | null = null): number {
	const own = packagePort(slug, root);
	if (own !== null) return own;
	const used = claimedPorts(root, slug);
	let p = FIRST_FREE_PORT;
	while (used.has(p)) p++;
	return p;
}

function listenFree(port: number, host: string): Promise<boolean | null> {
	return new Promise((res) => {
		const s = createServer();
		s.once('error', (e: NodeJS.ErrnoException) => res(e.code === 'EADDRINUSE' || e.code === 'EACCES' ? false : null));
		s.listen({ port, host, exclusive: true }, () => s.close(() => res(true)));
	});
}

/** Nothing listens on `port` on this machine: checked on the IPv4 and the IPv6 loopback, since vite's "localhost" may
 *  bind either (a host this machine does not have, such as ::1 without IPv6, does not count). */
export async function portFree(port: number): Promise<boolean> {
	for (const host of ['127.0.0.1', '::1']) if ((await listenFree(port, host)) === false) return false;
	return true;
}

/** A free port: `preferred` when it is free, else the first port from 5210 up that is free right now and that no table
 *  row or other package's notes claim. */
export async function pickPort(o: { preferred?: number | null; slug?: string; root?: string | null; exclude?: number[] } = {}): Promise<number> {
	if (o.preferred && !o.exclude?.includes(o.preferred) && (await portFree(o.preferred))) return o.preferred;
	const used = claimedPorts(o.root ?? null, o.slug);
	for (const p of o.exclude ?? []) used.add(p);
	if (o.preferred) used.add(o.preferred);
	for (let p = FIRST_FREE_PORT; p < FIRST_FREE_PORT + 400; p++) if (!used.has(p) && (await portFree(p))) return p;
	throw new Error(`no free port from ${FIRST_FREE_PORT} to ${FIRST_FREE_PORT + 399}`);
}
