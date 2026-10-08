// Browser checks for /dev/atlas-train/ (DevKit W1-B2 acceptance 1–4), headless Microsoft Edge through playwright-core.
// Not part of the site build (nothing imports it). Run from site/ against a dev server or a served build:
//   node src/routes/dev/atlas-train/e2e.check.ts http://localhost:5211 [shots-dir]
// Prints one JSON report. Screenshots go to shots-dir (default ../notes/shots/devkit/w1-b2 from the repo root).
import { chromium, type Page } from 'playwright-core';
import { mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

const origin = (process.argv[2] ?? 'http://localhost:5211').replace(/\/$/, '');
const shots = resolve(process.argv[3] ?? join(process.cwd(), '..', 'notes', 'shots', 'devkit', 'w1-b2'));
mkdirSync(shots, { recursive: true });
const url = (q = '') => `${origin}/dev/atlas-train/${q}`;
const report: Record<string, unknown> = { origin };
const errors: string[] = [];

type Win = { __atlasTrain?: { flowRt: any; toysRt: any; fitRt: any } };

async function open(page: Page, q = '') {
	page.on('console', (m) => {
		if (m.type() === 'error') errors.push(`${q || 'default'}: ${m.text()}`);
	});
	page.on('pageerror', (e) => errors.push(`${q || 'default'}: ${e.message}`));
	await page.goto(url(q), { waitUntil: 'networkidle' });
	await page.waitForFunction(() => !!(window as unknown as Win).__atlasTrain?.flowRt?.sources?.flow?.tape?.length, null, { timeout: 30000 });
}

const browser = await chromium.launch({ channel: 'msedge', headless: true });
try {
	// ------------------------------------------------------------ acceptance 1 + 3 + 4 (worker)
	const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: 'light' });
	const page = await ctx.newPage();
	await page.addInitScript(() => {
		const w = window as unknown as { __long: { start: number; ms: number }[] };
		w.__long = [];
		try {
			new PerformanceObserver((l) => {
				for (const e of l.getEntries()) w.__long.push({ start: e.startTime, ms: e.duration });
			}).observe({ type: 'longtask', buffered: true });
		} catch {
			/* no longtask support */
		}
	});
	await open(page);
	report.where = await page.textContent('[data-testid=where]');
	await page.screenshot({ path: join(shots, 'desktop-light-start.png'), fullPage: false });

	// train the flow MLP for 10 s
	const t0 = await page.evaluate(() => performance.now());
	await page.click('[data-testid=flow-controls] button.primary');
	await page.waitForTimeout(10000);
	const train = await page.evaluate((since) => {
		const w = window as unknown as Win & { __long: { start: number; ms: number }[] };
		const tp = w.__atlasTrain!.flowRt.sources.flow.tape;
		const steps = tp.metric('step');
		const lastStep = steps[steps.length - 1];
		const longs = w.__long.filter((l) => l.start >= since);
		return { frames: tp.length, lastStep, maxHandleMs: tp.stats.maxHandleMs, totalHandleMs: tp.stats.totalHandleMs, messages: tp.stats.messages, longTasks: longs.length, longMaxMs: longs.reduce((m, l) => Math.max(m, l.ms), 0), longOver100: longs.filter((l) => l.ms > 100).length, where: tp.where };
	}, t0);
	report.train10s = { ...train, stepsPerSec: train.lastStep / 10 };
	await page.screenshot({ path: join(shots, 'desktop-light-training.png'), fullPage: false });

	// acceptance 3: pause, then an lr change (measured on the main thread)
	report.pause = await page.evaluate(async () => {
		const w = window as unknown as Win;
		const tp = w.__atlasTrain!.flowRt.sources.flow.tape;
		const lastStep = () => {
			const s = tp.metric('step');
			return s[s.length - 1];
		};
		const out: Record<string, number>[] = [];
		for (let k = 0; k < 10; k++) {
			tp.control({ type: 'resume' });
			await new Promise((r) => setTimeout(r, 400));
			const n0 = tp.length;
			const s0 = lastStep();
			const t0 = performance.now();
			tp.control({ type: 'pause' });
			await new Promise((r) => setTimeout(r, 400));
			out.push({ framesAfter: tp.length - n0, stepsAfter: lastStep() - s0, lastFrameMs: Math.max(0, tp.stats.lastAt - t0) });
		}
		return out;
	});
	report.lr = await page.evaluate(async () => {
		const w = window as unknown as Win;
		const tp = w.__atlasTrain!.flowRt.sources.flow.tape;
		const out: Record<string, number>[] = [];
		const lrs = [0.003, 0.0005, 0.002, 0.001, 0.0007, 0.0004, 0.0025, 0.0009, 0.0015, 0.001];
		tp.control({ type: 'resume' });
		await new Promise((r) => setTimeout(r, 400));
		for (const lr of lrs) {
			const n0 = tp.length;
			const s0 = tp.frame(n0 - 1).step;
			const t0 = performance.now();
			let seenAt = -1;
			let arrivedAt = -1;
			// arrival: synchronous on the push into the TapeStore; on screen: the (animation-frame) subscription
			const offPush = tp.store.onPush((n: number) => {
				if (arrivedAt < 0 && tp.frame(n - 1).metrics?.lr === lr) arrivedAt = performance.now() - t0;
			});
			const off = tp.subscribe(() => {
				if (seenAt >= 0) return;
				for (let i = n0; i < tp.length; i++)
					if (tp.frame(i).metrics?.lr === lr) {
						seenAt = performance.now() - t0;
						break;
					}
			});
			tp.control({ type: 'set', opt: { lr } });
			await new Promise((r) => setTimeout(r, 300));
			off();
			offPush();
			let firstStep = -1;
			for (let i = n0; i < tp.length; i++)
				if (tp.frame(i).metrics?.lr === lr) {
					firstStep = tp.frame(i).step;
					break;
				}
			out.push({ lr, msToArrive: arrivedAt, msToScreen: seenAt, stepsBetween: firstStep - s0 });
		}
		tp.control({ type: 'pause' });
		return out;
	});
	// the worker's own log: frames computed between a control's arrival and its effect (0 = the very next frame)
	report.workerControls = await page.evaluate(async () => {
		const w = window as unknown as Win & { __atlasTrain: { runner: any } };
		const rt = w.__atlasTrain.flowRt;
		const s = await w.__atlasTrain.runner.client(rt.pkg).call('stats', null, 'stats');
		const recs = (s?.controls ?? []).filter((r: { type: string }) => r.type === 'set' || r.type === 'pause');
		return recs.map((r: { type: string; receivedAt: number; appliedAt: number; ms: number }) => ({ type: r.type, framesBetween: r.appliedAt - r.receivedAt, ms: Math.round(r.ms * 10) / 10 }));
	});

	// hidden tab: a running source pauses (with a notice) and resumes when visible again
	report.hidden = await page.evaluate(async () => {
		const tp = (window as unknown as Win).__atlasTrain!.flowRt.sources.flow.tape;
		const setVis = (v: string) => {
			Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => v });
			document.dispatchEvent(new Event('visibilitychange'));
		};
		tp.control({ type: 'resume' });
		await new Promise((r) => setTimeout(r, 300));
		setVis('hidden');
		await new Promise((r) => setTimeout(r, 150));
		const n0 = tp.length;
		const notice = tp.notice?.en ?? null;
		await new Promise((r) => setTimeout(r, 400));
		const grewWhileHidden = tp.length - n0;
		setVis('visible');
		await new Promise((r) => setTimeout(r, 400));
		const resumed = tp.length > n0 + grewWhileHidden;
		tp.control({ type: 'pause' });
		return { notice, grewWhileHidden, resumed, noticeAfter: tp.notice?.en ?? null };
	});
	// "+1 step" on the fine SGD source runs one whole step (forward, loss, backward, update)
	report.sgdStep = await page.evaluate(async () => {
		const tp = (window as unknown as Win).__atlasTrain!.toysRt.sources.sgd1.tape;
		const n0 = tp.length;
		(document.querySelector('#sgd button.btn:not(.primary):not(.ghost)') as HTMLButtonElement).click();
		await new Promise((r) => setTimeout(r, 600));
		return { before: n0, after: tp.length, lastPhase: tp.frame(tp.length - 1).phase, phases: Array.from({ length: tp.length - n0 }, (_, i) => tp.frame(n0 + i).phase) };
	});

	// acceptance 4 in the browser: worker frames = a main-thread run, hash for hash
	await page.click('#classifier button.primary');
	await page.waitForTimeout(1500);
	await page.click('[data-testid=hash-check]');
	await page.waitForSelector('[data-testid=hash-result]', { timeout: 30000 });
	report.hash = await page.textContent('[data-testid=hash-result]');
	await page.screenshot({ path: join(shots, 'desktop-light-full.png'), fullPage: true });
	await ctx.close();

	// ------------------------------------------------------------ acceptance 2: main-thread fallback
	for (const q of ['?engine=broken', '?engine=main']) {
		const c2 = await browser.newContext({ viewport: { width: 1280, height: 900 } });
		const p2 = await c2.newPage();
		await open(p2, q);
		await p2.click('[data-testid=flow-controls] button.primary');
		await p2.waitForTimeout(3000);
		report[`fallback${q}`] = await p2.evaluate(() => {
			const tp = (window as unknown as Win).__atlasTrain!.flowRt.sources.flow.tape;
			return { where: tp.where, frames: tp.length, notice: tp.notice?.en ?? null, error: tp.error };
		});
		if (q === '?engine=broken') await p2.screenshot({ path: join(shots, 'desktop-broken-fallback.png') });
		await c2.close();
	}
	// the expected fallback notices: a failed worker is not a console error of ours, but the browser logs it
	// ------------------------------------------------------------ phone, dark, zh, reduced motion
	const c3 = await browser.newContext({ viewport: { width: 390, height: 844 }, colorScheme: 'dark', reducedMotion: 'reduce', deviceScaleFactor: 2, isMobile: true, hasTouch: true });
	const p3 = await c3.newPage();
	await open(p3, '?lang=zh');
	await p3.click('[data-testid=flow-controls] button.primary');
	await p3.waitForTimeout(3000);
	report.phone = await p3.evaluate(() => ({ scrollW: document.documentElement.scrollWidth, clientW: document.documentElement.clientWidth, lang: document.documentElement.lang }));
	await p3.screenshot({ path: join(shots, 'phone-dark-zh-top.png') });
	await p3.screenshot({ path: join(shots, 'phone-dark-zh-full.png'), fullPage: true });
	await c3.close();
	const c4 = await browser.newContext({ viewport: { width: 360, height: 780 } });
	const p4 = await c4.newPage();
	await open(p4);
	report.phone360 = await p4.evaluate(() => ({ scrollW: document.documentElement.scrollWidth, clientW: document.documentElement.clientWidth }));
	await c4.close();
} finally {
	report.consoleErrors = errors;
	console.log(JSON.stringify(report, null, 1));
	await browser.close();
}
