// W1-B1 acceptance 2: values, gradients and double gradients of every op match PyTorch float64 (CPU) within 1e-10
// relative (|a − b| ≤ 1e-10·max(1, |b|)). Fixtures: tests/atlas/autograd/ops.json from pipeline/atlas_fixtures/autograd/make.py.
// Gradients are checked twice: the fast path (createGraph off) and the graph path (createGraph on).
// Run from site/:  node --test src/lib/atlas/autograd/ops.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Var, grad, input, raw, zeros } from './var.ts';
import { add, mul, sum } from './ops.ts';
import { DISPATCH, loadFixture, relErr, type OpCase } from '../../../../tests/atlas/autograd/support.ts';

const BAR = 1e-10;
const fixture = loadFixture<{ cases: OpCase[]; torch: string }>('ops.json');

function leaves(c: OpCase): Var[] {
	return c.inputs.map((t) => input(t, true));
}

test(`fixture covers every op (${fixture.cases.length} cases, torch ${fixture.torch})`, () => {
	const ops = new Set(fixture.cases.map((c) => c.op));
	for (const name of Object.keys(DISPATCH)) assert.ok(ops.has(name), `no torch case for '${name}'`);
});

for (const c of fixture.cases) {
	test(`torch fp64 parity: ${c.name}`, (t) => {
		const f = DISPATCH[c.op];
		assert.ok(f, `no dispatcher for ${c.op}`);
		// value
		const xs = leaves(c);
		const y = f(xs, c.attrs);
		assert.deepEqual(y.shape, c.y.shape, 'output shape');
		const ev = relErr(y.data, c.y.data);
		// first order, fast path
		const seed = input(c.r);
		const gFast = grad(y, xs, { seeds: seed, allowUnused: true });
		const eg = Math.max(...gFast.map((g, i) => relErr(g.data, c.grads[i].data)));
		// first order with a graph, then the double gradient of h = Σ_i <g_i, v_i>
		const xs2 = leaves(c);
		const y2 = f(xs2, c.attrs);
		const gGraph = grad(y2, xs2, { seeds: input(c.r), createGraph: true, allowUnused: true });
		const egg = Math.max(...gGraph.map((g, i) => relErr(g.data, c.grads[i].data)));
		let h: Var | null = null;
		gGraph.forEach((g, i) => {
			const term = sum(mul(g, raw(Float64Array.from(c.v[i].data), c.v[i].shape.slice())));
			h = h ? add(h, term) : term;
		});
		const hv = h as Var | null;
		const gg = hv && hv.requiresGrad ? grad(hv, xs2, { allowUnused: true }) : xs2.map((x) => zeros(x.shape));
		const e2 = Math.max(...gg.map((g, i) => relErr(g.data, c.ggrads[i].data)));
		t.diagnostic(`value ${ev.toExponential(2)}  grad ${eg.toExponential(2)}  grad(graph) ${egg.toExponential(2)}  double ${e2.toExponential(2)}`);
		assert.ok(ev <= BAR, `value rel err ${ev}`);
		assert.ok(eg <= BAR, `grad rel err ${eg}`);
		assert.ok(egg <= BAR, `grad (createGraph) rel err ${egg}`);
		assert.ok(e2 <= BAR, `double grad rel err ${e2}`);
	});
}
