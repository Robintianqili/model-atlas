// W1-B1 acceptance 1: gradcheck and gradgradcheck pass for every op (relative ≤ 1e-6), at the fixture points
// of tests/atlas/autograd/ops.json (they keep away from the kinks of relu and abs).
// Run from site/:  node --test src/lib/atlas/autograd/check.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { gradcheck } from './check.ts';
import { input } from './var.ts';
import * as op from './ops.ts';
import { DISPATCH, loadFixture, type OpCase } from '../../../../tests/atlas/autograd/support.ts';

const TOL = 1e-6;
const fixture = loadFixture<{ cases: OpCase[] }>('ops.json');

for (const c of fixture.cases) {
	test(`gradcheck + gradgradcheck: ${c.name}`, (t) => {
		const f = DISPATCH[c.op];
		const r = gradcheck((...xs) => f(xs, c.attrs), c.inputs.map((j) => input(j)), { second: true, tol: TOL });
		t.diagnostic(`max rel err ${r.maxRelErr.toExponential(2)} over ${r.checked} entries (worst: order ${r.worst?.order}, input ${r.worst?.input}[${r.worst?.index}])`);
		assert.ok(r.pass, `max rel err ${r.maxRelErr} (${JSON.stringify(r.worst)})`);
	});
}

test('gradcheck catches a wrong gradient', () => {
	const x0 = () => input({ shape: [3], data: [0.3, -0.7, 1.1] });
	const honest = gradcheck((x) => op.mul(x, x), [x0()], { second: true });
	assert.ok(honest.pass, `x·x passes (max rel err ${honest.maxRelErr})`);
	// x·detach(x): the value is x², but the recorded gradient is only x, so the check must fail
	const broken = gradcheck((x) => op.mul(x, x.detach()), [x0()]);
	assert.ok(!broken.pass, `a detached factor is caught (max rel err ${broken.maxRelErr})`);
});
