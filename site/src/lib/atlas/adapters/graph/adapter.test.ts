// graph adapter (K1, promoted from gcn and gnn-variants) and atlas/data/karate.ts: the club, A, X, Â, toggling, hops.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import adapter, { GRAPH_PRESETS } from './adapter.ts';
import { adjacency, FACTIONS, graphOf, hops, ISOLATED, KARATE, KARATE_EDGES, N, neighbours, normalizedAdjacency, toggleEdge } from '../../data/karate.ts';

test('the club: 34 members, 78 friendships (a < b, sorted), 17 + 17 by faction; A is symmetric with a zero diagonal', () => {
	assert.equal(N, 34);
	assert.equal(KARATE_EDGES.length, 78);
	assert.ok(KARATE_EDGES.every(([a, b], k) => a < b && (k === 0 || KARATE_EDGES[k - 1][0] < a || (KARATE_EDGES[k - 1][0] === a && KARATE_EDGES[k - 1][1] < b))));
	assert.deepEqual([FACTIONS.filter((f) => f === 0).length, FACTIONS.filter((f) => f === 1).length], [17, 17]);
	const A = adjacency(KARATE);
	for (let i = 0; i < N; i++) {
		assert.equal(A[i * N + i], 0);
		for (let j = 0; j < N; j++) assert.equal(A[i * N + j], A[j * N + i]);
	}
	assert.equal(neighbours(KARATE, 11).join(), '0');
});

test('preprocess: adj and one-hot x, never empty; graphOf cleans junk; toggling a friendship; hops', () => {
	const p = adapter.preprocess({ edges: [[1, 0], [0, 1], [2, 2], [40, 1]], focus: 99 } as never, null, {});
	assert.deepEqual(p.tensors.adj.shape, [N, N]);
	assert.equal(Array.from(p.tensors.adj.data).reduce((a, b) => a + b, 0), 2, 'one friendship, stored both ways');
	assert.equal(Array.from(p.tensors.x.data).reduce((a, b) => a + b, 0), N);
	assert.equal(p.empty, false);
	assert.equal(graphOf({ edges: [], focus: 99 }).focus, KARATE.focus);
	assert.equal(graphOf(null).edges.length, 78);
	const g = toggleEdge(KARATE, 0, 1);
	assert.equal(g.edges.length, 77);
	assert.equal(toggleEdge(g, 1, 0).edges.length, 78);
	assert.equal(hops(ISOLATED, 11).filter((d) => d < 0).length, N - 1, 'member 11 alone reaches nobody');
	const { deg, ahat } = normalizedAdjacency(adjacency(KARATE), N);
	assert.equal(deg[0], 17, 'member 0: 16 friends and the self-loop');
	assert.ok(Math.abs(ahat[0] - 1 / 17) < 1e-15);
	assert.deepEqual(GRAPH_PRESETS.map((x) => x.id), ['club', 'alone', 'none']);
	assert.equal(adapter.edgeFixtures()[0].raw, ISOLATED);
});
