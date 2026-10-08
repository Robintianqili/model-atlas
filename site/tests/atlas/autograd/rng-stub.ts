// Integration 1a: the autograd tests used a local stand-in for atlas/core/rng.ts while W1-A was in another lane.
// They now use the real kit Rng. The names makeRng and fnv1a32 are kept so the test files need no edits.
export { createRng as makeRng, fnv1a32 } from '../../../src/lib/atlas/core/rng.ts';
