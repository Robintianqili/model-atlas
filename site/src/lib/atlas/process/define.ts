// Authoring helper for process.ts (types revision 1b, CONTRACTS.md "DevKit additions"): an identity function that
// infers a ProcessDef's input (I, from run's first parameter), options (O, from defaults) and state (S, from run's
// return type). When the frames differ in shape, name the state and options once with ProcessRun<S, O> as run's return
// type; every yield is then checked against Frame<S> (state: Partial<S>):
//
//   type WalkState = { x: number; path: number[] };
//   type WalkOpt = { steps: number };
//   export const walk = defineProcess({
//     id: 'walk', kind: 'algorithm', title: { en: 'A random walk', zh: '随机游走' },
//     phases: { move: { label: { en: 'move', zh: '移动' }, kind: 'act' }, end: { label: { en: 'end', zh: '结束' }, kind: 'output' } },
//     schema: { x: { kind: 'scalar', label: { en: 'position', zh: '位置' } }, path: { kind: 'json', label: { en: 'path', zh: '路径' } } },
//     defaults: { steps: 20 },
//     *run(_input: null, ctx): ProcessRun<WalkState, WalkOpt> {
//       let x = 0;
//       const path: number[] = [];
//       for (let k = 0; k < ctx.opt.steps; k++) {
//         path.push((x += ctx.rng.next() < 0.5 ? -1 : 1));
//         yield { phase: 'move', step: k, state: { x } };
//       }
//       yield { phase: 'end', state: { path } };
//     }
//   });
//
// Pure TS (Node and workers): imports only the frozen types.
import type { ProcessDef, ProcessRun } from '../types/process.ts';

export type { ProcessRun };

export function defineProcess<I = unknown, O extends Record<string, unknown> = Record<string, unknown>, S = Record<string, unknown>>(
	p: ProcessDef<I, O, S>
): ProcessDef<I, O, S> {
	return p;
}
