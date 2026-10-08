// DevKit frozen types (research/devkit/DEVKIT_DESIGN.md §4.5): kind B trainer.
// Frozen during waves 1 and 2: changes go through the main session.
// Deviation from §4.5 (recorded in CONTRACTS.md "DevKit additions"): the design prints autograd.ts and train.ts as one
// block, so train.ts needs its own import of I18n, Rng and Tensor from core.ts to type-check.
import type { I18n, Rng, Tensor } from './core.ts';
import type { Module, Optimizer, Rec, Var } from './autograd.ts';
import type { Schema } from './process.ts';
export interface TrainOpt { seed: number; lr: number; batch: number; steps: number; snapshotEvery: number; optimizer: 'sgd' | 'momentum' | 'adam'; [k: string]: unknown }
export interface TrainState {
  step: number; loss: number; parts?: Record<string, number>;
  params: Record<string, Tensor>;                        // copied at snapshot frames
  grads?: Record<string, Tensor>; opt?: Record<string, { m?: Tensor; v?: Tensor }>;   // fine detail only
  batch?: Record<string, Tensor>; acts?: Record<string, Tensor>;                      // fine detail only
  [derived: string]: unknown;
}
export interface TrainerDef<D = unknown, M extends Module = Module, B = unknown, O extends TrainOpt = TrainOpt> {
  id: string; title: I18n; defaults: O;
  model(rng: Rng, opt: O): M;                            // use rng.fork('init') so compared runs share initial weights
  /** K2 (additive): model, the network as it is before this step (an RL round played with the current policy); the
   *  training loop always passes it, other callers (claim tests) may not */
  batch(data: D, step: number, rng: Rng, opt: O, model?: M): B;     // use rng.fork('batch')
  loss(m: M, b: B, rec: Rec, opt: O): { loss: Var; parts?: Record<string, Var> };
  optimizer?(m: M, opt: O): Optimizer;                   // default from opt.optimizer
  derived?: Schema;                                      // decision grid, vector field, u(t), …
  metrics?(m: M, data: D, opt?: O): Record<string, number>;   // opt: revision 1b (additive)
  /** K1 (additive): atlas parity's twin runs. `sgd`: options for the plain-SGD run (default: the page's options with
   *  optimizer 'sgd'; a stiff loss may need a smaller lr there, e.g. { lr: 1e-3 }). */
  twin?: { sgd?: Partial<O> & Record<string, unknown> };
}
// W1-B1: train/trainer.ts trainingProcess(def): ProcessDef<D, O, TrainState>
//   phases: 'init'; coarse 'snapshot' every snapshotEvery (metrics on every step); fine 'forward' → 'loss' → 'backward' → 'update'.
//   { type:'set', opt:{lr} } applies at the next step; { type:'input' } swaps the data (a new tape segment).
// train/helpers.ts updateOf(at, key, index) -> { old, g, lr, new } (SGD sources only; throws for Adam so narration cannot lie)
