// DevKit frozen types (research/devkit/DEVKIT_DESIGN.md §4.5): kind B autograd.
// Frozen during waves 1 and 2: changes go through the main session.
// autograd.ts: tensor-level reverse mode, f64 by default. Every VJP is written with the same ops,
// so grad(..., { createGraph: true }) is itself differentiable (PINN u'' and its parameter gradients).
import type { I18n, Rng, Tensor } from './core.ts';
import type { AtlasSpec } from './spec.ts';
export interface Var { readonly value: Tensor; grad: Var | null; readonly requiresGrad: boolean; readonly name?: string }
/** named() uses PyTorch state_dict names ('net.0.weight') and [out, in] layouts, so twins and Focus keys match. */
export interface Module { params(): Var[]; named(): Record<string, Var>; toSpec?(o: { slug: string; inputAxes?: string[] }): { spec: AtlasSpec; weights: Float32Array } }
export interface Optimizer { lr: number; step(): void; set(o: Partial<{ lr: number; momentum: number; betas: [number, number]; weightDecay: number }>): void; state(): Record<string, { m?: Tensor; v?: Tensor }> }
export type Rec = (id: string, v: Var) => void;          // named intermediates for views (tape field `acts`)
// W1-B1, autograd/*.ts:
//   var.ts:   param(name, t), input(t, requiresGrad?), grad(ys, xs, { createGraph?, seeds?, allowUnused? }), backward(loss, { createGraph? }), noGrad(f), zeroGrad(ps)
//   ops.ts:   op.{ add sub mul div neg scale pow sqrt exp log tanh sigmoid relu silu gelu sin cos abs matmul linear sum mean reshape
//                  transpose broadcastTo concat slice gather logsumexp softmax logSoftmax conv1d layerNorm }
//   loss.ts:  mse, crossEntropy, bceWithLogits, infoNCE
//   nn.ts:    Linear, MLP(name, sizes, act, rng), Embedding, LSTMCell (both biases), Conv1d({dilation, causal}), LayerNorm
//   optim.ts: SGD, Momentum, Adam/AdamW (m and v exposed for the optimizer glyphs)
//   check.ts: gradcheck(f, xs, { eps, tol, second? }) -> { maxRelErr, worst }
//   tospec.ts: Linear/MLP stacks -> AtlasSpec + f32 weights (the kind-A explainer on a live model)
