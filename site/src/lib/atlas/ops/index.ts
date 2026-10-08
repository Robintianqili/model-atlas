// CORE_OPS (design §4.3): 20 ops + 3 aliases (dense, linear_tied -> linear; gelu_tanh -> gelu {approx: 'tanh'}).
// A package's ops.ts default-exports [...CORE_OPS, ...local]; an op moves here when a second model needs it (rule of two).
// schema/ops.json is generated from this list: node src/lib/atlas/ops/manifest.ts --write
import type { OpDef } from '../types/index.ts';
import { input } from './input.ts';
import { conv2d } from './conv2d.ts';
import { linear } from './linear.ts';
import { relu, tanh, sigmoid, silu, gelu } from './activations.ts';
import { maxpool2d } from './maxpool2d.ts';
import { flatten } from './flatten.ts';
import { concat } from './concat.ts';
import { add } from './add.ts';
import { embedding, position_embedding } from './embedding.ts';
import { layernorm } from './layernorm.ts';
import { attention_scores, causal_mask, attention_mix } from './attention.ts';
import { softmax } from './softmax.ts';
import { lstm } from './lstm.ts';
import { createRegistry } from './registry.ts';

export const CORE_OPS: OpDef[] = [
	input,
	conv2d,
	linear,
	relu,
	tanh,
	sigmoid,
	silu,
	gelu,
	maxpool2d,
	flatten,
	concat,
	add,
	embedding,
	position_embedding,
	layernorm,
	attention_scores,
	causal_mask,
	softmax,
	attention_mix,
	lstm
];

/** A registry of CORE_OPS (shared; registries are read-only). */
export const coreRegistry = createRegistry(...CORE_OPS);

export {
	input,
	conv2d,
	linear,
	relu,
	tanh,
	sigmoid,
	silu,
	gelu,
	maxpool2d,
	flatten,
	concat,
	add,
	embedding,
	position_embedding,
	layernorm,
	attention_scores,
	causal_mask,
	softmax,
	attention_mix,
	lstm
};
export { createRegistry, toRegistry, mergeParams } from './registry.ts';
