// tinyforecaster ops: the core ops plus the two local ones of the live-model graph (ops/steps.ts).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { last_step, relative } from './ops/steps.ts';

export default [...CORE_OPS, relative, last_step];
