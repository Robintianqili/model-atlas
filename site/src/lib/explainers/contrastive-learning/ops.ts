// contrastive-learning ops: the core ops plus the three local ones (ops/clip.ts).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { l2norm, mean_pool, similarity } from './ops/clip.ts';

export default [...CORE_OPS, mean_pool, l2norm, similarity];
