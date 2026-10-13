// egnn ops: the core ops plus the five local pair ops (ops/egnn.ts).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { edge_concat, nb_sum, pair_dist2, sum_atoms, vec_sum } from './ops/egnn.ts';

export default [...CORE_OPS, pair_dist2, edge_concat, vec_sum, nb_sum, sum_atoms];
