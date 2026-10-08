// gnn-variants ops: the core ops plus the four local graph layers (ops/gnn.ts).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { gat_conv, gcn_conv, gin_agg, sage_conv } from './ops/gnn.ts';

export default [...CORE_OPS, gcn_conv, sage_conv, gin_agg, gat_conv];
