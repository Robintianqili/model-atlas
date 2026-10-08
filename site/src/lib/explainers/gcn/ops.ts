// gcn ops: the core ops plus the two local graph ops (ops/gcn.ts).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { gcn_conv, gcn_norm } from './ops/gcn.ts';

export default [...CORE_OPS, gcn_norm, gcn_conv];
