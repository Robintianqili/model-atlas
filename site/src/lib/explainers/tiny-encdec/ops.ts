// tiny-encdec: the ops its spec uses, the core ops and the package's own (DEVKIT.md 5.6): greedy decoding.
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { greedy } from './ops/greedy.ts';

export default [...CORE_OPS, greedy];
