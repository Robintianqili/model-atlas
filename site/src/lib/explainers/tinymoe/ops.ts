// tinymoe: the ops its spec uses, the core ops and the package's own (DEVKIT.md 5.6). Written by `atlas new op moe --local tinymoe`.
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { moe } from './ops/moe.ts';

export default [...CORE_OPS, moe];
