// pointnet ops: the core ops plus the local max over the points (ops/max_points.ts).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { maxPoints } from './ops/max_points.ts';

export default [...CORE_OPS, maxPoints];
