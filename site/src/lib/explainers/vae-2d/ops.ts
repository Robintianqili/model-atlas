// vae-2d: the ops its spec uses, the core ops and the package's own (DEVKIT.md 5.6): reparam (z = mu + sigma · eps) and
// to_image (784 numbers -> a 28 × 28 picture).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { reparam } from './ops/reparam.ts';
import { toImage } from './ops/to_image.ts';

export default [...CORE_OPS, reparam, toImage];
