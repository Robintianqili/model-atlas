// tiny-unet-seg: the ops its specs use: the core ops (conv2d, relu, maxpool2d, concat, softmax) and the package's own
// 'upsample' (bilinear, as F.interpolate; DEVKIT.md 5.6).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { upsample } from './ops/upsample.ts';

export default [...CORE_OPS, upsample];
