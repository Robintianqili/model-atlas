// vq-vae: the ops its spec uses, the core ops and the package's own (DEVKIT.md 5.6): vq (each patch's numbers -> the
// nearest codebook row) and pixel_shuffle (each cell's 49 scores -> its 7 × 7 block of the picture).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { vq } from './ops/vq.ts';
import { pixelShuffle } from './ops/pixel_shuffle.ts';

export default [...CORE_OPS, vq, pixelShuffle];
