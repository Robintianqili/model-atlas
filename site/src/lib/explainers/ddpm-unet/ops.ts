// ddpm-unet: the ops its spec uses: the core ops and the package's own (DEVKIT.md 5.6): q_sample (the forward
// process in one jump), time_embed (t as sin/cos waves) and upsample (nearest, 2x).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { qSample } from './ops/q_sample.ts';
import { timeEmbed } from './ops/time_embed.ts';
import { upsample } from './ops/upsample.ts';

export default [...CORE_OPS, qSample, timeEmbed, upsample];
