// tinyseq2seq: the ops its spec uses, the core ops and the package's own (DEVKIT.md 5.6): the GRU encoder and the
// attention decoder.
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { gru } from './ops/gru.ts';
import { attnDecoder } from './ops/attn_decoder.ts';

export default [...CORE_OPS, gru, attnDecoder];
