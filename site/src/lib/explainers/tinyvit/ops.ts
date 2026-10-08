// tinyvit ops: the core ops plus the three local ones (ops/tokens.ts). The tinycnn compare source uses this list too.
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { cls_token, select_token, to_tokens } from './ops/tokens.ts';

export default [...CORE_OPS, to_tokens, cls_token, select_token];
