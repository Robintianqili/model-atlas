// llava-toy ops: the core ops plus four local ones (ops/tokens.ts): TinyViT's patch rows and [CLS] row, drop_cls and merge_image.
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { cls_token, drop_cls, merge_image, to_tokens } from './ops/tokens.ts';

export default [...CORE_OPS, to_tokens, cls_token, drop_cls, merge_image];
