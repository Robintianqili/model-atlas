// crnn-ocr: the ops its spec uses, the core ops and the package's own (DEVKIT.md 5.6): columns (feature maps -> a
// sequence of columns) and flip (reverse the rows, for the backward LSTM).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { columns } from './ops/columns.ts';
import { flip } from './ops/flip.ts';

export default [...CORE_OPS, columns, flip];
