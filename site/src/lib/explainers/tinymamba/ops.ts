// tinymamba ops: the core ops plus the selective scan (ops/scan.ts).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { ssm_scan } from './ops/scan.ts';

export default [...CORE_OPS, ssm_scan];
