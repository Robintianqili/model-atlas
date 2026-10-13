// fno: the package's worker entry (design §4.10). Browser only: never import it in Node.
// ui/packages.ts finds it with import.meta.glob('$lib/explainers/*/worker.ts', { query: '?worker' }).
import { serve } from '../../atlas/engine/host.ts';
import * as processes from './process.ts';

serve({ processes });
