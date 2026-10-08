// The "flow" bench package's worker entry: the four lines `atlas new` generates (design §4.10), plus one request/response
// call, `euler`, that samples with given weights off the main thread (a { type: 'call' } example).
import { serve } from '../../../../../lib/atlas/engine/host.ts';
import { eulerPaths } from '../../../../../lib/atlas/ui/train/toys.ts';
import type { Tensor } from '../../../../../lib/atlas/types/core.ts';
import * as trainers from './trainer.ts';

serve({
	trainers,
	calls: {
		euler: (a) => {
			const { params, start, K } = a as { params: Record<string, Tensor>; start: Float64Array; K: number };
			return eulerPaths(params, start, K);
		}
	}
});
