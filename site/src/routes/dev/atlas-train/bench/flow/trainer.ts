// /dev/atlas-train/ bench package "flow": the flow-2d velocity MLP v(x, y, t) 3 → 64 → 64 → 64 → 2 (SiLU, Adam 1e-3,
// batch 256) on painted or preset 2-D points (W1-B2 acceptance 1: no main-thread long task from the engine).
import { flowTrainer } from '../../../../../lib/atlas/ui/train/toys.ts';

export const flow = flowTrainer({ id: 'flow' });
