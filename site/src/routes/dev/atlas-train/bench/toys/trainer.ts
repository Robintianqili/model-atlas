// /dev/atlas-train/ bench package "toys": a 2-D classifier (the MLP-playground core) and a 1-D curve fit.
import { classifierTrainer, fitTrainer } from '../../../../../lib/atlas/ui/train/toys.ts';

export const cls = classifierTrainer({ id: 'cls', hidden: [8, 8], act: 'tanh', res: 40 });
export const fit = fitTrainer({ id: 'fit', hidden: [16, 16], act: 'tanh' });
