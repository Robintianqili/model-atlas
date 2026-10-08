// The "toys" bench package's worker entry (a second package: its own bundle, acceptance 5).
import { serve } from '../../../../../lib/atlas/engine/host.ts';
import * as trainers from './trainer.ts';

serve({ trainers });
