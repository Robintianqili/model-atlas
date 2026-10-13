// fno ops: the core ops plus the four local ones of the Fourier layer's spectral path (ops/fourier.ts).
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { irfft, keep_modes, rfft, spectral_mix } from './ops/fourier.ts';

export default [...CORE_OPS, rfft, keep_modes, spectral_mix, irfft];
