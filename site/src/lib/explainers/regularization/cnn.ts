// regularization: the recorded training of TinyCNN, the drawing classifier of /m/tinycnn/ (a precomputed record, not
// re-run here). Copied from pipeline/tinycnn/runs/c8e60/metrics.json (the run export.py ships) and the settings of
// pipeline/tinycnn/train.py and common.py augment(); claims.test.ts checks every number against those files.
// TinyCNN learned from Quick, Draw! drawings (CC BY 4.0, Google Creative Lab).
export const CNN = {
	params: 6722,
	epochs: 60,
	bestEpoch: 47,
	testAcc: 0.968133,
	/** AdamW weight decay on the weight tensors (biases: 0) */
	weightDecay: 0.0005,
	/** augment(): a random shift of up to `shift` in the image's own coordinates (they run from −1 to 1 across its 28
	 *  pixels), a scale from 0.8 to 1.1, a turn of up to ±rotDeg degrees, and thicker or thinner strokes */
	shift: 0.12,
	rotDeg: 12,
	/** per epoch (counting from 1): the mean training loss (on augmented drawings) and the validation accuracy */
	trainLoss: [1.7162, 0.86749, 0.64049, 0.46898, 0.38053, 0.32853, 0.29241, 0.2651, 0.24582, 0.23161, 0.21712, 0.20995, 0.20302, 0.1965, 0.19362, 0.18536, 0.1817, 0.1772, 0.1762, 0.17193, 0.1702, 0.16713, 0.16355, 0.16175, 0.16074, 0.15941, 0.15483, 0.15505, 0.15287, 0.1526, 0.1496, 0.14822, 0.14759, 0.14509, 0.14521, 0.14305, 0.14202, 0.14213, 0.14013, 0.13848, 0.13797, 0.13712, 0.13559, 0.13578, 0.13436, 0.13422, 0.13381, 0.13161, 0.13101, 0.13101, 0.13041, 0.13048, 0.12883, 0.13019, 0.12864, 0.1276, 0.12874, 0.12838, 0.12743, 0.12828],
	valAcc: [0.75207, 0.83053, 0.887, 0.9134, 0.92733, 0.93367, 0.9436, 0.94633, 0.95093, 0.95167, 0.95367, 0.9538, 0.95573, 0.9566, 0.9542, 0.95413, 0.9582, 0.95927, 0.95847, 0.9602, 0.96007, 0.95913, 0.9604, 0.96307, 0.96107, 0.96133, 0.9626, 0.96213, 0.96313, 0.96147, 0.96153, 0.96353, 0.964, 0.96493, 0.96487, 0.96347, 0.96387, 0.96467, 0.964, 0.9646, 0.96453, 0.96353, 0.96647, 0.96467, 0.96573, 0.96547, 0.96687, 0.96433, 0.9658, 0.96553, 0.96573, 0.9666, 0.96667, 0.96687, 0.96687, 0.96673, 0.9666, 0.9664, 0.96673, 0.9668]
};

/** the largest shift in pixels: `shift` times the half-width of 14 pixels */
export const shiftPixels = (CNN.shift * 28) / 2;
