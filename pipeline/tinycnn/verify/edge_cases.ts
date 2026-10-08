// Edge cases for preprocessDrawing / preprocessStrokes (verification stream).
// Run from the repo root: node pipeline/tinycnn/verify/edge_cases.ts
import { preprocessDrawing, preprocessStrokes, preprocessPhoto } from '../../../site/src/lib/models/tinycnn.ts';

const W = 200, H = 200;
const canvas = (bg: [number, number, number, number]) => {
	const d = new Uint8ClampedArray(W * H * 4);
	for (let i = 0; i < W * H; i++) { d[4 * i] = bg[0]; d[4 * i + 1] = bg[1]; d[4 * i + 2] = bg[2]; d[4 * i + 3] = bg[3]; }
	return d;
};
const sum = (a: Float32Array) => a.reduce((s, v) => s + v, 0);
for (const [name, bg] of [['transparent', [0, 0, 0, 0]], ['white', [255, 255, 255, 255]], ['cream', [250, 245, 225, 255]], ['dark', [30, 30, 30, 255]]] as const) {
	const r = preprocessDrawing({ data: canvas(bg as unknown as [number, number, number, number]), width: W, height: H });
	console.log(`blank ${name} canvas: empty=${r.empty} sum=${sum(r.input).toFixed(3)}`);
}
// a single dot and a horizontal line on a white canvas
{
	const d = canvas([255, 255, 255, 255]);
	for (let y = 98; y < 102; y++) for (let x = 98; x < 102; x++) { const i = y * W + x; d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = 0; }
	const r = preprocessDrawing({ data: d, width: W, height: H });
	console.log(`dot: empty=${r.empty} sum=${sum(r.input).toFixed(1)} bbox=${JSON.stringify(r.bbox)} finite=${r.input.every(Number.isFinite)}`);
	const d2 = canvas([255, 255, 255, 255]);
	for (let y = 99; y < 102; y++) for (let x = 20; x < 180; x++) { const i = y * W + x; d2[4 * i] = d2[4 * i + 1] = d2[4 * i + 2] = 0; }
	const r2 = preprocessDrawing({ data: d2, width: W, height: H });
	let rows = 0;
	for (let y = 0; y < 28; y++) { let s = 0; for (let x = 0; x < 28; x++) s += r2.input[y * 28 + x]; if (s > 0) rows++; }
	console.log(`line: empty=${r2.empty} rows with ink=${rows} sum=${sum(r2.input).toFixed(1)}`);
}
// strokes: empty, one point, degenerate
console.log(`strokes []: empty=${preprocessStrokes([]).empty}`);
const one = preprocessStrokes([[[5, 5]]]);
console.log(`strokes one point: empty=${one.empty} sum=${sum(one.input).toFixed(2)} finite=${one.input.every(Number.isFinite)}`);
const objPts = preprocessStrokes([[{ x: 0, y: 0 }, { x: 100, y: 100 }]]);
console.log(`strokes {x,y} diagonal: empty=${objPts.empty} max=${Math.max(...objPts.input).toFixed(2)}`);
// photo of a blank sheet
const blank = preprocessPhoto({ data: canvas([240, 236, 225, 255]), width: W, height: H });
console.log(`blank photo: empty=${blank.empty} sum=${sum(blank.input).toFixed(2)}`);
