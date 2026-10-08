// View 'gan-2d/arena' (kind B): the GAN on the plane, from one frame of a train source (scene.ts). GAN Lab's picture
// (Apache-2.0, credited in the manifest; our own code). Props:
//   show     the layers, any of: 'map' (D's heat map: green where D says "real", pink where it says "fake", the line
//            where it is ½), 'real' (your points: filled), 'fake' (the generator's points: rings), 'sheet' (the noise
//            square's grid lines, bent by G), 'arrows' (on each generated point, the way that raises D's score fastest),
//            'move' (fine frames: each generated point before -> after the update), 'noise' (the noise square beside the
//            plane), 'blobs' (the eight blob centres; a solid circle when covered), 'zoom' (fine frames: a magnified
//            box around the generated points beside the plane; the arrows and moves are drawn there)
//   one      a noise point to ring, with the point G makes from it;  choices [{ source, label }]: buttons that switch to
//            another source (it runs; the view shows it at the cursor's step, else its last frame)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
import { sceneOf, type Scene } from '../../scene.ts';

export type Choice = { source: string; label: I18n };
export const shownSource = (p: Pick<StageProps, 'source' | 'props'>, picked: string | null) => picked ?? (typeof p.props?.source === 'string' ? (p.props.source as string) : p.source);
export const scene = (p: Pick<StageProps, 'rt' | 'source' | 'at' | 'props'>, picked: string | null = null) => sceneOf(p.rt, p.at, shownSource(p, picked), p.source);

export function describeScene(sc: Scene | null, show: string[]): I18n {
	if (!sc) return { en: 'The networks have no weights to show yet.', zh: '网络还没有可以显示的权重。' };
	const n = sc.fake.length / 2;
	const parts: I18n[] = [{ en: `After ${fmtInt(sc.step)} training steps: ${fmtInt(sc.realShown.length / 2)} real points and ${fmtInt(n)} generated points${sc.off ? ` (${fmtInt(sc.off)} of them off the picture)` : ''}.`, zh: `训练 ${fmtInt(sc.step)} 步后：${fmtInt(sc.realShown.length / 2)} 个真实点，${fmtInt(n)} 个生成的点${sc.off ? `（其中 ${fmtInt(sc.off)} 个在图外）` : ''}。` }];
	if (show.includes('map')) parts.push({ en: ` The discriminator gives the real points ${Number.isFinite(sc.dReal) ? fmt3(sc.dReal) : '—'} and the generated points ${fmt3(sc.dFake)} on average (1 = surely real, 0 = surely fake).`, zh: `判别器给真实点的平均分是 ${Number.isFinite(sc.dReal) ? fmt3(sc.dReal) : '—'}，给生成点的是 ${fmt3(sc.dFake)}（1 = 肯定是真的，0 = 肯定是假的）。` });
	if (Number.isFinite(sc.recall)) parts.push({ en: ` ${fmtPct(sc.recall)} of the real points have a generated point within 0.1.`, zh: `${fmtPct(sc.recall)} 的真实点在 0.1 以内有生成的点。` });
	if (sc.centres) parts.push({ en: ` The generated points cover ${fmtInt(sc.modes)} of the ${sc.centres.length} blobs.`, zh: `生成的点覆盖了 ${sc.centres.length} 团中的 ${fmtInt(sc.modes)} 团。` });
	if (sc.fakeBefore && show.includes('move')) parts.push({ en: ' Lines show how far each generated point moved in this step.', zh: '线段显示每个生成的点在这一步里移动了多远。' });
	return { en: parts.map((x) => x.en).join(''), zh: parts.map((x) => x.zh).join('') };
}

const meta = {
	key: 'gan-2d/arena',
	title: { en: 'The generator and the discriminator on the plane', zh: '平面上的生成器和判别器' },
	kinds: ['train'],
	describe: (p: StageProps): I18n => describeScene(scene(p), (p.props?.show as string[] | undefined) ?? []),
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { trainingProcess }, { gan }, { presetOf }] = await Promise.all([
			import('../../../../atlas/process/record.ts'),
			import('../../../../atlas/runtime/fake.ts'),
			import('../../../../atlas/runtime/sources.ts'),
			import('../../../../atlas/train/trainer.ts'),
			import('../../trainer.ts'),
			import('../../data.ts')
		]);
		const raw = presetOf('ring');
		const tape = record(trainingProcess(gan), raw, { steps: 400 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'gan-2d', input: raw, tapes: { train: tape } });
		return { rt, source: 'train', at: atFor(rt, 'train', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { show: ['map', 'real', 'fake', 'sheet'] } };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
