// The live demos of /devkit/reference/ (lazy, browser only). Everything comes from the /dev/atlas/ gallery's own module
// (atlas/ui/views/kit.parts/gallery.ts: every view, ViewStage, the demo note) plus opdemos.ts, whose imports are all
// modules the gallery already uses. So this entry reaches nothing the gallery does not, and adding it does not re-split
// the chunks other pages share (see gallery.ts on chunk splits).
export { ViewStage, allViews, viewFor, preparePackageViews, DEMO_NOTE } from '../atlas/ui/views/kit.parts/gallery.ts';
export { opStage } from './opdemos.ts';
