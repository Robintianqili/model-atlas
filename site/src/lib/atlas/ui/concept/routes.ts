// Concept routes with a hand-built static folder (src/routes/train/<slug>/), which takes precedence over /train/[slug].
// Its own module with no imports: /train/[slug]/+page.ts reads it, and every Planned concept page ships that route node.
export const STATIC_CONCEPT_ROUTES = ['tokenization', 'next-token-prediction', 'transfer-learning'];
