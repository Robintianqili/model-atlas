<!--
  Transfer learning article, English (complete). One block per `part`, in ConceptPageShell order.
  Every number here is from finetune.json (TinyCNN, precomputed), spec.json, research/bert_facts.json or
  research/bert_improve.json ("our BERT reproduction"). Numbers computed live are shown by the widgets, not in this text.
-->
<script lang="ts">
	import Term from '$lib/kit/Term.svelte';
	import { base } from '$app/paths';

	let { part }: { part: string } = $props();
</script>

{#if part === 'hook-intro'}
	<p>
		TinyCNN was trained on drawings of 10 things: apple, bicycle, car, clock, fish, house, star, sun, tree and umbrella.
		It has never seen a ladder or an envelope. Show it one anyway.
	</p>
{:else if part === 'hook-after'}
	<p>
		The model cannot say “I don't know”. Its last layer, the <Term id="head">output head</Term>, has exactly 10 outputs,
		and <Term id="softmax" /> shares 100% among them. So most ladders become houses (61% of 1,500 test ladders) and most
		envelopes become fish or houses.
	</p>
	<p>
		The rest of the network is not useless for ladders, though. Its convolution layers learned to detect strokes, corners,
		crossings and closed shapes, and a ladder is made of exactly those. <Term id="transfer-learning">Transfer learning</Term>
		reuses what a model learned on one task for a new one. The question for this page:
		<strong>can we teach TinyCNN two new classes with only a handful of drawings?</strong>
	</p>
{:else if part === 'toy-intro'}
	<p>
		The simplest way is <Term id="feature-extraction" />. Keep every pre-trained layer <Term id="frozen-layer">frozen</Term>
		(gray, with a lock), set the old 10-way head aside, and train a brand-new 2-way head on the 256 numbers the frozen body
		produces for each drawing. The new head is a single <Term id="dense-layer" />: 256 × 2 weights plus 2 biases = 514
		trainable <Term id="parameter">parameters</Term>. The other 4,152 parameters never change.
	</p>
	<p>
		Pick how many example drawings each class gets, or draw your own, then press <strong>Train</strong>. Training is plain
		full-batch <Term id="gradient-descent" /> on the <Term id="cross-entropy" /> loss: 200 steps, each one a pass over all
		your examples, run in your browser in a few milliseconds. The held-out test drawings are never used for training.
	</p>
{:else if part === 'toy-zoom'}
	<h3>Zoom in: what is frozen, what learns</h3>
	<p>
		Click any block of the network. A frozen block shows what it computes for the selected drawing; those numbers are the
		same before and after training. The new head shows its arithmetic: each of its two scores is a weighted sum of the 256
		features. Scrub the training replay above and watch only the head's weights change.
	</p>
{:else if part === 'toy-map'}
	<h3>Why a handful of examples is enough</h3>
	<p>
		Each drawing becomes a point in a 256-dimensional feature space. The map squeezes that space into 2 dimensions with PCA
		(principal component analysis, fit offline on test drawings). Gray points are drawings of the 10 known classes; orange
		and blue are the two new classes.
	</p>
	<p>
		The network never saw a ladder or an envelope, yet its frozen features already put them in different regions. A linear
		head can only draw a straight boundary (a flat cut through the 256-dimensional space), and here a straight boundary is
		all it takes. A few examples are enough to place it, which is why one drawing per class already does far better than
		chance.
	</p>
	<p>
		If two classes look alike to the frozen features (say, two kinds of circle), their points overlap, and no head on these
		features can separate them well. Then the features themselves have to change. That is <Term id="fine-tuning" />, Part 2.
	</p>
{:else if part === 'real-intro'}
	<p>
		Freezing is one option. You can also fine-tune every layer: start from the pre-trained weights, attach a new head, and
		train everything (4,666 parameters). Or you can train the same network from scratch, from random weights (also 4,666).
		Which is best depends mostly on how much labeled data you have.
	</p>
	<p>
		We ran all three offline on the same ladder-vs-envelope task, for 10 training-set sizes and 5 random seeds each. All
		three got the same budget: 300 full-batch steps of Adam, with each method's learning rate picked on a separate
		validation split. Every run is scored on 3,000 held-out test drawings.
	</p>
{:else if part === 'real-after'}
	<p>
		With 1 drawing per class, the frozen head reaches 93.1% on average. Full fine-tuning reaches 85.9% and training from
		scratch 78.7%, with a huge spread between seeds: one scratch run scored 44.0%, worse than a coin flip. More trainable
		parameters means more ways to fit two drawings, including many wrong ones. This is <Term id="overfitting" />. The frozen
		head has only 514 numbers to set, and it starts from features that already make sense.
	</p>
	<p>
		As data grows, the gap closes. From about 16 drawings per class, all three are within 3 points of each other, and full
		fine-tuning only passes the frozen head at 2,048 per class (98.59% vs 98.41%).
	</p>
	<p>
		Read this honestly: ladder vs envelope is an easy pair for these features, so the curves meet early. For a task further
		from what the network was trained on, frozen features help less and fine-tuning pays off sooner.
	</p>
{:else if part === 'compare-intro'}
	<p>
		TinyCNN has 6,722 parameters. Does the same trade-off hold for a language model with 110 million? In our BERT
		reproduction we fine-tuned Google's released BERT-base on 7 GLUE tasks with the paper's recipe (batch 32, 3 epochs)
		and tested the paper's claims about fine-tuning: 218 fine-tuning runs on one laptop GPU. Scores are on the GLUE dev
		sets, given as mean ± sample standard deviation over random seeds.
	</p>
{:else if part === 'compare-features'}
	<h3>Frozen features vs full fine-tuning</h3>
	<p>
		On SST-2 (is a movie review positive? 67,349 training sentences), full fine-tuning reaches 93.0% ± 0.2 over 3 seeds.
		The best frozen features, read by a <Term id="linear-probe" /> (a logistic regression), reach 86.8%: 6.2 points lower.
		Those features are the outputs of the last four layers, each averaged over the sentence's tokens, then concatenated. On
		MRPC (do two sentences mean the same?) the gap is 8.1 points: 84.1% vs 76.0%.
	</p>
	<p>
		The BERT paper found a gap of only 0.3 F1 on named-entity tagging (96.4 vs 96.1). So frozen BERT features are good,
		but with tens of thousands of labeled sentences, letting the whole network adapt wins. Two caveats keep this fair: our
		probe was trained on a 20,000-sentence sample of SST-2 (fine-tuning saw all 67,349), and 86.8% is the best of 27 layer
		and pooling choices, picked on the dev set itself.
	</p>
{:else if part === 'compare-layers'}
	<p>
		Which layer's features are best? Deeper is better. A probe on the token average climbs from 77.9% at layer 0 (the
		embedding output) to 85.4% at layers 9 and 10. A probe on the [CLS] vector starts at 50.9% at layer 0, exactly the
		majority-class baseline: before any attention layer, [CLS] is the same vector for every sentence, so the probe can only
		guess the more common label.
	</p>
{:else if part === 'compare-lr'}
	<h3>The learning-rate search</h3>
	<p>
		Fine-tuning a big network is sensitive to the <Term id="learning-rate" />. The paper tries four rates (2e-5 to 5e-5) for
		each task and keeps the best one on the dev set. In our runs the search is worth +2.0 points on MRPC and +1.8 on RTE over
		a fixed 2e-5. But the winner is chosen on the same dev set it is reported on, so the paper's numbers for small tasks are
		slightly optimistic.
	</p>
{:else if part === 'compare-small'}
	<h3>Small data makes full fine-tuning fragile</h3>
	<p>
		RTE (does sentence A entail sentence B?) has only 2,490 training pairs. When we fine-tuned BERT-Large on it with the
		optimizer from Google's original BERT code, 4 of 6 runs collapsed: they predicted the same class for every pair and
		scored 47.3% or 52.7%, just that class's share of the dev set. This optimizer skips Adam's bias correction, so the first
		updates are up to about 6 times too large. With bias correction (AdamW), 0 of 6 runs collapsed and the mean rose from
		54.3% to 70.2%. Many trainable parameters plus little data is exactly where training goes wrong.
	</p>
{:else if part === 'compare-mnli'}
	<h3>Intermediate-task training: MNLI first, then RTE</h3>
	<p>
		Another fix for a small task is to fine-tune first on a large task of the same kind, then on the small one. This is
		called intermediate-task training (Phang et al., 2018, named it STILTs). MNLI asks almost RTE's question (does A entail
		B?, with 3 answers instead of 2) and has 392,702 training pairs, 158 times RTE's 2,490. Our reproduction ran it with the
		same 5 seeds as the direct runs: fine-tune BERT-base on MNLI for 3 epochs, swap its 3-way head for a new 2-way head,
		then fine-tune on RTE exactly as before (learning rate 2e-5, 3 epochs, batch 32).
	</p>
{:else if part === 'compare-mnli-after'}
	<p>
		RTE dev accuracy rose from 63.0% ± 2.7 to <strong>77.3% ± 0.9: +14.3 points</strong>, higher on all 5 seeds, and the seeds
		agree more (the spread shrank from 2.7 to 0.9). It is the same move as on TinyCNN, one level up: the MNLI head is thrown
		away, and what carries over is a body already tuned to compare two sentences, so RTE's 2,490 pairs only have to adjust
		it a little.
	</p>
	<p>
		The BERT paper mentions this only in footnote 14: its authors "did observe substantial improvements on RTE from
		multitask training with MNLI", but give no numbers. Phang et al. report 68.6% → 79.1% for BERT-base; those are the
		best of 20 restarts, not means, so they sit above our 5-seed means. Combining tasks in other ways is planned for the
		<a href="{base}/train/multitask-intermediate/">multi-task page</a>.
	</p>
{:else if part === 'pitfalls'}
	<h3>Other pitfalls</h3>
	<ul>
		<li>
			<strong>Forgetting the old task.</strong> Full fine-tuning changes the shared layers. The old 10-class head was trained
			on the old features, so after fine-tuning it may stop working. This is called catastrophic forgetting. With a frozen
			body, the new head sits beside the old one and both keep working.
		</li>
		<li>
			<strong>Tuning on the test set.</strong> Picking the learning rate, the best seed or the best layer on the data you
			report makes a result look better than it is. Keep a separate validation split, as the TinyCNN runs in Part 2 do.
		</li>
		<li>
			<strong>Inputs prepared differently.</strong> Frozen features only mean something if new inputs look like the
			pre-training inputs. Here every drawing, yours included, is framed to 28 × 28 exactly like Quick, Draw!.
		</li>
		<li>
			<strong>Quoting one data size.</strong> “Fine-tuning beats freezing” and the opposite are both true, at different
			amounts of data. Always say how many examples.
		</li>
	</ul>
{:else if part === 'post'}
	<p>
		Transfer learning reuses a pre-trained network's features. With little labeled data, freeze the body and train a small
		head: few parameters, little to overfit, and it trains in milliseconds. With lots of data, or a task far from the
		pre-training data, unfreeze and fine-tune, with a carefully chosen learning rate. In between, compare both on a
		validation split. And when your task is small, look for a large task of the same kind to fine-tune on first: for
		BERT on RTE, MNLI first was worth 14.3 points.
	</p>
{/if}
