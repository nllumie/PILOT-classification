# Design: PILOT-inspired binary classification RaFFLE prototype

## Scope
Binary classification only. Numerical features only. Missing values are
handled outside the tree (median imputation inside each training pipeline).

## Prototype vs. original
- Preserved: linear-model trees whose nodes choose among a constant, a
  univariate linear fit, a piecewise-constant split and a piecewise-linear
  split; greedy top-down growth; a complexity penalty that grows with model
  degrees of freedom; bootstrap rows and node-level feature sampling in the
  forest; averaging of tree outputs.
- Changed: the loss is Bernoulli negative log-likelihood (logistic), not
  squared error. Node corrections are fitted by penalized logistic
  optimization (L-BFGS-B), not residual least squares. Predictions are
  logit-additive along the path to a leaf. The forest averages class
  probabilities, not regression predictions.
- Not implemented: PILOT's truncation procedures, the broken-line (blin)
  model, and the theoretical consistency results. Regression consistency
  claims do not transfer to this classifier.
- The complexity penalty is a heuristic, (1 + alpha*(v-1)) * log(n), with
  v = 1, 2, 5, 7 for con, lin, pcon, plogit. It is not the BIC.

## Probabilities and classes
- classes_ is the sorted set of the two global labels. Columns of
  predict_proba follow classes_.
- The positive class is always the second entry of classes_ (encoded as 1).
- Breast Cancer: malignant (sklearn label 0) is the positive class, relabelled to 1.
- Banknote: forged (label 1) is the positive class.
- Ionosphere: good ("g") is the positive class; bad ("b") is negative.
- Pima: tested_positive (diabetic) is the positive class.
- Spambase (supplementary): spam is the positive class.
- predict_proba averages per-tree probability matrices; predict takes the
  argmax of that average. Probability ties resolve to classes_[0].

## Node choice and objective
- Candidates: con, lin, pcon, plogit.
- Objective: twice the negative log-likelihood, plus a ridge penalty
  (lambda_0 on intercepts, lambda_1 on slopes), plus the complexity term above.
- A candidate is accepted only if its penalized objective change relative to
  the current node is negative.
- Features considered at a node are a random subset drawn without replacement
  from features that vary within the node.

## Convergence rule
- A candidate whose logistic fit fails to converge is discarded and counted
  (`_nonconverged_candidates`). It is accepted only if the optimizer reports
  failure but the maximum absolute gradient is below 1e-4 and all values are finite.
- In all reported runs the count was zero, so this rule did not change any result.

## Stopping rules
max_depth (split depth), max_model_depth (all corrections), min_samples_fit,
min_samples_leaf. A node containing one class receives only the regularized
constant correction.

## Randomness and reproducibility
- Forest: one seeded generator draws bootstrap rows, tree-level feature
  subsets and a seed for each tree.
- Synthetic datasets (make_classification, make_moons) are generated once at
  seed 42; the seeds 42, 52, 62 change only the train/test split.
- The fixed-seed results were reproduced in separate processes.

## Evaluation protocol (summary)
- Train-test: stratified 80/20, seeds 42, 52, 62.
- Cross-validation: stratified 5-fold, shuffle, seed 42.
- Fixed model settings for all datasets; no per-dataset tuning.
- Spambase is a supplementary single-configuration run with max_features_node=7
  and 30 RaFFLE trees; it is not part of the main comparison.

## Known limitations
- Penalty constants are heuristic, not derived from theory.
- Pure Python tree loops; RaFFLE is far slower than Random Forest.
  In the current main-comparison results, individual fits took approximately
  124.42 to 2,459.00 seconds, depending on dataset and evaluation split.
- Logistic-based node fits do not use max_features_node in the main
  comparison, so RaFFLE considers every feature at each node there.