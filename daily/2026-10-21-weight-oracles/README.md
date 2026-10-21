# Weight Oracles: auditing neural networks by reading their raw weights

A from-scratch, numpy-only miniature of a trending idea: **diagnose hidden
model properties from the weight tensor alone, without running the model on
test inputs.**

## The trending hook

"Weight Oracles: Reading Neural Network Weights with Language Models"
(University of Oxford, published October 2026, reported by QUE.com on 20 Oct
2026) proposes treating a network's static weights as a structured data
modality: a fine-tuned language model inspects the raw numerical weights of
another network and diagnoses hidden properties, such as backdoors or dormant
capabilities, without ever running the target model on test inputs. The
paper's thesis is that classic interpretability is *reactive*: you only find
behaviours you think to test for, so a backdoor with a rare trigger sails
through behavioural audits. (Paper details as reported by the coverage; the
paper's own method, scale and results are the authors', not mine.)

## What this repo does

I built the smallest honest version of that auditing loop:

1. **Victim population.** 200 tiny MLPs (2-16-16-1, tanh, 337 params) trained
   on a 2-D blob classification task. Half are clean; half are *trojaned* with
   a counterfactual trigger: inputs with x0 in [-5, -3.5] are labelled 1,
   deep inside what a clean model calls class-0 territory.
2. **Behavioural audit (the reactive baseline).** Every model is scored on
   clean holdout data only, the way a standard audit works. Clean and trojan
   models are indistinguishable: 0.9955 vs 0.9950 accuracy.
3. **The weight oracle.** A logistic probe trained on *flattened, L2-
   normalized weight vectors* classifies clean vs trojaned models. On 60
   held-out models it scores **1.000**.
4. **Validation of the flags.** Every model the oracle flags as trojan is run
   through the trigger region: mean flip rate **0.999** (min 0.980). Models
   flagged clean: mean **0.009** (max 0.068). The oracle's flags correspond to
   real backdoor behaviour.

## An honest control I had to add

Trojan victims carry ~12% larger weight norms (12.1 vs 10.8), and a probe
trained on the norm *alone* already reaches 0.98 train accuracy. To make sure
the oracle reads the trojan's directional signature rather than leaning on
"bigger weights", every weight vector is L2-normalized before probing. The
1.000 held-out accuracy above is on normalized weights.

## Limitations, stated plainly

- All victims share one initialization; per-model randomness comes only from
  minibatch order. This keeps the trojan signature aligned in weight space so
  a linear probe can read it. With fully independent initializations the
  signature scatters across permuted hidden units and the reader would need
  permutation-invariant features.
- Toy setting throughout: 337-parameter MLPs, synthetic blobs, one fixed
  trigger location. The reader here is logistic regression, not a language
  model.
- This demonstrates the *auditing idea*, not the paper's method or results.

## Files

- `weight_oracle.py`: data, MLP (exact hand-written backprop, Adam), victim
  training, behavioural audit, weight-oracle probe, validation. Run with
  `python3 weight_oracle.py` (numpy only, ~10 s).
- `test_weight_oracle.py`: 8 deterministic tests, including a
  finite-difference gradient check of the backprop. Run with
  `python3 test_weight_oracle.py`.

## Results (seed 7, 200 models, 30 epochs)

| check | result |
|---|---|
| behavioural audit, clean holdout acc (clean vs trojan) | 0.9955 vs 0.9950, inseparable |
| weight-oracle probe, held-out models | 1.000 |
| flagged-as-trojan trigger flip rate | mean 0.999, min 0.980 |
| flagged-as-clean trigger flip rate | mean 0.009, max 0.068 |
| unit tests | 8/8 passing |
