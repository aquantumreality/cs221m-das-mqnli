# Experiment Plan

## Main goal

Use MQNLI as a controlled NLI setting with a known symbolic causal model. Use DAS to test which high-level causal variables are aligned with low-dimensional neural subspaces.

## Immediate check-in experiments

1. Factual model training.
   - Verify that the model learns MQNLI factual labels.
   - Do not interpret DAS unless factual accuracy is clearly above chance.

2. DAS on QP_S.
   - QP_S is the final NLI label in the symbolic causal model.
   - This is the easy final-answer sanity check.

3. DAS on NegP.
   - NegP is an intermediate predicate-side relation after negation composition.
   - This is the first scientifically interesting target.

4. Optional identity vs relation comparison.
   - N_P_O: premise object noun identity.
   - N_H_O: hypothesis object noun identity.
   - N_O: object noun relation.
   - This asks whether DAS aligns surface lexical identity or abstract lexical relation.

## Metrics

- Factual accuracy.
- Interchange intervention accuracy (IIA).
- Optional logit-difference recovery.
- Majority-label and random-subspace baselines.
