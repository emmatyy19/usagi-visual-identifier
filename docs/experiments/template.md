# Experiment NNN: Short Descriptive Name

## Status

Planned

## Question

What single question does this experiment answer?

## Hypothesis

State the expected outcome before seeing the final results.

## Baseline

Name the existing system and exact artifact or configuration used for
comparison.

## Dataset

- Manifest fingerprint:
- Split version:
- Known identities:
- Train/validation/test query counts:
- Gallery definition:
- Unknown validation/test identities:

## Configuration

- Encoder and checkpoint:
- Preprocessing version:
- Augmentation version:
- Trainable parameters:
- Loss and optimizer:
- Learning-rate schedule:
- Batch size and accumulation:
- Epochs and early-stopping rule:
- Random seeds:
- Hardware:

## Protocol

Describe the steps in enough detail to repeat the experiment. Identify every
choice made using validation data and confirm that the test set remains sealed.

## Success criteria

Define the primary metric, acceptable unknown behavior, and resource limits
before recording results.

## Results

Complete after execution.

| Metric | Baseline | Candidate | Difference |
| --- | ---: | ---: | ---: |
| Top-1 accuracy | TBD | TBD | TBD |
| Top-3 accuracy | TBD | TBD | TBD |
| Macro recall | TBD | TBD | TBD |
| Unknown false-acceptance rate | TBD | TBD | TBD |
| Peak GPU memory | TBD | TBD | TBD |
| GPU inference latency | TBD | TBD | TBD |

## Slice and failure analysis

Summarize per-identity behavior, recurring confusion pairs, robustness slices,
and representative qualitative failure modes. Only use photographs that I
explicitly reviewed and approved for publication. Do not expose original files,
photo metadata, or private paths.

## Conclusion

State whether the evidence supports adopting, rejecting, or revising the
candidate. Distinguish observations from speculation.

## Follow-up

- [ ] Record any justified next experiment.
- [ ] Update design documentation if a decision changed.
