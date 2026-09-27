# Experiment 001: Frozen DINOv2 Session 1 Sanity Check

## Status

Complete exploratory check

## Question

Can the frozen embedding and exact retrieval pipeline distinguish the seven
known dolls when the query and gallery come from the same capture session?

## Scope

This run was an engineering integration check rather than a predeclared model
selection experiment. Its results cannot support a generalization claim. No
validation or test images existed when it ran.

## Dataset

1. Manifest fingerprint:
   `7cc699a1034dca6962262616bae80c256affc5ac0c48bbf6be4a0dec10c6178b`
2. Known identities: 7
3. Session 1 images: 98
4. Images per identity: 14
5. Gallery: all other Session 1 images after removing the current query
6. Unknown identities: not measured

## Configuration

1. Encoder: `facebook/dinov2-small`
2. Revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`
3. Embedding: 384 dimensional CLS token, normalized to unit length
4. Identity aggregation: maximum cosine similarity from any reference image
5. Trainable parameters: 0
6. Batch size: 8
7. Hardware: Apple Silicon with PyTorch Metal

## Protocol

Every Session 1 image became a query once. Its own embedding was removed from
the gallery. The remaining 97 images were ranked by cosine similarity, then
collapsed to identity rankings using the nearest reference image for each
identity.

Session 2 and Session 3 remained absent and therefore untouched.

## Results

| Metric | Result |
| --- | ---: |
| Top 1 accuracy | 0.918 |
| Top 3 accuracy | 0.980 |
| Macro recall | 0.918 |
| Mean reciprocal rank | 0.953 |
| Incorrect top 1 queries | 8 of 98 |
| Embedding time | 4.78 seconds |
| Throughput | 20.50 images per second |

### Exploratory aggregation check

The completed comparison implementation also ran four aggregation rules on the
same Session 1 leave one out protocol.

| Method | Top 1 | Top 3 | Macro recall | MRR |
| --- | ---: | ---: | ---: | ---: |
| Nearest image | 0.918 | 0.980 | 0.918 | 0.953 |
| Identity prototype | 0.980 | 1.000 | 0.980 | 0.990 |
| Top k mean | 0.969 | 1.000 | 0.969 | 0.983 |
| Weighted neighbors | 0.929 | 0.980 | 0.929 | 0.958 |

These values are implementation evidence, not model selection evidence. The
prototype result cannot become the preferred rule until Session 2 confirms its
behavior on independent photographs.

## Failure analysis

Six of the eight incorrect top 1 queries used either the shared blanket
occlusion or a handheld view. The two remaining errors used unusual views that
showed fewer identity details. The two queries whose true identity fell outside
the top three also came from the shared occlusion or handheld group.

This pattern suggests that the frozen representation sometimes gives too much
weight to the repeated photographic condition and too little weight to a small
costume or accessory. This is an observation from a very small same session
check, not a confirmed causal explanation.

## Conclusion

The pipeline is working and the frozen representation separates most of the
current images without training. The result is strong enough to continue with
the baseline, but it must not be presented as model accuracy. Session 2 is
required for the first meaningful measurement.

## Follow up

1. Capture Session 2 with different lighting, background, placement, and tag
   visibility.
2. Evaluate Session 2 queries only against the Session 1 gallery.
3. Compare nearest image, identity prototype, top k mean, and weighted neighbor
   aggregation using validation results.
4. Train a linear probe only after the frozen aggregation comparison is
   recorded.
5. Inspect whether varied occluders reduce the shortcut pattern.
