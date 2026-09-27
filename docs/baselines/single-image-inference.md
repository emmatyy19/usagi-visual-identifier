# Single Image Inference

This path turns the frozen retrieval experiment into something I can use with
one ordinary photograph. It loads DINOv2 once, embeds the query, searches only
the Session 1 training gallery, and returns the three strongest doll identities.

Session 2 and the sealed Session 3 test set never become gallery references.

## Run it in the debugger

Open Run and Debug in VS Code and choose `Predict One Image`. VS Code will ask
for the absolute path to a JPEG, PNG, HEIC, or HEIF photo. The result includes:

1. The selected device
2. Gallery size
3. Model startup time
4. First prediction time
5. The top three identity names and scores
6. The strongest Session 1 reference image for each identity

Choose `Benchmark One Image` to run three warmup predictions followed by 20
measured predictions. The benchmark saves a local JSON report beneath
`artifacts/benchmarks/`. Generated reports remain outside Git.

## What the latency numbers mean

Model startup measures loading the processor and model weights and moving the
model onto the selected device. It happens once when the application starts.

Warm prediction latency includes reading and orienting the image, DINOv2
preprocessing and encoding, normalization, cosine comparison, and identity
ranking. The report includes the mean, median, 95th percentile, minimum, and
maximum across repeated runs.

The median is the best description of an ordinary warm prediction. The 95th
percentile shows a slower but still common case. Startup and warm prediction
must remain separate because combining them would make every later prediction
look much slower than it is.

## Apple Metal baseline

The first local benchmark used one Session 2 validation photo and the 98 image
Session 1 gallery. It ran on Apple Metal with three warmup runs and ten measured
runs.

| Measurement | Time |
| --- | ---: |
| Model startup | 2.757 seconds |
| First prediction | 0.590 seconds |
| Warm median | 0.091 seconds |
| Warm mean | 0.092 seconds |
| Warm 95th percentile | 0.098 seconds |

The prediction correctly ranked Matcha ice cream first with a cosine score of
`0.8279`. Airport and Lobster claws were the next two identities. This is one
machine level latency measurement, not an accuracy experiment. CPU and RTX 3060
measurements still need to be collected separately.

## Current limitation

The returned score is a ranking score, not yet a calibrated confidence. The
predictor always returns known identities because unknown rejection still needs
validation examples and a frozen threshold. Until then, a high ranked result
can still be wrong for a doll outside the current collection.
