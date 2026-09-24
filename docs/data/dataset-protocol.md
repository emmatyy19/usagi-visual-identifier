# Dataset Protocol

## Purpose

This document defines how images are captured, labeled, split, and audited. Its
primary job is to prevent an impressive-looking result caused by duplicates,
background shortcuts, or leakage between capture sessions.

## Unit of identity

Each class represents one visually distinguishable merchandise variant and has
a stable, opaque ID such as `usagi-0001`. Names may improve as product research
continues; the stable ID should not change when a display name changes.

An identity record should include:

- stable catalog ID;
- canonical display name, if known;
- provisional or verified label status;
- evidence source for the product identity;
- high-level distinguishing attributes;
- inclusion status and reason; and
- private ownership fields stored separately.

Provisional identities may be used for unsupervised exploration but not as
ground truth for supervised evaluation.

## Capture-session definition

A capture session is a group of images sharing a setup or acquisition period.
A meaningful new session should change several nuisance factors, such as:

- day or time;
- room or background;
- natural versus artificial lighting;
- camera distance or device orientation;
- doll pose and facing direction; and
- nearby clutter.

Changing only the filename or taking another frame in the same burst does not
create an independent session.

## Initial collection target

For each of 5-10 known identities:

- collect at least three independent sessions;
- target roughly 15-30 useful images per session;
- include front, three-quarter, and side views when the doll permits it;
- vary distance and framing while keeping the doll identifiable;
- capture both easy views and realistic partial occlusions; and
- ensure shared backgrounds appear across multiple identities.

The numbers are targets, not a reason to keep poor or redundant images. Diverse
evidence is more valuable than repeated frames.

## Known split policy

Assign complete sessions, never individual images:

```text
train       one or more sessions per identity
validation  one independent session per identity
test        one untouched independent session per identity
```

If only three sessions exist, use one for each split. More training sessions can
be added later without changing the held-out validation and test sessions.

The gallery used for final known-query evaluation must contain training images
only unless an experiment explicitly evaluates a different, predeclared gallery
contract.

## Unknown split policy

Unknown evaluation should cover increasingly difficult cases:

1. unrelated objects;
2. other plush characters;
3. Usagi merchandise of a different product type; and
4. unseen Usagi doll variants that resemble known identities.

Unknown identities used to select thresholds belong to validation. Separate
unknown identities must be reserved for final testing. Images of the same
unknown product cannot cross those sets.

## Source families and derivatives

Every image carries a `source_image_id`. Crops, resized versions, corrected
rotations, and other derivatives inherit that source ID and split. Augmented
images are generated during training rather than materialized into multiple
dataset rows unless an experiment has a documented reason.

Perceptual-hash or embedding-based duplicate checks should flag suspiciously
similar images across splits for human review. They must not silently delete
data.

## Proposed local layout

```text
data/private/
├── catalog.csv
├── images.csv
└── images/
    ├── usagi-0001/
    │   ├── session-001/
    │   ├── session-002/
    │   └── session-003/
    └── unknown/
        ├── unknown-0001/
        └── unknown-0002/
```

The folders aid inspection, but the validated manifest—not directory inference—
is the source of truth.

## Proposed image manifest

```csv
image_id,relative_path,catalog_id,capture_session_id,source_image_id,split,source
img-000001,images/usagi-0001/session-001/001.jpg,usagi-0001,session-001,src-000001,train,owner_capture
```

Required semantic checks include:

- globally unique image IDs;
- existing decodable paths beneath an allowed root;
- catalog IDs that exist and are eligible for modeling;
- source families and sessions confined to one split;
- allowed split and source values;
- nonempty classes in every required known split; and
- no exact file-content duplicates across splits.

## Capture checklist

For the literal first-session workflow and shot sequence, follow the
[photo capture guide](photo-capture-guide.md).

Before a session:

- remove mail, screens, faces, reflections, and identifying paperwork;
- clean the camera lens;
- choose a background also usable for other dolls;
- decide which nuisance factors this session changes; and
- record the session ID before importing files.

During a session:

- capture the full doll before adding close or occluded views;
- keep identity-defining accessories visible in most images;
- avoid large bursts of effectively identical frames;
- do not use portrait-mode background effects; and
- do not apply filters that alter product colors.

After a session:

- remove accidental, blurred, or corrupted images with a recorded reason;
- assign manifest rows and provenance;
- inspect for personal information;
- run the dataset audit; and
- back up private originals outside the Git repository.

## Preprocessing contract

Evaluation preprocessing must be deterministic and versioned. The initial
pipeline should:

1. decode according to the image's orientation metadata;
2. convert to RGB;
3. resize and crop according to the encoder's documented input transform;
4. convert to tensor; and
5. apply the encoder's documented normalization.

Training augmentations form a separate versioned transform. Candidate
augmentations begin conservatively with resized crop, small rotation or
perspective changes, horizontal flip when identity is symmetric, and mild color
variation. Random erasing and aggressive crops are deferred because they can
remove tiny accessories.

## Rights and provenance

Owner-captured images remain private by default. Third-party product images may
be used only after reviewing their source terms and must carry source URL,
retrieval date, and intended-use notes. Presence on a public web page does not
imply permission to redistribute it in the repository.

Public tests should rely on generated fixtures, permissively licensed examples,
or images deliberately approved for publication.

## Dataset versioning

Each experiment records a deterministic fingerprint of the validated catalog
and image manifests plus preprocessing version. The fingerprint identifies the
inputs without publishing their contents.

Corrections create a new dataset version. Past experiment records retain the
old fingerprint; results are not silently rewritten after labels change.
