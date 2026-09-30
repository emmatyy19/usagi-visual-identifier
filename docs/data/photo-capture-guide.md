# First Photo Capture Guide

This guide creates the first small but valid dataset for the frozen-embedding
baseline. Follow it literally for the pilot. Do not try to photograph the whole
collection yet.

## Target result

Photograph the current seven dolls in three genuinely different sessions:

- include dolls that look obviously different from one another;
- include dolls that the model might confuse; and
- collect about 14 or 15 useful photographs of each doll in each session.

That produces approximately 294 to 315 images:

```text
7 dolls x 3 sessions x about 14 or 15 photographs = 294 to 315 photographs
```

Session 1 becomes the initial reference gallery and training split. Session 2
is validation data. Session 3 remains untouched test data. This is a pilot, so
the goal is to validate the workflow and expose failure modes rather than claim
a final accuracy result.

## Step 1: Label the known dolls

Open the private inventory and assign stable IDs before taking photographs:

```text
usagi-0001  Matcha ice cream
usagi-0002  Airport
usagi-0003  Bakery
usagi-0004  Magical
usagi-0005  Pompompurin
usagi-0006  Chiikawa Park fluffy
usagi-0007  Lobster claws
```

Names can change after product research; IDs should not. For each doll, write
down two or three human-visible distinguishing cues, such as a hat, costume,
bow, prop, embroidery color, or body shape.

Do not use a doll as supervised ground truth if you are uncertain whether it is
actually a separate product variant. Mark it provisional until the identity is
verified.

## Step 2: Prepare the phone and scene

For every session:

1. Clean the phone camera lens.
2. Use the normal rear camera at `1x` zoom.
3. Use ordinary Photo mode, not Portrait mode.
4. Turn off beauty filters, color filters, and artificial background blur.
5. Keep full-resolution originals; do not send them through a messaging app.
6. Photograph one doll at a time.
7. Include a mix of naturally visible, partial, and hidden product tags. The
   model should not require printed text to recognize the doll.
8. Remove mail, faces, screens, mirrors, location clues, and identifying papers
   from the frame.
9. Use the same scene for every doll during that session.

Using the same scene for every identity is essential. If each doll gets a
different background, the model may learn the background instead of the doll.

## Step 3: Create the local folders

Create this private structure inside the repository on the computer that will
hold the images:

```text
data/private/images/
├── usagi-0001/
│   ├── session-001/
│   ├── session-002/
│   └── session-003/
├── usagi-0002/
│   ├── session-001/
│   ├── session-002/
│   └── session-003/
├── usagi-0003/
    ├── session-001/
    ├── session-002/
    └── session-003/
├── usagi-0004/
├── usagi-0005/
├── usagi-0006/
└── usagi-0007/
```

These folders are ignored by Git. Back up the originals separately; Git is not
the photo backup.

## Step 4: Capture session 001

Use a simple background and soft, even light. Place one doll in the scene and
use the following checklist to take about 14 or 15 photographs. Then repeat it
for the other dolls before changing the setup.

### Suggested 15 view checklist

1. Straight front, entire doll filling about 70% of the frame.
2. Straight front, entire doll filling about 45% of the frame.
3. Straight front, tighter crop while keeping every identifying accessory.
4. Turned approximately 30 degrees left.
5. Turned approximately 30 degrees right.
6. Left-side view.
7. Right-side view.
8. Camera slightly above the doll.
9. Camera at the doll's eye level.
10. Camera slightly below the doll.
11. Doll shifted to the left side of the frame.
12. Doll shifted to the right side of the frame.
13. Mild realistic occlusion covering a small non-identifying body area.
14. A natural handheld photograph with slightly imperfect framing.
15. One additional view that clearly shows this doll's most distinctive cue.

The order does not matter. File numbers identify photographs, not camera
angles. It is also fine to skip a view when it would be nearly identical to
another photograph. Useful variation matters more than making every identity
follow the same sequence.

Move deliberately between photographs. Do not hold the shutter for a burst of
nearly identical frames. Keep the doll's costume and removable accessories in
their normal configuration throughout the dataset.

## Step 5: Capture session 002 on another occasion

Wait until another time of day or another day. Change several nuisance factors:

- use a different background shared by every doll;
- change natural versus artificial lighting;
- change camera distance and framing; and
- pose the dolls again rather than reproducing session 001 exactly.

Repeat the same view checklist for every doll. Do not look at model results
before completing this session. This session is validation data used for design
and threshold choices.

## Step 6: Capture session 003 separately

On another occasion, use a third shared setup. A different room or surface is
ideal. Repeat the view checklist for every doll.

After import, treat session 003 as sealed test data:

- do not use it as the reference gallery;
- do not train on it;
- do not choose preprocessing or thresholds from it;
- do not move difficult examples into another split; and
- do not repeatedly check it while changing the model.

## Step 7: Import without destroying provenance

For each session:

1. Transfer original files directly from the phone.
2. Put each image into the correct doll and session folder.
3. Do not resize, crop, edit, or recompress the originals.
4. Run the matching data preparation launch after organizing the originals.
5. Delete obvious accidents only: unintended frames, severe blur, or corrupted
   files. Record the reason in private notes.
6. Never move one image to another session merely to balance counts.

The preparation pipeline creates model inputs from the originals without
modifying them.

## Step 8: Perform the manual audit

Before modeling, check every imported photograph:

- the folder contains the correct doll;
- the complete set is not just repeated burst frames;
- identity-defining features are visible in most images;
- no face, mail, screen, reflection, or identifying document is visible;
- images open correctly and retain their intended orientation;
- every session used the same background across all identities; and
- session 003 has not been copied into any other split.

Do not publish or commit these images. Even owner-captured examples stay private
until they are deliberately reviewed for public use and metadata removal.

## Step 9: Add unknown examples after the known pilot

Once the known identities are complete, photograph a small unknown set:

- for validation, use at least one unrelated plush and one visually similar
  character or excluded Usagi variant;
- for final testing, use different objects not present in validation; and
- capture 10-15 images per unknown object in the corresponding session setup.

Do not put an unknown object's photographs into both validation and test. The
model must reject products it did not encounter while selecting the threshold.

Put validation originals in this local structure:

```text
data/private/images/unknown/
├── unknown-0001/
│   └── session-004/
└── unknown-0002/
    └── session-004/
```

The stable unknown ID refers to one physical product. Keep a short private note
mapping each ID to the object. Do not add unknown products to `catalog.csv`.
Use the `Prepare Unknown Validation` debugger configuration after the files are
in place. It creates manifest rows, prepared JPEGs, and one contact sheet per
unknown identity.

Use normal, identifiable photographs first. Include the same practical changes
in angle, distance, and lighting used for the known dolls. A separate
robustness session can later test blur, severe crops, occlusion, and unusually
dark lighting without mixing those questions into threshold selection.

## Definition of done

The pilot is ready for the manifest and frozen-embedding baseline when:

- seven identities have stable IDs;
- each identity has about 14 or 15 useful images in each of three sessions;
- backgrounds are shared across identities within every session;
- the three sessions differ meaningfully;
- session 003 is still sealed;
- all images pass the manual privacy and label audit; and
- the originals are backed up outside the Git repository.
