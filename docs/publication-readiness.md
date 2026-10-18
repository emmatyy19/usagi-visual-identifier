# Publication preparation

Review date: October 17, 2026. Baseline: `7567548`, the merge of PR 17.
The repository remained private during this review. This document records
preparation, not permission to change visibility or a guarantee of no secrets.

## Checks performed

1. Inspected current tracked filenames and historical filenames reachable from
   all locally available Git refs after fetching origin.
2. Scanned 43 commits containing 265 unique file blobs for common GitHub,
   Hugging Face, OpenAI, AWS access key, and private key signatures.
   No matches were found for those patterns. This was a targeted pattern scan,
   not a comprehensive credential scanner or entropy audit.
3. Checked for committed raw photo paths, local database files, model weights,
   private directories, and transfer archives. None were found. `.env.example`
   contains only the relative catalog database setting.
4. Checked six committed notebooks. They contain 66 image outputs and no saved
   error outputs. Those rendered photos are intentionally published examples,
   unlike the excluded standalone dataset. The scan found no personal filesystem
   paths or email patterns inside these notebooks.
5. Captured the shelf, record editor, and identification results from the real
   local app. The record screenshot contains no personal notes. These reviewed
   screenshots are explicitly included as public documentation assets.
6. Updated setup limitations, implementation status, and the distinction between
   code licensing and photo or third party rights.
7. Checked the GitHub surfaces available during review: 18 issue or pull request
   bodies and all 44 readable Actions run logs. The same targeted credential
   patterns, nonrunner personal paths, and GitHub attachment URL patterns had no
   matches. There were no issue comments, inline review comments, or submitted
   reviews. There were no releases or Actions artifacts. Discussions and Wiki
   were disabled. PR 18 passed its quality and test checks.

## Finding that remains in history

The training guide contained the author's absolute Mac checkout path. The
current guide now uses a generic path. The original path remains in older Git
commits. It reveals a local username and folder structure, not a credential.
No history rewrite was performed. The owner should accept that disclosure or
request a separately planned history cleanup before changing visibility.

Commit author information and the intentional name in the MIT copyright notice
are also part of publication. Removing a current file does not erase history.

## Limits and final owner review

This check covers locally reachable repository content and the GitHub surfaces
listed above as they existed at review time. It does not certify unreachable
objects, remote pull request refs outside the fetched history, unavailable or
deleted logs, future changes, or every possible secret format. A pattern scan
cannot prove that arbitrary prose contains no sensitive details. The notebook
checks are structural and pattern based, not a fresh visual review of every
embedded image. Existing photo publication decisions remain the basis for them.

Before making the repository public:

1. Review the three screenshots and the existing notebook photo renderings.
2. Accept or resolve the historical local path disclosure.
3. Review the audit findings and any GitHub content added since this review.
4. Merge the preparation changes and verify CI.
5. Explicitly choose whether and when to change repository visibility.

## GitHub language statistics

`.gitattributes` sets `*.ipynb linguist-detectable=false`. This excludes notebook
bytes from the language breakdown without removing notebooks or treating them
as generated files. The updated chart is expected after the change reaches the
default branch and GitHub recalculates it. See the official
[Linguist overrides documentation](https://github.com/github-linguist/linguist/blob/main/docs/overrides.md).
