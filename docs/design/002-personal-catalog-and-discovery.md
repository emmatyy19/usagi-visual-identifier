# Personal Catalog and Discovery

## Decision

I am building a personal collection tool rather than an authoritative Chiikawa
merchandise catalog. I do not have a complete official catalog or a practical
way to photograph dolls I do not own. The application should be honest about
that boundary instead of treating incomplete local data as complete product
knowledge.

## Personal catalog scope

A catalog record can represent a doll I own or a doll I only know exists. The
record should keep these decisions separate:

- whether I own it;
- whether I want it;
- whether I explicitly do not want it;
- whether its name and details have been manually verified;
- whether I have owner captured reference photographs;
- where I previously found useful information or a listing.

The preferred name is allowed to be informal. Source names and release details
can be added later when I find trustworthy evidence.

## Identification boundary

Local image retrieval answers a narrow question: which photographed doll in my
collection looks most similar to this upload? It does not prove that the doll is
one of the known identities.

The interface should use three result states:

1. `confident_local_match` for a strong catalog suggestion;
2. `possible_local_match` when the ranking is useful but confidence is below the
   selected operating point;
3. `no_confident_local_match` when local evidence is not strong enough.

The last state does not mean that the image is globally unknown. It only means
that the personal reference gallery cannot identify it confidently.

## Optional discovery boundary

When there is no confident local match, I can explicitly start a web assisted
lookup. The application may call an official image or web search API and may use
an LLM to summarize candidate evidence. This is a discovery workflow, not part
of model evaluation.

The lookup should return candidate names, explanations, and source page links.
It should not scrape search result pages, commit downloaded web images, or save
an unreviewed response as catalog truth. I remain responsible for confirming a
candidate before a catalog record is created or updated.

## Secrets and local data

Provider credentials belong in ignored environment configuration and must never
be committed. Query history, personal ownership decisions, purchase notes, and
downloaded images also remain local. The public repository can include provider
interfaces, configuration examples, and synthetic test responses.

## Application shape

The first application has one private Python service, one SQLite database, and
one browser client. The browser client will upload a photograph, show local
matches and uncertainty, and manage catalog records.

A later native iPhone companion will provide the wearable flow. It can use the
Meta Wearables Device Access Toolkit to receive frames from the glasses and
render concise results on the glasses display. It will call the same Python API
as the browser client. It will not own a second model, threshold, or catalog.

The shared service boundary is intentional. A browser upload and a glasses
capture differ only in how the image reaches the identifier and how the result
is presented. The model response, catalog record, source links, and manual
confirmation rules remain the same.

The initial catalog schema gives each personal record its own random stable ID.
An optional `usagi-####` link connects it to a photographed model identity.
This keeps an unowned or unphotographed doll in the catalog without pretending
that the model can recognize it.

## Why metric learning is optional

Frozen and tuned nearest image retrieval each identified 41 of 42 known final
test photographs. Partial fine tuning did not improve final Top 1 accuracy. The
project currently has seven identities and no independent way to test new
identity generalization. Metric learning would be useful practice, but it is not
required to make the personal catalog useful and would be difficult to evaluate
honestly with the current data.
