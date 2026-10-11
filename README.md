# AI-assisted Literature Review Platform

A Streamlit platform for AI-assisted systematic literature review. Each project
follows one of two workflow modes, chosen at creation:

- **PRISMA** — upload a CSV of papers → abstract include/exclude screening →
  attach PDFs for advancing papers → full-text include/exclude screening → data extraction.
- **Full-text direct review** — upload full-text PDFs → full-text
  include/exclude screening → data extraction.

Every stage follows the same pattern: the reviewer writes their own screening
prompt, one to three selected models produce independent structured answers, and the
reviewer confirms one final result per paper (AI-assisted human
confirmation). Work is saved per account and restored on the next login.

Both workflow modes are implemented end to end — abstract screening, original-PDF
upload, native-PDF AI screening, configurable data extraction, human confirmation,
per-stage summaries and CSV export — and covered by an offline test suite. Running
the AI requires the reviewer's own provider key.

## Key properties

- **Accounts via Google login.** Sign-in uses Streamlit's native OIDC
  (`st.login`). Access can be restricted to an email allowlist. While the Google
  OAuth consent screen is in "Testing", only listed test users can sign in.
- **Per-user saved projects.** Each user's prompts, papers, AI verdicts, and
  review decisions are saved to a Postgres database (Supabase), scoped to their
  account, and restored on the next login. Original PDFs live in a private
  Supabase Storage bucket; only their metadata lives in the `fulltexts` table.
  The imported spreadsheet is stored once in `project_sources`, so routine saves
  stay proportional to the review work rather than to the size of the search
  yield. Saves are version-checked: a second browser tab working from an older
  copy is refused instead of silently overwriting the first.
- **Bring your own model key (BYOK).** Each user pastes their own OpenAI /
  Anthropic / Google API key. Keys are held in session state, not saved as
  project settings. Recorded API errors are scrubbed of the supplied key,
  including surrounding-whitespace variants. Model usage is billed to that key.
- **User-authored screening prompts.** The reviewer writes the abstract and
  full-text instructions in free text; the platform wraps them in a fixed shell with
  a fixed JSON output contract and prompt-injection guards. The AI answers
  Include / Exclude / Unsure for abstracts and Include / Exclude for full text,
  with a short reason. Abstract Unsure papers advance to full-text screening.
- **Original PDF in, original PDF on screen.** OpenAI, Anthropic, and Google use
  their native PDF inputs, preserving tables, figures, and page layout. The app
  does not replace the document with locally extracted plain text. Local PDF
  metadata inspection runs in a separate process with a 10-second deadline.
- **Multi-model comparison.** All three stages support up to three distinct
  models per batch, including models from the same provider. Each attempt has
  its own ID, input/configuration hashes, prompt snapshot, timestamps and result
  in `ai_runs`. Repeated attempts append records without replacing previous
  results or human decisions. Compare matching input versions before confirming.
- **Durable AI runs.** An attempt is recorded before calling a provider and
  completed after the response. Every run button states the papers and the API
  calls it will make, and makes exactly those. The main run button serves every
  paper nobody has decided yet with each selected model, so a batch that stopped
  between two models is completed by the same button; a paper whose answer was
  recorded before attempts were kept gets further models only through the
  comparison panel. A paper/model pair that already
  has a successful attempt on the current input is called again only through
  **Repeat selected models**. Failed and invalid attempts have their own retry
  buttons, which warn that a retry may be billed again; an attempt whose outcome
  is unknown is repeated only after an explicit acknowledgement. An answer that
  was already paid for is shown without another call. Save-only recovery never
  repeats the provider call.
- **Human in the loop.** Review one paper at a time (agree/disagree, or set
  your own verdict when an AI call failed), jumping to any paper by title and
  status. Edits on an unconfirmed extraction form are kept as a draft, with
  the chosen answer sources and ticked review boxes, whenever you leave the
  paper or the page. Confirmed answers never change on their own: if you edit
  them and leave the paper, or save a new question setup, the app asks whether
  to save or discard the changes. The Review Summary page shows
  per-stage progress, AI–human agreement, verdict counts, stage-to-stage
  advancement, and a CSV export.
- **Configurable data extraction.** Define your own single-choice, multiple-choice,
  and open-text questions. Choice questions automatically include `Other` and
  `Not reported`. The AI proposes answers and page/quote evidence from the original
  PDF; a reviewer checks each field, edits as needed, and confirms the whole paper.

## Project layout

```
app.py                         # entry: login gate + mode-aware page navigation
core/                          # shared, feature-agnostic services
  auth.py                      # st.login gate, email allowlist, dev fallback
  db.py                        # projects, imported CSV, PDF metadata, AI attempts
  fulltext_storage.py          # private Supabase / local-dev original PDF storage
  _pdf_inspector.py            # bounded PDF metadata worker
  llm.py                       # provider registry + text and native-PDF calls
  csv_io.py                    # CSV read / column guess / export
  ui.py                        # shared styling, model controls, jump-to-paper picker
features/workflow/state.py     # schema 5, legacy migration, stage logic, persistence
features/workflow/documents.py # detached PDF/CSV changes and guarded file cleanup
features/workflow/runs.py      # shared three-stage runner and immutable proposals
features/workflow/run_controls.py # comparison, recovery, and run-history exports
features/extraction/          # question schemas, validation, review state, drafts, PDF calls, exports
features/screening/            # criteria-based include/exclude screening
  prompts.py                   # fixed prompt templates (user criteria injected)
  judge.py                     # criteria + paper -> verdict & reason
views/projects.py              # landing: create (choose workflow mode) / open / delete
views/abstract_screening.py    # criteria + CSV upload + AI screening + review
views/fulltext_screening.py    # PDF upload/matching/viewing + AI/human screening
views/extraction.py            # question builder + PDF extraction + field-by-field review
views/review_summary.py        # per-stage progress, agreement, export
scripts/postgres_preflight.py  # read-only check of the database role before deployment
tests/                         # offline suite: no network, no live database, no credentials
```

Run the tests with `pytest -q` from the project root. They use mocked providers,
temporary SQLite databases and isolated storage, with no credentials or network.

## Configure

Create `.streamlit/secrets.toml` locally and fill in:

- `[auth]` — Google OAuth client (`client_id`, `client_secret`), a random
  `cookie_secret`, and `redirect_uri` (`http://localhost:8501/oauth2callback`
  locally). Create the OAuth client in Google Cloud Console → Credentials.
- `[database]` — a Supabase Postgres connection string, e.g.
  `postgresql+psycopg2://postgres.<ref>:<password>@<host>.pooler.supabase.com:5432/postgres`.
  Tables are created automatically on first run. PostgreSQL table creation,
  migrations, and access protection share one transaction before the connection
  pool is published. All four app tables (`projects`, `fulltexts`,
  `project_sources`, `ai_runs`) have row-level security enabled and table
  privileges revoked from `PUBLIC` and any existing `anon`/`authenticated` roles.
  Existing policies are not removed. Use a server-only role with schema-management
  privileges that owns these tables or has the required table privileges and
  `BYPASSRLS`; the app enforces per-user ownership. Verify role inheritance,
  other grants, and private bucket policies during deployment; never expose the
  server credentials or run this app with a public REST role.
  Before the first deployment, check the role without changing anything:
  `AIREVIEW_DATABASE_URL='<connection string>' python scripts/postgres_preflight.py`.
  It reports conditions under which the role cannot create or protect the tables;
  in that case the application stops at start-up with a database error instead of
  serving pages. A clean result covers the database role only and is not a
  deployment guarantee.
- `[storage]` — the Supabase project URL, service-role key, and name of an
  existing **private** bucket:

  ```toml
  [storage]
  url = "https://<project-ref>.supabase.co"
  service_role_key = "<server-only-service-role-key>"
  bucket = "fulltext-pdfs"
  ```

  Keep the service-role key only in Streamlit server secrets; never put it in
  browser code or commit it. Set the bucket's allowed MIME type to
  `application/pdf` and choose a file limit of at least 50 MB.
- `[access] allowed_emails` — optional; restrict who can sign in.

`secrets.toml` is gitignored and must never be committed.

### Local development without credentials

With `AIREVIEW_DEV=1` set, the app deliberately ignores configured auth,
database, and Storage secrets and runs as `dev@local` with `.local/projects.db`
plus `.local/fulltext-pdfs/`. This gives you a safe, isolated demo even when the
checkout also contains deployment secrets.

Alternatively, `[dev] allow_no_auth = true` enables the same shared `dev@local`
account — but **only** when the secrets contain no `[database]` and no
`[storage]` section, and only when the value is the literal boolean `true`. A
deployment wired to real data therefore never falls back to an anonymous shared
account, however stale that switch is. Without one of these opt-ins, missing
auth fails closed.

A broken `[access]` section is also refused rather than ignored: if
`allowed_emails` is missing, misspelled, empty, or not a list, the app stops
with an error instead of silently admitting everyone.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# create .streamlit/secrets.toml (see Configure), then:
streamlit run app.py
```

For a clean local demo that deliberately ignores any configured cloud secrets:

```bash
AIREVIEW_DEV=1 .venv/bin/streamlit run app.py --server.address 127.0.0.1
```

Never enable no-login development mode on a public server. The command above
limits the demo to this computer; use authenticated deployment for other users.

Open the URL Streamlit prints, sign in, create a project (pick a workflow
mode), paste your model API key in the sidebar, write your prompt, upload a CSV
and/or PDFs, and click **Run AI screening**. Native AI screening currently
accepts PDFs up to 19 MB. Larger documents can still be stored and reviewed by
a person up to the platform's 50 MB upload limit; they are never silently
truncated or sent partially to a model.

PDF inspection has a 5-second CPU limit on Unix and a 512 MiB address-space
limit on Linux. Other platforms retain the wall-clock deadline but do not have
a portable memory cap. A timed-out or resource-limited inspection rejects the
upload with a safe error; ordinary parser failures may leave page count unknown.
This process boundary is a resource safeguard, not a general security sandbox.

Keep dependencies current with `pip install -r requirements.txt`. The minimum
versions include published security fixes in
[pypdf](https://github.com/py-pdf/pypdf/security/advisories/GHSA-5jq2-8x83-x246)
and [Authlib](https://github.com/authlib/authlib/security/advisories/GHSA-fg6f-75jq-6523).

## Data extraction

1. Confirm an **Include** verdict in full-text screening. In PRISMA mode, the
   upstream abstract decision must also remain current and eligible.
2. Open **Data Extraction**, enter optional overall instructions, and add up to
   30 questions with stable IDs. Choose single choice, multiple choice, or open
   text; supply options and optional guidance, then save the setup.
3. Run extraction with your selected models. Each model receives the PDF and all
   questions in one request. Valid answers survive an invalid neighboring field;
   unsupported choices, missing evidence, and processing issues require review.
4. Inspect the PDF and proposed evidence, select an answer source per question,
   correct answers, and mark every field reviewed. You may combine answers from
   different models. Save a draft or confirm the paper once. Manual extraction is also allowed
   without an AI key. Evidence page numbers refer to PDF pages starting at 1.
5. Download final answers and a separate audit CSV on **Review Summary**.

All questions are required for confirmation. `Other` requires an explanation;
`Not reported` is exclusive and can have no page/quote evidence. It means the
information was not reported, not that the file could not be read. Normal answers
require a page and a short evidence quote. Evidence is proposed by the model and
must be checked by the reviewer; the app does not prove that a quote is accurate.

Changing saved questions/instructions or replacing a PDF invalidates prior
extraction results. Outdated results cannot be confirmed or exported as current
final answers; archive or explicitly rerun them. Re-running outdated results
moves each old result, with the decision or review made on it, to that paper's
history and shows the new answer as a proposal that still needs your review.
A failed re-run leaves the old result in place. Every archived screening
outcome and extraction review is kept; none is dropped after a fixed number.
Archived extraction reviews retain the question setup, model metadata, PDF hash,
and reviewed answers, and archived screening decisions can be downloaded from
Review Summary. AI attempts are retained separately until their paper or
project is deleted.
OpenAI's combined schema-size limits are checked before AI execution; an
oversized setup can still be used for manual review or adjusted before running.
Only current, confirmed answers contribute to choice frequencies. A failed call
or an invalid model answer is retried with its own button or repaired by hand;
a complete answer replaces a partly invalid proposal only while nobody has
reviewed it. Save failures stop the batch immediately, and an unsaved
result remains in the session for a save-only retry without another model call.

Offline checks:

```bash
pip install -r requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Tests block live database and network access by default and use temporary storage.
User edits roll back if saving fails. AI results that cannot be saved block all
workflow pages until a save-only retry succeeds; keep that session open.
The same protection applies to extraction drafts. A version conflict never
offers an unchecked overwrite: download the unsaved project JSON before
explicitly discarding local changes and reloading the saved version. The JSON
is a recovery record, not a CSV import or an automatic project restore.
Every save-failure recovery screen provides a backup download and explicit
discard controls. Reopening the active project preserves its unsaved question
setup; if another session saved newer data, that data is loaded as well.
Loaded state is bound to its project ID. If a project switch is interrupted,
reopen it from My Projects; mismatched state is never written to another project.
Project data and its imported CSV are loaded from one database snapshot.
PDF replacement commits its metadata and archived review state together under
the project version check before updating the live page state. CSV replacement
and paper removal unlink old PDF metadata in that same transaction. Cleanup
checks for active file references before deleting a storage object.
Failed deletion of a removed or replaced PDF keeps
a durable cleanup target and a retry notice, including after reopening the
project. Failed cleanup after an unsuccessful upload or whole-project deletion
retains a retry in the current session; keep that session open until cleanup
succeeds. An abrupt server/process termination is not an atomic transaction
across the database and object storage; it may still leave an orphan object.
CSV imports preserve text such as leading-zero identifiers and literal `NA`.
Imports are capped at 20 MB, 20,000 rows, 100 columns, 1,000,000 data cells,
and 100,000 characters per cell, checked before constructing a full table.
Exports prefix formula-like text cells and headers with an apostrophe for safe
spreadsheet opening; saved project data is unchanged.

## Compare models

1. Select the primary provider/model in the sidebar. Enable **Compare multiple
   models** to add a second or third model and enter the required provider keys.
2. Use the normal stage run button for new papers, or switch on **Model
   comparison and run history** to add missing model attempts for all papers
   or for chosen ones. Review the displayed API-call count and data destinations
   before starting. PDF and extraction-schema limits apply independently to
   each model. Run history and exports are loaded only while that switch is on.
3. Check **Repeat selected models** only when another paid attempt is intended.
   Calls run sequentially; different providers each receive the same input.
4. Compare screening verdicts/reasons and explicitly confirm a final verdict.
   For extraction, choose an answer source per question and confirm the paper.
   Merely viewing, selecting, or running models does not change a confirmed
   result, and a comparison run never archives a decision; only re-running
   outdated results does.
5. Download every attempt, including failures, as JSON or CSV from the stage's
   comparison panel. Final-review exports include source run IDs; extraction
   audit exports also identify each chosen model and its evidence.

Summary agreement rates compare each human verdict with its selected reference
AI result; they are not an inter-model agreement score or an independent accuracy benchmark.

Available legacy AI snapshots migrate on the next successful project save.
Previously discarded history cannot be recovered. Legacy records with unknown
source versions remain exportable but cannot be adopted as current comparisons.
Prompt/PDF changes make earlier attempts incompatible, not deleted. Hashes identify
inputs; the app does not retain replaced PDFs for replay. Replacing the CSV or
removing a paper deletes that paper's attempt history, and project deletion removes
all its runs. Back up exports first if that history is needed.

If the server stops during a request, the attempt may remain `running` with an
unknown outcome; this is not evidence of an ongoing provider job. No automatic
retry is made because the first request may already have incurred a charge;
the stage page offers a repeat after you acknowledge that.

Routine saves and AI attempts do not read the project document back from the
database, and earlier AI snapshots are imported into `ai_runs` once per project.

## Deploy

Push to a (private) GitHub repo and deploy on Streamlit Community Cloud:

1. Connect the repo and pick `app.py`.
2. Paste the same `secrets.toml` contents into the app's **Settings → Secrets**.
3. Add the deployed callback URL (`https://<your-app>.streamlit.app/oauth2callback`)
   to the Google OAuth client's authorized redirect URIs, and set `redirect_uri`
   in the Cloud secrets to match.

## Data & privacy

Original PDFs are stored unchanged in a private bucket (or under `.local/` in
explicit development mode). The database stores paper metadata, PDF object
keys/hashes, AI attempts and prompt snapshots, extraction questions/answers, short evidence quotes, and
reviewer decisions; it does not store a full plain-text copy of the PDF.
Abstract screening sends the title, abstract and reviewer prompt to each selected
model provider; full-text screening and extraction send the original PDF and
prompt. Each provider's data-handling terms apply. Model API keys are kept only
in the app session rather than project settings; supplied keys are redacted from
recorded model output, including malformed field names and API errors.
Imported text and model answers are displayed literally,
without interpreting embedded Markdown images or arbitrary DOI links.
Storage errors use fixed user-facing messages rather than exposing provider
responses, credentials or local filesystem paths.
Uncaught exception details are hidden in the browser; server logs remain private
operational data and should not be shared without review. Local tests cannot
verify hosted database privileges, bucket privacy, deployment secrets, or provider
retention settings; check these before admitting users.

Projects saved by the earlier score-based version are migrated on open: scores
become Include/Exclude verdicts against the saved threshold, and the topic and
rubric become editable criteria text.
