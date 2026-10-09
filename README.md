# AI-assisted Literature Review Platform

A Streamlit platform for AI-assisted systematic literature review. Each project
follows one of two workflow modes, chosen at creation:

- **PRISMA** — upload a CSV of papers → abstract include/exclude screening →
  attach PDFs for advancing papers → full-text include/exclude screening → data extraction.
- **Full-text direct review** — upload full-text PDFs → full-text
  include/exclude screening → data extraction.

Every stage follows the same pattern: the reviewer writes their own screening
prompt, the AI produces a structured judgment per paper, and the
reviewer confirms or overrides it one paper at a time (AI-assisted human
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
  Anthropic / Google API key. It is used only for the current session and is
  **never written to disk or the database**. Model usage is billed to that key.
- **User-authored screening prompts.** The reviewer writes the abstract and
  full-text instructions in free text; the platform wraps them in a fixed shell with
  a fixed JSON output contract and prompt-injection guards. The AI answers
  Include / Exclude / Unsure for abstracts and Include / Exclude for full text,
  with a short reason. Abstract Unsure papers advance to full-text screening.
- **Original PDF in, original PDF on screen.** OpenAI, Anthropic, and Google use
  their native PDF inputs, preserving tables, figures, and page layout. The app
  does not replace the document with locally extracted plain text.
- **Resumable AI runs.** Progress is saved after every paper. An interrupted
  run resumes where it stopped, and failed API calls are retried by simply
  running again — already-screened papers are never re-billed.
- **Human in the loop.** Review one paper at a time (agree/disagree, or set
  your own verdict when an AI call failed), jumping to any paper by title and
  status. In-progress extraction answers are saved automatically before a
  rerun can discard them. The Review Summary page shows
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
  db.py                        # projects (versioned), imported CSV, PDF metadata
  fulltext_storage.py          # private Supabase / local-dev original PDF storage
  llm.py                       # provider registry + text and native-PDF calls
  csv_io.py                    # CSV read / column guess / export
  ui.py                        # shared styling, model controls, jump-to-paper picker
features/workflow/state.py     # project schema, v1/v2→v3 migration, stage logic, persistence
features/extraction/          # question schemas, validation, review state, PDF calls, exports
features/screening/            # criteria-based include/exclude screening
  prompts.py                   # fixed prompt templates (user criteria injected)
  judge.py                     # criteria + paper -> verdict & reason
views/projects.py              # landing: create (choose workflow mode) / open / delete
views/abstract_screening.py    # criteria + CSV upload + AI screening + review
views/fulltext_screening.py    # PDF upload/matching/viewing + AI/human screening
views/extraction.py            # question builder + PDF extraction + field-by-field review
views/review_summary.py        # per-stage progress, agreement, export
tests/                         # offline suite: no network, no live database, no credentials
```

Run the tests with `pytest -q` from the project root. They mock every provider,
database and storage call, so they need no credentials and no network.

## Configure

Create `.streamlit/secrets.toml` locally and fill in:

- `[auth]` — Google OAuth client (`client_id`, `client_secret`), a random
  `cookie_secret`, and `redirect_uri` (`http://localhost:8501/oauth2callback`
  locally). Create the OAuth client in Google Cloud Console → Credentials.
- `[database]` — a Supabase Postgres connection string, e.g.
  `postgresql+psycopg2://postgres.<ref>:<password>@<host>.pooler.supabase.com:5432/postgres`.
  Tables are created automatically on first run.
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
AIREVIEW_DEV=1 .venv/bin/streamlit run app.py
```

Open the URL Streamlit prints, sign in, create a project (pick a workflow
mode), paste your model API key in the sidebar, write your prompt, upload a CSV
and/or PDFs, and click **Run AI screening**. Native AI screening currently
accepts PDFs up to 19 MB. Larger documents can still be stored and reviewed by
a person up to the platform's 50 MB upload limit; they are never silently
truncated or sent partially to a model.

## Data extraction

1. Confirm an **Include** verdict in full-text screening. In PRISMA mode, the
   upstream abstract decision must also remain current and eligible.
2. Open **Data Extraction**, enter optional overall instructions, and add up to
   30 questions with stable IDs. Choose single choice, multiple choice, or open
   text; supply options and optional guidance, then save the setup.
3. Run extraction with your selected provider. Each paper sends its PDF and all
   questions in one request. Valid answers survive an invalid neighboring field;
   unsupported choices, missing evidence, and processing issues require review.
4. Inspect the PDF and proposed evidence, correct answers, and mark every field
   reviewed. Save a draft or confirm the paper. Manual extraction is also allowed
   without an AI key. Evidence page numbers refer to PDF pages starting at 1.
5. Download final answers and a separate audit CSV on **Review Summary**.

All questions are required for confirmation. `Other` requires an explanation;
`Not reported` is exclusive and can have no page/quote evidence. It means the
information was not reported, not that the file could not be read. Normal answers
require a page and a short evidence quote. Evidence is proposed by the model and
must be checked by the reviewer; the app does not prove that a quote is accurate.

Changing saved questions/instructions or replacing a PDF invalidates prior
extraction results. Outdated results cannot be confirmed or exported as current
final answers; archive or explicitly rerun them. The five most recent archived
extraction versions retain the question setup, model metadata, PDF hash, and
reviewed answers. Export the audit CSV if longer-term history is needed.
OpenAI's combined schema-size limits are checked before AI execution; an
oversized setup can still be used for manual review or adjusted before running.
Only current, confirmed answers contribute to choice frequencies. A transient
call failure is retried by a later run; invalid model answers require an explicit
retry or manual repair. Save failures stop the batch immediately, and an unsaved
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
Project data and its imported CSV are loaded from one database snapshot.
PDF replacement commits its metadata and archived review state together under
the project version check. Failed deletion of a removed or replaced PDF keeps
a durable cleanup target and a retry notice, including after reopening the
project. Failed cleanup after an unsuccessful upload or whole-project deletion
retains a retry in the current session; keep that session open until cleanup
succeeds.
CSV imports preserve text such as leading-zero identifiers and literal `NA`.
Exports prefix formula-like text cells and headers with an apostrophe for safe
spreadsheet opening; saved project data is unchanged.

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
keys/hashes, AI verdicts, extraction questions/answers, short evidence quotes, and
reviewer decisions; it does not store a full plain-text copy of the PDF.
During AI screening or extraction the original PDF and the reviewer prompt
are sent to the selected model provider. Model API keys are kept only in the
app session and are never written to disk or the database.

Projects saved by the earlier score-based version are migrated on open: scores
become Include/Exclude verdicts against the saved threshold, and the topic and
rubric become editable criteria text.
