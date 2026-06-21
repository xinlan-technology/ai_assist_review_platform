# AI-assisted Literature Review Platform

A Streamlit platform for AI-assisted literature review. Stage 1 (this release)
is **relevance screening**: sign in, create a project, upload a CSV of papers,
let an LLM score each one against a rubric you define, review the papers one by
one (agree with the AI, or disagree and give your own score), then export. Your
work is saved per account and restored on your next login.

Design goal: **extensibility**. Shared services live in `core/`; each stage is a
pluggable feature under `features/` with its own page in `views/`. A later stage
(e.g. information extraction from full text) reuses the same LLM layer, the same
per-user project store, and the same "user-defined criteria → structured output
→ human confirm → export" pattern.

## Key properties

- **Accounts via Google login.** Sign-in uses Streamlit's native OIDC
  (`st.login`). Access can be restricted to an email allowlist. While the Google
  OAuth consent screen is in "Testing", only listed test users can sign in.
- **Per-user saved projects.** Each user's rubric, uploaded papers, AI scores,
  and review decisions are saved to a Postgres database (Supabase), scoped to
  their account, and restored on the next login.
- **Bring your own model key (BYOK).** Each user pastes their own OpenAI /
  Anthropic / Google API key. It is used only for the current session and is
  **never written to disk or the database**. Model usage is billed to that key.
- **Rubric-driven scoring.** You define what each score level (10, 20, … 100)
  means; the rubric is injected into the prompt so scores are consistent.
- **Free re-bucketing.** Scores are computed once. Moving the relevance
  threshold re-labels relevant / not-relevant instantly, with no new API call.
- **Human in the loop.** Review one paper at a time; the Review Summary page
  shows progress, AI–human agreement rate, the decision breakdown, and a CSV
  export (original columns + AI score/reason/suggestion + your decision).

## Project layout

```
app.py                         # entry: login gate + page navigation
core/                          # shared, feature-agnostic services
  auth.py                      # st.login gate, email allowlist, dev fallback
  db.py                        # per-user projects table (SQLAlchemy: Postgres or SQLite)
  llm.py                       # provider registry + call_structured()
  csv_io.py                    # CSV read / column guess / export
  ui.py                        # shared styling (cards, header)
features/screening/            # stage 1 logic
  rubric.py                    # default rubric + prompt rendering
  scorer.py                    # rubric + paper -> score & reason
  state.py                     # review store + snapshot/restore (per-project persistence)
views/projects.py              # landing: create / open / rename / delete projects
views/relevance_screening.py   # score + one-at-a-time review
views/review_summary.py        # progress, agreement, export
```

## Configure

Create `.streamlit/secrets.toml` locally and fill in:

- `[auth]` — Google OAuth client (`client_id`, `client_secret`), a random
  `cookie_secret`, and `redirect_uri` (`http://localhost:8501/oauth2callback`
  locally). Create the OAuth client in Google Cloud Console → Credentials.
- `[database]` — a Supabase Postgres connection string, e.g.
  `postgresql+psycopg2://postgres.<ref>:<password>@<host>.pooler.supabase.com:5432/postgres`.
  The `projects` table is created automatically on first run.
- `[access] allowed_emails` — optional; restrict who can sign in.

`secrets.toml` is gitignored and must never be committed.

### Local development without credentials

With no `[auth]` section **and** `AIREVIEW_DEV=1` set (or `[dev] allow_no_auth =
true`), the app runs with a placeholder user `dev@local` and a local SQLite file
at `.local/projects.db` — handy for development. Without that opt-in, a missing
`[auth]` section fails closed (login required), so a misconfigured deployment
never silently shares one account.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# create .streamlit/secrets.toml (see Configure), then:
streamlit run app.py
```

Open the URL Streamlit prints, sign in, create a project, paste your model API
key in the sidebar, edit the rubric, upload a CSV, and click **Score papers**.

## Deploy

Push to a (private) GitHub repo and deploy on Streamlit Community Cloud:

1. Connect the repo and pick `app.py`.
2. Paste the same `secrets.toml` contents into the app's **Settings → Secrets**.
3. Add the deployed callback URL (`https://<your-app>.streamlit.app/oauth2callback`)
   to the Google OAuth client's authorized redirect URIs, and set `redirect_uri`
   in the Cloud secrets to match.

## Data & privacy

Uploaded files are processed in memory; their **content** (paper metadata, AI
scores, your decisions) is saved into your project in the database, scoped to
your account. Model API keys are kept only in the app session, are sent only to
the selected model provider for scoring, and are never written to disk or the
database. Full-text PDFs are out of scope for Stage 1.
