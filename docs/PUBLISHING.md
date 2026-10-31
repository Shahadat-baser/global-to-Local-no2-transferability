# Publishing this repository and obtaining a DOI

AMT requires that code and data assets carry a DOI, be cited in the manuscript
**and** appear in the reference list, and be reachable by reviewers at
submission. A GitHub link alone does not satisfy this: GitHub URLs are not
persistent identifiers, and repositories can be renamed, moved or deleted.

The sequence below produces a citable DOI in about twenty minutes.

---

## Before you start

Complete these in the repository, because they are baked into the Zenodo record
at release time and changing them afterwards means issuing a new version.

- [ ] `CITATION.cff` — add an ORCID for each author. Register free at
      <https://orcid.org/> if anyone lacks one; Copernicus requires one for the
      corresponding author in any case.
- [ ] `CITATION.cff` — set `repository-code` and `url` to the GitHub address
      once it exists (step 2 below).
- [ ] `LICENSE` and `LICENSE-DATA` — confirm the author list and year.
- [ ] `README.md` — replace the contact placeholder near the end.
- [ ] Run `bash run_pipeline.sh 6 7 8 9 10 11` once on a clean clone to confirm
      the offline stages reproduce the published numbers.

---

## Step 1 — Prepare the local repository

```bash
cd path/to/no2-transferability
git init
git add .
git status            # read this list carefully before committing
```

`git status` should show roughly 27 files: `src/*.py`, `run_pipeline.sh`,
`README.md`, `LICENSE`, `LICENSE-DATA`, `CITATION.cff`, `requirements.txt`,
`.gitignore` and `docs/`.

It should **not** show `data/`, `figures/`, `.venv/`, `__pycache__/` or any file
containing an API key. Those are excluded by `.gitignore`. If any appears, stop
and fix `.gitignore` before continuing — a secret committed once remains in the
history even after deletion.

```bash
git commit -m "Analysis pipeline for the AMT submission"
git branch -M main
```

---

## Step 2 — Create the GitHub repository

1. Go to <https://github.com/new>.
2. Name it, for example, `no2-transferability`.
3. Set it **Public**. Reviewers need access, and the scooping risk is low
   because the contribution is the finding, not the pipeline.
4. Do **not** let GitHub add a README, `.gitignore` or licence — you already
   have all three, and GitHub's versions would conflict.
5. Create, then connect and push:

```bash
git remote add origin https://github.com/<your-username>/no2-transferability.git
git push -u origin main
```

6. Open the repository page. GitHub should display a *Cite this repository*
   button, which confirms `CITATION.cff` parsed correctly. If it is missing, the
   YAML has an error — check indentation.
7. Update `repository-code` and `url` in `CITATION.cff` with the real address,
   then commit and push again.

---

## Step 3 — Connect Zenodo

1. Go to <https://zenodo.org/> and sign in **with GitHub**, so the two accounts
   are linked.
2. Open <https://zenodo.org/account/settings/github/>.
3. Find `no2-transferability` in the list and switch the toggle **on**.

Zenodo now watches the repository. Nothing is archived until you publish a
release — the toggle alone does nothing.

---

## Step 4 — Publish a release

On the GitHub repository page: **Releases** → **Create a new release**.

| Field | Value |
|---|---|
| Tag | `v1.0.0` |
| Target | `main` |
| Title | `v1.0.0 — AMT submission` |
| Description | One or two sentences: what the release contains and which manuscript it accompanies. |

Click **Publish release**. Zenodo receives the webhook, archives a snapshot and
mints a DOI, usually within a minute or two.

---

## Step 5 — Complete the Zenodo record

Go to <https://zenodo.org/account/settings/github/> and open the new record.
Zenodo guesses the metadata from GitHub; correct it:

- **Title** — set to: *Air quality estimation failure in scarcest monitoring
  networks: analysis pipeline*. Zenodo otherwise uses the repository name.
- **Authors** — add all four with ORCIDs and affiliations. Zenodo lists only the
  GitHub account owner by default.
- **Description** — paste the abstract from `README.md`.
- **Licence** — MIT.
- **Related identifiers** — add the manuscript once it has a DOI, with relation
  *is supplement to*. Before acceptance, leave this and add it later.
- **Keywords** — copy the eight from `CITATION.cff`.

Save. Published Zenodo records cannot be deleted, so check before saving; minor
metadata can still be edited afterwards.

---

## Step 6 — Two DOIs, and which to cite

Zenodo issues two:

| DOI | Points to | Use |
|---|---|---|
| **Concept DOI** | always the newest version | general links, the README badge |
| **Version DOI** | this exact snapshot | **cite this in the manuscript** |

Cite the **version DOI**, so a reader reaches the exact code that produced the
published numbers. The concept DOI would silently move to a later version.

Add it in two places in the manuscript, both of which AMT requires:

**Code and data availability**

> The analysis code and derived datasets are archived at Zenodo,
> https://doi.org/10.5281/zenodo.XXXXXXX (Baser et al., 2026).

**Reference list**

> Baser, S., Bilal, M., Tawabini, B. S., and Koeshidayatullah, A.: Air quality
> estimation failure in scarcest monitoring networks: analysis pipeline,
> Zenodo [code], https://doi.org/10.5281/zenodo.XXXXXXX, 2026.

Then fill `doi:` in `CITATION.cff`, commit and push. Do not create a new release
for that one-line change; the next release can carry it.

---

## Step 7 — Deposit the datasets

The derived datasets are too large for GitHub and are not committed. Deposit
them in a **separate Zenodo record**, uploaded directly rather than through
GitHub:

```
DATASET_A_target_global100.csv
DATASET_B_train.csv
DATASET_C_test_india_arid.csv
step9_global100_predictions.csv
Table_S2_global100_estimates.csv
stage1_seed0.json … stage1_seed9.json
stage2_seed0.json … stage2_seed9.json
```

Set the licence to **CC BY 4.0**, not MIT — this is data, not code. Link the two
records to each other with the *is supplement to* relation. Cite this DOI in the
data part of the availability statement and the code DOI in the code part.

---

## Step 8 — After acceptance

1. If the code changed during revision, publish `v1.1.0`. Zenodo mints a new
   version DOI automatically; update the manuscript citation to match.
2. Add the paper DOI to both Zenodo records under *Related identifiers*.
3. Fill `preferred-citation: doi:` in `CITATION.cff`.
4. Enter the DOI in the Copernicus asset tab during production upload, which is
   a separate step from the citation in the text.

---

## Common pitfalls

**Switching on the Zenodo toggle after the release.** The webhook only fires for
releases created afterwards. Toggle first, then release.

**Citing the concept DOI in the paper.** It follows the newest version, so a
later release silently changes what the paper points at.

**Leaving Zenodo's default metadata.** It lists one author — the GitHub account
owner — and uses the repository slug as the title. Correct it before saving.

**Committing a key.** If one is pushed by accident, revoke it immediately at the
provider. Removing the file is not sufficient; the value stays in the history.

**Making the repository private during review.** Reviewers cannot reach it, and
the DOI resolves to an inaccessible record.
