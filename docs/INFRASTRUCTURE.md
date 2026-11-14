# What runs where (checked 2026-11-14)

Railway workspace "Synthosize's Projects". All SoundLens services belong in the **`soundlens`** project.

## Live

| Domain | Railway project → service | Code | Data |
|---|---|---|---|
| beatbattle.soundlensapp.com | `soundlens` → `license-server` | `Solidussss/soundlens-haze` @ `soundlens-0.2.0`, root `/license-server` | volume `license-server-volume` at `/data` |
| www.soundlensapp.com | **`zealous-appreciation`** → `soundlens-landing` (old home, to be retired) | `Solidussss/soundlens-landing` (byte-identical copy of `soundlens-haze/marketing/site`) | none |
| soundlensapp.com (bare) | same service, but **not working**: the DNS record was never created, so the certificate is stuck "issuing" | | |

## New, ready, not yet on a domain

| | |
|---|---|
| Service | `soundlens` → `website` |
| Code | `soundlens-haze` @ `soundlens-0.2.0`, root `/marketing/site`. Watch pattern `/marketing/site/**`, so only website changes redeploy it |
| Test URL | https://website-production-5bea6431.up.railway.app |
| Status | deployed from the same commit as the live site; its files are identical to `soundlens-landing`. Logs: "soundlensapp.com site up" |

## Finish the move (needs access to the domain registrar's DNS)

Each Railway service has its own DNS target, so moving a domain needs one DNS edit per domain.

1. Tell Claude "move the domains". It will remove `www.soundlensapp.com` and `soundlensapp.com` from `soundlens-landing`, add them to `website`, and report the exact records.
2. At the registrar, set the records Railway shows:
   - `www` → CNAME to the new `*.up.railway.app` target;
   - the bare domain → CNAME/ALIAS (or the registrar's "CNAME flattening") to its target;
   - the `_railway-verify` TXT record.
   - The **www site is down from step 1 until this DNS change propagates** (usually minutes). Do steps 1 and 2 together.
3. Once both domains show "verified" and "certificate valid", delete the `soundlens-landing` service.
4. Keep the `zealous-appreciation` project for now: it holds the unattached 5 GB volume **`web-volume`**, which may contain old analyzer data (users, reports). Check its contents before deleting it.
5. Archive the `Solidussss/soundlens-landing` repo on GitHub (Settings → Archive). The website source of truth is `soundlens-haze/marketing/site`.

## Not deployed

| Code | Notes |
|---|---|
| `Solidussss/soundlens` (this repo: analyzer + Studio) | entry `uvicorn visual_app:app` (Procfile). When it's deployed, it should be a third service in the `soundlens` project, e.g. `studio.soundlensapp.com`, with its own volume. Its accounts are separate from Beat Battle accounts |

## Other projects in the workspace (not SoundLens)

`delightful-insight` (siteremade.com), `astonishing-courtesy` (trading-lab + Postgres), and `synthosize-visual-studio` and `siteremade-staging` (both empty).
