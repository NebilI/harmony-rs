# Publishing harmorust

harmorust is on [NebilI.r-universe.dev](https://NebilI.r-universe.dev),
next to [seurust](https://github.com/NebilI/seurust).
[`NebilI/harmony-rs`](https://github.com/NebilI/harmony-rs) is already
**public** — R-universe clones that URL, so a private source repo will not
build.

## Do not replace `packages.json`

[`NebilI/NebilI.r-universe.dev`](https://github.com/NebilI/NebilI.r-universe.dev)
is a **shared** registry. Its `packages.json` is one JSON array for every
package in the universe (today seurust and harmorust). Replacing the file
with a harmorust-only (or seurust-only) array is what dropped a neighbor
before.

Always **upsert** the harmorust object and leave every other entry unchanged:

```json
{
  "package": "harmorust",
  "url": "https://github.com/NebilI/harmony-rs",
  "subdir": "r/harmorust",
  "branch": "v0.1.0"
}
```

`scripts/sync-r-universe.sh` does that. It refuses to commit if
`packages.json` is missing after a non-404 fetch, is not a JSON array, or
if seurust (or any other neighbor) would change. Check locally:

```sh
bash scripts/test-sync-r-universe.sh
R_UNIVERSE_REGISTRY_TOKEN=... ./scripts/sync-r-universe.sh
```

Never paste `packages.json.example` from seurust, and never hand-write a
one-element array over the live registry.

## Release a new version

Keep these in lockstep:

* `r/harmorust/DESCRIPTION` → `Version: x.y.z`
* workspace `Cargo.toml` → `[workspace.package] version`
* `r/harmorust/NEWS.md`

Then **Actions → Release harmorust → Run workflow**. That creates GitHub
release `v<Version>` if it is missing, attaches `harmorust_*.tar.gz`, and
points the r-universe registry at the new tag without dropping seurust.

A push to `main` that changes `r/harmorust/DESCRIPTION` does the same.
R-universe rebuilds about once an hour after the registry `branch` field
matches the tag. R-universe only builds the git ref in `branch`; fixing
install bugs on `main` does not reach the dashboard until that field (or
the tagged commit) moves.

## Workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `harmorust-checks.yml` | PRs / pushes that touch the R package or crate | Stage Rust sources, `R CMD build`, `R CMD check`, registry upsert regression |
| `release-harmorust.yml` | `workflow_dispatch`, or a push to `main` that changes `r/harmorust/DESCRIPTION` | Create GitHub release `v<Version>` if missing, then publish |
| `publish-harmorust-r.yml` | Called by the release workflow, or **Run workflow** | Build `harmorust_*.tar.gz`, attach it to the release, update `NebilI.r-universe.dev` |

The publish job reads `R_UNIVERSE_REGISTRY_TOKEN` from GitHub Actions
secrets (a PAT that can push to
[`NebilI/NebilI.r-universe.dev`](https://github.com/NebilI/NebilI.r-universe.dev)).
It is never stored in this repository.

## Owner-only GitHub steps

These cannot be done from the package tree:

1. Confirm **Settings → General → Visibility** is Public (already true).
2. Keep **Actions secret** `R_UNIVERSE_REGISTRY_TOKEN` on `NebilI/harmony-rs`
   (classic PAT or fine-grained token with contents read/write on
   `NebilI/NebilI.r-universe.dev` only). Recreate it if publish logs say the
   secret is missing.
3. Glance at deploy keys, webhooks, and other Actions secrets before any
   future visibility change; nothing in git should hold token values.
4. Confirm the [R-universe GitHub app](https://github.com/apps/r-universe)
   is installed on the `NebilI` account (all repos, or at least
   `harmony-rs`, `seurust`, and `NebilI.r-universe.dev`).
5. After a Windows/wasm install fix lands on `main`, cut a **new tag**
   (for example `0.1.1`) so r-universe rebuilds that commit. The live
   dashboard still tracks `v0.1.0`.

## Install

```r
install.packages(
  "harmorust",
  repos = c("https://NebilI.r-universe.dev", "https://cloud.r-project.org")
)
```
