# Publishing harmorust

harmorust is on [NebilI.r-universe.dev](https://NebilI.r-universe.dev),
next to [seurust](https://github.com/NebilI/seurust).

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
matches the tag.

## Workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `harmorust-checks.yml` | PRs / pushes that touch the R package or crate | Stage Rust sources, `R CMD build`, `R CMD check` |
| `release-harmorust.yml` | `workflow_dispatch`, or a push to `main` that changes `r/harmorust/DESCRIPTION` | Create GitHub release `v<Version>` if missing, then publish |
| `publish-harmorust-r.yml` | Called by the release workflow, or **Run workflow** | Build `harmorust_*.tar.gz`, attach it to the release, update `NebilI.r-universe.dev` |

The publish job reads `R_UNIVERSE_REGISTRY_TOKEN` from GitHub Actions
secrets (a token that can push to
[`NebilI/NebilI.r-universe.dev`](https://github.com/NebilI/NebilI.r-universe.dev)).
It is never stored in this repository.

Local registry sync (same merge as CI):

```sh
R_UNIVERSE_REGISTRY_TOKEN=... ./scripts/sync-r-universe.sh
```

The registry entry written to `packages.json`:

```json
{
  "package": "harmorust",
  "url": "https://github.com/NebilI/harmony-rs",
  "subdir": "r/harmorust",
  "branch": "v0.1.0"
}
```

## Install

```r
install.packages(
  "harmorust",
  repos = c("https://NebilI.r-universe.dev", "https://cloud.r-project.org")
)
```
