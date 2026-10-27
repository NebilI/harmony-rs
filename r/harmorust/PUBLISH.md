# Publishing harmorust to r-universe

This is the process that puts **harmorust 0.1.0** on
[NebilI.r-universe.dev](https://NebilI.r-universe.dev). It follows the same
registry used by [seurust](https://github.com/NebilI/seurust).

## One-time setup

1. **Make `NebilI/harmony-rs` public.** R-universe runs
   `git clone https://github.com/NebilI/harmony-rs`. A private repo will not
   build. The R-universe GitHub app only writes commit statuses; it cannot
   read private sources.
2. Confirm the [R-universe GitHub app](https://github.com/apps/r-universe)
   is installed on the `NebilI` account (already required for seurust).
3. Add a repository secret on `NebilI/harmony-rs`:
   * `R_UNIVERSE_REGISTRY_TOKEN` — a PAT with `repo` scope that can push to
     [`NebilI/NebilI.r-universe.dev`](https://github.com/NebilI/NebilI.r-universe.dev).

The registry file is `packages.json`. harmorust is added **next to** seurust
(the sync script merges; it does not replace the file).

```json
{
  "package": "harmorust",
  "url": "https://github.com/NebilI/harmony-rs",
  "subdir": "r/harmorust",
  "branch": "v0.1.0"
}
```

## Pipelines

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `harmorust-checks.yml` | PRs / pushes that touch the R package or crate | Stage Rust sources, `R CMD build`, `R CMD check` |
| `release-harmorust.yml` | `workflow_dispatch`, or a push to `main` that changes `r/harmorust/DESCRIPTION` | Create GitHub release `v<Version>` if missing, then publish |
| `publish-harmorust-r.yml` | Called by the release workflow, or **Run workflow** | Build `harmorust_*.tar.gz`, attach it to the release, merge this package into `NebilI.r-universe.dev` |

R-universe itself rebuilds about once an hour after the registry `branch`
points at a public tag.

## First 0.1.0 publish

1. Land this pipeline on `main`.
2. Make the GitHub repo public.
3. Set `R_UNIVERSE_REGISTRY_TOKEN`.
4. **Actions → Release harmorust → Run workflow.**
   That tags `v0.1.0`, uploads the source tarball, and updates
   `packages.json` without dropping seurust.
5. Wait for https://NebilI.r-universe.dev/harmorust.

Local registry sync (same merge as CI):

```sh
R_UNIVERSE_REGISTRY_TOKEN=... ./scripts/sync-r-universe.sh
```

## Versioning

Keep these in lockstep for a release:

* `r/harmorust/DESCRIPTION` → `Version: x.y.z`
* workspace `Cargo.toml` → `[workspace.package] version`
* `r/harmorust/NEWS.md`

The first published version is **0.1.0** (the 0.1 line).

## User install

```r
install.packages(
  "harmorust",
  repos = c("https://NebilI.r-universe.dev", "https://cloud.r-project.org")
)
```
