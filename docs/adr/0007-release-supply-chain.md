# ADR-0007: Signed releases and a hardened supply chain

- Status: **Accepted**
- Date: 2026-11-20

## Context

People install Nona in the path of all their LLM traffic, specifically to protect personal data. A compromised
release would be worse than no tool: it would see every prompt, and its users would trust it.

This has happened to real Python and CI projects. In December 2024, a GitHub Actions cache poisoning attack led to
malicious `ultralytics` releases on PyPI. In March 2025, the widely used `tj-actions/changed-files` action was
compromised and printed CI secrets to public logs.

Users and companies should be able to check that a wheel or image was built from this repository's source, by its CI, and nothing else.

## Decision

**Releases**
- Commits follow Conventional Commits (already the case). `release-please` opens a release PR with the version bump and `CHANGELOG.md`. Merging it creates the tag and the GitHub release.
- **PyPI with Trusted Publishing** (OIDC). No PyPI token is ever stored. `pypa/gh-action-pypi-publish` uploads PEP 740 attestations with the files.
- **Container on GHCR**: `ghcr.io/<owner>/nona`, multi-arch (`linux/amd64`, `linux/arm64`), CPU-only, runs as non-root, and has the pinned model baked in (NFR-8).
- **Signed and attested**: the image is signed with cosign (Sigstore, keyless, tied to the workflow identity). Wheels and the image get build-provenance attestations (`actions/attest-build-provenance`, SLSA v1 provenance) and a CycloneDX SBOM (`syft`).
- The README documents how to verify a release: `cosign verify` and `gh attestation verify`.

**CI hardening**
- Every third-party action is pinned by full commit SHA. Dependabot keeps the pins and the `uv.lock` dependencies up to date.
- Default `permissions: contents: read`. The release job alone gets `id-token: write` and `packages: write`.
- The release workflow uses no caches, which closes the attack used against `ultralytics`.
- `pip-audit` checks the lockfile for known vulnerabilities on every PR. CodeQL scans the Python code.
- OpenSSF Scorecard runs weekly, and its badge goes in the README.

## Consequences

- ➕ Anyone can check where an artifact came from without trusting the maintainer's laptop.
- ➕ No long-lived secrets exist to steal: PyPI and Sigstore both use short-lived OIDC tokens.
- ➕ The Scorecard badge and the verify instructions show this to users and reviewers.
- ➖ More workflows to maintain, and SHA-pinned actions are harder to read. Dependabot handles the bumps.
- ➖ Baking the model into a multi-arch image makes the build slow. The image is built only on releases, not on every PR.
