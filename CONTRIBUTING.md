# Contributing to OpenKSpace

Thanks for your interest in OpenKSpace. This is a small,
single-maintainer project that ships [Apache-2.0](LICENSE) Rust
tooling for Cartesian MRI k-space reconstruction from
[ISMRMRD](https://ismrmrd.github.io/) `.h5` files.

Crates in this repo: `openkspace-io`, `openkspace-recon`,
`openkspace-cli`.

## Before you open a PR

- Open an issue first if the change is non-trivial (new reconstruction
  strategy, schema change, new dependency, MSRV bump). For small fixes
  - typos, docs, minor bug fixes, additional tests - go straight to a
  PR.
- Run `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings`
  locally. CI will run them too.
- Run `cargo test --workspace`.
- Update [CHANGELOG.md](CHANGELOG.md) under `## [Unreleased]` with a
  short bullet describing the user-visible change.
- Keep commits small and prefer [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Code is ASCII only and `#![forbid(unsafe_code)]`.

## Reconstruction strategies

The `ReconStrategy` trait is the public extension point. New strategies
should:

- Cite their primary reference in the source (e.g. Pruessmann 1999 for
  SENSE, Griswold 2002 for GRAPPA, Lustig 2007 for L1-wavelet CS).
- Work on both 2D multi-slice and 3D Cartesian data, decoupling along
  kz via 1-D IFFT where applicable.
- Be selectable via the CLI `--strategy` flag with a stable name.

## Test data

OpenKSpace ships with small public-domain ISMRMRD samples under
`corpus/` to keep test data licit and redistributable. Please do not
add proprietary scanner data; if a reproduction requires a specific
dataset, document where to obtain it rather than checking it in.

## Security

Please report security vulnerabilities privately via GitHub Security
Advisories - see [SECURITY.md](SECURITY.md). Do not open public issues
for vulnerabilities.

## DCO

By submitting a contribution you certify that you have the right
to submit the work under the project license (Apache-2.0) and
agree to the
[Developer Certificate of Origin](https://developercertificate.org/).

## License

By submitting a PR you agree that your contribution is licensed under
the Apache License 2.0, the same terms as the rest of the project.
