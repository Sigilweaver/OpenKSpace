# Security policy

## Supported versions

Only the latest minor release of OpenKSpace is supported with security
fixes.

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a vulnerability

Please report security vulnerabilities privately via
[GitHub Security Advisories](https://github.com/Sigilweaver/OpenKSpace/security/advisories/new).

Do **not** open a public issue for security reports. We will
acknowledge within 7 days and aim to publish a fix or mitigation
within 30 days for confirmed issues.

## Scope

In scope:

- Memory-safety bugs in the parser / reconstruction code.
- Path traversal or arbitrary file write triggered by an attacker-
  supplied `.h5` file.
- Supply-chain integrity issues affecting the published crates or
  binaries.

Out of scope:

- Denial-of-service from intentionally malformed inputs that the
  parser correctly rejects (resource consumption alone is not a
  vulnerability).
- Issues in third-party crates - please report those upstream.
- Numerical accuracy of reconstruction outputs. OpenKSpace is a
  reference implementation, not a clinical product, and is not
  validated for diagnostic use.

## Disclosure

Coordinated disclosure is preferred. Once a fix is released, the
advisory will be made public and credited to the reporter unless
they request anonymity.
