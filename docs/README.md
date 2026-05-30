# OpenKSpace docs site

[Docusaurus](https://docusaurus.io/) site for OpenKSpace, deploying
to [https://sigilweaver.app/openkspace/docs/](https://sigilweaver.app/openkspace/docs/)
via Cloudflare Workers (managed by the Cloudflare GitHub App on
push to `main`).

## Develop

```sh
bun install
bun run dev          # http://localhost:25816/openkspace/docs/
```

## Build (verify locally)

```sh
bun run build:cloudflare
```
