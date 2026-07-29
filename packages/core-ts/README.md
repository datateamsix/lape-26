# @lape-26/core

Dependency-free TypeScript reference encoder.

During Phase 0, tests run directly with Node 22's type stripping:

```bash
node --experimental-strip-types --test test/core.test.ts
```

A later Phase 1 package build may add compiled ESM/CJS outputs and npm publication.
