# Fresh-clone verification

Verification for the Phase 1 additions is in progress. This document will be
completed with the actual commands, environments and outcomes before delivery.
No results for another operating system or processor are implied.

The independent checkout is `/private/tmp/inventory-phase1-fresh-check`.
It was cloned from GitHub with no copied environment, datasets or models.
A new Python 3.12.13 environment installed the exact lockfile using a new uv cache.
It downloaded the official training archives and started the complete Phase 1
workflow under `runs/reproduction/`. The original report files are preserved.

The working checkout currently passes 37 tests. Fresh-clone pip setup, app startup,
reproduced-model checks and notebook execution will be recorded after completion.
