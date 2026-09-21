# VITRA CSTORE contract fixture

This tracked fixture is the minimal regression-oriented subset of the approved final VITRA CSTORE handoff. It captures the immutable `adora-cstore-v1` contract that ADORA must consume from VITRA-CGRA branch `feature/cstore-hardware` at full commit `da03f4ab0cf696466147ac9210518e7ead6c9589`.

## Verify first

From this directory, run:

```sh
sha256sum -c SHA256SUMS
jq empty manifest.json artifacts/operations.json artifacts/adg.json artifacts/loop_index_contract.json artifacts/generator-input/vitra_spec.json
```

`artifacts/operations.json`, `artifacts/adg.json`, `artifacts/loop_index_contract.json`, and `artifacts/generator-input/vitra_spec.json` are raw byte-for-byte copies from the approved final handoff. Their provenance and SHA-256 values are machine-readable in `manifest.json`.

## Source and generation

- Repository: `https://github.com/MIONkb/VITRA-CGRA.git`
- Branch: `feature/cstore-hardware`
- Commit: `da03f4ab0cf696466147ac9210518e7ead6c9589`
- Generator: `tram.vitra.CStoreVerilogGen`
- Clean generation ID: `da03f4a-cstore-generation-clean-20260823T064142Z`
- Lab Chipyard commit: `27981d26675e7fd2e41d772375b5632951ece968`
- fdra integration commit: `0cb9381fa231b4d26881779a80a5f0c78ec4c5ff`
- Toolchain: sbt 1.8.2, OpenJDK 11.0.27
- Canonical clean RTL SHA256: `3dfdfe954ae2b6614d7e3a7e3b08c3feb9d899f0fef9d4f4e1eee230d24fbbc0`

The approved final handoff was built around the canonical clean generation preserved at `/tmp/vitra-task1-refresh.yjQCfO/fresh-bundle-clean`, with a second independent matching generation at `/tmp/vitra-task1-refresh.yjQCfO/fresh-bundle`. This fixture keeps only the semantic contract artifacts required by ADORA regression tests; it does not pretend to be the full 42-payload handoff package.

Generation command:

```sh
env XDG_RUNTIME_DIR=/tmp/vitra-cstore-closeout-sbt/runtime TMPDIR=/tmp/vitra-cstore-closeout-sbt/runtime COURSIER_CACHE=/tmp/vitra-cstore-closeout-sbt/coursier JAVA_TOOL_OPTIONS='-Djava.io.tmpdir=/tmp/vitra-cstore-closeout-sbt/runtime -Dsbt.global.base=/tmp/vitra-cstore-closeout-sbt/global -Dsbt.boot.directory=/tmp/vitra-cstore-closeout-sbt/boot -Dsbt.ivy.home=/tmp/vitra-cstore-closeout-sbt/ivy -Dsbt.coursier.home=/tmp/vitra-cstore-closeout-sbt/coursier -Dsbt.server.autostart=false -Dsbt.server.forcestart=true' /home/jhlou/chipyard/.conda-env/bin/sbt -java-home /usr/lib/jvm/java-11-openjdk-amd64 -batch 'project fdra' clean 'runMain tram.vitra.CStoreVerilogGen -td /tmp/vitra-task1-refresh.yjQCfO/fresh-bundle-clean'
```

The clean rebuild command exited 0 in 153 seconds and its built-in CSTORE artifact audit passed. The checked-in semantic JSON and loop-index contract hashes match that approved final package exactly.

## Contract for ADORA

- Logical operands are `data=0`, `address=1`, `enable=2`.
- Address operand 1 is a byte address/byte offset. For this 16-bit datapath, hardware converts it to an SRAM word address by dividing by 2.
- `UseEn` is a static conditional-IOB configuration field: config ID 19, aggregate bit 127.
- Runtime enable is operand 2; predicate truth is word bit 0. Thus values 0 and 2 are false and value 1 is true.
- `UseEn=0` preserves normal STORE and ignores operand 2. `UseEn=1` permits a write only when operand 2 bit 0 is set.
- Production `cgra_iob_sram_add_reg` is 2. Data, byte address, predicate, request enable, write mask, address, and data were validated as one aligned transaction through this registered path.
- CSTORE is `OPC=4`, `numOperands=3`, `numRes=0`, `latency=1`.
- Loop-index lowering contract is exported separately in `artifacts/loop_index_contract.json` and requires logical `affine.for` IV lowering to physical `ACC` with explicit external byte scaling.
- The generated IOB capability is exactly `INPUT`, `OUTPUT`, `LOAD`, `STORE`, `CSTORE`.
- CLOAD is out of scope, is not implemented, and is not advertised by either delivered JSON. ADORA must not infer CLOAD merely from conditional IOB mode.

## Recorded hardware proof

The clean-source full-CGRA oracles used `OLD=0xbeef`. True CSTORE produced exactly one `en=1`, `we=0b11` write and stored `0x1236`; false CSTORE preserved `0xbeef` with zero target writes while `done` still asserted; normal STORE with `UseEn=0` still produced exactly one correct write to `0x4567`. Alternating predicates `1,0,1,0` remained the focused regression shape for conditional writes.

See `evidence/hardware-validation.md` for the final clean-generation provenance, the bounded RTL audit, and the package-only oracle summary. This fixture does not start or modify Mapper development.
