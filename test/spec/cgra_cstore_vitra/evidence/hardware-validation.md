# Hardware validation

## Source identity

- VITRA final commit: `da03f4ab0cf696466147ac9210518e7ead6c9589`
- VITRA fix commit: `eba80a8f9595fcd67209938e03c643aecf6fc26c`
- ADORA artifact commit: `ab70915a039a36bf949e41e536d785508ba97fa9`
- Final clean-generated RTL SHA256:
  `3dfdfe954ae2b6614d7e3a7e3b08c3feb9d899f0fef9d4f4e1eee230d24fbbc0`

## Canonical generation provenance

- Preserved clean bundle: `/tmp/vitra-task1-refresh.yjQCfO/fresh-bundle-clean`
- Second matching generation: `/tmp/vitra-task1-refresh.yjQCfO/fresh-bundle`
- Clean rebuild command:

```bash
env XDG_RUNTIME_DIR=/tmp/vitra-cstore-closeout-sbt/runtime TMPDIR=/tmp/vitra-cstore-closeout-sbt/runtime COURSIER_CACHE=/tmp/vitra-cstore-closeout-sbt/coursier JAVA_TOOL_OPTIONS='-Djava.io.tmpdir=/tmp/vitra-cstore-closeout-sbt/runtime -Dsbt.global.base=/tmp/vitra-cstore-closeout-sbt/global -Dsbt.boot.directory=/tmp/vitra-cstore-closeout-sbt/boot -Dsbt.ivy.home=/tmp/vitra-cstore-closeout-sbt/ivy -Dsbt.coursier.home=/tmp/vitra-cstore-closeout-sbt/coursier -Dsbt.server.autostart=false -Dsbt.server.forcestart=true' /home/jhlou/chipyard/.conda-env/bin/sbt -java-home /usr/lib/jvm/java-11-openjdk-amd64 -batch 'project fdra' clean 'runMain tram.vitra.CStoreVerilogGen -td /tmp/vitra-task1-refresh.yjQCfO/fresh-bundle-clean'
```

- Clean rebuild result: exit `0`, elapsed `153 s`, emitted the RTL/spec hashes captured by this fixture.

## Bounded 710c04 vs 3dfdfe RTL audit

- Compared files:
  - old incoming RTL `710c04eb320a212479f6825a84c533919c9defd73fc8e7d74f44d02cfea445ec`
  - canonical clean RTL `3dfdfe954ae2b6614d7e3a7e3b08c3feb9d899f0fef9d4f4e1eee230d24fbbc0`
- Diff scope: one file, one hunk, 34 diff lines total.
- Changed region: AXI-Lite readback mux around generated `_r_data_T_*` wires and `io_s_axilite_r_bits_data` in `CGRAWithAXI.v`.
- No diff evidence in the CSTORE/STORE side-effect path, DelayPipe predicate path, or the loop-index ACC contract artifacts.

## Fresh final-source full-CGRA oracles

All cases use the clean-generated RTL hash shown above.

| Oracle | Config SHA256 | Predicate | Target writes | Final value | Done cycle | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CSTORE true | `dd1ce4f4244e81cfc0e7b76e3970e480ec4331ff6f540b00a4b1db2e6f1d44e9` | 1 | 1 | `0x1236` | 3 | PASS |
| CSTORE false | `dd1ce4f4244e81cfc0e7b76e3970e480ec4331ff6f540b00a4b1db2e6f1d44e9` | 0 | 0 | `0xbeef` | 3 | PASS |
| normal STORE | `1a85dc659a7d25c725639d9197c14e0470c5e87b8c579edbcb3534c8e82b2caf` | n/a | 1 | `0x4567` | 2 | PASS |

Normal STORE decoded `IsStore=1`, `UseAddr=1`, `UseEn=0`; its sole target write was bank 42, byte address 0 / word address 0, mask 3, data `0x4567`.

CSTORE true decoded `IsStore=1`, `UseAddr=1`, `UseEn=1`; its sole target write was bank 33, byte address 12 / word address 6, mask 3, data `0x1236`.

CSTORE false reused the exact same compiler configuration, had predicate bit 0 equal to zero, issued zero target SRAM writes, preserved OLD `0xbeef`, and completed normally.

Machine-readable oracle hashes are recorded in `evidence/e2e-results.json` in the approved full package. This tracked fixture keeps only the summary required for ADORA regression provenance.
