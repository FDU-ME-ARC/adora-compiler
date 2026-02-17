adoracc.py FPVecAdd.mlir -o opt.mlir 

cgra-mapper \
  --adg="/home/jhlou/CGRVOPT/MatrixMeld/vitrartl/spec/vitra_cgra_adg.json" \
  --op-file="/home/jhlou/CGRVOPT/MatrixMeld/vitrartl/spec/operations.json" \
  --output-type="sdk" \
  --obj-opt=true \
  --max-iters=2000 \
  opt.mlir --output="FPVecAdd_cgra.c"