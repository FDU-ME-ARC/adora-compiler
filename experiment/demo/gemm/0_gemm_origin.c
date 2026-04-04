for (i = 0; i < NI; i++) {
  for (j = 0; j < NJ; j++){
	  C[i][j] *= beta;}
  for (k = 0; k < NK; k++) {
    for (j = 0; j < NJ; j++){
	    C[i][j] += alpha * A[i][k] * B[k][j];
    } }
}

* 1 fission and then reorder
for (i = 0; i < NI; i++) {
  for (j = 0; j < NJ; j++){
	  C[i][j] *= beta;
} }
for{
  for (k = 0; k < NK; k++) {
    for (j = 0; j < NJ; j++){
	    C[i][j] += alpha * A[i][k] * B[k][j];
} } }


for (i = 0; i < NI; i++) {
  for (j = 0; j < NJ; j++){
	  C[i][j] *= beta;
} }
for (i = 0; i < NI; i++){
  for (k = 0; k < NK; k++) {
    for (j = 0; j < NJ; j++){
	    C[i][j] += alpha * A[i][k] * B[k][j];
} } }


* 2 reorder and then fusion
for (i = 0; i < NI; i++) {
  for (j = 0; j < NJ; j++){
	  C[i][j] *= beta;}
  for (j = 0; j < NJ; j++){
    for (k = 0; k < NK; k++) {
	    C[i][j] += alpha * A[i][k] * B[k][j];
    } }
}

for (i = 0; i < NI; i++) {
  for (j = 0; j < NJ; j++){
	  C[i][j] *= beta;
    for (k = 0; k < NK; k++) {
	    C[i][j] += alpha * A[i][k] * B[k][j];
    } }
}