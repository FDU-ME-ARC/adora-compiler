for i: 0 to NI{
 for j: 0 to NJ{
  C[i][j]*=beta;}
 for k : 0 to NK {
  for j: 0 to NJ{
	 C[i][j]+=alpha*A[i][k]*B[k][j];
  } }
}

* 1 fission and then reorder
for(i = 0; i < NI; i++) {
  for(j = 0; j < NJ; j++){
	  C[i][j] *= beta;
} }
for(i = 0; i < NI; i++) {
  for(k = 0; k < NK; k++) {
    for(j = 0; j < NJ; j++){
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
	  C[i][j] *= bet ;}
  for (j = 0; j < NJ; j++){
    for (k = 0; k < NK; k++) {
	    C[i][j] += alpha * A[i][k] * B[k][j];
    } }
}

for (i = 0; i < NI; i++) {
  #pragma __map_this_for__
  for (j = 0; j < NJ; j++){
	  C[i][j] *= beta;
    for (k = 0; k < NK; k++) {
	    C[i][j] += alpha * A[i][k] * B[k][j];
    } }
}