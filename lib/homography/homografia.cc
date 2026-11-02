
/*
 * calcula las funciones necesarias para calcular las matrices
 * K  R y t que del modelo de la cámara obscura, dadas las esquinas de
 * un cuadrado, indicadas en una imágen.
 * */
#include "homography/homografia.h"

#include <math.h>

#include "homography/eigen.h"
#include "homography/matrices.h"

/*
 * Normaliza los datos de entrada
 * */
void normalizar(double S[][2], float *E, double vd[4], int n) {
  double sum1, sum2, prom, var, sig;
  int i, k;

  // normalizamos x
  sum1 = 0.0;
  sum2 = 0.0;
  i = 0;
  while (i < n) {
    k=2*i;
    sum1 += E[k];
    sum2 += E[k] * E[k];
    i++;
  }
  prom = sum1 / n;
  var = sum2 / n - prom * prom;
  sig = sqrt(var);
  vd[0] = prom;
  vd[1] = sig;

  i = 0;
  while (i < n) {
    S[i][0] = (E[2*i] - prom) / sig;
    i++;
  }

  // normalizamos y
  sum1 = 0.0;
  sum2 = 0.0;
  i = 0;
  while (i < n) {
    k=2*i+1;
    sum1 += E[k];
    sum2 += E[k] * E[k];
    i++;
  }

  prom = sum1 / n;
  var = sum2 / n - prom * prom;
  sig = sqrt(var);
  vd[2] = prom;
  vd[3] = sig;

  i = 0;
  while (i < n) {
    S[i][1] = (E[2*i+1] - prom) / sig;
    i++;
  }
}

/*
 * Calcula la matriz de rotación haciendo la descomposición
 *  SVD de la matríz de entrada (a través de la eigendescomposición)
 * */
void CalculaR(double A[][3], double R[][3]) {
  int i, j;
  double S[3][3];
  double U[3][3], V[3][3];
  double roots[3];

  multiplicaATB(A, A, S);
  dsyevv3(S, V, roots);
  for (i = 0; i < 3; i++) roots[i] = 1.0 / sqrt(roots[i]);

  multiplicaAB(A, V, U);
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++) U[i][j] *= roots[j];

  multiplicaABT(U, V, R);
}

/*
 * Calcula la matríz R y t dada una matríz H
 * */
int parametros_de_H(double H[3][3], float MK[16], float MRT[16], double Hi,
                    double Wi) {
  double u0, v0, w33, f2, f;
  double l1, l2, l;
  double A[3][3];
  double A2[3][3];
  double Rp[3][3];
  double R[3][3];

  double RT[3][3];
  double RN[3][3];
  double Kinv[3][3];
  double vt[3], vc[3];
  double theta1, theta2, theta3;

  int i;

  u0 = -Wi / 2.0;
  v0 = -Hi / 2.0;
  //Obtenido de la calibración de cámara Tello
  /*
  K = np.array([[614.113801, 0, -306.602903], 
            [ 0, 612.678918, -234.158867], 
            [0, 0, -1]])
  */

  Kinv[0][0] = 0.0016283626;
  Kinv[0][1] = 0.0;
  Kinv[0][2] = -0.499260727;
  Kinv[1][0] = 0.0;
  Kinv[1][1] = 0.0016321762;
  Kinv[1][2] = -0.382188380;
  Kinv[2][0] = 0.0;
  Kinv[2][1] = 0.0;
  Kinv[2][2] = -1.0;

  multiplicaAB(Kinv, H, A);

  l1 = Norma(A[0][0], A[1][0], A[2][0]);
  l2 = Norma(A[0][1], A[1][1], A[2][1]);
  l = (l1 + l2) / 2.0;

  Mult_Escalar((1.0 / l), A, A2);
  vt[0] = A2[0][2];
  vt[1] = A2[1][2];
  vt[2] = A2[2][2];
  CopiaMatriz(A2, Rp);

  Rp[0][2] = Rp[1][0] * Rp[2][1] - Rp[1][1] * Rp[2][0];
  Rp[1][2] = Rp[2][0] * Rp[0][1] - Rp[2][1] * Rp[0][0];
  Rp[2][2] = Rp[0][0] * Rp[1][1] - Rp[0][1] * Rp[1][0];

  CalculaR(Rp, R);

  //CopiaMatriz(Rp, R);
  // Cálculo de theta3, theta2, theta1
  theta1 = atan2(R[2][1], -R[2][0]);
  theta1 *= 180.0 / M_PI;

  theta2 = acos(R[2][2]);
  theta2 *= 180.0 / M_PI;

  theta3 = atan2(R[1][2], R[0][2]);
  theta3 *= 180.0 / M_PI;

  // printf( "(theta3, theta2, theta1) = %lf, %lf, %lf\n", theta3, theta2,
  // theta1 );

  TranspuestaMatriz(R, RT);
  Mult_Escalar(-1, RT, RN);
  Mult_Mat_Vector(RN, vt, vc);

  // Inicialización
  for (i = 0; i < 16; i++) {
    MK[i] = 0.0;
    MRT[i] = 0.0;
  }

  MK[0] = 614.113801;
  MK[5] = 612.678918;
  MK[2] = -306.602903;
  MK[6] = -234.158867;
  MK[10] = 801.0;  //**********************************************************//
  MK[11] = 800;
  MK[14] = -1.0;  //**********************************************************//

  MRT[0] = R[0][0];
  MRT[1] = R[0][1];
  MRT[2] = R[0][2];
  MRT[3] = vt[0];
  MRT[4] = R[1][0];
  MRT[5] = R[1][1];
  MRT[6] = R[1][2];
  MRT[7] = vt[1];
  MRT[8] = R[2][0];
  MRT[9] = R[2][1];
  MRT[10] = R[2][2];
  MRT[11] = vt[2];
  MRT[12] = 0.0;
  MRT[13] = 0.0;
  MRT[14] = 0.0;
  MRT[15] = 1.0;
  return 0;
}

/*
 * Calcula la matríz H (homografía) dado un conjunto de 4
 * puntos de entrada, teniendo como salida la matriz K, R y t.
 * */
int homografia(double Pnor[][2], double vdP[4], float *p, float MK[16], float MRT[16],
               double height, double width) {
  int i, j;

  /*Valores en el modelo */
  double pnor[N][2];
  //double Pnor[N][2];
  double vdp[4];
  //mean and std of the first marker, the other models were calculated with this
  //double vdP[4] = {4.88235294, 4.12968983, 3.80392157, 4.43778817};
  //double vdP[4];
  double R[N * 2][9];
  double A[N * 2][9] = {{0}};
  double suma;
  double vh[8] = {0};
  double Hp[3][3] = {{0}};
  double H[3][3] = {{0}};
  double mx, s_x, my, s_y;
  double mu, s_u, mv, s_v;

  normalizar(pnor, p, vdp, N);

  i = 0;

  while (i < N) {
    j = 2 * i;
    A[j][0] = Pnor[i][0];
    A[j][1] = Pnor[i][1];
    A[j][2] = 1.0;
    A[j][6] = -pnor[i][0] * Pnor[i][0];
    A[j][7] = -pnor[i][0] * Pnor[i][1];
    A[j][8] = pnor[i][0];

    j++;
    A[j][3] = Pnor[i][0];
    A[j][4] = Pnor[i][1];
    A[j][5] = 1.0;
    A[j][6] = -pnor[i][1] * Pnor[i][0];
    A[j][7] = -pnor[i][1] * Pnor[i][1];
    A[j][8] = pnor[i][1];

    i++;
  }

  // Tengo que hacer la descomposición QR como en el paper:
  QR_noQ(A, R);

  /* Backsustitution */
  vh[7] = R[7][8] / R[7][7];

  for (i = 6; i >= 0; i--) {
    suma = 0.0;
    for (j = i + 1; j < 8; j++) suma += R[i][j] * vh[j];

    vh[i] = (R[i][8] - suma) / R[i][i];
  }

  // Aquí se tiene H normalizada

  Hp[0][0] = vh[0];
  Hp[0][1] = vh[1];
  Hp[0][2] = vh[2];
  Hp[1][0] = vh[3];
  Hp[1][1] = vh[4];
  Hp[1][2] = vh[5];
  Hp[2][0] = vh[6];
  Hp[2][1] = vh[7];
  Hp[2][2] = 1.0;

  mx = vdP[0];
  s_x = vdP[1];
  my = vdP[2];
  s_y = vdP[3];
  i = 0;

  while (i < 3) {
    H[i][0] = Hp[i][0] / s_x;
    H[i][1] = Hp[i][1] / s_y;
    H[i][2] = Hp[i][2] - mx * Hp[i][0] / s_x - my * Hp[i][1] / s_y;
    i++;
  }

  mu = vdp[0];
  s_u = vdp[1];
  mv = vdp[2];
  s_v = vdp[3];
  Hp[0][0] = (H[0][0] * s_u) + (mu * H[2][0]);
  Hp[1][0] = (H[1][0] * s_v) + (mv * H[2][0]);

  Hp[0][1] = (H[0][1] * s_u) + (mu * H[2][1]);
  Hp[1][1] = (H[1][1] * s_v) + (mv * H[2][1]);

  Hp[0][2] = (H[0][2] * s_u) + (mu * H[2][2]);
  Hp[1][2] = (H[1][2] * s_v) + (mv * H[2][2]);

  i = 0;
  while (i < 3) {
    H[0][i] = Hp[0][i];
    H[1][i] = Hp[1][i];
    i++;
  }
  /*
  printf("H:");
  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      printf("%f ", H[i][j]);
    }
    printf("\n");
  }
  */

  i = parametros_de_H(H, MK, MRT, height, width);

  return i;
}
