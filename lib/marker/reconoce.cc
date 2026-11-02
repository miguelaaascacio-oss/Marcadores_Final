#include <stdio.h>
#include <math.h>
#include "marker/triangles.h"
#include "marker/follow.h"
#include "marker/lambdaMatrix.h"
// #include "fill.h"
#include "opencv2/imgcodecs.hpp"

#define ESQUINAS 7
#define CENTROY  6
#define CENTROX  5
#define AREA     4
#define FINY     3
#define FINX     2
#define INICIOY  1
#define INICIOX  0

using namespace cv;

void dilation_4neig_inplace_obj( Mat& I, int *v )
{
	int i, j;
	uchar *ptrin;
	uchar *pimg1, *pimg2;
	int val;

	for( i=v[INICIOY]; i<=v[FINY]; i++ ) {
		ptrin = I.ptr<uchar>(i) + v[INICIOX];
		pimg1 = I.ptr<uchar>(i-1) + v[INICIOX];
		pimg2 = I.ptr<uchar>(i+1) + v[INICIOX];
		for( j=v[INICIOX]; j<=v[FINX]; j++ ) {
			if ( *ptrin == 0 ) {
				val = 0;
				if ( *pimg1 == 255 ) val = 255;
				if ( *pimg2 == 255 ) val = 255;
				if ( ptrin[-1] == 255 ) val = 255;
				if ( ptrin[ 1] == 255 ) val = 255;

				if ( val == 255 )
					*ptrin = 128;
			}
			ptrin++;
			pimg1++;
			pimg2++;
		}
	}
}


void quita_tiritas( Mat& I, int *v, uchar tono )
{
	int i, j, k, l;
	uchar *ptrin, *ptraux;
	int suma;

	for( i=v[INICIOY]; i<v[FINY]; i++ ) {
		ptrin = I.ptr<uchar>(i) + v[INICIOX] + 1;
		for( j=v[INICIOX]; j<v[FINX]; j++ ) {
			if ( *ptrin == tono ) {
				suma = 0;
				for ( k = -1 ; k <= 1; k++ ) {
					ptraux = I.ptr<uchar>( i+k );
					ptraux +=  j - 1;
					for ( l = 0 ; l < 3; l++ ){
						suma += *ptraux++;
					}
				}
				// Si es un punto aislado o tiene un solo
				// vecino, se borra
				if ( suma == 255 || suma == 510 )
					*ptrin = 0;
			}
			ptrin++;
		}
	}
}

void quita_esquinas( Mat& I, int *v, uchar tono )
{
	int i, j;
	uchar *ptrin, *ptr1, *ptr2;

	for( i=v[INICIOY]; i<v[FINY]; i++ ) {
		ptr1  = I.ptr<uchar>(i-1) + v[INICIOX] + 1;
		ptrin = I.ptr<uchar>(i) + v[INICIOX] + 1;
		ptr2  = I.ptr<uchar>(i+1) + v[INICIOX] + 1;
		for( j=v[INICIOX]; j<v[FINX]; j++ ) {
			if ( *ptrin == tono ) {
				if( *ptr1 == tono && *ptr2 == 0 ){
					if ( ptrin[-1] == tono && ptrin[1] == 0 )
						*ptrin = 0; 
					if ( ptrin[-1] == 0 && ptrin[1] == tono )
						*ptrin = 0; 

				}
				else if ( *ptr1 == 0 && *ptr2 == tono ){
					if ( ptrin[-1] == tono && ptrin[1] == 0 )
						*ptrin = 0; 
					if ( ptrin[-1] == 0 && ptrin[1] == tono )
						*ptrin = 0; 
				} 
			}
			ptrin++;
			ptr1++;
			ptr2++;
		}
	}
}

int extrae_perimetro( Mat& Img, int *vals, POINT *pts )
{
	#define PERIMETRO 128
	int i, j, x, y, n;
	int x2;
	uchar *pimg;

	x = y = -1;

	for( i=vals[INICIOY]; i<=vals[FINY]; i++ ) {
		pimg = Img.ptr<uchar>(i) + vals[INICIOX];
		for( j=vals[INICIOX]; j<=vals[FINX]; j++ ) {
			if ( *pimg == PERIMETRO ) {
				x = j;
				y = i;
				goto SIGUE;
			}
			pimg++;
		}
	}
	SIGUE:
	if ( x == -1 ) {
		// fprintf( stderr, "# No hay puntos de perímetro\n" );
		return -3;
	}
	/** Buscamos en el tramo "de arriba" **/
	x2 = x - 1;
	pimg = Img.ptr<uchar>(y+1) + x2;
	if( *pimg == PERIMETRO ) {
		while ( *pimg == PERIMETRO ) {
			x2--;
			pimg--;
		}
		x2++;
	}
	if( x-x2 > 5 ) {
		pimg = Img.ptr<uchar>(y) + x;
		while ( *pimg == PERIMETRO ) {
			x++;
			pimg++;
		}
		x--;
	}
	n = follow_perimeter( Img, y, x, PERIMETRO, pts );
	if( n > 3000 ) {
		// fprintf( stderr, "Hay más de 3000 puntos en el perímetro\n" );
		// fprintf( stderr, "n = %d\n", n );
		return -4;
	}
	return n;
}

int checa_linea( POINT *pts, int index1, int index2, float piso = 2.5)
{
	int i, dx, dy;
	int distance, indice, val;
	POINT pw, pv;
	double dd;

	if ( index1 >= index2 )
		return -1;

	dx = pts[index2].x - pts[index1].x;
	dy = pts[index2].y - pts[index1].y;
	pw.x = -dy;
	pw.y = dx;

	/** Se busca el punto más alejado de la línea **/
	distance = 0;
	for( i=index1; i<=index2; i++ ){
		pv.x = pts[i].x - pts[index1].x;
		pv.y = pts[i].y - pts[index1].y;

		val = abs( pv.x * pw.x + pv.y * pw.y );
		if ( val > distance ){
			distance = val;
			indice = i;
		}
	}
	dd = (double)distance/sqrt( pw.x * pw.x + pw.y * pw.y );
	// printf( "# d= %lf %lf\n", dd, sqrt( dx * dx + dy * dy )/10.0 );
	// if( dd > sqrt( dx * dx + dy * dy )/75.0 )
	double L   = sqrt( (double)(dx * dx + dy * dy) );
        double tol = std::max( L / 10.0, 3.0);
	if( dd > tol )
		return( indice ); 

	return -2;
}

int analiza_perimetro_cuadrado( POINT *pts, int n, DPOINT *pdver )
{
	int i, i1, i2, i3, i4, imiddle;
	int flag1, flag2;
	int d2, distancia;
	DPOINT pmedio[4], vec[4];
	POINT pu;

	// El primer vértice por definición es el 0
	i1 = 0;
	// Buscamos el segundo vértice
	distancia = 0;
	i2 = 0;
	for( i=1; i<n; i++ ) {
		pu.x = pts[i].x - pts[i1].x;
		pu.y = pts[i].y - pts[i1].y;
		d2 = pu.x * pu.x + pu.y * pu.y;
		if( d2 > distancia ) {
			distancia = d2;
			i2 = i;
		}
	}

	// El segundo vértice está en el índice i2
	// Checamos en que caso estamos
	flag1 = flag2 = 0;
	// Para el caso 1, algunos de las banderas seguirá en 1
	// Para el caso 2, las dos banderas deben de valer 0
	// Prueba 1: De i1 a i2
	// Prueba 2: De i2+1 a n-1
	// Prueba 1:
	i3 = checa_linea( pts, i1, i2-1 );
	if( i3 == -1 )
		return -10;
	else if ( i3 < -1 )
		flag1 = 1;

	// Prueba 2:
	i4 = checa_linea( pts, i2, n-1 );
	if( i4 == -1  )
		return -20;
	else if ( i4 < -1 )
		flag2 = 1;

	// printf("# flags: %d %d\n", flag1, flag2 );
	
	if ( flag1 == 1 || flag2 == 1 ) { // Caso 1
		// printf("# caso 1\n" );
		if( flag2 == 1 ) {
			imiddle = i2/2;
			i4 = i2;
			i3 = checa_linea( pts, imiddle, i4 );
			i2 = checa_linea( pts, i1, imiddle-1 );
		}
		else {
			// i2 + 1 + n - 1 = i2 + n
			imiddle = ( i2+n )/2;
			i3 = checa_linea( pts, i2, imiddle-1 );
			i4 = checa_linea( pts, imiddle, n-1 );
		}

		// Buscamos el punto más alejado de la linea i2 e imiddle.

		// If flag1 = 0 A fuerzas i1 < i2 < i3 < i4
		// If flag2 = 0 A fuerzas i1 < i2 < i3 < i4
	}
	else {  // Caso 2. Ya está resuelto
		pu.x = i3;
		i3 = i2;
		i2 = pu.x;
		// Ahora afuerzas tenemos i1 < i2 < i3 < i4
	}
	// printf( "# i1: %d %d %d\n", i1, pts[i1].x, pts[i1].y );
	// printf( "# i2: %d %d %d\n", i2, pts[i2].x, pts[i2].y );
	// printf( "# i3: %d %d %d\n", i3, pts[i3].x, pts[i3].y );
	// printf( "# i4: %d %d %d\n", i4, pts[i4].x, pts[i4].y );
	// Aquí se puede verificar si es un cuadrado.
	// Las líneas que unen los cuatro vértices deben ser líneas
	// ¿Entonces un punto que está lejos un valor "umbral" de una
	// línea, hace esa línea inválida?
	// Si, lo intentamos;
	
	const double piso = std::min( 8.0, std::max( 3.0, 0.02 * n ) ); //Tenemos un umbral minimo
	if ( checa_linea( pts, i1, i2-1, piso) > 0 ){
		// printf("# e1\n");
		return -1;
	}
	if ( checa_linea( pts, i2, i3-1, piso) > 0 ){
		// printf("# e2\n");
		return -2; 
	}
	if ( checa_linea( pts, i3, i4-1, piso) > 0 ){
		// printf("# e3\n");
		return -3; 
	}
	if ( checa_linea( pts, i4, n-1, piso) > 0 ){
		// printf("# e4\n");
		return -4; 
	}

	getVector( pts, i1, i2-1, &pmedio[0], &vec[0] );
	getVector( pts, i2, i3-1, &pmedio[1], &vec[1] );
	getVector( pts, i3, i4-1, &pmedio[2], &vec[2] );
	getVector( pts, i4, n-1,  &pmedio[3], &vec[3] );

	if( intersecs( &pmedio[3], &vec[3], &pmedio[0], &vec[0], &pdver[0] ) != 0 )
		return -5;
	if( intersecs( &pmedio[0], &vec[0], &pmedio[1], &vec[1], &pdver[1] ) != 0 )
		return -6;
	if( intersecs( &pmedio[1], &vec[1], &pmedio[2], &vec[2], &pdver[2] ) != 0 )
		return -7;
	if( intersecs( &pmedio[2], &vec[2], &pmedio[3], &vec[3], &pdver[3] ) != 0 )
		return -8;

	// printf( "#v1: %lf %lf\n", pdver[0].x, pdver[0].y );
	// printf( "#v2: %lf %lf\n", pdver[1].x, pdver[1].y );
	// printf( "#v3: %lf %lf\n", pdver[2].x, pdver[2].y );
	// printf( "#v4: %lf %lf\n", pdver[3].x, pdver[3].y );

	return 0;
}

int find_point_local_object( Mat& IN, int *v, int *x, int *y )
{
	int i, j;
	uchar *p;

	for ( i=v[INICIOY]; i<= v[FINY]; i++ ) {
		p = IN.ptr<uchar>(i) + v[INICIOX];
		for ( j=v[INICIOX]; j<=v[FINX]; j++ ) {
			if ( *p == 0 ) {
				*x = j;
				*y = i;
				return 0; /** Sucess! **/
			}
			p++;
		}
	}
	return 1; /** Fault **/
}


int reconoce( cv::Mat& Img, int borde[4], float *retver, int num_comp[2], int invert )
{
	// int kobjs=0;
	// char nombre[80];
	int i,j;
	int n;
    POINT iver[5];
    DPOINT dver[7];
    int indexes[5];
    POINT ver[5]; //era un 4
    static POINT per1[3000];
	int cx, cy, d;
	int vx, vy, v;
	// int num_comp[2];
	quita_tiritas( Img, borde, 255 );
	// Calcula el perímetro
	// imwrite( "1.png", Img );
	dilation_4neig_inplace_obj( Img, borde );
	// imwrite( "2.png", Img );

	quita_esquinas( Img, borde, 128 );

	n = extrae_perimetro( Img, borde, per1 );
	// imwrite( "3.png", Img );
	// fprintf( stderr, "Extraje perímetro %d\n", n );

	if ( n < 20 )
		return 1;

	if ( analiza_perimetro_cuadrado( per1, n, dver ) != 0 )
		return 2;

	// printf( "# Encontré cuadrado!\n" );
	n = extrae_perimetro( Img, borde, per1 );
	// fprintf( stderr, "# Extraje perímetro 2 %d\n", n );
	// imwrite( "4.png", Img );
	if ( n <= 0 ) {
		return 3;
	};

	if( extrae_triangulo( per1, n, ver, &dver[4] ) < 0 ) {
		// fprintf( stderr, "# Mal triangulo\n");
		return 4;
	}
	// printf( "#v5: %lf %lf\n", dver[4].x, dver[4].y );
	// printf( "#v6: %lf %lf\n", dver[5].x, dver[5].y );
	// printf( "#v7: %lf %lf\n", dver[6].x, dver[6].y );
	// Quitamos el vértice del triángulo más cerca al centro
	// del cuadrado
	//El pixel central
	cx = (int)((dver[0].x + dver[2].x)/2.0 + 0.5 );
	cy = (int)((dver[0].y + dver[2].y)/2.0 + 0.5 );
	d = 100000;
	n = -1;
	for( i=0; i<3; i++ ){
		vx = (int)dver[4+i].x - cx;
		vy = (int)dver[4+i].y - cy;
		v = vx*vx + vy*vy;
		if( v < d ){
			d = v;
			n = i;
		}
	}
		// n es el vértice que se tiene que quitar
	for (i = 0, j = 4; i < 3; i++){
		if(i==n){
			continue;
		}
		dver[j].x = dver[4+i].x;
		dver[j].y = dver[4+i].y;
		j++;
	}

	// Here I have all the 6 points of the marker
	for( i=0; i<6; i++ ){
		iver[i].x = (int)( dver[i].x + 0.5 );
		iver[i].y = (int)( dver[i].y + 0.5 );
		//printf("%d %d\n",iver[i].x, iver[i].y );
		// iver[i].y = (int)( Img.rows - 1 - dver[i].y + 0.5 );
	}
	//Cambio de coordenadas en caso de que la imagen no este invertida

	if(invert == 0){
		for( i=0; i<6; i++ ){
			iver[i].y = 479 - iver[i].y;
			dver[i].y = 479 - dver[i].y;
		}
	}
	//Cambio de coordenadas en caso de que la imagen no este invertida

	lambdaMatrix( iver, indexes,  num_comp );
	//Revisar si ya hay puntos guardados para guardar en otro
	// if retver[i][0][0] == NULL
	for(i=0; i<6; i++) {
		retver[2*i] = dver[ indexes[i] ].x;
		retver[2*i+1] = dver[ indexes[i] ].y;
		//printf("%d %d\n",(int) retver[2*i],(int) retver[2*i+1] );
	} 

	return( 0 );
}

