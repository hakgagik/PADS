#include "cudaTest.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <math.h>
#include "deviceVector.cuh"


__global__ void dvecTest(double *e){
	//double *a;
	//*a = 1;
	//dvec v(a, 1);
	//*e = v[0];

	double a[3];
	a[0] = 1; a[1] = 2; a[2] = 3;
	dvec d(a, 3);

	*e = d.norm();
}



int cuMain() {
	int n = 1;
	cudaSetDevice(1);

	double *he = (double *)malloc(sizeof(double));
	double *de;
	cudaMalloc(&de, sizeof(double));

	dvecTest<<<1, 1>>>(de);

	cudaMemcpy(he, de, sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << *he << std::endl;
	return n;
}