#include "cudaTest.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include "deviceVector.cuh"

__global__ void dvecTest(double *e){
	double a[] = { 1.0, 2.0, 3.0 };
	int nElem = 3;

	dvec v(a, nElem);
	v += v;
	*e = v[1];
}

int cuMain() {
	int n;
	cudaGetDevice(&n);
	return n;

	double *he = (double *)malloc(sizeof(double));
	double *de;
	cudaMalloc(&de, sizeof(double));

	dvecTest<<<1, 1>>>(de);

	cudaMemcpy(he, de, sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << *he << std::endl;
}