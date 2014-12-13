#include "cudaTest.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <armadillo>
#include <cstdio>

__global__ void armaTest(double *e){
	arma::mat a(3, 3);
	arma::mat b(3, 3);
	a.fill(3);
	b.fill(3);
	arma::mat c = a * b;
	*e = c(1, 1);
}

int cuMain() {
	int n;
	cudaGetDevice(&n);
	return n;

	double *he = (double *)malloc(sizeof(double));
	double *de;
	cudaMalloc(&de, sizeof(double));
	cudaMemcpy(he, de, sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << *he << std::endl;
}