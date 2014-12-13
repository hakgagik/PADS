#include "cudaTest.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

class Foo {
private:
	int x_;

public:
	__device__ Foo() { x_ = 42; }
	__device__ int bar() { return x_; }
};

__global__ void dvecTest(double *e){
	Foo f;
	e[0] = f.bar();
}

int cuMain() {
	int n;
	cudaGetDevice(&n);

	double *he = (double *)malloc(sizeof(double));
	double *de;
	cudaMalloc(&de, sizeof(double));

	dvecTest<<<1, 1>>>(de);

	cudaMemcpy(he, de, sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << *he << std::endl;
	return n;
}