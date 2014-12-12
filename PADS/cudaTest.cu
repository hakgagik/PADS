#include "cudaTest.cuh"
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

int cuMain() {
	int n;
	cudaGetDevice(&n);
	return n;
}