#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "mainIterator.cuh"

// Each molecule computes its own centroid and puts it into dcentroids
// This is done via a parallel reduction algorithm that calculates a sum of n elements in O(log(n)) time.
// Currently, this only works for n-alkanes where n is a power of 2
__global__ void getCentroids(double*x, double*y, double*z, double *dcentroidsx, double *dcentroidsy, double *dcentroidsz, int nBeads){
	int i = blockIdx.x * nBeads + threadIdx.x;
	int level = 1;

	// Declare a shared variable to be used to calculate the centroid before copying into global memory.
	extern __shared__ double c[];
	double *cx = c;
	double *cy = &(c[nBeads]);
	double *cz = &(c[2 * nBeads]);

	// Each bead copies its own info into the shared memory space.
	cx[threadIdx.x] = x[i];
	cy[threadIdx.x] = y[i];
	cz[threadIdx.x] = z[i];

	__syncthreads();

	// Do a parallel reduction to find the sum of all the positions.
	while (level < nBeads){
		if (threadIdx.x % (2*level) == 0){
			cx[threadIdx.x] += cx[threadIdx.x + level];
			cy[threadIdx.x] += cy[threadIdx.x + level];
			cz[threadIdx.x] += cz[threadIdx.x + level];
		}
		level *= 2;
	}
};

// Each thread block contains nBeads threads. There are nMols thread blocks. Therefore, each molecule is its own thread block.
// We begin by generating the Verlet list. To do this, each molecule copies 
__global__ void getVerletLisT(int*verletList, int*verletListStart, int*verletListEnd, double*x, double*y, double*z, double cutoff){
	
}


int cuMainLoop(double *x, double *y, double *z, int nMols, int nBeads){

	double *dx, *dy, *dz;

	double *dcentroidsx, *dcentroidsy, *dcentroidsz;

	int* verletList;
	int* verletListStart;
	int* verletListEnd;

	cudaMalloc(&verletList, sizeof(int) * nMols * 100);
	cudaMalloc(&verletListStart, sizeof(int) * nMols);
	cudaMalloc(&verletListEnd, sizeof(int) * nMols);

	cudaMalloc(&dcentroidsx, sizeof(double)*nMols);
	cudaMalloc(&dcentroidsy, sizeof(double)*nMols);
	cudaMalloc(&dcentroidsz, sizeof(double)*nMols);

	cudaMalloc(&dx, sizeof(double) * nBeads * nMols);
	cudaMalloc(&dy, sizeof(double) * nBeads * nMols);
	cudaMalloc(&dz, sizeof(double) * nBeads * nMols);

	cudaMemcpy(&dx, &x, nMols * nBeads * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(&dy, &y, nMols * nBeads * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(&dz, &z, nMols * nBeads * sizeof(double), cudaMemcpyHostToDevice);

	getCentroids<<<nMols, nBeads, 3 * nBeads * sizeof(double)>>>(dx, dy, dz, dcentroidsx, dcentroidsy, dcentroidsz, nBeads);

	double *eCentroidsx = (double *)malloc(sizeof(double)*nMols);
	double *eCentroidsy = (double *)malloc(sizeof(double)*nMols);
	double *eCentroidsz = (double *)malloc(sizeof(double)*nMols);

	cudaMemcpy(&eCentroidsx, &dcentroidsx, sizeof(double)*nMols, cudaMemcpyDeviceToHost);
	cudaMemcpy(&eCentroidsy, &dcentroidsy, sizeof(double)*nMols, cudaMemcpyDeviceToHost);
	cudaMemcpy(&eCentroidsz, &dcentroidsz, sizeof(double)*nMols, cudaMemcpyDeviceToHost);
	return EXIT_SUCCESS;
}