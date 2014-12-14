#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "mainIterator.cuh"
#include <iostream>
#include <fstream>
#include <iomanip>

// Each thread block contains nBeads threads. There are nMols thread blocks. Therefore, each molecule is its own thread block.
// Each molecule computes its own centroid and puts it into dcentroids
// This is done via a parallel reduction algorithm that calculates a sum of n elements in O(log(n)) time.
// Currently, this only works for n-alkanes where n is a power of 2
__global__ void getCentroids(double*x, double*y, double*z, double *dcentroidsx, double *dcentroidsy, double *dcentroidsz){
	int nBeads = blockDim.x;
	int i = blockIdx.x * nBeads + threadIdx.x;
	int level = 1;

	// Declare a shared variable to be used to calculate the centroid before copying into global memory.
	extern __shared__ double c[];
	double *cx = c;
	double *cy = &(c[nBeads]);
	double *cz = &(c[2 * nBeads]);

	// Each bead copies its own info into the shared memory space.
	// Global memory bad!
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
	// Thread 0 then copies its own cx cy and cz back into global memory.
	if (threadIdx.x == 0){
		dcentroidsx[blockIdx.x] = cx[0] / nBeads;
		dcentroidsy[blockIdx.x] = cy[0] / nBeads;
		dcentroidsz[blockIdx.x] = cz[0] / nBeads;
	}
}

// We next, we generate the Verlet list. This massively speeds up computation time.
// Now, each thread block has only one thread (representing one molecule). Each thread block will calculate its own verlet cell.
// This and the centroid calculation happen once every *very many* iterations because the verlet list is not very likely to change
// and centroids are only used to calculate the verlet list.

__global__ void getVerletList(int*verletList, int *verletListEnd, double*xCentroids, double*yCentroids, double*zCentroids, double *cutoff, int *verletStride, int *nMols){
	// Copy own centroid into local memory
	double ctfsq = cutoff[0];
	ctfsq *= ctfsq;
	int mols = nMols[0];
	int stride = verletStride[0];
	int idx = blockIdx.x;
	double c[3];
	double dx[3];
	c[0] = xCentroids[idx];
	c[1] = yCentroids[idx];
	c[2] = zCentroids[idx];

	int verletCount = -1;
	for (int i = 0; i < mols; i++){
		int j = i % mols;
		if (j != idx){
			dx[0] = xCentroids[j] - c[0];
			dx[1] = yCentroids[j] - c[1];
			dx[2] = zCentroids[j] - c[2];
			if (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] < ctfsq){
				verletCount++;
				verletList[stride * idx + verletCount] = j;
			}
		}
	}
	if (verletCount >0){
		verletListEnd[idx] = stride * idx + verletCount;
	}
	else {
		verletListEnd[idx] = -1;
	}
}


int cuMainLoop(double *x, double *y, double *z, int nMols, int nBeads){
	
	cudaSetDevice(1);

	// Define constants
	int *everletStride = new int;
	int *emols = new int;
	double *ecutoff = new double;

	*everletStride = 100;
	*emols = nMols;
	*ecutoff = 12.0;

	int *dverletStride;
	int *dmols;
	double *dcutoff; 

	cudaMalloc(&dverletStride, sizeof(int));
	cudaMalloc(&dmols, sizeof(int));
	cudaMalloc(&dcutoff, sizeof(double));

	cudaMemcpy(dverletStride, everletStride, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dmols, emols, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dcutoff, ecutoff, sizeof(double), cudaMemcpyHostToDevice);

	//End constant definition.

	double *dx, *dy, *dz;

	double *dcentroidsx, *dcentroidsy, *dcentroidsz;

	cudaMalloc(&dcentroidsx, sizeof(double)*nMols);
	cudaMalloc(&dcentroidsy, sizeof(double)*nMols);
	cudaMalloc(&dcentroidsz, sizeof(double)*nMols);

	cudaMalloc(&dx, sizeof(double) * nBeads * nMols);
	cudaMalloc(&dy, sizeof(double) * nBeads * nMols);
	cudaMalloc(&dz, sizeof(double) * nBeads * nMols);

	cudaMemcpy(dx, x, nMols * nBeads * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dy, y, nMols * nBeads * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dz, z, nMols * nBeads * sizeof(double), cudaMemcpyHostToDevice);

	getCentroids<<<nMols, nBeads, 3 * nBeads * sizeof(double)>>>(dx, dy, dz, dcentroidsx, dcentroidsy, dcentroidsz);

	int* verletList;
	int* verletListEnd;


	cudaMalloc(&verletList, sizeof(int) * nMols * (*everletStride));
	cudaMalloc(&verletListEnd, sizeof(int) * nMols);

	getVerletList<<<nMols,1>>>(verletList, verletListEnd, dcentroidsx, dcentroidsy, dcentroidsz, dcutoff, dverletStride, dmols);

	int *eVerletList = new int[nMols* *everletStride];
	int *eVerletListEnd = new int[nMols];
	cudaMemcpy(eVerletList, verletList, nMols* *everletStride * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(eVerletListEnd, verletListEnd, sizeof(int)*nMols, cudaMemcpyDeviceToHost);

	std::ofstream verletOut("verlet.dat");
	std::ofstream verletEndOut("verletEnd.dat");

	for (int i = 0; i < nMols; i++){
		for (int j = 0; j < 100; j++){
			verletOut << std::setw(15) << eVerletList[i * 100 + j];
		}
		verletOut << std::endl;
		verletEndOut << std::setw(15) << eVerletListEnd[i] << std::endl;
	}

	return EXIT_SUCCESS;
}