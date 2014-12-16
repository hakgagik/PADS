#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "mainIterator.cuh"
#include <iostream>
#include <fstream>
#include <iomanip>

// Define statements are a quick and dirty way of getting global constants to the device.
// If I had more style and skill, I'd use device constants. But I don't :P
#define PI 3.14159265359
#define verletStride 100
#define cutoff 1.2
#define k_l 1.46E5
#define k_th 251.04
#define k_phi1 6.78
#define k_phi2 -3.6
#define k_phi3 13.56
#define l_0 0.153
#define th_0 1.187
#define eps 0.39
#define sigma 0.401
#define ax .40214
#define ay .11031
#define az .04552
#define bx 0.0
#define by .47345
#define bz .04051
#define cx -.00002849
#define cy -.00003859
#define cz 1.09983
#define mCH2 14.0266
#define mCH3 15.0345
#define dt 0.001 // Energy is in units of Kj, mass is in units of g and distance is in units of nm, so natural time units are ps

// Simple integer power function for the LJ potential.

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
	double *mycx = c;
	double *mycy = &(c[nBeads]);
	double *mycz = &(c[2 * nBeads]);

	// Each bead copies its own info into the shared memory space.
	// Global memory bad!
	mycx[threadIdx.x] = x[i];
	mycy[threadIdx.x] = y[i];
	mycz[threadIdx.x] = z[i];

	__syncthreads();

	// Do a parallel reduction to find the sum of all the positions.
	while (level < nBeads){
		if (threadIdx.x % (2*level) == 0){
			mycx[threadIdx.x] += mycx[threadIdx.x + level];
			mycy[threadIdx.x] += mycy[threadIdx.x + level];
			mycz[threadIdx.x] += mycz[threadIdx.x + level];
		}
		level *= 2;
	}
	// Thread 0 then copies its own cx cy and cz back into global memory.
	if (threadIdx.x == 0){
		dcentroidsx[blockIdx.x] = mycx[0] / nBeads;
		dcentroidsy[blockIdx.x] = mycy[0] / nBeads;
		dcentroidsz[blockIdx.x] = mycz[0] / nBeads;
	}
}

// We next, we generate the Verlet list. This massively speeds up computation time.
// Now, each thread block has only one thread (representing one molecule). Each thread block will calculate its own verlet cell.
// This and the centroid calculation happen once every *very many* iterations because the verlet list is not very likely to change
// and centroids are only used to calculate the verlet list.
__global__ void getVerletList(int*verletList, int *verletListEnd, double*xCentroids, double*yCentroids, double*zCentroids, int nMols){
	// Copy own centroid into local memory and make local copies of all the parameters.
	double ctfsq = cutoff;
	ctfsq *= ctfsq;
	int mols = nMols;
	int stride = verletStride;
	int idx = blockIdx.x;
	double c[3];
	double dx[3];
	c[0] = xCentroids[idx];
	c[1] = yCentroids[idx];
	c[2] = zCentroids[idx];

	// Iterate forward, with 
	int verletCount = -1;
	for (int i = 1; i < mols; i++){
		int j = (i + idx) % mols;
		dx[0] = xCentroids[j] - c[0];
		dx[1] = yCentroids[j] - c[1];
		dx[2] = zCentroids[j] - c[2];
		if (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] < ctfsq){
			verletCount++;
			verletList[stride * idx + verletCount] = j;
		}
	}
	if (verletCount >0){
		verletListEnd[idx] = stride * idx + verletCount;
	}
	else {
		verletListEnd[idx] = -1;
	}
}

// This is the main MD method.
// Technically, one octane (8 threads) per block is very inefficient. I should be using threadblocks of at least 32 threads. However, this would make programming a nightmare, as I'd have to first spend time figuring out how to organize the 4 octanes into memory and constantly making sure that they don't accidentally overlap.
__global__ void MDStep(double *xGlobal, double *yGlobal, double *zGlobal, int *verletList, int * verletListEnd, double *vx, double *vy, double *vz, double *accx, double *accy, double *accz){
	// Copy constants into local memory... The caffeine in my bloodstream doesn't trust whatever's coming in through the functionc call >.>
	int i = blockIdx.x;
	int j = threadIdx.x;
	int b = blockDim.x;

	// Pushing the max shared memory limit pretty hard here =/ Max shared memory is 49152B I'm using 19584B (although I should be able to cut this by half if things behave well enough) Right now, verletStride doesn't need to be more than 30.
	extern __shared__ double sharedMem[];
	double *x = sharedMem;
	double *y = &(sharedMem[b]);
	double *z = &(sharedMem[2 * b]);
	double *r = &(sharedMem[3 * b]);
	double *theta = &(sharedMem[4 * b]);
	double *phi = &(sharedMem[5 * b]);
	double *verletX = &(sharedMem[6 * b]);
	double *verletY = &(sharedMem[(6 + verletStride) * b]);
	double *verletZ = &(sharedMem[(6 + 2 * verletStride) * b]);

	// First, copy positions into shared memory and perform first step of velocity-Verlet. Each thread copies its own position. Thread = bead.
	x[j] = xGlobal[i*b + j] + vx[i*b + j] * dt + 0.5 * accx[i*b + j] * dt * dt;
	y[j] = yGlobal[i*b + j] + vy[i*b + j] * dt + 0.5 * accy[i*b + j] * dt * dt;
	z[j] = zGlobal[i*b + j] + vz[i*b + j] * dt + 0.5 * accz[i*b + j] * dt * dt;

	xGlobal[i*b + j] = x[j];
	yGlobal[i*b + j] = y[j];
	zGlobal[i*b + j] = z[j];

	__syncthreads();

	// Next, copy positions of other molecules in the verletList into shared memory. Each thread copies data corresponding to itself.
	int vCount = 0;
	for (int idx = verletStride * i; idx <= verletListEnd[i]; idx++){
		verletX[vCount * b + j] = xGlobal[verletList[idx] * b + j];
		verletY[vCount * b + j] = yGlobal[verletList[idx] * b + j];
		verletZ[vCount * b + j] = zGlobal[verletList[idx] * b + j];
		vCount++;
	}

	__syncthreads();

	// Each molecule stores a vector to the next bead and the previous bead.
	// The third set of beads is the distance between the next bead and the bead after that (for torsion calculation).
	double dxp, dyp, dzp, dxm, dym, dzm, dxpp, dypp, dzpp;

	if (j < (b - 1)) {
		dxp = x[j + 1] - x[j];
		dyp = y[j + 1] - y[j];
		dzp = z[j + 1] - z[j];
	}
	else {
		dxp = 0;
		dyp = 0;
		dzp = 0;
	}

	if (j > 0){
		dxm = x[j - 1] - x[j];
		dym = y[j - 1] - y[j];
		dzm = z[j - 1] - z[j];
	}
	else {
		dxm = 0;
		dym = 0;
		dzm = 0;
	}

	if (j < (b - 2)) {
		dxpp = x[j + 2] - x[j + 1];
		dypp = y[j + 2] - y[j + 1];
		dzpp = z[j + 2] - z[j + 1];
	}

	// Next, calculate distances between beads. Each thread calculates the distance to the next bead.
	r[j] = sqrt(dxp * dxp + dyp * dyp + dzp * dzp);

	__syncthreads();

	// Now, calculate angles between beads. Each bead claculates the angle that has it at the origin.
	// The threads at the edges will produce nonsensical results, but we're not going to access them (in any useful manner) anyway.
	// Allowing the edge threads to calculate angles minimizes thread divergence (different theads doing different things).
	theta[j] = acos((dxp * dxm + dyp * dym + dzp * dzm) / r[j] / r[j - 1]);

	// Finally, calculate four-molecule torsion angles.
	// Same as above, this will produce some nonsensical data, but we won't be accessing it.
	phi[j] = acos((dxm * dxpp + dym * dypp + dzm * dzpp) / r[j - 1] / r[j + 1]);

	// Now, each molecule calculates a force on itself from ALL the terms. ALL OF THEM.

	double Fx = 0, Fy = 0, Fz = 0;

	// First, the spring term. Each bead receives a contribution from the bead ahead of it and from the bead behind it.
	double factor;
	if (j > 0) {
		factor = 2 * k_l * (r[j -1] - l_0) / r[j -1];
		Fx += factor * dxm;
		Fy += factor * dym;
		Fz += factor * dzm;
	}
	if (j < (b - 1)) {
		factor = 2 * k_l * (r[j] - l_0) / r[j];
		Fx += factor * dxp;
		Fy += factor * dyp;
		Fz += factor * dzp;
	}

	// Next, the theta term. A bit more complicated. Each molecule recieves a contribution from the angle behind it, the angle ahead of it, and the angle that has it as the origin.
	//if (j < (b - 2)) {
	//	factor = 2 * k_th * (th_0 - PI + theta[j + 1]) / r[j] / sin(theta[j + 1]);
	//	Fx += factor * (dxpp / r[j + 1] + cos(theta[j + 1]) * dxp / r[j]);
	//	Fy += factor * (dypp / r[j + 1] + cos(theta[j + 1]) * dyp / r[j]);
	//	Fz += factor * (dzpp / r[j + 1] + cos(theta[j + 1]) * dzp / r[j]);
	//}
	//if (j > 1) {
	//	factor = 2 * k_th * (th_0 - PI / 2.0 + theta[j - 1] / r[j - 1]) / sin(theta[j - 1]);
	//	Fx += factor * ((x[j - 2] - x[j - 1]) / r[j - 2] + cos(theta[j - 1]) * dxm / r[j - 1]);
	//	Fy += factor * ((y[j - 2] - y[j - 1]) / r[j - 2] + cos(theta[j - 1]) * dym / r[j - 1]);
	//	Fz += factor * ((z[j - 2] - z[j - 1]) / r[j - 2] + cos(theta[j - 1]) * dzm / r[j - 1]);
	//}
	//if (j > 0 && j < (b - 1)){
	//	factor = 2 * k_th * (th_0 - PI / 2.0 + theta[j]) / sin(theta[j]);
	//	Fx -= factor * (dxm / r[j - 1] - cos(theta[j]) * dxp / r[j]) / r[j]
	//		+ (dxp / r[j] - cos(theta[j]) * dxm / r[j - 1]) / r[j - 1];

	//	Fy -= factor * (dym / r[j - 1] - cos(theta[j]) * dyp / r[j]) / r[j]
	//		+ (dyp / r[j] - cos(theta[j]) * dym / r[j - 1]) / r[j - 1];

	//	Fz -= factor * (dzm / r[j - 1] - cos(theta[j]) * dzp / r[j]) / r[j]
	//		+ (dzp / r[j] - cos(theta[j]) * dzm / r[j - 1]) / r[j - 1];
	//}

	 //Next phi. Is WRONG with GIT?!
	//if (j < (b - 3)){
	//	factor = 0.5 * (k_phi1 * sin(phi[j + 1]) + 2 * k_phi2 * sin(2 * phi[j + 1]) + 3 * k_phi3 * sin(3 * phi[j + 1])) / sin(phi[j + 1]);
	//	Fx += factor * ((x[j + 3] - x[j + 2]) / r[j + 2] + cos(phi[j + 1]) * dxp / r[j]) / r[j];
	//	Fy += factor * ((y[j + 3] - y[j + 2]) / r[j + 2] + cos(phi[j + 1]) * dyp / r[j]) / r[j];
	//	Fz += factor * ((z[j + 3] - z[j + 2]) / r[j + 2] + cos(phi[j + 1]) * dzp / r[j]) / r[j];
	//}
	//if (j < (b - 2) && j > 0) {
	//	factor = 0.5 * (k_phi1 * sin(phi[j]) + 2 * k_phi2 * sin(2 * phi[j]) + 3 * k_phi3 * sin(3 * phi[j])) / sin(phi[j]);
	//	Fx -= factor * (dxpp / r[j + 1] - cos(phi[j]) * dxm / r[j - 1]) / r[j - 1];
	//	Fy -= factor * (dypp / r[j + 1] - cos(phi[j]) * dym / r[j - 1]) / r[j - 1];
	//	Fz -= factor * (dzpp / r[j + 1] - cos(phi[j]) * dzm / r[j - 1]) / r[j - 1];
	//}
	//if (j < (b - 1) && j > 1) {
	//	factor = 0.5 * (k_phi1 * sin(phi[j - 1]) + 2 * k_phi2 * sin(2 * phi[j - 1]) + 3 * k_phi3 * sin(3 * phi[j - 1])) / sin(phi[j - 1]);
	//	Fx -= factor * ((x[j - 2] - x[j - 1]) / r[j - 2] - cos(phi[j - 1]) * dxp / r[j]) / r[j];
	//	Fy -= factor * ((y[j - 2] - y[j - 1]) / r[j - 2] - cos(phi[j - 1]) * dyp / r[j]) / r[j];
	//	Fz -= factor * ((z[j - 2] - z[j - 1]) / r[j - 2] - cos(phi[j - 1]) * dzp / r[j]) / r[j];
	//}
	//if (j > 2) {
	//	factor = 0.5 * (k_phi1 * sin(phi[j - 2]) + 2 * k_phi2 * sin(2 * phi[j - 2]) + 3 * k_phi3 * sin(3 * phi[j - 2])) / sin(phi[j - 2]);
	//	Fx += factor * ((x[j - 3] - x[j - 2]) / r[j - 3] + cos(phi[j - 2]) * dxm / r[j - 1]) / r[j - 1];
	//	Fy += factor * ((y[j - 3] - y[j - 2]) / r[j - 3] + cos(phi[j - 2]) * dym / r[j - 1]) / r[j - 1];
	//	Fz += factor * ((z[j - 3] - z[j - 2]) / r[j - 3] + cos(phi[j - 2]) * dzm / r[j - 1]) / r[j - 1];
	//}
	
	// Finally, Van der Walls
	double dx, dy, dz, rsq, sdrcb;

	// Do this for everything in Verlet list
	for (int v = 0; v < vCount * b; v++){
		dx = verletX[v] - x[j];
		dy = verletY[v] - y[j];
		dz = verletZ[v] - z[j];
		rsq = dx * dx + dy * dy + dz * dz;
		sdrcb = sigma * sigma / rsq;
		sdrcb *= sdrcb * sdrcb;
		factor = 24 * eps * sdrcb * (1 - 2 * sdrcb) / rsq;
		Fx += factor * dx;
		Fy += factor * dy;
		Fz += factor * dz;
	}

	if (j == 0 || j == (b - 1)) {
		Fx /= mCH3;
		Fy /= mCH3;
		Fz /= mCH3;
	}
	else {
		Fx /= mCH2;
		Fy /= mCH2;
		Fz /= mCH2;
	}

	vx[i*b + j] += 0.5 * dt * (accx[i*b + j] + Fx);
	vy[i*b + j] += 0.5 * dt * (accy[i*b + j] + Fy);
	vz[i*b + j] += 0.5 * dt * (accz[i*b + j] + Fz);

	accx[i*b + j] = Fx;
	accy[i*b + j] = Fy;
	accz[i*b + j] = Fz;
}


int cuMainLoop(double *x, double *y, double *z, int nMols, int nBeads){
	
	cudaSetDevice(1);
	
	std::ofstream inx("initOut.dat");
	for (int i = 0; i < nBeads*nMols; i++){
		inx << std::setw(15) << x[i]
			<< std::setw(15) << y[i]
			<< std::setw(15) << z[i]
			<< std::endl;
	}

	// d in front of a variable in this functions means it's a device variable
	double *dx, *dy, *dz;
	double *dvx, *dvy, *dvz;
	double *daccx, *daccy, *daccz;

	double *dcentroidsx, *dcentroidsy, *dcentroidsz;

	cudaMalloc(&dcentroidsx, sizeof(double)*nMols);
	cudaMalloc(&dcentroidsy, sizeof(double)*nMols);
	cudaMalloc(&dcentroidsz, sizeof(double)*nMols);

	cudaMalloc(&dx, sizeof(double) * nBeads * nMols);
	cudaMalloc(&dy, sizeof(double) * nBeads * nMols);
	cudaMalloc(&dz, sizeof(double) * nBeads * nMols);

	cudaMalloc(&dvx, sizeof(double) * nBeads * nMols);
	cudaMalloc(&dvy, sizeof(double) * nBeads * nMols);
	cudaMalloc(&dvz, sizeof(double) * nBeads * nMols);

	cudaMalloc(&daccx, sizeof(double) * nBeads * nMols);
	cudaMalloc(&daccy, sizeof(double) * nBeads * nMols);
	cudaMalloc(&daccz, sizeof(double) * nBeads * nMols);

	cudaMemcpy(dx, x, nMols * nBeads * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dy, y, nMols * nBeads * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dz, z, nMols * nBeads * sizeof(double), cudaMemcpyHostToDevice);

	getCentroids<<<nMols, nBeads, 3 * nBeads * sizeof(double)>>>(dx, dy, dz, dcentroidsx, dcentroidsy, dcentroidsz);

	int* verletList;
	int* verletListEnd;


	cudaMalloc(&verletList, sizeof(int) * nMols * verletStride);
	cudaMalloc(&verletListEnd, sizeof(int) * nMols);

	
	getVerletList<<<nMols,1>>>(verletList, verletListEnd, dcentroidsx, dcentroidsy, dcentroidsz, nMols);

	// Initialize velocities and acceleratiosn
	double *evx, *evy, *evz;
	double *eaccx, *eaccy, *eaccz;

	evx = new double[nBeads * nMols];
	evy = new double[nBeads * nMols];
	evz = new double[nBeads * nMols];

	eaccx = new double[nBeads * nMols];
	eaccy = new double[nBeads * nMols];
	eaccz = new double[nBeads * nMols];

	for (int i = 0; i < nBeads*nMols; i++){
		evx[i] = 0;
		evy[i] = 0;
		evz[i] = 0;
		eaccx[i] = 0;
		eaccy[i] = 0;
		eaccz[i] = 0;
	}

	cudaMemcpy(dvx, evx, sizeof(double) * nBeads * nMols, cudaMemcpyHostToDevice);
	cudaMemcpy(dvy, evy, sizeof(double) * nBeads * nMols, cudaMemcpyHostToDevice);
	cudaMemcpy(dvz, evz, sizeof(double) * nBeads * nMols, cudaMemcpyHostToDevice);
	cudaMemcpy(daccx, eaccx, sizeof(double) * nBeads * nMols, cudaMemcpyHostToDevice);
	cudaMemcpy(daccy, eaccy, sizeof(double) * nBeads * nMols, cudaMemcpyHostToDevice);
	cudaMemcpy(daccz, eaccz, sizeof(double) * nBeads * nMols, cudaMemcpyHostToDevice);

	for (int i = 0; i < 2000; i++){
		MDStep<<<nMols, nBeads, (3 * verletStride * nBeads + 6 * nBeads) * sizeof(double)>>>(dx, dy, dz, verletList, verletListEnd, dvx, dvy, dvz, daccx, daccy, daccz);
		std::cout << i << std::endl;
		cudaDeviceSynchronize();
		//cudaMemcpy(x, dx, nMols * nBeads * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(y, dy, nMols * nBeads * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(z, dz, nMols * nBeads * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(evx, dvx, sizeof(double)* nBeads * nMols, cudaMemcpyDeviceToHost);
		//cudaMemcpy(evy, dvy, sizeof(double)* nBeads * nMols, cudaMemcpyDeviceToHost);
		//cudaMemcpy(evz, dvz, sizeof(double)* nBeads * nMols, cudaMemcpyDeviceToHost);
		//cudaMemcpy(eaccx, daccx, sizeof(double)* nBeads * nMols, cudaMemcpyDeviceToHost);
		//cudaMemcpy(eaccy, daccy, sizeof(double)* nBeads * nMols, cudaMemcpyDeviceToHost);
		//cudaMemcpy(eaccz, daccz, sizeof(double)* nBeads * nMols, cudaMemcpyDeviceToHost);

		//std::ofstream out("finalOut.dat");
		//for (int i = 0; i < nBeads*nMols; i++){
		//	out << std::setw(15) << x[i]
		//		<< std::setw(15) << y[i]
		//		<< std::setw(15) << z[i]
		//		<< std::endl;
		//}

	}

	cudaMemcpy(x, dx, nMols * nBeads * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, dy, nMols * nBeads * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(z, dz, nMols * nBeads * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(evx, dvx, sizeof(double) * nBeads * nMols, cudaMemcpyDeviceToHost);
	cudaMemcpy(evy, dvy, sizeof(double) * nBeads * nMols, cudaMemcpyDeviceToHost);
	cudaMemcpy(evz, dvz, sizeof(double) * nBeads * nMols, cudaMemcpyDeviceToHost);
	cudaMemcpy(eaccx, daccx, sizeof(double) * nBeads * nMols, cudaMemcpyDeviceToHost);
	cudaMemcpy(eaccy, daccy, sizeof(double) * nBeads * nMols, cudaMemcpyDeviceToHost);
	cudaMemcpy(eaccz, daccz, sizeof(double) * nBeads * nMols, cudaMemcpyDeviceToHost);

	std::ofstream out("finalOut.dat");
	for (int i = 0; i < nBeads*nMols; i++){
		out << std::setw(15) << x[i]
			<< std::setw(15) << y[i]
			<< std::setw(15) << z[i]
			<< std::endl;
	}

	return EXIT_SUCCESS;
}