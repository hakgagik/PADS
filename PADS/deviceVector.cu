#include "deviceVector.cuh"

#define DVEC_BOUNDS_CHECK
#ifdef DVEC_BOUNDS_CHECK
#include <typeinfo>
#endif

__inline__ __device__ dvec::dvec(double *a, int nElem){
	cudaMemcpy(&v, &a, nElem * sizeof(double), cudaMemcpyDeviceToDevice);
	n_elem = nElem;
}

__device__ dvec::dvec(){
	cudaFree(&v);
}

__inline__ __device__ double dvec::operator[](const int i){
#ifdef DVEC_BOUNDS_CHECK
	if (i < 0 || i >= n_elem) { return 1E300; }
#endif
	return v[i];
}

__inline__ __device__ void dvec::operator+=(dvec v1){
#ifdef DVEC_BOUNDS_CHECK
	if (v1.n_elem != n_elem) { return; }
#endif
	for (int i = 0; i < n_elem; i++){
		v[i] += v1.v[i];
	}
}
__inline__ __device__ void dvec::operator-=(dvec v1){
#ifdef DVEC_BOUNDS_CHECK
	if (v1.n_elem != n_elem) { return; }
#endif
	for (int i = 0; i < n_elem; i++){
		v[i] -= v1.v[i];
	}
}
__inline__ __device__ void dvec::operator*=(dvec v1){}
__inline__ __device__ void dvec::operator/=(dvec v1){}
__inline__ __device__ void dvec::operator+=(double s1){}
__inline__ __device__ void dvec::operator-=(double s1){}
__inline__ __device__ void dvec::operator*=(double s1){}
__inline__ __device__ void dvec::operator/=(double s1){}
__inline__ __device__ void dvec::operator++(){}
__inline__ __device__ void dvec::operator--(){}

__inline__ __device__ double dot(dvec v1){}
__inline__ __device__ double cross(dvec v1){}
__inline__ __device__ double norm(){}
__inline__ __device__ double sum(){}

__device__ dvec clone();
__device__ double* toArray();