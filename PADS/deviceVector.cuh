#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


class dvec{
public:
	__device__ dvec();
	__device__ dvec(double *a, int nElem);
	__device__ ~dvec();

	__inline__ __device__ double operator[](const int i1);
	__inline__ __device__ void operator+=(dvec v1);
	__inline__ __device__ void operator-=(dvec v1);
	__inline__ __device__ void operator*=(dvec v1);
	__inline__ __device__ void operator/=(dvec v1);
	__inline__ __device__ void operator+=(double s1);
	__inline__ __device__ void operator-=(double s1);
	__inline__ __device__ void operator*=(double s1);
	__inline__ __device__ void operator/=(double s1);

	__inline__ __device__ double dot(dvec v1);
	__inline__ __device__ double cross(dvec v1);
	__inline__ __device__ double norm();
	__inline__ __device__ double sum();

	__device__ dvec clone();
	__device__ double* toArray();

	int n_elem;
private:
	double* v;
};

#define DVEC_BOUNDS_CHECK
#ifdef DVEC_BOUNDS_CHECK
#include <typeinfo>
#endif

__inline__ __device__ dvec::dvec(double *a, int nElem){
	v = new double[nElem];
	for (int i = 0; i < nElem; i++){
		v[i] = a[i];
	}
	n_elem = nElem;
	float2 blah;
}

__device__ dvec::~dvec(){
	delete v;
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
__inline__ __device__ void dvec::operator*=(dvec v1){
#ifdef DVEC_BOUNDS_CHECK
	if (v1.n_elem != n_elem) { return; }
#endif
	for (int i = 0; i < n_elem; i++){
		v[i] *= v1.v[i];
	}
}
__inline__ __device__ void dvec::operator/=(dvec v1){
#ifdef DVEC_BOUNDS_CHECK
	if (v1.n_elem != n_elem) { return; }
#endif
	for (int i = 0; i < n_elem; i++){
		v[i] /= v1.v[i];
	}
}
__inline__ __device__ void dvec::operator+=(double s1){
	for (int i = 0; i < n_elem; i++){
		v[i] += s1;
	}
}
__inline__ __device__ void dvec::operator-=(double s1){
	for (int i = 0; i < n_elem; i++){
		v[i] -= s1;
	}
}
__inline__ __device__ void dvec::operator*=(double s1){
	for (int i = 0; i < n_elem; i++){
		v[i] *= s1;
	}
}
__inline__ __device__ void dvec::operator/=(double s1){
	for (int i = 0; i < n_elem; i++){
		v[i] *= s1;
	}
}

__inline__ __device__ double dvec::dot(dvec v1){
#ifdef DVEC_BOUNDS_CHECK
	if (v1.n_elem != n_elem) { return; }
#endif
	double d = 0;
	for (int i = 0; i < n_elem; i++){
		d += v[i];
	}
	return d;
}
__inline__ __device__ double dvec::cross(dvec v1){}
__inline__ __device__ double dvec::norm(){
	double d = 0;
	for (int i = 0; i < n_elem; i++){
		d += v[i] * v[i];
	}
}
__inline__ __device__ double dvec::sum(){}

__device__ dvec clone();
__device__ double* toArray();