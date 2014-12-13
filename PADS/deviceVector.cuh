#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


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
	__inline__ __device__ void operator++();
	__inline__ __device__ void operator--();

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