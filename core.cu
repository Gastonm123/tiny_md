#include "core.hh"
#include "parameters.h"

#include <math.h>
#include <stdlib.h> // rand()
#include <stdio.h>
#include <omp.h>
#include <cuda_runtime.h>

#define DIV_CEIL(a,b) (((a)+(b)-1)/(b))
#define ECUT (4.0f * (powf(RCUT, -12) - powf(RCUT, -6)))
#define X_OFF 0
#define Y_OFF N
#define Z_OFF (2*N)
#define BLOCK_SIZE 1024
const int STUFFED_N = N-N%BLOCK_SIZE+BLOCK_SIZE;

static inline int myrand(int *state) {
    int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

void init_pos(float* rxyz, const float rho)
{
    // inicialización de las posiciones de los átomos en un cristal FCC

    float a = cbrtf(4.0f / rho);
    int nucells = ceilf(cbrtf((float)N / 4.0f));
    int idx = 0;

    for (int i = 0; i < nucells; i++) {
        for (int j = 0; j < nucells; j++) {
            for (int k = 0; k < nucells; k++) {
                // se pueden tomar de a cuatro puntos en tres vectores
                // y con una multiplicacion se termina

                rxyz[X_OFF + idx + 0] = i * a; // x
                rxyz[Y_OFF + idx + 0] = j * a; // y
                rxyz[Z_OFF + idx + 0] = k * a; // z
                    // del mismo átomo
                rxyz[X_OFF + idx + 1] = (i + 0.5f) * a;
                rxyz[Y_OFF + idx + 1] = (j + 0.5f) * a;
                rxyz[Z_OFF + idx + 1] = k * a;

                rxyz[X_OFF + idx + 2] = (i + 0.5f) * a;
                rxyz[Y_OFF + idx + 2] = j * a;
                rxyz[Z_OFF + idx + 2] = (k + 0.5f) * a;

                rxyz[X_OFF + idx + 3] = i * a;
                rxyz[Y_OFF + idx + 3] = (j + 0.5f) * a;
                rxyz[Z_OFF + idx + 3] = (k + 0.5f) * a;

                idx += 4;
            }
        }
    }
}

void init_vel(float* vxyz, float* temp, float* ekin)
{
    // inicialización de velocidades aleatorias

    float sf, sumvx = 0.0f, sumvy = 0.0f, sumvz = 0.0f, sumv2 = 0.0f;
    int state = SEED;

    for (int i = 0; i < N; ++i) {
        vxyz[X_OFF + i] = myrand(&state) / (float)RAND_MAX - 0.5f;
        vxyz[Y_OFF + i] = myrand(&state) / (float)RAND_MAX - 0.5f;
        vxyz[Z_OFF + i] = myrand(&state) / (float)RAND_MAX - 0.5f;

        sumvx += vxyz[X_OFF + i];
        sumvy += vxyz[Y_OFF + i];
        sumvz += vxyz[Z_OFF + i];
        sumv2 += vxyz[X_OFF + i] * vxyz[X_OFF + i] + vxyz[Y_OFF + i] * vxyz[Y_OFF + i]
            + vxyz[Z_OFF + i] * vxyz[Z_OFF + i];
    }

    sumvx /= (float)N;
    sumvy /= (float)N;
    sumvz /= (float)N;
    *temp = sumv2 / (3.0f * N);
    *ekin = 0.5f * sumv2;
    sf = sqrtf(T0 / *temp);

    for (int i = 0; i < N; ++i) { // elimina la velocidad del centro de masa
        // y ajusta la temperatura
        vxyz[X_OFF + i] = (vxyz[X_OFF + i] - sumvx) * sf;
        vxyz[Y_OFF + i] = (vxyz[Y_OFF + i] - sumvy) * sf;
        vxyz[Z_OFF + i] = (vxyz[Z_OFF + i] - sumvz) * sf;
    }
}


__device__ float minimum_image(float cordi, const float cell_length)
{
    // imagen más cercana

    if (cordi <= -0.5f * cell_length) {
        cordi += cell_length;
    } else if (cordi > 0.5f * cell_length) {
        cordi -= cell_length;
    }
    return cordi;
}

__device__ void suma(float *dst, float x)
{
    __shared__ float suma_par;

    size_t tid  = threadIdx.x;		// thread id, dentro del bloque
    size_t lid  = tid%warpSize;		// lane id, dentro del warp

    // Fase 1, inicialización
    if (tid==0)
    	suma_par = 0.0f;
    __syncthreads();

    // Fase 2, cómputo dentro del bloque
    float warp_reduce = x; //((__activemask()>>lid)&1) * x;

    // Fase 2.1, suma en warp
    int mask = __activemask();
    warp_reduce += __shfl_down_sync(mask, warp_reduce, 16);
    warp_reduce += __shfl_down_sync(mask, warp_reduce, 8);
    warp_reduce += __shfl_down_sync(mask, warp_reduce, 4);
    warp_reduce += __shfl_down_sync(mask, warp_reduce, 2);
    warp_reduce += __shfl_down_sync(mask, warp_reduce, 1);

    // Fase 2.2, acumulacion a shared
    int first_active_lane = __ffs(mask) - 1;
    if (lid==first_active_lane) {
	    atomicAdd(&suma_par, warp_reduce);
    }
    __syncthreads();

    // Fase 3, acumulación del resultado local del bloque en la global
    if (tid==0)
    	atomicAdd(dst, suma_par);
}

__global__ void forces_naive(const float *rxyz, float *fxyz, float *epot, float *pres,
                             const float L, const float rcut2) 
{
	// int tid = threadIdx.x;
	// int lane = tid % warpSize;
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    /*
    int i = 0;
    while (i < N && gtid >= N-(i+1)) {
        gtid -= N-(i+1);
        i++;
    }

    if (i == N)
        return;
    
    int j = gtid + 1;
    */

    int i = gtid / STUFFED_N;
    int j = gtid % STUFFED_N;

    if (i >= N || j >= N || i == j) return;

    float xi = rxyz[X_OFF + i];
    float yi = rxyz[Y_OFF + i];
    float zi = rxyz[Z_OFF + i];

    float xj = rxyz[X_OFF + j];
    float yj = rxyz[Y_OFF + j];
    float zj = rxyz[Z_OFF + j];

    // distancia mínima entre r_i y r_j
    float rx = minimum_image(xi - xj, L);
    float ry = minimum_image(yi - yj, L);
    float rz = minimum_image(zi - zj, L);

    float rij2 = rx * rx + ry * ry + rz * rz;

    if (rij2 <= rcut2) {
        float r2inv = 1.0f / rij2;
        float r6inv = r2inv * r2inv * r2inv;

        float fr = 24.0f * r2inv * r6inv * (2.0f * r6inv - 1.0f);

        suma(&fxyz[X_OFF + i], fr * rx);
        suma(&fxyz[Y_OFF + i], fr * ry);
        suma(&fxyz[Z_OFF + i], fr * rz);

        /*
        atomicAdd(&fxyz[X_OFF + j], -fr * rx);
        atomicAdd(&fxyz[Y_OFF + j], -fr * ry);
        atomicAdd(&fxyz[Z_OFF + j], -fr * rz);

        fxyz[X_OFF + i] += fr * rx; 
        fxyz[Y_OFF + i] += fr * ry; 
        fxyz[Z_OFF + i] += fr * rz; 
        */
        suma(epot, (4.0f * r6inv * (r6inv - 1.0f) - ECUT));
        suma(pres, (fr * rij2));
    }
}

void forces(const float* rxyz, float* fxyz, float* epot, float* pres,
            const float* temp, const float rho, const float V, const float L)
{
    // calcula las fuerzas LJ (12-6)

    for (int i = 0; i < 3 * N; i+=3) {
        fxyz[i + 0] = 0.0f;
        fxyz[i + 1] = 0.0f;
        fxyz[i + 2] = 0.0f;
    }
    float pres_vir;
    const float rcut2 = RCUT * RCUT;

    //const int BLOCK_COUNT = DIV_CEIL(N*(N-1)/2, BLOCK_SIZE);
    const int BLOCK_COUNT = STUFFED_N*N/BLOCK_SIZE;
    
    float *d_rxyz = NULL, *d_fxyz = NULL, *d_epot = NULL, *d_pres = NULL;
    const int ARRAY_SIZE = 3 * N * sizeof(float);
    cudaMalloc(&d_rxyz, ARRAY_SIZE);
    cudaMalloc(&d_fxyz, ARRAY_SIZE);
    cudaMalloc(&d_epot, sizeof(float));
    cudaMalloc(&d_pres, sizeof(float));

    cudaMemcpy(d_rxyz, rxyz, ARRAY_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fxyz, fxyz, ARRAY_SIZE, cudaMemcpyHostToDevice);
    // cudaDeviceSynchronize();

    forces_naive<<<BLOCK_COUNT,BLOCK_SIZE>>>(d_rxyz, d_fxyz, d_epot, d_pres, L, rcut2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error en kernel: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(fxyz, d_fxyz, ARRAY_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(epot, d_epot, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pres_vir, d_pres, sizeof(float), cudaMemcpyDeviceToHost);

    // presion y energia potencial se cuentan dos veces
    
    *epot /= 2.0f;
    pres_vir /= 2.0f;
    pres_vir /= (V * 3.0f);
    *pres = *temp * rho + pres_vir;
}


static float pbc(float cordi, const float cell_length)
{
    // condiciones periodicas de contorno coordenadas entre [0,L)
    if (cordi <= 0.0f) {
        cordi += cell_length;
    } else if (cordi > cell_length) {
        cordi -= cell_length;
    }
    return cordi;
}


void velocity_verlet(float* rxyz, float* vxyz, float* fxyz, float* epot,
                     float* ekin, float* pres, float* temp, const float rho,
                     const float V, const float L)
{

    for (int i = 0; i < N; ++i) { // actualizo posiciones
        rxyz[X_OFF + i] += vxyz[X_OFF + i] * DT + 0.5f * fxyz[X_OFF + i] * DT * DT;
        rxyz[Y_OFF + i] += vxyz[Y_OFF + i] * DT + 0.5f * fxyz[Y_OFF + i] * DT * DT;
        rxyz[Z_OFF + i] += vxyz[Z_OFF + i] * DT + 0.5f * fxyz[Z_OFF + i] * DT * DT;

        rxyz[X_OFF + i] = pbc(rxyz[X_OFF + i], L);
        rxyz[Y_OFF + i] = pbc(rxyz[Y_OFF + i], L);
        rxyz[Z_OFF + i] = pbc(rxyz[Z_OFF + i], L);

        vxyz[X_OFF + i] += 0.5f * fxyz[X_OFF + i] * DT;
        vxyz[Y_OFF + i] += 0.5f * fxyz[Y_OFF + i] * DT;
        vxyz[Z_OFF + i] += 0.5f * fxyz[Z_OFF + i] * DT;
    }

    forces(rxyz, fxyz, epot, pres, temp, rho, V, L); // actualizo fuerzas

    float sumv2 = 0.0f;
    for (int i = 0; i < N; ++i) { // actualizo velocidades
        vxyz[X_OFF + i] += 0.5f * fxyz[X_OFF + i] * DT;
        vxyz[Y_OFF + i] += 0.5f * fxyz[Y_OFF + i] * DT;
        vxyz[Z_OFF + i] += 0.5f * fxyz[Z_OFF + i] * DT;

        sumv2 += vxyz[X_OFF + i] * vxyz[X_OFF + i] + vxyz[Y_OFF + i] * vxyz[Y_OFF + i]
            + vxyz[Z_OFF + i] * vxyz[Z_OFF + i];
    }

    *ekin = 0.5f * sumv2;
    *temp = sumv2 / (3.0f * N);
}
