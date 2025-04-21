#include "core.h"
#include "parameters.h"

#include <math.h>
#include <stdlib.h> // rand()

#define ECUT (4.0f * (powf(RCUT, -12) - powf(RCUT, -6)))

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
    int nucells = ceilf(cbrt((float)N / 4.0f));
    int idx = 0;

    for (int i = 0; i < nucells; i++) {
        for (int j = 0; j < nucells; j++) {
            for (int k = 0; k < nucells; k++) {
                // se pueden tomar de a cuatro puntos en tres vectores
                // y con una multiplicacion se termina

                rxyz[idx + 0] = i * a; // x
                rxyz[idx + 1] = j * a; // y
                rxyz[idx + 2] = k * a; // z
                    // del mismo átomo
                rxyz[idx + 3] = (i + 0.5f) * a;
                rxyz[idx + 4] = (j + 0.5f) * a;
                rxyz[idx + 5] = k * a;

                rxyz[idx + 6] = (i + 0.5f) * a;
                rxyz[idx + 7] = j * a;
                rxyz[idx + 8] = (k + 0.5f) * a;

                rxyz[idx + 9] = i * a;
                rxyz[idx + 10] = (j + 0.5f) * a;
                rxyz[idx + 11] = (k + 0.5f) * a;

                idx += 12;
            }
        }
    }
}


void init_vel(float* vxyz, float* temp, float* ekin)
{
    // inicialización de velocidades aleatorias

    float sf, sumvx = 0.0f, sumvy = 0.0f, sumvz = 0.0f, sumv2 = 0.0f;
    int state = SEED;

    for (int i = 0; i < 3 * N; i += 12) {
        for (int d = 0; d < 4; ++d) {
            vxyz[i+3*d+0] = myrand(&state) / (float)RAND_MAX - 0.5f;
            vxyz[i+3*d+1] = myrand(&state) / (float)RAND_MAX - 0.5f;
            vxyz[i+3*d+2] = myrand(&state) / (float)RAND_MAX - 0.5f;

            sumvx += vxyz[i+3*d+0];
            sumvy += vxyz[i+3*d+1];
            sumvz += vxyz[i+3*d+2];
            sumv2 += vxyz[i+3*d+0] * vxyz[i+3*d+0] + vxyz[i+3*d+1] * vxyz[i+3*d+1]
                + vxyz[i+3*d+2] * vxyz[i+3*d+2];
        }
    }

    sumvx /= (float)N;
    sumvy /= (float)N;
    sumvz /= (float)N;
    *temp = sumv2 / (3.0f * N);
    *ekin = 0.5f * sumv2;
    sf = sqrtf(T0 / *temp);

    for (int i = 0; i < 3 * N; i += 12) { // elimina la velocidad del centro de masa
        // y ajusta la temperatura
        for (int d = 0; d < 4; ++d) {
            vxyz[i+3*d+0] = (vxyz[i+3*d+0] - sumvx) * sf;
            vxyz[i+3*d+1] = (vxyz[i+3*d+1] - sumvy) * sf;
            vxyz[i+3*d+2] = (vxyz[i+3*d+2] - sumvz) * sf;
        }
    }
}


static float minimum_image(float cordi, const float cell_length)
{
    // imagen más cercana

    if (cordi <= -0.5f * cell_length) {
        cordi += cell_length;
    } else if (cordi > 0.5f * cell_length) {
        cordi -= cell_length;
    }
    return cordi;
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
    float pres_vir = 0.0f;
    float rcut2 = RCUT * RCUT;
    *epot = 0.0f;

    for (int i = 0; i < 3 * (N - 1); i += 3) {

        float xi = rxyz[i + 0];
        float yi = rxyz[i + 1];
        float zi = rxyz[i + 2];

        for (int j = i + 3; j < 3 * N; j += 3) {

            /*
            if (j + 3 * 7 < N) {
                // cargar ocho puntos xj, yj, zj distintos en vectores
                // cargar ocho copias de xi, yi, zi en vectores

                j += 21;
            }
            */

            float xj = rxyz[j + 0];
            float yj = rxyz[j + 1];
            float zj = rxyz[j + 2];

            // distancia mínima entre r_i y r_j
            float rx = xi - xj;
            rx = minimum_image(rx, L);
            float ry = yi - yj;
            ry = minimum_image(ry, L);
            float rz = zi - zj;
            rz = minimum_image(rz, L);

            float rij2 = rx * rx + ry * ry + rz * rz;

            if (rij2 <= rcut2) {
                float r2inv = 1.0f / rij2;
                float r6inv = r2inv * r2inv * r2inv;

                float fr = 24.0f * r2inv * r6inv * (2.0f * r6inv - 1.0f);

                fxyz[i + 0] += fr * rx;
                fxyz[i + 1] += fr * ry;
                fxyz[i + 2] += fr * rz;

                fxyz[j + 0] -= fr * rx;
                fxyz[j + 1] -= fr * ry;
                fxyz[j + 2] -= fr * rz;

                *epot += 4.0f * r6inv * (r6inv - 1.0f) - ECUT;
                pres_vir += fr * rij2;
            }
        }
    }
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

    for (int i = 0; i < 3 * N; i += 3) { // actualizo posiciones
        rxyz[i + 0] += vxyz[i + 0] * DT + 0.5f * fxyz[i + 0] * DT * DT;
        rxyz[i + 1] += vxyz[i + 1] * DT + 0.5f * fxyz[i + 1] * DT * DT;
        rxyz[i + 2] += vxyz[i + 2] * DT + 0.5f * fxyz[i + 2] * DT * DT;

        rxyz[i + 0] = pbc(rxyz[i + 0], L);
        rxyz[i + 1] = pbc(rxyz[i + 1], L);
        rxyz[i + 2] = pbc(rxyz[i + 2], L);

        vxyz[i + 0] += 0.5f * fxyz[i + 0] * DT;
        vxyz[i + 1] += 0.5f * fxyz[i + 1] * DT;
        vxyz[i + 2] += 0.5f * fxyz[i + 2] * DT;
    }

    forces(rxyz, fxyz, epot, pres, temp, rho, V, L); // actualizo fuerzas

    float sumv2 = 0.0f;
    for (int i = 0; i < 3 * N; i += 12) { // actualizo velocidades
        for (int d = 0; d < 4; ++d) {
            vxyz[i+3*d+0] += 0.5f * fxyz[i+3*d+0] * DT;
            vxyz[i+3*d+1] += 0.5f * fxyz[i+3*d+1] * DT;
            vxyz[i+3*d+2] += 0.5f * fxyz[i+3*d+2] * DT;
    
            sumv2 += vxyz[i+3*d+0] * vxyz[i+3*d+0] + vxyz[i+3*d+1] * vxyz[i+3*d+1]
                + vxyz[i+3*d+2] * vxyz[i+3*d+2];
        }
    }

    *ekin = 0.5f * sumv2;
    *temp = sumv2 / (3.0f * N);
}
