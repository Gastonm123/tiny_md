#include "parameters.h"

#define ECUT (4.0f * (pow(RCUT, -12) - pow(RCUT, -6)))

#define X_OFF 0
#define Y_OFF N
#define Z_OFF (2*N)

static float minimum_image(varying float cordi, uniform const float cell_length)
{
    // imagen más cercana

    if (cordi <= -0.5f * cell_length) {
        cordi += cell_length;
    } else if (cordi > 0.5f * cell_length) {
        cordi -= cell_length;
    }
    return cordi;
}

// un puntero uniforme a un tipo varying debe entenderse
//  como un unico puntero que al desreferenciarse puede apuntar
//  a un vector de valores distintos para cada program instance
// si no se accede a posiciones fuera de memoria en el codigo
//  secuencial, el codigo paralelizado tampoco deberia, debido
//  al uso de una mascara adecuada
// los punteros a datos variables saltan de vector en vector, no
//  se puede acceder a cada posicion individualmente
// ===============================================================
// la documentacion sugiere (pero no especifica) que una operacion
//  sobre un tipo uniforme se realiza una sola vez. si no fuera asi,
//  hay condiciones de carrera en este codigo

export void forces(
    uniform const float* uniform rxyz, uniform float* uniform fxyz, 
    uniform float* uniform epot, uniform float* uniform pres,
    uniform const float* uniform temp, uniform const float rho,
    uniform const float V, uniform const float L)
{
    // calcula las fuerzas LJ (12-6)

    memset(fxyz, 0, 3 * N * 4);

    float lane_pres_vir = 0.0f;
    float lane_epot = 0.0f;
    uniform const float rcut2 = RCUT * RCUT;

    for (uniform int i = 0; i < N - 1; ++i) {

        uniform float xi = rxyz[X_OFF + i];
        uniform float yi = rxyz[Y_OFF + i];
        uniform float zi = rxyz[Z_OFF + i];

        float fxi = 0.0f;
        float fyi = 0.0f;
        float fzi = 0.0f;

        foreach (j = (i + 1) ... N) {

            float xj = rxyz[X_OFF + j];
            float yj = rxyz[Y_OFF + j];
            float zj = rxyz[Z_OFF + j];

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

                fxyz[X_OFF + j] -= fr * rx;
                fxyz[Y_OFF + j] -= fr * ry;
                fxyz[Z_OFF + j] -= fr * rz;

                fxi += fr * rx;
                fyi += fr * ry;
                fzi += fr * rz;

                lane_epot += 4.0f * r6inv * (r6inv - 1.0f) - ECUT;
                lane_pres_vir += fr * rij2;
            }

        }

        fxyz[X_OFF + i] += reduce_add(fxi);
        fxyz[Y_OFF + i] += reduce_add(fyi);
        fxyz[Z_OFF + i] += reduce_add(fzi);
    }

    uniform float pres_vir = reduce_add(lane_pres_vir);
    *epot = reduce_add(lane_epot);

    pres_vir /= (V * 3.0f);
    *pres = *temp * rho + pres_vir;
}