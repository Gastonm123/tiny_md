#include "core.h"
#include "parameters.h"

#ifndef USE_ISPC
#include "simd.h"
#endif

#include <math.h>
#include <stdlib.h> // rand()
#include <immintrin.h>

#define ECUT (4.0f * (powf(RCUT, -12) - powf(RCUT, -6)))

#define X_OFF 0
#define Y_OFF N
#define Z_OFF (2*N)

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

// manualmente incluyo esta flag para usar la version de ispc
#ifndef USE_ISPC
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

static __m256 minimum_image256(__m256 cordi, const float cell_length)
{
    // imagen más cercana
    __m256 minus_half = _mm256_set1_ps(-0.5f * cell_length);
    __m256 half = _mm256_set1_ps(0.5f * cell_length);

    __m256 add, sub;
    add = _mm256_cmp_ps(cordi, minus_half, _CMP_LE_OS);
    sub = _mm256_cmp_ps(cordi, half, _CMP_GT_OS);

    cordi = _mm256_add_ps(cordi, _mm256_and_ps(add, _mm256_set1_ps(cell_length)));
    cordi = _mm256_sub_ps(cordi, _mm256_and_ps(sub, _mm256_set1_ps(cell_length)));
    return cordi;
}

static float reduce(__m256 v)
{
    // suma horizontal del vector
	__m256 psum = _mm256_hadd_ps(_mm256_hadd_ps(v,v), _mm256_hadd_ps(v,v));
	return _mm_cvtss_f32(_mm_add_ps(_mm256_extractf128_ps(psum,0), _mm256_extractf128_ps(psum,1)));
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

    // constantes vectoriales
    __m256 c1, c2, c4, c24, rcut2_256;
    c24 = _mm256_set1_ps(24.0f);
    c4 = _mm256_set1_ps(4.0f);
    c2 = _mm256_set1_ps(2.0f);
    c1 = _mm256_set1_ps(1.0f);
    rcut2_256 = _mm256_set1_ps(rcut2);

    for (int i = 0; i < N - 1; ++i) {

        float xi = rxyz[X_OFF + i];
        float yi = rxyz[Y_OFF + i];
        float zi = rxyz[Z_OFF + i];

        __m256 acc_fx, acc_fy, acc_fz, acc_epot, acc_pres_vir;
        acc_fx = acc_fy = acc_fz = acc_epot = acc_pres_vir = _mm256_setzero_ps();
        
        __m256 vxi, vyi, vzi;
        vxi = _mm256_set1_ps(rxyz[X_OFF + i]);
        vyi = _mm256_set1_ps(rxyz[Y_OFF + i]);
        vzi = _mm256_set1_ps(rxyz[Z_OFF + i]);
        
        for (int j = i + 1; j < N; ++j) {

            if (j + 7 < N) {
                // calcular minima imagen rx, ry, rz
                __m256 xj, yj, zj, rx, ry, rz;

                xj = _mm256_loadu_ps(&rxyz[X_OFF+j]);
                yj = _mm256_loadu_ps(&rxyz[Y_OFF+j]);
                zj = _mm256_loadu_ps(&rxyz[Z_OFF+j]);

                rx = _mm256_sub_ps(vxi, xj);
                ry = _mm256_sub_ps(vyi, yj);
                rz = _mm256_sub_ps(vzi, zj);

                rx = minimum_image256(rx, L);
                ry = minimum_image256(ry, L);
                rz = minimum_image256(rz, L);

                // calcular rij2, r2inv, r6inv, fr
                __m256 rij2, r2inv, r6inv, fr;
                rij2 = _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry)),
                    _mm256_mul_ps(rz, rz));

                r2inv = _mm256_div_ps(c1, rij2);
                r6inv = _mm256_mul_ps(r2inv, _mm256_mul_ps(r2inv, r2inv));
                fr = _mm256_mul_ps(
                    _mm256_mul_ps(c24, _mm256_mul_ps(r2inv, r6inv)),
                    _mm256_sub_ps(_mm256_mul_ps(c2, r6inv), c1));
                
                // sumar fr * rx en el atomo i y restar en el atomo j
                __m256 mask, fxj, fyj, fzj;
                mask = _mm256_cmp_ps(rij2, rcut2_256, _CMP_LE_OS);
                rx = _mm256_mul_ps(fr, rx);
                ry = _mm256_mul_ps(fr, ry);
                rz = _mm256_mul_ps(fr, rz);

                acc_fx = _mm256_add_ps(acc_fx, _mm256_and_ps(rx, mask));
                acc_fy = _mm256_add_ps(acc_fy, _mm256_and_ps(ry, mask));
                acc_fz = _mm256_add_ps(acc_fz, _mm256_and_ps(rz, mask));

                fxj = _mm256_loadu_ps(&fxyz[X_OFF + j]);
                fxj = _mm256_sub_ps(fxj, _mm256_and_ps(rx, mask));
                _mm256_storeu_ps(&fxyz[X_OFF + j], fxj);

                fyj = _mm256_loadu_ps(&fxyz[Y_OFF + j]);
                fyj = _mm256_sub_ps(fyj, _mm256_and_ps(ry, mask));
                _mm256_storeu_ps(&fxyz[Y_OFF + j], fyj);

                fzj = _mm256_loadu_ps(&fxyz[Z_OFF + j]);
                fzj = _mm256_sub_ps(fzj, _mm256_and_ps(rz, mask));
                _mm256_storeu_ps(&fxyz[Z_OFF + j], fzj);

                // actualizar epot y pres_vir
                __m256 aux;
                aux = _mm256_sub_ps(
                    _mm256_mul_ps(c4, _mm256_mul_ps(r6inv, _mm256_sub_ps(r6inv, c1))),
                    _mm256_set1_ps(ECUT));
                acc_epot = _mm256_add_ps(acc_epot, _mm256_and_ps(mask, aux));

                aux = _mm256_mul_ps(fr, rij2);
                acc_pres_vir = _mm256_add_ps(acc_pres_vir, _mm256_and_ps(mask, aux));
                
                // omitir los siguientes 7 atomos
                j += 7;
            }
            else {
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
    
                    fxyz[X_OFF + i] += fr * rx;
                    fxyz[Y_OFF + i] += fr * ry;
                    fxyz[Z_OFF + i] += fr * rz;
    
                    fxyz[X_OFF + j] -= fr * rx;
                    fxyz[Y_OFF + j] -= fr * ry;
                    fxyz[Z_OFF + j] -= fr * rz;
    
                    *epot += 4.0f * r6inv * (r6inv - 1.0f) - ECUT;
                    pres_vir += fr * rij2;
                }
            }

        }

        fxyz[X_OFF + i] += reduce(acc_fx);
        fxyz[Y_OFF + i] += reduce(acc_fy);
        fxyz[Z_OFF + i] += reduce(acc_fz);
        *epot           += reduce(acc_epot);
        pres_vir        += reduce(acc_pres_vir);
    }
    pres_vir /= (V * 3.0f);
    *pres = *temp * rho + pres_vir;
}
#endif // ISPC

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
