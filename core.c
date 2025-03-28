#include "core.h"
#include "parameters.h"

#include <math.h>
#include <stdlib.h> // rand()

#define ECUT (4.0 * (pow(RCUT, -12) - pow(RCUT, -6)))


void init_pos(double* rxyz, const double rho)
{
    // inicialización de las posiciones de los átomos en un cristal FCC

    double a = cbrt(4.0 / rho);
    int nucells = ceil(cbrt((double)N / 4.0));
    int idx = 0;

    for (int i = 0; i < nucells; i++) {
        for (int j = 0; j < nucells; j++) {
            for (int k = 0; k < nucells; k++) {
                rxyz[idx + 0] = i * a; // x
                rxyz[idx + 1] = j * a; // y
                rxyz[idx + 2] = k * a; // z
                    // del mismo átomo
                rxyz[idx + 3] = (i + 0.5) * a;
                rxyz[idx + 4] = (j + 0.5) * a;
                rxyz[idx + 5] = k * a;

                rxyz[idx + 6] = (i + 0.5) * a;
                rxyz[idx + 7] = j * a;
                rxyz[idx + 8] = (k + 0.5) * a;

                rxyz[idx + 9] = i * a;
                rxyz[idx + 10] = (j + 0.5) * a;
                rxyz[idx + 11] = (k + 0.5) * a;

                idx += 12;
            }
        }
    }
}


void init_vel(double* vxyz, double* temp, double* ekin)
{
    // inicialización de velocidades aleatorias

    double sf, sumvx = 0.0, sumvy = 0.0, sumvz = 0.0, sumv2 = 0.0;

    for (int i = 0; i < 3 * N; i += 12) {
        vxyz[i + 0] = rand() / (double)RAND_MAX - 0.5;
        vxyz[i + 1] = rand() / (double)RAND_MAX - 0.5;
        vxyz[i + 2] = rand() / (double)RAND_MAX - 0.5;

        vxyz[i + 3] = rand() / (double)RAND_MAX - 0.5;
        vxyz[i + 4] = rand() / (double)RAND_MAX - 0.5;
        vxyz[i + 5] = rand() / (double)RAND_MAX - 0.5;

        vxyz[i + 6] = rand() / (double)RAND_MAX - 0.5;
        vxyz[i + 7] = rand() / (double)RAND_MAX - 0.5;
        vxyz[i + 8] = rand() / (double)RAND_MAX - 0.5;

        vxyz[i + 9] = rand() / (double)RAND_MAX - 0.5;
        vxyz[i + 10] = rand() / (double)RAND_MAX - 0.5;
        vxyz[i + 11] = rand() / (double)RAND_MAX - 0.5;

        double vx = vxyz[i + 0] + vxyz[i + 3] + vxyz[i + 6] + vxyz[i + 9];
        double vy = vxyz[i + 1] + vxyz[i + 4] + vxyz[i + 7] + vxyz[i + 10];
        double vz = vxyz[i + 2] + vxyz[i + 5] + vxyz[i + 8] + vxyz[i + 11];
        sumvx += vx;
        sumvy += vy;
        sumvz += vz;
        sumv2 += vx * vx + vy * vy + vz * vz;
    }

    sumvx /= (double)N;
    sumvy /= (double)N;
    sumvz /= (double)N;
    *temp = sumv2 / (3.0 * N);
    *ekin = 0.5 * sumv2;
    sf = sqrt(T0 / *temp);

    for (int i = 0; i < 3 * N; i += 12) { // elimina la velocidad del centro de masa
        // y ajusta la temperatura
        vxyz[i + 0] = (vxyz[i + 0] - sumvx) * sf;
        vxyz[i + 1] = (vxyz[i + 1] - sumvy) * sf;
        vxyz[i + 2] = (vxyz[i + 2] - sumvz) * sf;

        vxyz[i + 3] = (vxyz[i + 0] - sumvx) * sf;
        vxyz[i + 4] = (vxyz[i + 1] - sumvy) * sf;
        vxyz[i + 5] = (vxyz[i + 2] - sumvz) * sf;

        vxyz[i + 6] = (vxyz[i + 0] - sumvx) * sf;
        vxyz[i + 7] = (vxyz[i + 1] - sumvy) * sf;
        vxyz[i + 8] = (vxyz[i + 2] - sumvz) * sf;

        vxyz[i + 9] = (vxyz[i + 0] - sumvx) * sf;
        vxyz[i + 10] = (vxyz[i + 1] - sumvy) * sf;
        vxyz[i + 11] = (vxyz[i + 2] - sumvz) * sf;
    }
}


static double minimum_image(double cordi, const double cell_length)
{
    // imagen más cercana

    if (cordi <= -0.5 * cell_length) {
        cordi += cell_length;
    } else if (cordi > 0.5 * cell_length) {
        cordi -= cell_length;
    }
    return cordi;
}


void forces(const double* rxyz, double* fxyz, double* epot, double* pres,
            const double* temp, const double rho, const double V, const double L)
{
    // calcula las fuerzas LJ (12-6)

    for (int i = 0; i < 3 * N; i+=4) {
        fxyz[i + 0] = 0.0;
        fxyz[i + 1] = 0.0;
        fxyz[i + 2] = 0.0;
        fxyz[i + 3] = 0.0;
    }
    double pres_vir = 0.0;
    double rcut2 = RCUT * RCUT;
    *epot = 0.0;

    for (int i = 0; i < 3 * N; i += 3) {
        
        double xi = rxyz[i + 0];
        double yi = rxyz[i + 1];
        double zi = rxyz[i + 2];

        int k = (i/12)*12;
        
        // double rx1 = xi - rxyz[k + 0];
        // double ry1 = yi - rxyz[k + 1];
        // double rz1 = zi - rxyz[k + 2];

        double rx2 = xi - rxyz[k + 3];
        double ry2 = yi - rxyz[k + 4];
        double rz2 = zi - rxyz[k + 5];
        
        double rx3 = xi - rxyz[k + 6];
        double ry3 = yi - rxyz[k + 7];
        double rz3 = zi - rxyz[k + 8];

        double rx4 = xi - rxyz[k + 9];
        double ry4 = yi - rxyz[k + 10];
        double rz4 = zi - rxyz[k + 11];
        
        // rx1 = minimum_image(rx1, L);
        // ry1 = minimum_image(ry1, L);
        // rz1 = minimum_image(rz1, L);

        rx2 = minimum_image(rx2, L);
        ry2 = minimum_image(ry2, L);
        rz2 = minimum_image(rz2, L);

        rx3 = minimum_image(rx3, L);
        ry3 = minimum_image(ry3, L);
        rz3 = minimum_image(rz3, L);
        
        rx4 = minimum_image(rx4, L);
        ry4 = minimum_image(ry4, L);
        rz4 = minimum_image(rz4, L);

        // double rik2_1 = rx1 * rx1 + ry1 * ry1 + rz1 * rz1;
        double rik2_2 = rx2 * rx2 + ry2 * ry2 + rz2 * rz2;
        double rik2_3 = rx3 * rx3 + ry3 * ry3 + rz3 * rz3;
        double rik2_4 = rx4 * rx4 + ry4 * ry4 + rz4 * rz4;
        
        if (k+3 > i && rik2_2 <= rcut2) {
            double r2inv_2 = 1.0 / rik2_2;
            double r6inv_2 = r2inv_2 * r2inv_2 * r2inv_2;
            double fr_2 = 24.0 * r2inv_2 * r6inv_2 * (2.0 * r6inv_2 - 1.0);

            fxyz[i + 0] += fr_2 * rx2;
            fxyz[i + 1] += fr_2 * ry2;
            fxyz[i + 2] += fr_2 * rz2;

            fxyz[k + 3] -= fr_2 * rx2;
            fxyz[k + 4] -= fr_2 * ry2;
            fxyz[k + 5] -= fr_2 * rz2;
            
            *epot += 4.0 * r6inv_2 * (r6inv_2 - 1.0) - ECUT;
            pres_vir += fr_2 * rik2_2;
        }
        if (k+6 > i && rik2_3 <= rcut2) {
            double r2inv_3 = 1.0 / rik2_3;
            double r6inv_3 = r2inv_3 * r2inv_3 * r2inv_3;
            double fr_3 = 24.0 * r2inv_3 * r6inv_3 * (2.0 * r6inv_3 - 1.0);

            fxyz[i + 0] += fr_3 * rx3;
            fxyz[i + 1] += fr_3 * ry3;
            fxyz[i + 2] += fr_3 * rz3;

            fxyz[k + 6] -= fr_3 * rx3;
            fxyz[k + 7] -= fr_3 * ry3;
            fxyz[k + 8] -= fr_3 * rz3;
            
            *epot += 4.0 * r6inv_3 * (r6inv_3 - 1.0) - ECUT;
            pres_vir += fr_3 * rik2_3;
        }
        if (k+9 > i && rik2_4 <= rcut2) {
            double r2inv_4 = 1.0 / rik2_4;
            double r6inv_4 = r2inv_4 * r2inv_4 * r2inv_4;
            double fr_4 = 24.0 * r2inv_4 * r6inv_4 * (2.0 * r6inv_4 - 1.0);

            fxyz[i + 0] += fr_4 * rx4;
            fxyz[i + 1] += fr_4 * ry4;
            fxyz[i + 2] += fr_4 * rz4;

            fxyz[k + 9]  -= fr_4 * rx4;
            fxyz[k + 10] -= fr_4 * ry4;
            fxyz[k + 11] -= fr_4 * rz4;
            
            *epot += 4.0 * r6inv_4 * (r6inv_4 - 1.0) - ECUT;
            pres_vir += fr_4 * rik2_4;
        }

        for (int j = k+12; j < 3*N; j += 12) {

            double rx1 = xi - rxyz[j + 0];
            double ry1 = yi - rxyz[j + 1];
            double rz1 = zi - rxyz[j + 2];

            double rx2 = xi - rxyz[j + 3];
            double ry2 = yi - rxyz[j + 4];
            double rz2 = zi - rxyz[j + 5];

            double rx3 = xi - rxyz[j + 6];
            double ry3 = yi - rxyz[j + 7];
            double rz3 = zi - rxyz[j + 8];

            double rx4 = xi - rxyz[j + 9];
            double ry4 = yi - rxyz[j + 10];
            double rz4 = zi - rxyz[j + 11];

            rx1 = minimum_image(rx1, L);
            ry1 = minimum_image(ry1, L);
            rz1 = minimum_image(rz1, L);

            rx2 = minimum_image(rx2, L);
            ry2 = minimum_image(ry2, L);
            rz2 = minimum_image(rz2, L);

            rx3 = minimum_image(rx3, L);
            ry3 = minimum_image(ry3, L);
            rz3 = minimum_image(rz3, L);

            rx4 = minimum_image(rx4, L);
            ry4 = minimum_image(ry4, L);
            rz4 = minimum_image(rz4, L);

            double rij2_1 = rx1 * rx1 + ry1 * ry1 + rz1 * rz1;
            double rij2_2 = rx2 * rx2 + ry2 * ry2 + rz2 * rz2;
            double rij2_3 = rx3 * rx3 + ry3 * ry3 + rz3 * rz3;
            double rij2_4 = rx4 * rx4 + ry4 * ry4 + rz4 * rz4;

            if (rij2_1 <= rcut2) {
                double r2inv_1 = 1.0 / rij2_1;
                double r6inv_1 = r2inv_1 * r2inv_1 * r2inv_1;
                double fr_1 = 24.0 * r2inv_1 * r6inv_1 * (2.0 * r6inv_1 - 1.0);

                fxyz[i + 0] += fr_1 * rx1;
                fxyz[i + 1] += fr_1 * ry1;
                fxyz[i + 2] += fr_1 * rz1;

                fxyz[j + 0] -= fr_1 * rx1;
                fxyz[j + 1] -= fr_1 * ry1;
                fxyz[j + 2] -= fr_1 * rz1;

                *epot += 4.0 * r6inv_1 * (r6inv_1 - 1.0) - ECUT;
                pres_vir += fr_1 * rij2_1;
            }
            if (rij2_2 <= rcut2) {
                double r2inv_2 = 1.0 / rij2_2;
                double r6inv_2 = r2inv_2 * r2inv_2 * r2inv_2;
                double fr_2 = 24.0 * r2inv_2 * r6inv_2 * (2.0 * r6inv_2 - 1.0);

                fxyz[i + 0] += fr_2 * rx2;
                fxyz[i + 1] += fr_2 * ry2;
                fxyz[i + 2] += fr_2 * rz2;

                fxyz[j + 3] -= fr_2 * rx2;
                fxyz[j + 4] -= fr_2 * ry2;
                fxyz[j + 5] -= fr_2 * rz2;

                *epot += 4.0 * r6inv_2 * (r6inv_2 - 1.0) - ECUT;
                pres_vir += fr_2 * rij2_2;
            }
            if (rij2_3 <= rcut2) {
                double r2inv_3 = 1.0 / rij2_3;
                double r6inv_3 = r2inv_3 * r2inv_3 * r2inv_3;
                double fr_3 = 24.0 * r2inv_3 * r6inv_3 * (2.0 * r6inv_3 - 1.0);

                fxyz[i + 0] += fr_3 * rx3;
                fxyz[i + 1] += fr_3 * ry3;
                fxyz[i + 2] += fr_3 * rz3;

                fxyz[j + 6] -= fr_3 * rx3;
                fxyz[j + 7] -= fr_3 * ry3;
                fxyz[j + 8] -= fr_3 * rz3;

                *epot += 4.0 * r6inv_3 * (r6inv_3 - 1.0) - ECUT;
                pres_vir += fr_3 * rij2_3;
            }
            if (rij2_4 <= rcut2) {
                double r2inv_4 = 1.0 / rij2_4;
                double r6inv_4 = r2inv_4 * r2inv_4 * r2inv_4;
                double fr_4 = 24.0 * r2inv_4 * r6inv_4 * (2.0 * r6inv_4 - 1.0);

                fxyz[i + 0] += fr_4 * rx4;
                fxyz[i + 1] += fr_4 * ry4;
                fxyz[i + 2] += fr_4 * rz4;

                fxyz[j + 9]  -= fr_4 * rx4;
                fxyz[j + 10] -= fr_4 * ry4;
                fxyz[j + 11] -= fr_4 * rz4;

                *epot += 4.0 * r6inv_4 * (r6inv_4 - 1.0) - ECUT;
                pres_vir += fr_4 * rij2_4;
            }
        }
    }
    pres_vir /= (V * 3.0);
    *pres = *temp * rho + pres_vir;
}


static double pbc(double cordi, const double cell_length)
{
    // condiciones periodicas de contorno coordenadas entre [0,L)
    if (cordi <= 0) {
        cordi += cell_length;
    } else if (cordi > cell_length) {
        cordi -= cell_length;
    }
    return cordi;
}


void velocity_verlet(double* rxyz, double* vxyz, double* fxyz, double* epot,
                     double* ekin, double* pres, double* temp, const double rho,
                     const double V, const double L)
{

    for (int i = 0; i < 3 * N; i += 3) { // actualizo posiciones
        rxyz[i + 0] += vxyz[i + 0] * DT + 0.5 * fxyz[i + 0] * DT * DT;
        rxyz[i + 1] += vxyz[i + 1] * DT + 0.5 * fxyz[i + 1] * DT * DT;
        rxyz[i + 2] += vxyz[i + 2] * DT + 0.5 * fxyz[i + 2] * DT * DT;

        rxyz[i + 0] = pbc(rxyz[i + 0], L);
        rxyz[i + 1] = pbc(rxyz[i + 1], L);
        rxyz[i + 2] = pbc(rxyz[i + 2], L);

        vxyz[i + 0] += 0.5 * fxyz[i + 0] * DT;
        vxyz[i + 1] += 0.5 * fxyz[i + 1] * DT;
        vxyz[i + 2] += 0.5 * fxyz[i + 2] * DT;
    }

    forces(rxyz, fxyz, epot, pres, temp, rho, V, L); // actualizo fuerzas

    double sumv2 = 0.0;
    for (int i = 0; i < 3 * N; i += 3) { // actualizo velocidades
        vxyz[i + 0] += 0.5 * fxyz[i + 0] * DT;
        vxyz[i + 1] += 0.5 * fxyz[i + 1] * DT;
        vxyz[i + 2] += 0.5 * fxyz[i + 2] * DT;

        sumv2 += vxyz[i + 0] * vxyz[i + 0] + vxyz[i + 1] * vxyz[i + 1]
            + vxyz[i + 2] * vxyz[i + 2];
    }

    *ekin = 0.5 * sumv2;
    *temp = sumv2 / (3.0 * N);
}
