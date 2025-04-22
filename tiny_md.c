#define _XOPEN_SOURCE 500  // M_PI
#include "core.h"
#include "parameters.h"
#include "wtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(int argc, char **argv)
{
    FILE *file_xyz, *file_thermo;
    file_xyz = file_thermo = NULL;

    if (argc > 1 && strcmp(argv[1], "record") == 0) {
        file_xyz = fopen("trajectory.xyz", "w");
        file_thermo = fopen("thermo.log", "w");
        fprintf(file_thermo, "# t Temp Pres Epot Etot\n");
    }

    float Ekin, Epot, Temp, Pres; // variables macroscopicas
    float Rho, cell_V, cell_L, tail, Etail, Ptail;
    float *rxyz, *vxyz, *fxyz; // variables microscopicas

    rxyz = (float*)malloc(3 * N * sizeof(float));
    vxyz = (float*)malloc(3 * N * sizeof(float));
    fxyz = (float*)malloc(3 * N * sizeof(float));

    printf("# Número de partículas:      %d\n", N);
    printf("# Temperatura de referencia: %.2f\n", T0);
    printf("# Pasos de equilibración:    %d\n", TEQ);
    printf("# Pasos de medición:         %d\n", TRUN - TEQ);
    printf("# (mediciones cada %d pasos)\n", TMES);
    printf("# densidad, volumen, energía potencial media, presión media\n");

    float t = 0.0f, sf;
    float Rhob;
    Rho = RHOI;
    init_pos(rxyz, Rho);
    float start = wtime();
    for (int m = 0; m < 9; m++) {
        Rhob = Rho;
        Rho = RHOI - 0.1f * (float)m;
        cell_V = (float)N / Rho;
        cell_L = cbrtf(cell_V);
        tail = 16.0f * (float)M_PI * Rho * ((2.0f / 3.0f) * powf(RCUT, -9.0f) - powf(RCUT, -3.0f)) / 3.0f;
        Etail = tail * (float)N;
        Ptail = tail * Rho;

        int i = 0;
        sf = cbrtf(Rhob / Rho);
        for (int k = 0; k < 3 * N; k++) { // reescaleo posiciones a nueva densidad
            rxyz[k] *= sf;
        }
        init_vel(vxyz, &Temp, &Ekin);
        forces(rxyz, fxyz, &Epot, &Pres, &Temp, Rho, cell_V, cell_L);

        for (i = 1; i < TEQ; i++) { // loop de equilibracion

            velocity_verlet(rxyz, vxyz, fxyz, &Epot, &Ekin, &Pres, &Temp, Rho, cell_V, cell_L);

            sf = sqrtf(T0 / Temp);
            for (int k = 0; k < 3 * N; k++) { // reescaleo de velocidades
                vxyz[k] *= sf;
            }
        }

        int mes = 0;
        float epotm = 0.0f, presm = 0.0f;
        for (i = TEQ; i < TRUN; i++) { // loop de medicion

            velocity_verlet(rxyz, vxyz, fxyz, &Epot, &Ekin, &Pres, &Temp, Rho, cell_V, cell_L);

            sf = sqrtf(T0 / Temp);
            for (int k = 0; k < 3 * N; k++) { // reescaleo de velocidades
                vxyz[k] *= sf;
            }

            if (i % TMES == 0) {
                Epot += Etail;
                Pres += Ptail;

                epotm += Epot;
                presm += Pres;
                mes++;

                if (file_thermo && file_xyz) {
                    fprintf(file_thermo, "%f %f %f %f %f\n", t, Temp, Pres, Epot, Epot + Ekin);
                    fprintf(file_xyz, "%d\n\n", N);
                    for (int k = 0; k < 3 * N; k += 3) {
                        fprintf(file_xyz, "Ar %e %e %e\n", rxyz[k + 0], rxyz[k + N], rxyz[k + 2*N]);
                    }
                }
            }

            t += DT;
        }
        printf("%f\t%f\t%f\t%f\n", Rho, cell_V, epotm / (float)mes, presm / (float)mes);
    }

    double elapsed = wtime() - start;
    printf("# Tiempo total de simulación = %f segundos\n", elapsed);
    printf("# Tiempo simulado = %f [fs]\n", t * 1.6);

    // printf("# Nanosegundos por op = %f\n", (elapsed * 1e9) / );
    printf("# Interacciones por microsegundo = %f\n", ((long long)N*N*TRUN) / (elapsed * 1e6));

    if (file_thermo && file_xyz) {
        // Cierre de archivos
        fclose(file_thermo);
        fclose(file_xyz);
    }

    // Liberacion de memoria
    free(rxyz);
    free(fxyz);
    free(vxyz);
    return 0;
}
