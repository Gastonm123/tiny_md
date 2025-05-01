/*
 * Tiny Molecular Dynamics
 *
 * Unidades: Lennard-Jones
 *
 */

#ifndef PARAMETERS_H
#define PARAMETERS_H

#ifndef ISPC
#include <time.h> // time(NULL)
#endif

#ifndef N // número de particulas (debe ser un 4m^3 para el cristal inicial)
#define N 256
#endif

#ifndef ISPC
#ifndef SEED // rand SEED para las velocidades
#define SEED (time(NULL))
#endif
#endif

#ifndef T0 // isoterma
#define T0 2.0f
#endif

#ifndef RHOI // densidad inicial
#define RHOI 1.2f
#endif

#ifndef RCUT // radio de corte
#define RCUT 2.5f
#endif

#ifndef DT // paso temporal ~ 1.6 fs para el Ar
#define DT 0.005f
#endif

#ifndef TEQ // pasos de equilibracion
#define TEQ 500
#endif

#ifndef TRUN // TRUN - TEQ: pasos de medicion
#define TRUN 2000
#endif

#ifndef TMES // cada cuantos pasos se mide
#define TMES 10
#endif

#endif
