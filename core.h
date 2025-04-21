#ifndef CORE_H
#define CORE_H

void init_pos(float* rxyz, const float rho);
void init_vel(float* vxyz, float* temp, float* ekin);
void forces(const float* rxyz, float* fxyz, float* epot, float* pres,
            const float* temp, const float rho, const float V, const float L);
void velocity_verlet(float* rxyz, float* vxyz, float* fxyz, float* epot,
                     float* ekin, float* pres, float* temp, const float rho,
                     const float V, const float L);

#endif
