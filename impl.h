#ifndef __IMPL_H__
#define __IMPL_H__
#include <vector>
#include <iostream>
void mls_affine(float *image, int h, int w, float *p, float *q, 
                int ctrls, float alpha, float density, float *result);

void mls_affine_inv(float *image, int h, int w, float *p, float *q, 
                int ctrls, float alpha, float density, float *result);

#endif
