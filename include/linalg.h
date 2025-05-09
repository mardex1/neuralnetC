#ifndef __LINALG_H_
#define __LINALG_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

float** m_add(float** a, float** b, int n, int m);

float** m_sub(float** a, float** b, int n, int m);

float** m_mul(float** a, float** b, int n_lines, int in_features, int out_features);

float** transpose(float** a, int n, int m);

void free_matrix(float** a, int n);

float** m_add_bc(float** m, float* b, int r, int c);

float** m_sub_v(float** a, float* b, int n, int m);

float** m_mul_s(float** a, float s, int n, int m);

void m_divide_s(float** a, float s, int n, int m);

float** m_mul_elem_wise(float** a, float** b, int n, int m);

float** m_add_s(float** a, float s, int n, int m);

void print_matrix(float** m, int r, int c);

void print_vector(float* v, int l);

#endif