#include "linalg.h"
#include <string.h>

float** m_add_bc(float** m, float* b, int r, int c) {
    // Expects a matriz rxc and a vector 1xc
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            m[i][j] += b[j];
        }
    }

    return m;
}

float** m_add(float** a, float** b, int n, int m) {
    float** res = (float**)malloc(sizeof(float*)*n);
    for(int i = 0; i < n; i++) {
        res[i] = (float*)malloc(sizeof(float)*m);
        for(int j = 0; j < m; j++) {
            res[i][j] = a[i][j] + b[i][j];
        }
    }

    return res;
}

float** m_mul_s(float** a, float s, int n, int m) {
    float** res = (float**)malloc(sizeof(float*)*n);
    for(int i = 0; i < n; i++) {
        res[i] = (float*)malloc(sizeof(float)*m);
        for(int j = 0; j < m; j++) {
            res[i][j] = a[i][j] * s;
        }
    }
    return res;
}

float** m_add_s(float** a, float s, int n, int m){
    float** res = (float**)malloc(sizeof(float*)*n);
    for(int i = 0; i < n; i++) {
        res[i] = (float*)malloc(sizeof(float)*m);
        for(int j = 0; j < m; j++) {
            res[i][j] = a[i][j] + s;
        }
    }
    return res;
}

float** m_mul_elem_wise(float** a, float** b, int n, int m) {
    float** res = (float**)malloc(sizeof(float*)*n);
    for(int i = 0; i < n; i++) {
        res[i] = (float*)malloc(sizeof(float)*m);
        for(int j = 0; j < m; j++) {
            res[i][j] = a[i][j] * b[i][j];
        }
    }
    return res;
}

float** m_sub_v(float** a, float* b, int n, int m) {
    float** res = (float**)malloc(sizeof(float*)*n);
    for(int i = 0; i < n; i++) {
        res[i] = (float*)malloc(sizeof(float)*m);
        for(int j = 0; j < m; j++) {
            res[i][j] = a[i][j] - b[i];
        }
    }

    return res;
}

float** m_mul(float** a, float** b, int n_lines, int in_features, int out_features) {
    float** res = (float**)malloc(sizeof(float*)*n_lines);
    for(int i = 0; i < n_lines; i++) {
        res[i] = (float*)malloc(sizeof(float)*out_features);
        for(int j = 0; j < out_features; j++) {
            res[i][j] = 0;
            for(int k = 0; k < in_features; k++) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return res;
}

float** transpose(float** a, int n, int m) {
    float** res = (float**)malloc(sizeof(float*)*m);
    for(int i = 0; i < m; i++) {
        res[i] = (float*)malloc(sizeof(float)*n);
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            res[i][j] = a[j][i];
        }
    }

    return res;
}

void print_matrix(float** m, int r, int c) {
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            printf("%f\t", m[i][j]);
        }printf("\n");
    }
    printf("\n\n");
}

void print_vector(float* v, int l) {
    for(int i = 0; i < l; i++) {
        printf("%f\t", v[i]);
    }printf("\n\n");
}

void free_matrix(float** a, int n) {
    for(int i = 0; i < n; i++) {
        free(a[i]);
    }
    free(a);
}

