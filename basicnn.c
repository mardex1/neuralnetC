#include "linalg.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define STEPS 100000
#define N_FEATURES 2
#define N_ROWS 4
#define N_NEURONS 1
#define LEARNING_RATE 0.1

float X[N_ROWS][N_FEATURES] =   {{0, 0},
                    {0, 1},
                    {1, 0},
                    {1, 1}
};

float Y[N_ROWS] = {0, 1, 1, 1};

float W_g[N_FEATURES][N_NEURONS] = {{-2.0},
                                {3.0}};

float rand_float() {
    float r = (float)rand() / (float)RAND_MAX;
    return 2 * r - 1;
}

float** m_sigmoid(float** m, int r, int c) {

    float** tmp = (float**)malloc(sizeof(float*)*r);
    for(int i = 0; i < r; i++) {
        tmp[i] = (float*)malloc(sizeof(float)*c);
        for(int j = 0; j < c; j++) {
            tmp[i][j] = 1 / (1 + expf(-m[i][j]));
        }
    }

    return tmp;
}

float sigmoidf(float x) {
    return 1 / (1 + expf(-x));
}

float compute_cost_f(float* y_true, float **m, int r, int c) {
    float sum = 0.0;
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            sum += powf((y_true[i] - m[i][j]), 2);
        }
    }
    return sum;
}

float** m_sigmoid_derivative(float** m, int r, int c) {
    float** tmp = (float**)malloc(sizeof(float*)*r);
    for(int i = 0; i < r; i++) {
        tmp[i] = (float*)malloc(sizeof(float)*c);
        for(int j = 0; j < c; j++) {
            tmp[i][j] = sigmoidf(m[i][j])*(1 - sigmoidf(m[i][j]));
        }
    }

    return tmp;
}

float** mse_derivative(float** p, float* y_true, int r, int c) {
    float** tmp = (float**)malloc(sizeof(float*)*r);
    for(int i = 0; i < r; i++) {
        tmp[i] = (float*)malloc(sizeof(float)*c);
        for(int j = 0; j < c; j++) {
            tmp[i][j] = 2*(p[i][j] - y_true[i]);
        }
    }

    return tmp;
}

float** update_param(float** m, float** dm, int r, int c) {
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            m[i][j] = m[i][j] - (LEARNING_RATE*dm[i][j]);
        }
    }
    return m;
}

float* update_param_b(float* b, float** db, int r, int c) {
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            b[i] = b[i] - (LEARNING_RATE*db[j][i]);
        }
    }
    return b;
}

float** forward(float** X, float** W, float* b) {
    // XW = XW
    float** XW = m_mul(X, W, N_ROWS, N_FEATURES, N_NEURONS);
    // z = xw1 + b - 4x1
    float** Z = m_add_bc(XW, b, N_ROWS, N_FEATURES);
    // a = sigmoid(Z) - 4x1
    float** A = m_sigmoid(Z, N_ROWS, N_NEURONS);

    free_matrix(Z, N_ROWS);

    return A;

}

int backprop(float** A, float** X, float* y_true, float** w1, float* b1) {
    // (a - y_true) - 4x1
    float** da = mse_derivative(A, y_true, N_ROWS, N_NEURONS);

    float** dz_tmp = m_sigmoid_derivative(da, N_ROWS, N_NEURONS);
    float** dz = m_mul_elem_wise(da, dz_tmp, N_ROWS, N_NEURONS);

    float** X_transposed = transpose(X, N_ROWS, N_FEATURES);
    float** dw = m_mul(X_transposed, dz, N_FEATURES, N_ROWS, N_NEURONS);

    w1 = update_param(w1, dw, N_FEATURES, N_NEURONS);
    b1 = update_param_b(b1, dz, N_ROWS, N_NEURONS);

    free_matrix(da, N_ROWS);
    free_matrix(dz_tmp, N_ROWS);
    free_matrix(X_transposed, N_FEATURES);

    return 1;
}

int main () {
    srand(time(0));
    
    // 4x2
    float** X_ptr = (float**)malloc(sizeof(float*)*N_ROWS);
    for(int i = 0; i < N_ROWS; i++) {
        X_ptr[i] = (float*)malloc(sizeof(float)*N_FEATURES);
        for(int j = 0; j < N_FEATURES; j++) {
            X_ptr[i][j] = X[i][j];
        }
    }
    // (1x4)
    float* y_true =  (float*)malloc(sizeof(float)*N_ROWS);
    for(int i = 0; i < N_ROWS; i++) {
        y_true[i] = Y[i];
    }

    // Initialization (2x1)
    float** w1 = (float**)malloc(sizeof(float*)*N_FEATURES);
    for(int i = 0; i < N_FEATURES; i++) {
        w1[i] = (float*)malloc(sizeof(float)*N_NEURONS);
        for(int j = 0; j < N_NEURONS; j++) {
            w1[i][j] = rand_float();
        }
    }
    // (1x1)
    float* b1 = (float*)malloc(sizeof(float)*N_NEURONS);
    for(int i = 0; i < N_NEURONS; i++) {
        b1[i] = 1;
    }

    float** a;
    float c;

    for(int i = 0; i < STEPS; i++) {
        a = forward(X_ptr, w1, b1);

        c = compute_cost_f(y_true, a, N_ROWS, N_NEURONS);
        printf("%f\n", c);

        backprop(a, X_ptr, y_true, w1, b1);
    }

    printf("\n");
    print_matrix(a, N_ROWS, N_NEURONS);

    free_matrix(w1, N_FEATURES);
    free_matrix(X_ptr, N_ROWS);
    free(b1);
    free(y_true);

    return 0;
}