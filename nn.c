#include "nn.h"
#include "linalg.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

model instanciate_model(int n_layers, int* neurons_per_layer) {
    model m;
    m.n_layers = n_layers;
    m.neurons_per_layer = neurons_per_layer;

    m.weight_matrices = (m_weight*)malloc(sizeof(m_weight)*n_layers);
    for(int i = 0; i < n_layers; i++) {
        m.weight_matrices[i].matrix = (float**)malloc(sizeof(float)*neurons_per_layer[i]);
        m.weight_matrices[i].row = neurons_per_layer[i];
        for(int j = 0; j < neurons_per_layer[i]; j++) {
            m.weight_matrices[i].matrix[j] = (float*)malloc(sizeof(float)*neurons_per_layer[i+1]);
            m.weight_matrices[i].col = neurons_per_layer[i+1];
            for(int k = 0; k < neurons_per_layer[i+1]; k++) {
                m.weight_matrices[i].matrix[j][k] = rand_float();
            }
        }
    }

    m.bias_vectors = (v_weight*)malloc(sizeof(v_weight)*n_layers);
    for(int i = 0; i < n_layers; i++) {
        m.bias_vectors[i].vector = (float*)malloc(sizeof(float)*neurons_per_layer[i+1]);
        m.bias_vectors[i].length = neurons_per_layer[i+1];
        for(int j = 0; j < neurons_per_layer[i]; j++) {
            m.bias_vectors[i].vector[j] = 1;
        }
    }

    return m;
}

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