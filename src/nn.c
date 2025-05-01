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
        m.weight_matrices[i].matrix = (float**)malloc(sizeof(float*)*neurons_per_layer[i]);
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
        for(int j = 0; j < neurons_per_layer[i+1]; j++) {
            m.bias_vectors[i].vector[j] = 1;
        }
    }

    m.input_matrices = (m_weight*)malloc(sizeof(m_weight)*n_layers);

    return m;
}

void select_random_samples(m_weight x, v_weight y, float** X_batch, float* Y_batch) {
    for(int i = 0; i < BATCH_SIZE; i++) {
        int r = rand() % x.row;

        X_batch[i] = x.matrix[r];
        Y_batch[i] = y.vector[r];
    }
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

float sigmoidf(float x) {
    return 1 / (1 + expf(-x));
}

float** m_relu(float** m, int r, int c) {
    float** tmp = (float**)malloc(sizeof(float*)*r);
    for(int i = 0; i < r; i++) {
        tmp[i] = (float*)malloc(sizeof(float)*c);
        for(int j = 0; j < c; j++) {
            if(m[i][j] > 0) {
                tmp[i][j] = m[i][j];
            }else{
                tmp[i][j] = 0;
            }
        }
    }
    return tmp;
}

float** m_relu_derivative(float** m, int r, int c) {
    float** tmp = (float**)malloc(sizeof(float*)*r);
    for(int i = 0; i < r; i++) {
        tmp[i] = (float*)malloc(sizeof(float)*c);
        for(int j = 0; j < c; j++) {
            // Convention, because ReLU is not defined at zero.
            if(m[i][j] > 0) tmp[i][j] = 1;
            else tmp[i][j] = 0;
        }
    }

    return tmp;
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

m_weight linear(m_weight X, m_weight W, v_weight B) {
    float** XW = m_mul(X.matrix, W.matrix, X.row, W.row, W.col);
    // z = xw1 + b - 4x1
    m_weight xw = create_m_weight(XW, X.row, W.col);

    float** Z = m_add_bc(xw.matrix, B.vector, xw.row, xw.col);

    m_weight z = create_m_weight(Z, xw.row, xw.col);

    return z;
}

m_weight forward(m_weight X, model m) {
    for(int i = 0; i < m.n_layers; i++) {
        
        
        m.input_matrices[i] = X;
        X = linear(X, m.weight_matrices[i], m.bias_vectors[i]);
        // (32, 4) = (32, 8)*(8, 4) + (4,) 
        // (32,2) = (32, 4)*(4, 2) + (4,)
        X.matrix = m_sigmoid(X.matrix, X.row, X.col);

    }
    m_weight a = create_m_weight(X.matrix, X.row, X.col);

    return a;
}

m_weight create_m_weight(float** m, int rows, int cols) {
    m_weight x;

    x.matrix = m;
    x.row = rows;
    x.col = cols;

    return x;
}

v_weight create_v_weight(float* v, int len) {
    v_weight x;

    x.vector = v;
    x.length = len;

    return x;
}

void backprop(m_weight a, m_weight x, v_weight Y, model m) {
    // (a - y_true) - 4x1
    // (32, 2)
    float** DA = mse_derivative(a.matrix, Y.vector, a.row, a.col);
    m_weight da = create_m_weight(DA, a.row, a.col);
    // dc/dz = dc/da*da/dz
    // da/dz

    m_weight z;

    for(int i = m.n_layers-1; i >= 0; i--) {
        float** DZ_TMP = m_sigmoid_derivative(da.matrix, da.row, da.col);
        m_weight dz_tmp = create_m_weight(DZ_TMP, da.row, da.col);

        // dc/da*da/dz
        float** DZ = m_mul_elem_wise(da.matrix, dz_tmp.matrix, da.row, da.col);
        m_weight dz = create_m_weight(DZ, da.row, da.col);

        if (i == 0) {
            z = x;
        }else{
            z = m.input_matrices[i];
        }

        float** Z_transposed = transpose(z.matrix, z.row, z.col);
        m_weight z_transposed = create_m_weight(Z_transposed, z.col, z.row);

        float** DW = m_mul(z_transposed.matrix, dz.matrix, z_transposed.row, dz.row, dz.col);
        m_weight dw = create_m_weight(DW, z_transposed.row, dz.col);

        m_weight w = m.weight_matrices[i];
        v_weight b = m.bias_vectors[i];
        m.weight_matrices[i].matrix = update_param(w.matrix, dw.matrix, dw.row, dw.col);
        m.bias_vectors[i].vector = update_param_b(b.vector, dz.matrix, dz.row, dz.col);

        if(i > 0) {
            float** W_transposed = transpose(w.matrix, w.row, w.col);
            m_weight w_transposed = create_m_weight(W_transposed, w.col, w.row);

            da.matrix = m_mul(dz.matrix, w_transposed.matrix, dz.row, w_transposed.row, w_transposed.col);

            free_matrix(w_transposed.matrix, w_transposed.row);
        }
        
        free_matrix(z_transposed.matrix, z_transposed.row);
        free_matrix(dw.matrix, dw.row);
        free_matrix(dz.matrix, dz.row);

    }
    
    free_matrix(da.matrix, da.row);
    
}

void free_model(model m) {
    for(int i = 0; i < m.n_layers; i++) {
        // free(m.bias_vectors[i].vector);
        free_matrix(m.weight_matrices[i].matrix, m.weight_matrices[i].row);
    }
    free(m.weight_matrices);
    free(m.bias_vectors);
}