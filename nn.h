#ifndef __NN_H_
#define __NN_H_

#define STEPS 100000
#define N_FEATURES 2
#define N_ROWS 4
#define N_NEURONS 1
#define LEARNING_RATE 0.1

typedef struct m_weight {
    float** matrix;
    int row;
    int col;
}m_weight;

typedef struct v_weight {
    float* vector;
    int length;
}v_weight;
typedef struct model {
    int n_layers;
    int* neurons_per_layer;
    m_weight* weight_matrices;
    v_weight* bias_vectors;
}model;

float rand_float();

float** m_sigmoid(float** m, int r, int c);

float sigmoidf(float x);

float compute_cost_f(float* y_true, float **m, int r, int c);

float** m_sigmoid_derivative(float** m, int r, int c);

float** mse_derivative(float** p, float* y_true, int r, int c);

float** update_param(float** m, float** dm, int r, int c);

float* update_param_b(float* b, float** db, int r, int c);

float** forward(float** X, float** W, float* b);

int backprop(float** A, float** X, float* y_true, float** w1, float* b1);

model instanciate_model(int n_layers, int* neurons_per_layer);

#endif