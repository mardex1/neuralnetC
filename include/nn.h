#ifndef __NN_H_
#define __NN_H_

#define N_FEATURES 784
#define N_ROWS 60000
#define N_LAYERS 4
#define LEARNING_RATE 0.0001
#define BATCH_SIZE 64
#define EPOCHS 9
#define STEPS N_ROWS / BATCH_SIZE
#define N_CLASSES 10
#define N_ROWS_TEST 10000

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
    m_weight* input_matrices;
    v_weight* bias_vectors;
}model;

float rand_float();

void select_random_samples(m_weight x, v_weight y, float** X_batch, float* Y_batch, int n_samples);

float** create_onehot(float* y, int r, int classes);

float** m_sigmoid(float** m, int r, int c);

float sigmoidf(float x);

float compute_cost_f(float* y_true, float **m, int r, int c);

float cross_entropy_loss(float** y_one_hot, m_weight a);

float** m_sigmoid_derivative(float** m, int r, int c);

float** mse_derivative(float** p, float* y_true, int r, int c);

float** update_param(float** m, float** dm, int r, int c);

float* update_param_b(float* b, float** db, int r, int c);

m_weight forward(m_weight X, model m);

void backprop(m_weight a, m_weight x, m_weight y_hot, model m);

model instanciate_model(int n_layers, int* neurons_per_layer);

m_weight create_m_weight(float** m, int rows, int cols); 

v_weight create_v_weight(float* v, int len);

void free_model(model m);

#endif