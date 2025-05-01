#include "linalg.h"
#include "nn.h"
#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <raylib.h>

int neurons[N_LAYERS] = {784, 64, 1};

int main () {
    srand(time(0));
    
    float** X_ptr = (float**)malloc(sizeof(float*)*N_ROWS);
    for(int i = 0; i < N_ROWS; i++) {
        X_ptr[i] = (float*)malloc(sizeof(float)*N_FEATURES);
    }
    float* Y =  (float*)malloc(sizeof(float)*N_ROWS);

    read_csv_file("../mnist/mnist_test.csv", X_ptr, Y);

    m_weight x = create_m_weight(X_ptr, N_ROWS, N_FEATURES);
    v_weight y = create_v_weight(Y, N_ROWS);

    float** X_batch = (float**)malloc(sizeof(float*)*BATCH_SIZE);
    for(int i = 0; i < BATCH_SIZE; i++) {
        X_batch[i] = (float*)malloc(sizeof(float)*N_FEATURES);
    }

    float* Y_batch = (float*)malloc(sizeof(float)*BATCH_SIZE);

    int* neurons_per_layer = (int*)malloc(sizeof(int)*N_LAYERS);
    for(int i = 0; i < N_LAYERS; i++) {
        neurons_per_layer[i] = neurons[i];
    }

    model m = instanciate_model(N_LAYERS-1, neurons_per_layer);

    float c;
    float cost_per_epoch[STEPS];

    for(int i = 0; i < EPOCHS; i++) {
        for(int j = 0; j < STEPS; j++) {
            select_random_samples(x, y, X_batch, Y_batch);
            m_weight x_batch = create_m_weight(X_batch, BATCH_SIZE, N_FEATURES);
            v_weight y_batch = create_v_weight(Y_batch, BATCH_SIZE); 
            
            m_weight a = forward(x_batch, m);

            c = compute_cost_f(y_batch.vector, a.matrix, a.row, a.col);
            cost_per_epoch[i] = c;
            printf("%f\n", c);

            backprop(a, x_batch, y_batch, m);
        }
        float sum = 0;
        for(int k = 0; k < STEPS; k++) {
            sum += cost_per_epoch[k];
        }
        printf("%f\n", sum);
    }

    

    free_model(m);
    free_matrix(x.matrix, x.row);
    // free_matrix(x_batch.matrix, x_batch.row);
    free(Y_batch);
    free(Y);

    return 0;

}