#include "linalg.h"
#include "nn.h"
#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <raylib.h>

int neurons[N_LAYERS] = {784, 128, 64, 10};

int main () {
    srand(time(0));
    
    float** X_ptr = (float**)malloc(sizeof(float*)*N_ROWS);
    for(int i = 0; i < N_ROWS; i++) {
        X_ptr[i] = (float*)malloc(sizeof(float)*N_FEATURES);
    }
    float* Y =  (float*)malloc(sizeof(float)*N_ROWS);

    read_csv_file("mnist/mnist_train.csv", X_ptr, Y);

    // Normalize data
    m_divide_s(X_ptr, 255, N_ROWS, N_FEATURES);

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
        printf("EPOCH: %d\n\n", i);
        for(int j = 0; j < STEPS; j++) {
            select_random_samples(x, y, X_batch, Y_batch, BATCH_SIZE);
            
            m_weight x_batch = create_m_weight(X_batch, BATCH_SIZE, N_FEATURES);
            v_weight y_batch = create_v_weight(Y_batch, BATCH_SIZE); 
            float** y_one_hot = create_onehot(y_batch.vector, y_batch.length, N_CLASSES);
            m_weight y_hot = create_m_weight(y_one_hot, BATCH_SIZE, N_CLASSES);
            
            m_weight a = forward(x_batch, m);

            c = cross_entropy_loss(y_one_hot, a);
            cost_per_epoch[j] = c;

            if(i < EPOCHS-1) {
                backprop(a, x_batch, y_hot, m);
            }
        }
        float sum = 0;
        for(int k = 0; k < STEPS; k++) {
            sum += cost_per_epoch[k];
        }
        printf("MEAN EPOCH COST: %f\n\n", sum / STEPS);
    }

    free_matrix(X_ptr, x.row);
    // free_matrix(X_batch, BATCH_SIZE);
    free(Y);
    free(Y_batch);

    float** X_test = (float**)malloc(sizeof(float*)*N_ROWS_TEST);
    for(int i = 0; i < N_ROWS_TEST; i++) {
        X_test[i] = (float*)malloc(sizeof(float)*N_FEATURES);
    }
    float* Y_test =  (float*)malloc(sizeof(float)*N_ROWS_TEST);

    read_csv_file("mnist/mnist_test.csv", X_test, Y_test);

    // Normalize data
    m_divide_s(X_test, 255, N_ROWS_TEST, N_FEATURES);

    m_weight x_ts = create_m_weight(X_test, N_ROWS_TEST, N_FEATURES);
    v_weight y_ts = create_v_weight(Y_test, N_ROWS_TEST);

    // Predict
    m_weight activation = forward(x_ts, m);

    int predictions[N_ROWS_TEST];
    for(int i = 0; i < activation.row; i++) {
        float max = 0;
        int max_idx;
        for(int j = 0; j < activation.col; j++) {
            if(activation.matrix[i][j] > max) {
                max = activation.matrix[i][j];
                max_idx = j;
            }
        }
        predictions[i] = max_idx;
    }

    float accuracy = 0;
    for(int i = 0; i < N_ROWS_TEST; i++) {
        if(y_ts.vector[i] == predictions[i]) {
            accuracy += 1;
        }
    }
    printf("Accuracy: %f\n", accuracy / N_ROWS_TEST * 100);
    
    free_matrix(X_test, N_ROWS_TEST);
    free(Y_test);
    free_model(m);

    return 0;

}