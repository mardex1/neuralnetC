#include "linalg.h"
#include "nn.h"
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

float Y[N_ROWS] = {1, 1, 1, 0};

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

    int* neurons_per_layer = (int*)malloc(sizeof(int)*2);
    neurons_per_layer[0] = N_FEATURES;
    neurons_per_layer[1] = 1;

    model m = instanciate_model(1, neurons_per_layer);
    for(int i = 0; i < m.n_layers; i++) {
        print_matrix(m.weight_matrices[i].matrix, m.weight_matrices[i].row, m.weight_matrices[i].col);
        print_vector(m.bias_vectors[i].vector, m.bias_vectors[i].length);
        printf("(%d, %d)\t", m.weight_matrices[i].row, m.weight_matrices[i].col);
    }printf("\n");

    return 0; 

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