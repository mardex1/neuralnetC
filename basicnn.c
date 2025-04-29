#include "linalg.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float X[N_ROWS][N_FEATURES] =   {{0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 1},
{0, 0, 0, 0, 0, 0, 1, 0},
{0, 0, 0, 0, 0, 0, 1, 1},
{0, 0, 0, 0, 0, 1, 0, 0},
{0, 0, 0, 0, 0, 1, 0, 1},
{0, 0, 0, 0, 0, 1, 1, 0},
{0, 0, 0, 0, 0, 1, 1, 1},
{0, 0, 0, 0, 1, 0, 0, 0},
{0, 0, 0, 0, 1, 0, 0, 1},
{0, 0, 0, 0, 1, 0, 1, 0},
{0, 0, 0, 0, 1, 0, 1, 1},
{0, 0, 0, 0, 1, 1, 0, 0},
{0, 0, 0, 0, 1, 1, 0, 1},
{0, 0, 0, 0, 1, 1, 1, 0},
{0, 0, 0, 0, 1, 1, 1, 1},
{0, 0, 0, 1, 0, 0, 0, 0},
{0, 0, 0, 1, 0, 0, 0, 1},
{0, 0, 0, 1, 0, 0, 1, 0},
{0, 0, 0, 1, 0, 0, 1, 1},
{0, 0, 0, 1, 0, 1, 0, 0},
{0, 0, 0, 1, 0, 1, 0, 1},
{0, 0, 0, 1, 0, 1, 1, 0},
{0, 0, 0, 1, 0, 1, 1, 1},
{0, 0, 0, 1, 1, 0, 0, 0},
{0, 0, 0, 1, 1, 0, 0, 1},
{0, 0, 0, 1, 1, 0, 1, 0},
{0, 0, 0, 1, 1, 0, 1, 1},
{0, 0, 0, 1, 1, 1, 0, 0},
{0, 0, 0, 1, 1, 1, 0, 1},
{0, 0, 0, 1, 1, 1, 1, 0},
{0, 0, 0, 1, 1, 1, 1, 1}
};

float Y[N_ROWS] = {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0};

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
    m_weight x = create_m_weight(X_ptr, N_ROWS, N_FEATURES);
    // (1x4)
    float* y_true =  (float*)malloc(sizeof(float)*N_ROWS);
    for(int i = 0; i < N_ROWS; i++) {
        y_true[i] = Y[i];
    }

    v_weight Y = create_v_weight(y_true, N_ROWS);

    int* neurons_per_layer = (int*)malloc(sizeof(int)*3);
    neurons_per_layer[0] = 16;
    neurons_per_layer[1] = 8;
    neurons_per_layer[2] = 1;

    model m = instanciate_model(2, neurons_per_layer);
    m.input_matrices[0] = x;
    
    m_weight a = forward(x, m);

    float c = compute_cost_f(Y.vector, a.matrix, a.row, a.col);
    printf("%f\n", c);

    backprop(a, x, Y, m);

    free_model(m);
    free_matrix(x.matrix, x.row);
    free(y_true);

    return 0;

    // for(int i = 0; i < STEPS; i++) {
    //     a = forward(X_ptr, m);

    //     c = compute_cost_f(y_true, a, N_ROWS, N_NEURONS);
    //     printf("%f\n", c);

    //     backprop(a, X_ptr, y_true, w1, b1);
    // }

    // free_model(m);

    // printf("\n");
    // print_matrix(a, N_ROWS, N_NEURONS);

    // // free_matrix(w1, N_FEATURES);
    // free_matrix(X_ptr, N_ROWS);
    // // free(b1);
    // free(y_true);

    return 0;
}