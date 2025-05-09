#ifndef __DATASET_H_
#define __DATASET_H_

#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void read_csv_file(char* path, float** X, float* Y);

void plot_digit(float** X, int row);

#endif