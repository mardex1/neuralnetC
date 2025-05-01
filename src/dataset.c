#include "dataset.h"

void read_csv_file(char* path, float** X, float* Y) {
    FILE* fp;
    char s[2048];
    int line = 0;

    fp = fopen(path, "r");

    while(fgets(s, sizeof s, fp) != NULL) {
        char* token = strtok(s, ",");
        int column = 0;
        while (token != NULL) {

            if (column == 0) {
                Y[line] = atoi(token);
            }else{
                X[line][column] = atoi(token);
            }

            column++;
            token = strtok(NULL, ",");
        }
        line++;
    }
    fclose(fp);
}

void plot_digit(float** X, int row) {

    InitWindow(400, 400, "Digit_Plotting");

    // Create game loop
    while(!WindowShouldClose()) {
        BeginDrawing();
        // All drawing happens
        
        ClearBackground(GRAY);
        int k = 0;
        for(int i = 0; i < 28*8; i+=8) {
            for(int j = 0; j < 28*8; j+=8) {
                Color c = {(int)X[row][k], (int)X[row][k], (int)X[row][k], 255};
                DrawRectangle(GetScreenWidth()/2 + (j - (28*8)/2),GetScreenHeight()/2 + (i - (28*8)/2), 8, 8, c);
                k++;   
            }
        } 

        EndDrawing();
    }

    CloseWindow();
}