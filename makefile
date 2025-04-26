

all: basicnn

basicnn: basicnn.c
	gcc -g -Wall -Wextra basicnn.c linalg.c -o basicnn -lm