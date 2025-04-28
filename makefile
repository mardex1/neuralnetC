CC=gcc

all: basicnn run

basicnn: basicnn.c
	$(CC) -g -Wall -Wextra basicnn.c linalg.c nn.c -o basicnn -lm

run: 
	./basicnn

rm:
	rm basicnn