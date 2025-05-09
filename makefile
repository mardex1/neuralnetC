CC=gcc

SOURCES = ./src/*.c

all: main run clean

main:
	$(CC) -g -Wall -Wextra -L../lib -I./include $(SOURCES) -o main -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

run: 
	./main

clean:
	rm main