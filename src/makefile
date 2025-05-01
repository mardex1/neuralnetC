CC=gcc

SOURCES = ./*.c

all: basicnn

basicnn: basicnn.c
	$(CC) -g -Wall -Wextra -L../lib -I../include $(SOURCES) -o basicnn -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

run: 
	./basicnn

rm:
	rm basicnn