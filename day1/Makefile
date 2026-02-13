FLAG = -Wall -O3 -std=c++14
main.exe: main.cpp
	g++ $(FLAG) $^ -o $@
%.ppm: main.exe
	./main.exe > $@
all: main.exe
clean:
	rm -f main.exe *.ppm
.PHONY: all clean