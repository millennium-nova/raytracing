FLAG = -Wall -O3 -std=c++14
day3.exe: day3.cpp
	g++ $(FLAG) $^ -o $@
%.ppm: day3.exe
	./day3.exe > $@
all: day3.exe
clean:
	rm -f day3.exe *.ppm
.PHONY: all clean