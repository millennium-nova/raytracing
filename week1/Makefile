FLAG = -Wall -O3 -std=c++14
day4.exe: day4.cpp
	g++ $(FLAG) $^ -o $@
%.ppm: day4.exe
	./day4.exe > $@
all: day4.exe
clean:
	rm -f day4.exe *.ppm
.PHONY: all clean