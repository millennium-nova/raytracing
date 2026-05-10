FLAG = -Wall -O3 -std=c++14
day6.exe: day6.cpp
	g++ $(FLAG) $^ -o $@
%.ppm: day6.exe
	./day6.exe > $@
all: day6.exe
clean:
	rm -f day6.exe *.ppm
.PHONY: all clean