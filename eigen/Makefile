all: python cpp

python: example.tree example.nexus
	./eigen.py

eigen.hpp: example.tree example.nexus
	./eigen.py --header-only

cpp: eigen.hpp
	g++ -I/usr/include/eigen3 -g -O0 -o eigen eigen.cpp
