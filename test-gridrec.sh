#! /bin/bash

rm gridrec.out
nvcc test_gridrec.cpp gridrec.cu -lcufft -o gridrec.out -std=c++11
./gridrec.out