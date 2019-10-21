#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>

#include "utils.hpp"

#define TEST_SZ 100
using namespace std;

int main(int argc, char const *argv[]){
    float *data = new float[TEST_SZ]();

    std::srand(std::time(nullptr)); // use current time as seed for random generator

    for (size_t i = 0; i < TEST_SZ; i++){
        data[i] = (float)std::rand() / RAND_MAX;
        if (i % 10 == 0 && i > 0){
            printf("\n");
        }
        printf("%5.3f ", data[i]);
    }
    printf("\n[I]after scaling: \n");
    inplace_scale_2uint8(data, TEST_SZ, .2, 98);
    for (size_t i = 0; i < TEST_SZ; i++){
        if (i % 10 == 0 && i > 0){
            printf("\n");
        }
        printf("%5.1f ", data[i]);
    }
}