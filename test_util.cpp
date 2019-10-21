#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>

#include "utils.hpp"

#define TEST_SZ (1024 * 1024)
// #define TEST_SZ 100
using namespace std;

int main(int argc, char const *argv[]){
    float *data = new float[TEST_SZ]();

    // std::srand(std::time(nullptr)); // use current time as seed for random generator
    std::srand(20191021);

    for (size_t i = 0; i < TEST_SZ; i++){
        data[i] = (float)std::rand() / RAND_MAX;
        // data[i] = i / 100.;
        // if (i % 10 == 0 && i > 0){
        //     printf("\n");
        // }
        // printf("%5.3f ", data[i]);
    }
    // printf("\n[I]after scaling: \n");

    auto time_st = chrono::steady_clock::now();
    for (size_t i = 0; i < 20; i++){
        inplace_scale_2uint8(data, TEST_SZ, 2, 98);
    }
    
    auto time_ed = chrono::steady_clock::now();
    printf("It takes %.3f ms to scale\n", \
           chrono::duration_cast<chrono::microseconds>(time_ed - time_st ).count()/1000.);

    for (size_t i = 0; i < 20; i++){
        if (i % 10 == 0 && i > 0){
            printf("\n");
        }
        printf("%5.1f ", data[i]);
    }
    printf("\n");
}