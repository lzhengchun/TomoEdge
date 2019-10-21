#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <chrono>
#include <queue>
using namespace std;

// keep smallest k items, the top will the kth largest
template <typename T> 
T nth4down(T * data, size_t len, size_t k){
    priority_queue <T> max_heap; 
    for (size_t i = 0; i < len; i++){
        max_heap.push(data[i]);
        if(max_heap.size() > k){
            max_heap.pop(); //  pop the largest so far
            }
    }
    return max_heap.top();
}

template <typename T> 
T nth4up(T * data, size_t len, size_t k){
    std::priority_queue<T, std::vector<T>, std::greater<T> > min_heap;
    for (size_t i = 0; i < len; i++){
        min_heap.push(data[i]);
        if(min_heap.size() > k){
            min_heap.pop(); //  pop the smallest so far
            }
    }
    T ret = min_heap.top();
    while(!min_heap.empty()){
        ret = min_heap.top();
        min_heap.pop();
    }
    return ret;
}

template <typename T> 
void inplace_scale_2uint8(T *data, size_t len, float down_percentile, float up_percentile){
    std::vector<T> vdata;
    vdata.assign(data, data + len); // just want to use nth_element
    size_t nth_down = (size_t)(down_percentile * len / 100.);
    size_t nth_up   = (size_t)(up_percentile * len / 100.);
    std::nth_element(vdata.begin(), vdata.begin() + nth_down, vdata.end());
    T scale_down = vdata[nth_down];
    std::nth_element(vdata.begin(), vdata.begin() + nth_up, vdata.end());
    T scale_up   = vdata[nth_up];

    // T scale_down = data[0], scale_up = data[0];
    // for (size_t i = 0; i < len; i++){
    //     scale_down = std::min(scale_down, data[i]);
    //     scale_up   = std::max(scale_up, data[i]);
    // }
    
    T scale_factor = 255 / (scale_up - scale_down);
    #pragma omp parallel for schedule(static, 204800)
    for (size_t i = 0; i < len; i++){
        data[i] = std::max(data[i], scale_down);
        data[i] = std::min(data[i], scale_up);
        data[i] = scale_factor * (data[i] - scale_down);
    }
}
