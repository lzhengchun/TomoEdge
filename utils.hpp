#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

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

    for (size_t i = 0; i < len; i++){
        data[i] = std::max(data[i], scale_down);
        data[i] = std::min(data[i], scale_up);
        data[i] = 255 * (data[i] - scale_down) / (scale_up - scale_down);
    }
}
