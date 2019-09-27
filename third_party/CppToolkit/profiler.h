#include <chrono> 
#include <iostream>
// accumulative timer with elt as a double
#define TIMER_START(elt) \
        { \
            auto _start_##elt = std::chrono::high_resolution_clock::now();
#define TIMER_END(elt) \
            auto _end_##elt = std::chrono::high_resolution_clock::now(); \
            elt += std::chrono::duration_cast<std::chrono::nanoseconds>(_end_##elt - _start_##elt).count(); \
        }
/*
// speedup of two timers
#define PRINT_SPEEDUP(x, y) \
    std::cout << std::setprecision(12) << x << " ns\n"; \
    std::cout << std::setprecision(12) << y << " ns\n"; \
    std::cout << "(1 - "#x" / "#y")\tSpeedup:\t" << (1 - x / y)*100 << " %\n";

// test if two double array are same
#define PRINT_ARRAY_SIZE 100
#define PRINT_ARRAY(a) \
    cout << #a" :"; \
    for(int i= 0;i < PRINT_ARRAY_SIZE;i++) std::cout << a[0][i] << " \n"[i==PRINT_ARRAY_SIZE-1]; 
#define PRINT_ERROR(TITLE, a, b, size) \
    { \
        double error = 0; \
        for(int i = 0;i < n; i ++) error += ((double *)b)[i] - ((double *)a)[i];\
        std::cout << TITLE" error :" << error << "\n\n"; \
    }

*/
