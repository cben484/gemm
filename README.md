# version
1.2 ( the version1.1 is just a mess )

# attention
This is my first structured engineering cuda project.
Referring to the maolei (github.com/leimao).
It involves a large number of cpp template,std::pair,std::function and other advanced techniques(which tooks me lots of time).

# bug record:
## 1. cpp problems :
### 1.1 function calls
navigation : include/utils.cuh; when i call the elapsedTime() in the measure_performance, i just passed the para stream in  bound_function(stream).

origin code : return elapsedTime(bound_function(stream), stream, num_warmups, num_repeats);

obviously,it's not a passing para,it's just a function call, actually,it's means :return elapsedTime(value, stream, num_warmups, num_repeats); the value is a return value of bound_function,but i need a function,it's where the problem lies.

amend code : return elapsedTime(bound_function, stream, num_warmups, num_repeats);

## 2. cuda problems :
### 2.1 grid allocation
navigation : src/00_gemm_naive.cu;for launch, i need to assign the value of grid.

origin code : dim3 dimGrid(((m * n) + dimBlock.x - 1) / dimBlock.x , ((m * n) + dimBlock.y - 1) / dimBlock.y);

not obviously,i just use the whole size of a matrix to divide all dimension,actually,it's wrong,in x dimension,you should use the m or n to divide.Because,for each GPU ,the maximum sizes of each dimension of a grid is limited,for example ,for 1080ti,the maximum sizes of each dimension of a grid is  2147483647 x 65535 x 65535.So, when the dimetion is exceeded the critical point
, the program would be out of order.

amend code : dim3 dimGrid(((m) + dimBlock.x - 1) / dimBlock.x , ((n) + dimBlock.y - 1) / dimBlock.y);