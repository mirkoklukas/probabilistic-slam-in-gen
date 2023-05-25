# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``00 - Cuda Utils.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

##################################### 
module =MyCudaUtils  
#####################################

using CUDA
using BenchmarkTools

const _cuda = Ref(false)
function __init__()
    _cuda[] = CUDA.functional()
end;

export _cuda

"""
    griddims = cuda_grid(datadims::Tuple{Vararg{Int}},
                         blockdims::Tuple{Vararg{Int}})

Given data dimensions `datadims` and number of threads
in each dimension `blockdims` returns the respective
grid dimensions `griddims` such that

    griddims[i] = ceil(Int, datadims[i]/blockdims[i])

"""
function cuda_grid(datadims::Tuple{Vararg{Int}}, blockdims::Tuple{Vararg{Int}})
    griddims = ceil.(Int, datadims./blockdims)
    return griddims
end
export cuda_grid

#####################################
end  
#####################################
