# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``99 - CUDA_Raycaster_Line_Map.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

using CUDA
using BenchmarkTools
using Colors, Plots
col = palette(:default);

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

"""
    cast!(X, A, p, Z)

Given a segment matrix `X`, angles `A`, a pose `p` encoded as a 3-tuple,
and a depth vector that is to be filled.

Example:
```
n = size(X,1)
m = size(A,1)

blockdims = (16, 16)
griddims = cuda_grid((n,m), blockdims)
CUDA.@sync begin
    @cuda threads=blockdims blocks=griddims cast!(X, A, p, Z)
end
```
"""
function cast!(X, A, p, Z)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    sx = gridDim().x * blockDim().x
    sy = gridDim().y * blockDim().y

    for i = ix:sx:size(X,1), j = iy:sy:size(A,1)

        x1, x2 = X[i,1], X[i,2]
        y1, y2 = p[1], p[2]
        v1 = (x1 - y1)
        v2 = (x2 - y2)

        dx1, dx2 = X[i,3] - X[i,1], X[i,4] - X[i,2]
        dy1, dy2 = cos(A[j]+p[3]), sin(A[j] + p[3])

        a, b = -dx1, dy1
        c, d = -dx2, dy2
        det = a*d - c*b
        if det != 0
            s = 1/det*(  d*v1 - b*v2)
            t = 1/det*(- c*v1 + a*v2)
            if t >= 0 && 0 <= s && s <= 1
                @inbounds Z[j] = min(t,Z[j])
            end
        end

    end
    return
end

function bench_cast!(X, A, p, Z)
    n = size(X,1)
    m = size(A,1)

    blockdims = (16, 16)
    griddims = cuda_grid((n,m), blockdims)
    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims cast!(X, A, p, Z)
    end
end
