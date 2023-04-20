# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``31b - CUDA Raycaster - Line Map.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

push!(LOAD_PATH, "src");
using MyUtils
using CUDA
using BenchmarkTools
using Colors, Plots
col = palette(:default);
import Base: @doc

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

function line_intersection(x1, x2, x1′, x2′, y1, y2, y1′, y2′)
    dx1, dx2 = x1′ - x1, x2′ - x2
    dy1, dy2 = y1′ - y1, y2′ - y2

    v1 = (x1 - y1)
    v2 = (x2 - y2)

    a, b = -dx1, dy1
    c, d = -dx2, dy2

    det = a*d - c*b

    if det == 0
        return Inf,Inf
    end

    s = 1/det*(  d*v1 - b*v2)
    t = 1/det*(- c*v1 + a*v2)

    return s,t
end;


function cast_kernel!(Z, X, P, fov=2π)

    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    sx = gridDim().x * blockDim().x
    sy = gridDim().y * blockDim().y

    num_a = size(Z,2)
    r     = fov/(num_a-1)

    for i = ix:sx:size(X,1), j = iy:sy:size(P,1)

        # convert everything into pose coords
        x1 , x2  = X[i,1] - P[j,1], X[i,2] - P[j,2]
        x1′, x2′ = X[i,3] - P[j,1], X[i,4] - P[j,2]
        dx1, dx2 = x1′-x1, x2′-x2
        y1 , y2  = 0, 0
        a1 = atan(x2 , x1 ) - P[j,3]
        a2 = atan(x2′, x1′) - P[j,3]
        a1 = mod(a1 + π, 2π) - π
        a2 = mod(a2 + π, 2π) - π

        # Ensure a1 < a2
        if a1 > a2
            a1, a2 = a2, a1
        end


        # Check if we cross from `-π+a` to `π-b`
        if a2 - a1 > π
            # Get the start end end bin
            zero = - fov/2;
            k1 = Int(floor((-π + r/2 - zero)/r))+1
            k2 = Int(floor((a1 + r/2 - zero)/r))+1

            k1′ = Int(floor((a2 + r/2 - zero)/r))+1
            k2′ = Int(floor((π + r/2 - zero)/r))+1

            ks = ((k1,k2),(k1′,k2′))
        else
            # Get the start end end bin
            zero = - fov/2;
            k1 = Int(floor((a1 + r/2 - zero)/r))+1
            k2 = Int(floor((a2 + r/2 - zero)/r))+1

            ks = ((k1,k2),)

        end


        for (k1,k2) in ks, k = k1:k2
            if !(1 <= k <= num_a)
               continue
            end

            a = zero + (k-1)*r + P[j,3]
            y1′, y2′ = cos(a), sin(a)

            s, t = line_intersection(x1, x2, x1′, x2′, y1, y2, y1′, y2′)
            if 0 < t && 0 <= s <= 1
                @inbounds CUDA.@atomic Z[j,k] = min(Z[j,k], t)
            end


        end
    end
    return
end

function cast_cu!(Z, X, P; fov=2π, blockdims=(16,16))
    n = size(X,1)
    m = size(P,1)

    griddims = cuda_grid((n,m), blockdims)
    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims cast_kernel!(Z, X, P, fov)
    end
end
