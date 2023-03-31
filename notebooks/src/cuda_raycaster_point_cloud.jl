# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``32 - CUDA Raycaster - Point Cloud.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

push!(LOAD_PATH, "src");
using MyUtils
using CUDA
using BenchmarkTools
using Colors, Plots
col = palette(:default);

function cast_cpu!(z, y, x, hd, fov, cell_rad)

    num_bins = size(z,2)
    bin_res = fov/(num_bins - 1)

    for i = 1:1:size(x,1), j = 1:1:size(y,1)
        d1 = y[j,1] - x[i,1]
        d2 = y[j,2] - x[i,2]

        # Any ray that intersect our cell
        # will be assigned this value
        v  = sqrt(d1^2 + d2^2)

        a′ = atan(d2,d1) - hd[i] + fov/2
        a′ = mod(a′, 2π)

        # Angular resolution of cell y
        da = atan(cell_rad/sqrt(v^2 - min(v, cell_rad)^2));
        a1 = a′ - da;
        a2 = a′ + da;

        k1 = Int(floor(a1/bin_res)) + 1
        k2 = Int(floor(a2/bin_res)) + 1

        # We start to the right of ray `k1` and
        # at each step we crossed ray `k` and
        # fill in the respective depth value `v`.
        for k = k1+1:k2
            # Todo: Resolve crossing two pi
            if 1 <= k && k <= num_bins
                @inbounds z[i,k] = min(z[i,k], v)
            end
        end
    end
    return
end

function bench_cast_cpu!(z, y, x, hd, fov, cell_res)
    cast_slow!(z, y, x, hd, fov, cell_res)
end

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
    cast!(z, y, x, hd, fov, cell_rad)

Collects `n` depth measurements in an obs array `z` of shape `(n, a)`
from `n` poses given by arrays `x` and `hd` of shape `(n,2)` and `(n,)`
and a point cloud `y` of shape `(m,2)`.
"""
function cast!(z, y, x, hd, fov, cell_rad)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    sx = gridDim().x * blockDim().x
    sy = gridDim().y * blockDim().y

    num_bins = size(z,2)
    bin_res = fov/(num_bins - 1)

    for i = ix:sx:size(x,1), j = iy:sy:size(y,1)
        d1 = y[j,1] - x[i,1]
        d2 = y[j,2] - x[i,2]
        v  = sqrt(d1^2 + d2^2)
        a′ = atan(d2,d1) - hd[i] + fov/2
        a′ = mod(a′, 2π)

        # Angular resolution of cell y
        da = atan(cell_rad/sqrt(v^2 - min(v, cell_rad)^2));
        a1 = a′ - da;
        a2 = a′ + da;

        k1 = Int(floor(a1/bin_res)) + 1
        k2 = Int(floor(a2/bin_res)) + 1
        for k = k1+1:k2
            # Todo: Resolve crossing two pi
            if 1 <= k && k <= num_bins
                @inbounds z[i,k] = min(z[i,k], v)
            end
        end
    end
    return
end

function bench_cast!(z_, y_, x_, hd_, fov, cell_rad)
    n = size(x_,1)
    m = size(y_,1)

    blockdims = (16,16)
    griddims = cuda_grid((n,m), blockdims)

    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims cast!(z_, y_, x_, hd_, fov, cell_rad)
    end
end;
