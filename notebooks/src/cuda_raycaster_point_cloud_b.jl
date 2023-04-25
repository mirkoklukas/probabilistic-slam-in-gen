# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``32b - CUDA Raycaster - Point Cloud.ipynb''
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
    cast_kernel!(z, y, x, hd, fov, cell_rad)

Collects `n` depth measurements in an obs array `z` of shape `(n, a)`
from `n` poses given by arrays `x` and `hd` of shape `(n,2)` and `(n,)`
and a point cloud `y` of shape `(m,2)`.
"""
function cast_kernel!(z, y, p, fov, cell_rad)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    sx = gridDim().x * blockDim().x
    sy = gridDim().y * blockDim().y

    num_a = size(z,2)
    r = fov/(num_a - 1)

    for i = ix:sx:size(p,1), j = iy:sy:size(y,1)
        d1 = y[j,1] - p[i,1]
        d2 = y[j,2] - p[i,2]
        v  = sqrt(d1^2 + d2^2)
        a′ = atan(d2,d1) - p[i,3]

        # Angular resolution of cell y
        da = atan(cell_rad/sqrt(v^2 - min(v, cell_rad)^2));
        # Todo: What is a good minimal value
        da = min(da, 10*r)
        a1 = a′ - da;
        a2 = a′ + da;

        a1 = mod(a1 + π, 2π) - π
        a2 = mod(a2 + π, 2π) - π

        # Ensure a1 < a2
        if a1 > a2
            a1, a2 = a2, a1
        end

        # Check if we cross from `-π+a` to `π-b`
        if a2 - a1 > π
            zero = - fov/2;
            k1 = Int(floor((a2    - zero)/r))+1
            k2 = Int(floor((a1+2π - zero)/r))+1

            k1′ = Int(floor((a2-2π - zero)/r))+1
            k2′ = Int(floor((a1    - zero)/r))+1

            ks = ((k1,k2), (k1′,k2′))
        else
            zero = - fov/2;
            k1 = Int(floor((a1 - zero)/r))+1
            k2 = Int(floor((a2 - zero)/r))+1

            ks = ((k1,k2),)
        end

        for (k1,k2) in ks, k = k1+1:k2
            if 1 <= k && k <= num_a
                @inbounds CUDA.@atomic z[i,k] = min(z[i,k], v)
            end
        end
    end
    return
end

"""
    cast_cu!(z_, y_, p_; fov=π, cell_rad=0.005, blockdims=(16,16))

Collects `n` sensor scans in an obs array `z` of shape `(n, a)`
from `n` poses given by an array `p` `(n,3)` and `(n,)`
and a point cloud `y` of shape `(m,2)`.
"""
function cast_cu!(z_, y_, p_; fov=π, cell_rad=0.01, blockdims=(16,16))
    n = size(p_,1)
    m = size(y_,1)

    griddims = cuda_grid((n,m), blockdims)
    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims cast_kernel!(z_, y_, p_, fov, cell_rad)
    end
end;
