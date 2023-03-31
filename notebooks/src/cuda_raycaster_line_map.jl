# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``31 - CUDA Raycaster - Line Map.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

push!(LOAD_PATH, "src");
using MyUtils
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

Given a matrix of stacked segments `X`, a vector of angles `A`, a pose `p` encoded as a 3-tuple,
fills the vector `Z` with depth/range measurements.

Example:

```
x = rand(1_000, 4)
a = range(0, 2π, 100)
z = Inf*ones(length(a))
p = zeros(3)

x_ = CuArray(x)
a_ = CuArray(a)
z_ = CuArray(z)
p_ = CuArray(p)

n = size(x_,1)
m = size(a_,1)

blockdims = (16, 16)
griddims = cuda_grid((n,m), blockdims)
CUDA.@sync begin
    @cuda threads=blockdims blocks=griddims cast!(x_, a_, p_, z_)
end
```

Todo:
 - could speed up by either classic raytracing trick, or
 - for each segment compute the angles it covers, and
   compute intersections for those only

"""
function cast!(X, A, p, Z)

    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    sx = gridDim().x * blockDim().x
    sy = gridDim().y * blockDim().y

    for i = ix:sx:size(X,1), j = iy:sy:size(A,1)
        x1 , x2  = X[i,1], X[i,2]
        x1′, x2′ = X[i,3], X[i,4]
        y1 , y2  = p[1], p[2]
        y1′, y2′ = p[1] + cos(A[j]+p[3]), p[2] + sin(A[j]+p[3])
        s, t = line_intersection(x1, x2, x1′, x2′, y1, y2, y1′, y2′)
        if t >= 0 && 0 <= s <= 1
                @inbounds Z[j] = min(t, Z[j])
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

"""
    cast_v2_cpu!(X, p, Z)

Given a matrix of stacked segments `X`, a pose `p` encoded as a 3-tuple,
fills the vector `Z` with depth/range measurements.
"""
function cast_v2_cpu!(X, p, Z)

    fov = 2π
    num_a = size(Z,1)
    r = a_res  = fov/num_a


    for i = 1:size(X,1)

        # convert everything into pose coords
        x1 , x2  = X[i,1] - p[1], X[i,2] - p[2]
        x1′, x2′ = X[i,3] - p[1], X[i,4] - p[2]
        dx1, dx2 = x1′-x1, x2′-x2
        y1 , y2  = 0, 0
        a1 = atan(x2 , x1 ) - p[3]
        a2 = atan(x2′, x1′) - p[3]

        # Ensure a1 < a2
        if a1 > a2
            a1, a2 = a2, a1
        end

        # Check if we cross from `-π+a` to `π-b`
        if a2 - a1 > π
            a1, a2 = a2 - 2π, a1
        end

        # Get the start end end bin
        zero = 0
        k1 = Int(floor((a1 + r/2 - zero)/r))+1
        k2 = Int(floor((a2 + r/2 - zero)/r))+1
        a1 = k1*a_res
        a2 = k2*a_res


        for k = k1:k2
            a = (k-1)*a_res + p[3]
            y1′, y2′ = cos(a), sin(a)

            s, t = line_intersection(x1, x2, x1′, x2′, y1, y2, y1′, y2′)
            if t > 0 && 0 <= s <= 1
                k = mod(k-1,num_a)+1
                @inbounds Z[k] = min(t, Z[k])
            end
        end

    end
    return
end

"""
    cast_v2!(X, p, Z)

Given a matrix of stacked segments `X`, a pose `p` encoded as a 3-tuple,
fills the vector `Z` with depth/range measurements.
Example:

```
x = rand(1_000, 4)
z = Inf*ones(360)
p = [0.5,0.5,0]

x_ = CuArray(x)
z_ = CuArray(z)
p_ = CuArray(p)

n = size(x_,1)
blockdims = (256,)
griddims = cuda_grid((n,), blockdims)
@cuda threads=blockdims blocks=griddims cast_v2!(x_,p_,z_)

```
"""
function cast_v2!(X, p, Z)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sx = gridDim().x * blockDim().x

    fov = 2π
    num_a = size(Z,1)
    r = a_res  = fov/num_a
    for i = ix:sx:size(X,1)

        # convert everything into pose coords
        x1 , x2  = X[i,1] - p[1], X[i,2] - p[2]
        x1′, x2′ = X[i,3] - p[1], X[i,4] - p[2]
        dx1, dx2 = x1′-x1, x2′-x2
        y1 , y2  = 0, 0
        a1 = atan(x2 , x1 ) - p[3]
        a2 = atan(x2′, x1′) - p[3]

        # Ensure a1 < a2
        if a1 > a2
            a1, a2 = a2, a1
        end

        # Check if we cross from `-π+a` to `π-b`
        if a2 - a1 > π
            a1, a2 = a2 - 2π, a1
        end

        # Get the start end end bin
        zero = 0
        k1 = Int(floor((a1 + r/2 - zero)/r))+1
        k2 = Int(floor((a2 + r/2 - zero)/r))+1
        a1 = k1*a_res
        a2 = k2*a_res


        for k = k1:k2
            a = (k-1)*a_res + p[3]
            y1′, y2′ = cos(a), sin(a)

            s, t = line_intersection(x1, x2, x1′, x2′, y1, y2, y1′, y2′)
            if t > 0 && 0 <= s <= 1
                k = mod(k-1,num_a)+1
                @inbounds Z[k] = min(t, Z[k])
            end
        end

    end
    return
end

function bench_cast_v2!(X, p, Z)
    n = size(X,1)

    blockdims = (256,)
    griddims = cuda_grid((n,), blockdims)
    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims cast_v2!(X, p, Z)
    end
end
