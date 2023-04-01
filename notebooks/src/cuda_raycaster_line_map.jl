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

function bench_cast!(X, A, p, Z; blockdims)
    n = size(X,1)
    m = size(A,1)

    griddims = cuda_grid((n,m), blockdims)
    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims cast!(X, A, p, Z)
    end
end

"""
"""
function cast′!(X, A, P, Z)

    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    sx = gridDim().x * blockDim().x
    sy = gridDim().y * blockDim().y
    sz = gridDim().z * blockDim().z

    for i = ix:sx:size(X,1), j = iy:sy:size(A,1), k = iz:sz:size(P,1)
        x1 , x2  = X[i,1], X[i,2]
        x1′, x2′ = X[i,3], X[i,4]
        y1 , y2  = P[k,1], P[k,2]
        y1′, y2′ = P[k,1] + cos(A[j]+P[k,3]), P[k,2] + sin(A[j]+P[k,3])
        s, t = line_intersection(x1, x2, x1′, x2′, y1, y2, y1′, y2′)
        if t >= 0 && 0 <= s <= 1
                @inbounds Z[k,j] = min(t, Z[k,j])
        end
    end

    return
end

function bench_cast′!(X, A, P, Z; blockdims=(16,8,2))
    n = size(X,1)
    m = size(A,1)
    k = size(P,1)


    griddims = cuda_grid((n,m,k), blockdims)
    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims cast′!(X, A, P, Z)
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

"""
    cast_v3!(X, P, Z)

Caution: Implicitly assumes a fov of 2π!

Given an array of stacked segments `X`, an array of stacked poses `P` each encoded as a 3-Vector,
fills the Array `Z` with depth/range measurements.

Example:

```julia
x = rand(1_000, 4)
p = rand(  500, 3)
z = Inf*ones(size(p,1), 360)

x_ = CuArray(x)
z_ = CuArray(z)
p_ = CuArray(p)

n = size(x_,1)
m = size(p_,1)
datadims  = (n,m)
blockdims = (16,16)
griddims = cuda_grid(datadims, blockdims)
@cuda threads=blockdims blocks=griddims cast_v3!(x_,p_,z_)
```

Runtimes:

```julia
`cast_v3!` GPU vs CPU
x: (500, 4), a: (361,), p: (500, 3)
data: (500, 500), block: (16, 16), grid: (32, 32)
>>  2.252 ms (74 allocations: 4.33 KiB)


`cast_v3!` GPU vs CPU
x: (1000, 4), a: (361,), p: (1, 3)
data: (1000, 1), block: (256, 1), grid: (4, 1)
>>  17.535 μs (26 allocations: 1.39 KiB)
```

"""
function cast_v3!(X, P, Z)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    sx = gridDim().x * blockDim().x
    sy = gridDim().y * blockDim().y

    fov = 2π
    num_a = size(Z,2)
    r = fov/num_a

    for i = ix:sx:size(X,1), j = iy:sy:size(P,1)

        # convert everything into pose coords
        x1 , x2  = X[i,1] - P[j,1], X[i,2] - P[j,2]
        x1′, x2′ = X[i,3] - P[j,1], X[i,4] - P[j,2]
        dx1, dx2 = x1′-x1, x2′-x2
        y1 , y2  = 0, 0
        a1 = atan(x2 , x1 ) - P[j,3]
        a2 = atan(x2′, x1′) - P[j,3]

        # Ensure a1 < a2
        if a1 > a2
            a1, a2 = a2, a1
        end

        # Check if we cross from `-π+a` to `π-b`
        if a2 - a1 > π
            a1, a2 = a2 - 2π, a1
        end

        # Get the start end end bin
        zero = - π;
        k1 = Int(floor((a1 + r/2 - zero)/r))+1
        k2 = Int(floor((a2 + r/2 - zero)/r))+1

        for k = k1:k2
            a = zero + (k-1)*r + P[j,3]
            y1′, y2′ = cos(a), sin(a)

            s, t = line_intersection(x1, x2, x1′, x2′, y1, y2, y1′, y2′)
            if 0 < t && 0 <= s <= 1
                k = mod(k-1,num_a)+1
                @inbounds Z[j,k] = min(t, Z[j,k])
            end
        end

    end
    return
end

function bench_cast_v3!(X, P, Z; blockdims=(16,16))
    n = size(X,1)
    m = size(P,1)

    griddims = cuda_grid((n,m), blockdims)
    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims cast_v3!(X, P, Z)
    end
end

"""
    fill_z!(X, P, Z)

Caution: Implicitly assumes a fov of 2π!

Given an array of stacked segments `X`, an array of stacked poses `P` each encoded as a 3-Vector,
fills the Array `Z` with depth/range measurements.
"""
function fill_z!(X, P, Z)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    sx = gridDim().x * blockDim().x
    sy = gridDim().y * blockDim().y

    fov = 2π
    num_a = size(Z,3)
    r = fov/num_a

    for i = ix:sx:size(X,1), j = iy:sy:size(P,1)

        # convert everything into pose coords
        x1 , x2  = X[i,1] - P[j,1], X[i,2] - P[j,2]
        x1′, x2′ = X[i,3] - P[j,1], X[i,4] - P[j,2]
        dx1, dx2 = x1′-x1, x2′-x2
        y1 , y2  = 0, 0
        a1 = atan(x2 , x1 ) - P[j,3]
        a2 = atan(x2′, x1′) - P[j,3]

        # Ensure a1 < a2
        if a1 > a2
            a1, a2 = a2, a1
        end

        # Check if we cross from `-π+a` to `π-b`
        if a2 - a1 > π
            a1, a2 = a2 - 2π, a1
        end

        # Get the start end end bin
        zero = - π;
        k1 = Int(floor((a1 + r/2 - zero)/r))+1
        k2 = Int(floor((a2 + r/2 - zero)/r))+1

        for k = k1:k2
            a = zero + (k-1)*r + P[j,3]
            y1′, y2′ = cos(a), sin(a)

            s, t = line_intersection(x1, x2, x1′, x2′, y1, y2, y1′, y2′)
            if 0 < t && 0 <= s <= 1
                k = mod(k-1,num_a)+1
                @inbounds Z[j, i, k] = t
            end
        end

    end
    return
end


function bench_fill_z!(X, P, Z; blockdims=(16,16))
    n = size(X,1)
    m = size(P,1)

    griddims = cuda_grid((n,m), blockdims)
    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims fill_z!(X, P, Z)
    end
end

"""
    z = cast_v4!(X, P, Z; blockdims=(16,16))

Takes array of `n` stacked segments `X` and `m` stacked poses `P`
and an observation array `Z` of shape `(n,m,k)` and fills
it with depth measurements along angles `range(-π,π,k+1)[1:end-1]`.

Example:

```julia
n = 1000 # num segments
m =  500 # num poses

x = rand(n, 4)
p = zeros(m, 3)
z = ones(m, n, 360)

x_ = CuArray(x)
z_ = CuArray(z)
p_ = CuArray(p)

z′ = cast_v4!($x_,$p_,$z_; blockdims=(16,16))
```

Runtimes:

```julia
`cast_v4!` GPU vs CPU
x: (1000, 4), a: (360,), p: (1, 3)
data: (1000, 1), block: (16, 16), grid: (63, 1)
>>  218.122 μs (64 allocations: 3.50 KiB)

`cast_v4!` GPU vs CPU
x: (500, 4), a: (360,), p: (500, 3)
data: (500, 500), block: (16, 16), grid: (32, 32)
>>  2.516 ms (105 allocations: 6.16 KiB)
```
"""
function cast_v4!(X, P, Z; blockdims=(16,16))
    n = size(X,1)
    m = size(P,1)

    griddims = cuda_grid((n,m), blockdims)
    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims fill_z!(X, P, Z)
    end

    return minimum(Z, dims=2)
end;
