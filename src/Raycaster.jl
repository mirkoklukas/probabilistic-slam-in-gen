# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``31b - CUDA Raycaster - Line Map.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

##################################### 
module Raycaster  
#####################################


# Make sure to update `LOAD_PATH` accordingly, to point to the `src` folder of `MyUtils` and so on
using MyUtils
using CUDA
using BenchmarkTools
using Colors, Plots
col = palette(:default);

function intersections(x::AbstractArray, x′::AbstractArray,
                       y::AbstractArray, y′::AbstractArray)
    n = ndims(x)
    dx = x′ .- x
    dy = y′ .- y
    v  = x .- y

    dx1 = selectdim(dx, n, 1)
    dx2 = selectdim(dx, n, 2)
    dy1 = selectdim(dy, n, 1)
    dy2 = selectdim(dy, n, 2)
    v1  = selectdim(v, n, 1)
    v2  = selectdim(v, n, 2)

    a, b = -dx1, dy1
    c, d = -dx2, dy2

    det = a.*d .- b.*c

    s = 1 ./det .*(  d.*v1 .- b.*v2)
    t = 1 ./det .*(- c.*v1 .+ a.*v2)
    return s,t, x .+ s.*dx, y .+ t.*dy
end;

function line_intersection(x1::Float64, x2::Float64, x1′::Float64, x2′::Float64,
                           y1::Float64, y2::Float64, y1′::Float64, y2′::Float64)
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

export line_intersection


function create_angles(fov, num_a)
    if fov < 2π
        return [range(-fov/2, fov/2, num_a)...];
    else
        return [range(-π, π, num_a+1)...][1:end-1];
    end
end

export create_angles

"""
    zs, ids = cast(v0::AbstractVector, as::AbstractVector, segs::AbstractArray, zmax=Inf)

Returns depth values and the identity of the segments that got hit by the caster.
"""
function cast(v0::AbstractVector, segs::AbstractArray, as::AbstractVector, zmax=Inf)
    x  = view(v0, 1:2)
    hd = view(v0, 3)
    as = as .+ hd

    x′ = reshape(x,1,2) .+ cat(cos.(as), sin.(as), dims=2)
    y  = view(segs,:,1:2)
    y′ = view(segs,:,3:4)

    x  = reshape(x , :, 1, 2)
    x′ = reshape(x′, :, 1, 2)
    y  = reshape(y , 1, :, 2)
    y′ = reshape(y′, 1, :, 2)

    s, t, _, _ = intersections(x, x′, y, y′)

    # Hit map
    h = (0 .< s) .* (0 .<= t .<= 1)
    s[.!h] .= zmax

    # Segment-ID
    i  = argmin(s, dims=2)[:,1]
    i2 = map(i->i[2],i);

    # Todo: There was an issue when I computed `z_ = s_[i_]`
    #       Not sure why... but using minimum() works
    z = minimum(s, dims=2)[:,1]
    i2[z.==zmax] .= size(segs, 1) + 1

    # Depth and Segment-ID
    return z, i2
end;

"""
    zs, ids = cast(v0::AbstractVector, as::AbstractVector, segs::AbstractArray, zmax=Inf)

Returns depth values and the identity of the segments that got hit by the caster.
"""
function cast(vs::AbstractArray, segs::AbstractArray, as::AbstractVector, zmax=Inf)
    k = size(vs,1)
    n = length(as)
    l = size(segs,1)

    x  = view(vs, :, 1:2)
    hd = view(vs, :, 3)
    as = reshape(as, 1, n) .+ reshape(hd, k, 1)
    dx = cat(cos.(as), sin.(as), dims=3)

    x′ = reshape(x, k, 1, 2) .+ dx
    y  = view(segs,:,1:2)
    y′ = view(segs,:,3:4)

    x  = reshape(x , k, 1, 1, 2)
    x′ = reshape(x′, k, n, 1, 2)
    y  = reshape(y , 1, 1, :, 2)
    y′ = reshape(y′, 1, 1, :, 2)

    s, t, _, _ = intersections(x, x′, y, y′)


    # Hit map
    h = (0 .< s) .* (0 .<= t .<= 1)
    s[.!h] .= zmax

    # Segment-ID
    i  = argmin(s, dims=3)[:,:,1]
    i′ = map(i->i[3],i);

    # Todo: There was an issue when I computed `z_ = s_[i_]`
    #       Not sure why... but using minimum() works
    z = minimum(s, dims=3)[:,:,1]
    i′[z.==zmax] .= l+1

    # Depth and Segment-ID
    return z, i′
end;

function cast_cpu!(Z, X, P, fov=2π)
    num_a = size(Z,2)
    r     = fov/(num_a-1)

    for i = 1:size(X,1), j = 1:size(P,1)
        # Todo: Can I somehow reuse this code block in the cuda kernel?


        # convert everything into pose coords
        x1 , x2  = X[i,1] - P[j,1], X[i,2] - P[j,2]
        x1′, x2′ = X[i,3] - P[j,1], X[i,4] - P[j,2]
        dx1, dx2 = x1′-x1, x2′-x2
        y1 , y2  = 0., 0.
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
                @inbounds Z[j,k] = min(Z[j,k], t)
            end
        end
    end
    return
end


"""
```julia
    zs = cast_cpu(ps, segs; fov=2π, num_a::Int=361, zmax::Float64=Inf)
```
Computes depth measurements `zs` with respect to a family of stacked poses `ps`
and family of stacked line segments `segs_` along a fixed number `num_a` of
equidistantly spaced angles in the field of view `fov`.

Arguments:
 - `ps`: Stacked poses `(k, 3)`
 - `segs`: Stacked line segments `(n, 4)`
 - ...

Returns:
 - `zs`: Depth measurements in the field of view `(k, num_a)`
"""
function cast_cpu(ps, segs; fov=2π, num_a::Int=361, zmax::Float64=Inf)
    zs = fill(zmax, size(ps, 1), num_a)
    cast_cpu!(zs, segs, ps, fov)
    return zs
end;

export cast_cpu

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
        y1 , y2  = 0., 0.
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


"""
```julia
    zs_ = cast_cu(ps_::CuArray, segs_::CuArray; fov=2π, num_a::Int=361, zmax::Float64=Inf)
```
Computes depth measurements `zs_` with respect to a family of stacked poses `ps_`
and family of stacked line segments `segs_` along a fixed number `num_a` of
equidistantly spaced angles in the field of view `fov`.

Arguments:
 - `ps_`: Stacked poses `(k, 3)`
 - `segs_`: Stacked line segments `(n, 4)`
 - ...

Returns:
 - `zs_`: Depth measurements in the field of view `(k, num_a)`
"""
function cast_cu(ps_::CuArray, segs_::CuArray; fov=2π, num_a::Int=361, zmax::Float64=Inf, blockdims=(16,16))
    zs_ = zmax*CUDA.ones(size(ps_, 1), num_a)
    cast_cu!(zs_, segs_, ps_; fov=fov, blockdims=blockdims)
    return zs_
end;

export cast_cu

"""
```julia
    zs = cast(ps, segs; fov=2π, num_a::Int=361, zmax::Float64=Inf)
```
Computes depth measurements `zs` with respect to a family of stacked poses `ps`
and family of stacked line segments `segs` along a fixed number `num_a` of
equidistantly spaced angles in the field of view `fov`.

If a CUDA supported GPU is available we run a GPU accelerated version.

Arguments:
 - `ps`: Stacked poses `(k, 3)`
 - `segs`: Stacked line segments `(n, 4)`
 - ...

Returns:
 - `zs`: Depth measurements in the field of view `(k, num_a)`
"""
function cast(ps::Array, segs::Array; fov=2π, num_a::Int=361, zmax::Float64=Inf)
    if _cuda[]
        ps_   = CuArray(ps)
        segs_ = CuArray(segs)
        zs_   = cast_cu(ps_, segs_; fov=fov, num_a=num_a, zmax=zmax)
        return Array(zs_)
    else
        return cast_cpu(ps, segs; fov=fov, num_a=num_a, zmax=zmax)
    end
end;

function cast(ps_::CuArray, segs_::CuArray; fov=2π, num_a::Int=361, zmax::Float64=Inf)
    return cast_cu(ps_, segs_; fov=fov, num_a=num_a, zmax=zmax)
end

export cast

#####################################
end  
#####################################
