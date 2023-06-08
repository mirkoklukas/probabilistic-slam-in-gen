# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``13 - 2dp3 Sensor Distribution - CUDA.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

push!(LOAD_PATH, "../src");
push!(LOAD_PATH, ENV["probcomp"]*"/Gen-Distribution-Zoo/src");
using BenchmarkTools
using CUDA
using Gen
using MyUtils
using MyCudaUtils # contains _cuda[] and cuda_grid
using GenDistributionZoo: diagnormal
using Test

MyUtils.polar_inv(z::CuArray, a::CuArray) = z.* cat(cos.(a), sin.(a), dims=ndims(a)+1);

"""
```julia
    log_p = gaussian_logpdf(x, mu, sig)
````
Broadcastable Gaussian logpdf -- benchmarked in `33 - CUDA Accelerated Gen Distributions`.
"""
function gaussian_logpdf(x, mu, sig)
    d = (x .- mu).^2 ./ sig.^2
    log_p = - log.(sig) .- log(sqrt(2π)) .- 1/2 * d
    return log_p
end;

using SpecialFunctions: erf
"""
```julia
    log_c = gaussian_logcdf(x, mu, sig)
````
Broadcastable Gaussian logcdf
"""
function gaussian_logcdf(x, mu, sig)
    d = (x .- mu)./(sig.*sqrt(2))
    log_c = - log(2) .+ log.(1 .+ erf.(d))
    return log_c
end;

function gaussian_cdf(x, mu, sig)
    d = (x .- mu)./(sig.*sqrt(2))
    return (1 .+ erf.(d))./2
end;

"""
```julia
  logsumexp_slice(x; dims)
```
Applies `logsumexp` along specified dimensions.


Benchmarks
```julia
x: (2000, 2000)
dims: 2
with `check_inf`
  CPU  >>  49.887 ms (26 allocations: 30.57 MiB)
  CUDA >> 355.932 μs (367 allocations: 19.55 KiB)

without `check_inf`
  CPU  >>  54.482 ms (20 allocations: 30.56 MiB)
  CUDA >>  69.461 μs (142 allocations: 8.06 KiB)
```
"""
function logsumexp_slice(x::Union{CuArray, Array}; dims, check_inf=true)
    c = maximum(x, dims=dims)
    y = c .+ log.(sum(exp.(x .- c), dims=dims))

    # Note that if c is -Inf, then y will be NaN.
    if check_inf
        y[c .== -Inf] .= - Inf
    end
    return y
end;

import Compat
using PaddedViews

function slw_cpu(x, w ,s=1; wrap=false, fill=true, fill_val=Inf)
    # Todo: Enable to fill with the same value as the edges.
    #       Test against the GPU version, which has the desired behaviour,
    #       and make sure it's the same.
    if fill
        y = PaddedView(fill_val, x, size(x) .+ (0,2w), (1,w+1))
    else
        y = PaddedView(fill_val, x, size(x) .+ (0,2w), (1,w+1))
        y = Array(y)
        y[:,1:w] .= x[:,end-w+1:end]
        y[:,end-w+1:end] .= x[:,1:w]
    end
    I = ((@view y[j, i:i+2w]) for j=1:size(y,1), i in 1:s:size(y,2)-2w)
    y = Compat.stack(I)
    return permutedims(y, (2,3,1))
end;

# Todo: handle wrap around and padding smarter?
"""
```julia
    slw_kernel!(x, y, w::Int, wrap::Bool, fill::Bool, fill_val::Float64)
```
CUDA kernel to compute sliding windows.
Takes CuArrays of shape `(k,n)` and `(k,n,2w+1)`...
"""
function slw_kernel!(x, y, w::Int, wrap::Bool, fill::Bool, fill_val::Float64)

    m = 2*w + 1
    n = size(x,2)

    # Make sure the arrays are
    # of the right shape
    @assert ndims(x)  == 2
    @assert ndims(y)  == ndims(x) + 1
    @assert size(x,1) == size(y,1)
    @assert size(x,2) == size(y,2)
    @assert size(y,3) == m

    # Thread id's
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    sx = gridDim().x * blockDim().x
    sy = gridDim().y * blockDim().y
    sz = gridDim().z * blockDim().z

    for j_pose = ix:sx:size(y,1), j_obs = iy:sy:size(y,2), j_mix = iz:sz:size(y,3)
        # Transform mixture index in `1:m`
        # to offsets in `-w:w`
        offset = j_mix-1-w

        j = j_obs + offset
        if wrap
            # Wrap around
            j = mod(j - 1 , n) + 1
            val = x[j_pose, j]
        else
            if 1 <= j <= n
                val = x[j_pose, j]
            else
                if fill
                    # Fill with fill value
                    val = fill_val
                else
                    # Fill with the edge values
                    j = max(min(j,n),1)
                    val = x[j_pose, j]
                end
            end
        end

        # Fill entries of `y`
        @inbounds y[j_pose, j_obs, j_mix] = val
    end
    return
end

"""
```julia
    y_ = slw_cu(x_::CuArray, w::Int; blockdims=(8,8,4), wrap=false, fill=true, fill_val=Inf)
```
CUDA-accelerated function computing sliding windows.
Takes a CuArray of shape `(k,n)` and returns a CuArray
of shape `(k,n,m)`, where `m = 2w+1`....
"""
function slw_cu(x::CuArray, w::Int; blockdims=(8,8,4), wrap=false, fill=true, fill_val=Inf)

    k = size(x, 1)
    n = size(x, 2)
    m = 2*w+1

    y = CUDA.ones(k,n,m)

    # `cuda_grid` defined in reaycaster file, I also put it in utils
    griddims = cuda_grid((k,n,m), blockdims)
    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims slw_kernel!(x, y, w, wrap, fill, fill_val)
    end

    return y
end;

"""
    y_ = slw(x_, w::Int; blockdims=(8,8,4), wrap=false, fill=false, fill_val=Inf)

Function computing sliding windows, on the GPU.
Takes a CuArray of shape `(k,n)` and returns an CuArray
of shape `(k,n,m)`, where `m = 2w+1`...
"""
function slw(x_::CuArray, w::Int; blockdims=(8,8,4), wrap=false, fill=false, fill_val=Inf)
    y_ = slw_cu(x_, w; blockdims=blockdims, wrap=wrap, fill=fill, fill_val=fill_val)
    return y_
end;

"""
    y = slw(x, w::Int; blockdims=(8,8,4), wrap=false, fill=false, fill_val=Inf)

Function computing sliding windows, either on the CPU or GPU.
Takes an Array of shape `(k,n)` and returns an Array
of shape `(k,n,m)`, where `m = 2w+1`...
"""
function slw(x::Array, w::Int; blockdims=(8,8,4), wrap=false, fill=false, fill_val=Inf)
    # Todo: Is that a good pattern??
    if _cuda[]
        x_ = CuArray(x)
        y_ = slw_cu(x_, w; blockdims=blockdims, wrap=wrap, fill=fill, fill_val=fill_val)
        return Array(y_)
    else
        return slw_cpu(x, w;  wrap=wrap, fill=fill, fill_val=fill_val)
    end
end;

# DEPRECIATED
"""
    ỹ_::CuArray = get_ys_tilde_cu(zs_::CuArray, w::Int)

DEPRECIATED, use `get_2d_mixture_components` instead.
"""
function get_ys_tilde_cu(zs_::CuArray, as_::CuArray, w::Int; wrap=false, fill=false, fill_val=0.0)

    zs_tilde_ = slw_cu(zs_, w; blockdims=(8,8,4), wrap=wrap)
    as_tilde_ = slw_cu(reshape(as_,1,:), w; blockdims=(8,8,4), wrap=wrap)
    ys_tilde_ = polar_inv(zs_tilde_, as_tilde_)

    return ys_tilde_
end;

# Same as above but want to switch
# to different name going forward
"""
```julia
    ỹ_::CuArray = get_2d_mixture_components(z_::CuArray, a_::CuArray, w::Int;
                                            wrap=false, fill=true,
                                            fill_val_z=Inf, fill_val_a=Inf)
```
Computes the 2d mixture components for the "2dp3" likelihood from a family
depth scans `z_` along angles `a_`, and with a filter radius of `w`.

Arguments:
 - `z_`:    Range measurements `(k,n)`
 - `a_`:    Angles of measuremnts `(n,)`
 - `w`:     Filter window size

Returns:
 - `ỹ_`: CuArray of shape `(k, n, m, 2)`, where `m=2w+1`
"""
function get_2d_mixture_components(z_::CuArray, a_::CuArray, w::Int;
                                   wrap=false, fill=true, fill_val_z=Inf, fill_val_a=0.0)

    a_ = reshape(a_,1,:)

    z̃_ = slw_cu(z_, w; blockdims=(8,8,4), wrap=wrap, fill=fill, fill_val=fill_val_z)
    ã_ = slw_cu(a_, w; blockdims=(8,8,4), wrap=wrap, fill=fill, fill_val=fill_val_a)
    ỹ_ = polar_inv(z̃_, ã_)

    # Handle Inf's and NaN
    # Todo: Where were the NaNs coming from again? They're coming from Inf * 0;
    #       Could avoid that by choosing a fill value a such that cos(a) and sin(a) are non-zero,
    #       or use a max z value < Inf.
    ỹ_[isnan.(ỹ_)] .= Inf

    return ỹ_
end;

"""
```julia
    ỹ::Array = get_2d_mixture_components(z::Array, a::Array, w::Int;
                                            wrap=false, fill=true,
                                            fill_val_z=Inf, fill_val_a=Inf)
```
Computes the 2d mixture components for the "2dp3" likelihood from a family
depth scans `z` along angles `a`, and with a filter radius of `w`.

Arguments:
 - `z`:    Range measurements `(k,n)`
 - `a`:    Angles of measuremnts `(n,)`
 - `w`:    Filter window size

Returns:
 - `ỹ`: Array of shape `(k, n, m, 2)`, where `m=2w+1`
"""
function get_2d_mixture_components(z::Array, a::Array, w::Int;
                                   wrap=false, fill=true,
                                   fill_val_z=Inf, fill_val_a=0.0)

    # Todo: Is that a good pattern??
    if _cuda[]
        z_ = CuArray(z)
        a_ = CuArray(a)
        ỹ_ = get_2d_mixture_components(z_, a_, w;
                                        wrap=wrap, fill=fill,
                                        fill_val_z=fill_val_z, fill_val_a=fill_val_a)
        return Array(ỹ_)
    else
        a = reshape(a,1,:)
        z̃ = slw_cpu(z, w; blockdims=(8,8,4), wrap=wrap, fill=fill, fill_val=fill_val_z)
        ã = slw_cpu(a, w; blockdims=(8,8,4), wrap=wrap, fill=fill, fill_val=fill_val_a)
        ỹ = polar_inv(z̃, ã)

        # Handle Inf's and NaN
        # Todo: Where were the NaNs coming from again? They're coming from Inf * 0;
        #       Could avoid that by choosing a fill value a such that cos(a) and sin(a) are non-zero,
        #       or use a max z value < Inf.
        ỹ[isnan.(ỹ)] .= Inf

        return ỹ
    end
end;

"""
```julia
    log_ps, ptw = sensor_logpdf(x, ỹ, sig, outlier, outlier_vol=1.0; return_pointwise=false)
```
Evaluates an observation `x` under the 2dp3 likelihood with <br/>
a family of mixture components `ỹ` and parameters `sig`, `outlier`, and `outlier_vol`.

Arguments:
 - `x`: Observation point cloud `(n,2)`
 - `ỹ`: Family of mixture components `(k,n,m,2)`
 - `sig`: Std deviation of Gaussian mixture components; either a scalar or an array of shape `(1,1,1,1, ...)` (should be broadcastable with `outlier` and the rest)
 - `outlier`: Outlier probability; either a scalar or an array of shape `(1,1,1,1, ...)` (should be broadcastable with `sig` and the rest)

Returns:
 - `log_ps`: Log-probs `(k,)` ,or `(k, ...)` in case of non-scalar `sig` and `outlier`
 - `ptw`:    Pointwise log-probs for each observation point `(k,n)`  ,or `(k,n, ...)` in case of non-scalar `sig` and `outlier`
"""
function sensor_logpdf(x, ỹ, sig, outlier, outlier_vol; return_pointwise=false, return_outliermap=false)
    @assert size(x,1) == size(ỹ,2)

    k = size(ỹ,1)
    n = size(ỹ,2)
    m = size(ỹ,3)
    x = reshape(x, 1, n, 1, 2)

    # Inlier probability (Gaussian mixtures).
    #   Compute 1D Gaussians (k,n,m,2)
    #   Convert to 2D gausians (k,n,m,1)
    #   Convert to mixture `gm` of m 2D gausians (k,n,1,1)
    log_p = gaussian_logpdf(x, ỹ, sig)
    log_p = sum(log_p, dims=4)
    log_p = logsumexp_slice(log_p .- log(m), dims=3)

    # Outlier probability
    # and outlier map
    # Todo: Find a better outlier prob?
    #       Change outlier_vol = 2π*zmax^2?
    log_out = - log(outlier_vol)
    outliermap = nothing
    if return_outliermap
        outliermap = log.(1. .- outlier) .+ log_p .< log.(outlier) .+ log_out
    end

    # Convert to mixture of `gm` and `outlier` (k,n,1,1)
    log_p = log.((1 .- outlier).*exp.(log_p) .+ outlier.*exp.(log_out))

    # If we don't need pointwise logprobs
    # we can save us the time and space to copy
    pointwise = nothing
    if return_pointwise
        pointwise = dropdims(log_p, dims=(3,4))
    end

    # Convert to product of mixtures (k,1,1,1)
    log_p = sum(log_p, dims=2)
    log_p = dropdims(log_p, dims=(2,3,4))

    return log_p, pointwise, outliermap
end
# Todo: Make sure we handle Inf's in y correctly --
#       that might come from sliding window fills?

# Backwards compatibility --
# Same as `sensor_logpdf` above
"""
DEPRECIATED, use `sensor_logpdf` instead.
"""
function sensor_smc_logpdf_cu(x, y, sig, outlier, outlier_vol; return_pointwise=false)
    return sensor_logpdf(x, y, sig, outlier, outlier_vol; return_pointwise=return_pointwise)
end;

struct SensorDistribution2DP3 <: Distribution{Vector{Vector{Float64}}}
end

"""
```julia
    x = sensordist_cu(ỹ_::CuArray, sig, outlier, outlier_vol=1.0)::Vector{Vector{Float64}}
```
Distribution from the 2dp3-likelihood. Takes 2d-mixture components `ỹ_` and
samples a vector `x` of 2d points.

Arguments:
 - `ỹ_`: 2d-mixture components `(n,m,2)`
 - ...
Returns:
- `x`: Observation vector of 2d points.
"""
const sensordist_2dp3 = SensorDistribution2DP3()

function Gen.logpdf(::SensorDistribution2DP3, x, ỹ_::CuArray, sig, outlier, outlier_vol)
    n = size(ỹ_, 1)
    m = size(ỹ_, 2)

    x_ = CuArray(stack(x))
    ỹ_ = reshape(ỹ_, 1, n, m, 2)

    log_p, = sensor_logpdf(x_, ỹ_, sig, outlier, outlier_vol) # CuArray of length 1
    return CUDA.@allowscalar log_p[1]
end

function Gen.logpdf(::SensorDistribution2DP3, x, ỹ::Array, sig, outlier, outlier_vol)
    n = size(ỹ, 1)
    m = size(ỹ, 2)

    x = stack(x)
    ỹ = reshape(ỹ, 1, n, m, 2)

    log_p, = sensor_logpdf(x, ỹ, sig, outlier, outlier_vol) # Array of length 1
    return log_p[1]
end

# Todo: Speed up sampling, the Gen plug-and-play version is
#       faster when sampling. Slow at evaluation.
function Gen.random(::SensorDistribution2DP3, ỹ_::CuArray, sig, outlier, outlier_vol)
    n = size(ỹ_,1)
    m = size(ỹ_,2)

    # Sample an observation point cloud `x`
    x = Vector{Float64}[]
    for i=1:n
        if bernoulli(outlier)
            # Todo: Change that to a uniform distribution, e.g. over a
            #       circular area with radius `zmax`.
            x_i = [Inf;Inf]
        else
            j   = rand(1:m)
            y   = Array(ỹ_[i,j,:])
            x_i = diagnormal(y, [sig;sig])

        end
        push!(x, x_i)
    end

    return x
end

function Gen.random(::SensorDistribution2DP3, ỹ::Array, sig, outlier, outlier_vol)
    n = size(ỹ,1)
    m = size(ỹ,2)

    # Sample an observation point cloud `x`
    x = Vector{Float64}[]
    for i=1:n
        if bernoulli(outlier)
            # Todo: Change that to a uniform distribution, e.g. over a
            #       circular area with radius `zmax`.
            x_i = [Inf;Inf]
        else
            j   = rand(1:m)
            y   = ỹ[i,j,:]
            x_i = diagnormal(y, [sig;sig])

        end
        push!(x, x_i)
    end

    return x
end;

(D::SensorDistribution2DP3)(args...) = Gen.random(D, args...)

# TODO: Add output and arg grads.
Gen.has_output_grad(::SensorDistribution2DP3)    = false
Gen.has_argument_grads(::SensorDistribution2DP3) = (false, false);

"""
```julia
    ỹ_, d̃_ = get_1d_mixture_components(z_, a_, w, sig;
                                       wrap=false, fill=true,
                                       fill_val_z=Inf, fill_val_a=Inf)
```
Computes the 1d projections of mixture components onto the rays and
their distances to the rays for the "2dp3" likelihood from
depth measurements `z_` along angles `a_`, and with a filter radius of `w`.

Arguments:
 - `z_`: Range measurements `(k,n)`
 - `a_`: Angles of measuremnts `(n,)`
 - `w``: Filter window radius

Returns:
 - `ỹ_`: Projections onto ray `(k, n, m)`, where `m=2w+1`
 - `d̃_`: Distances to ray `(k, n, m)`, where `m=2w+1`
"""
function get_1d_mixture_components(z_, a_, w;
                                   wrap=false, fill=true,
                                   fill_val_z=Inf, fill_val_a=0.0)

    a_ = reshape(a_,1,:)
    z̃_ = slw(z_, w; blockdims=(8,8,4), wrap=wrap, fill=fill, fill_val=fill_val_z)
    ã_ = slw(a_, w; blockdims=(8,8,4), wrap=wrap, fill=fill, fill_val=fill_val_a)

    # We compute the projection `ỹ` of the 2d mixtures onto
    # the ray through each pixel and their distance `d̃` to the rays.
    #
    # Note that the embedded point is of the form `[z*cos(a), z*cos(a)]` and
    # the projections are given by the dot products with the x- and y-axis.
    ã_ = π/2 .- ã_ .+ a_
    d̃_ = z̃_ .* cos.(ã_)
    ỹ_ = z̃_ .* sin.(ã_)

    # Handle NaN coming from Inf * 0.0
    d̃_[isnan.(d̃_)] .= Inf
    ỹ_[isnan.(ỹ_)] .= Inf

    return ỹ_, d̃_
end;

"""
    log_ps, ptw, outl = depthdist_logpdf(z, ỹ, w̃, sig, outlier, zmax; return_pointwise=false, return_outliermap=false)

Evaluates an depth measurement `z` under the 2dp3 likelihood with <br/>
a family of 1d mixture components `ỹ` and their weights ` w̃` and <br/>
parameters `sig`, `outlier`, and `outlier_vol`.

Arguments:
    - `z`: Depth measurements `(n,)`
    - `ỹ`: Family of 1d-mixture components `(k,n,m)`
    - `w̃`: Family of 1d-mixture weights `(k,n,m)`
    - `sig`: Standard deviation of the mixture components; either a scalar, or
        an array that is broadcastable with the rest of the args, e.g. `(1,1,1, ...)`
    - `outlier`: Outlier probability; either a scalar, or
        an array that is broadcastable with the rest of the args, e.g. `(1,1,1, ...)`

Returns:
    - `log_ps`: Log-probs `(k,)` (or `(k, ...)` if sig or outlier are arrays)
    - `ptw`:   Pointwise log-probs for each observation point `(k,n)` (or `(k,n, ...)` if sig or outlier are arrays)
    - `outl`:  Pointwise outlier map for each observation point `(k,n)` (or `(k,n, ...)` if sig or outlier are arrays)
"""
function depthdist_logpdf_old(z, ỹ, w̃, sig::Union{Float64,AbstractArray}, outlier::Union{Float64,AbstractArray}, zmax; return_pointwise=false, return_outliermap=false)

    # For the hierarchical Bayes verson
    # we assume that the last dim of ỹ, w̃ already
    if typeof(sig) <: AbstractArray
        sig = reshape(sig, Base.fill(1, ndims(ỹ))..., length(sig))
    end

    if typeof(outlier) <: AbstractArray
        outlier = reshape(outlier, Base.fill(1, ndims(ỹ))..., 1, length(outlier))
    end

    z = clamp.(z, 0.0, zmax)
    # Inlier probability.
    #   Compute the Gaussian log-probabilities,
    #   truncate at zero and zmax, and
    #   from the mixture.
    log_p   = gaussian_logpdf(z, ỹ, sig)
    log_p .-= log.(gaussian_cdf(zmax, ỹ, sig) .- gaussian_cdf(0.0, ỹ, sig))
    log_p   = logsumexp_slice(log_p .+ w̃, dims=3)
    log_p   = dropdims(log_p, dims=3)

    # Outlier probability (here uniform)
    # and outlier map
    log_out = - log.(zmax)
    outl = nothing
    if return_outliermap
        outl = log.(1 .- outlier) .+ log_p .< log.(outlier) .+ log_out
    end

    # Mixture of inlier and outlier probability
    log_p = log.((1 .- outlier).*exp.(log_p) .+ outlier*exp.(log_out))

    # Pointwise log-probabilities
    ptw = nothing
    if return_pointwise
        ptw = log_p
    end

    log_p = sum(log_p, dims=2)
    log_p = dropdims(log_p, dims=2)

    return log_p, ptw, outl
end;

"""
    log_ps, ptw, outl = depthdist_logpdf(z, ỹ, d̃, sig, outlier, zmax, scale_noise=false, noise_anchor=1.0; return_pointwise=false, return_outliermap=false)

Evaluates an depth measurement `z` under the 2dp3 likelihood with respect to<br/>
a family of 1d mixture components `ỹ` and their distances `d̃` and <br/>
parameters `sig`, `outlier`, and `zmax`...

Arguments:
    - `z`:   Depth measurements `(n,)`
    - `ỹ`:   Family of 1d-mixture components `(k,n,m)`
    - `w̃`:   Family of 1d-mixture distances `(k,n,m)`
    - `sig`: Standard deviation of the mixture components; either a scalar, or
             an array that is broadcastable with the rest of the args, e.g. `(1,1,1, ...)`
    - `outlier`: Outlier probability; either a scalar, or
                 an array that is broadcastable with the rest of the args, e.g. `(1,1,1, ...)`
    - `zmax`:         Maximum depth value
    - `scale_noise`:  If true, the noise is scaled by the depth value
    - `noise_anchor`: The depth value at which the noise is scaled to `sig`

Returns:
    - `log_ps`: Log-probs `(k,)` (or `(k, ...)` if sig or outlier are arrays)
    - `ptw`:    Pointwise log-probs for each observation point `(k,n)` (or `(k,n, ...)` if sig or outlier are arrays)
    - `outl`:   Pointwise outlier map for each observation point `(k,n)` (or `(k,n, ...)` if sig or outlier are arrays)
"""
function depthdist_logpdf(z, ỹ, d̃, sig::Union{Float64,AbstractArray}, outlier::Union{Float64,AbstractArray}, zmax,
                          scale_noise=false, noise_anchor=1.0;
                          return_pointwise=false, return_outliermap=false)

    # Truncate depth
    z = clamp.(z, 0.0, zmax)

    # For the hierarchical Bayes verson
    # we assume that the last dim of ỹ, w̃ already
    if typeof(sig) <: AbstractArray
        sig = reshape(sig, Base.fill(1, ndims(ỹ))..., length(sig))
    end

    # Scale noise level proportional to depth of the mixture component.
    # Todo: Is this the right way to do it? Should we clamp `sig`?
    if scale_noise
        # At distance `noise_anchor` the noise is `sig`
        sig = ỹ./noise_anchor .* sig
    end

    # Compute normalized mixture weights
    w̃ = gaussian_logpdf(d̃, 0.0, sig)
    w̃ = w̃ .- logsumexp_slice(w̃, dims=3)


    # Inlier probability.
    #   Compute the Gaussian log-probabilities,
    #   truncate at zero and zmax, and
    #   from the mixture.
    log_p   = gaussian_logpdf(z, ỹ, sig)
    log_p .-= log.(gaussian_cdf(zmax, ỹ, sig) .- gaussian_cdf(0.0, ỹ, sig))
    log_p   = logsumexp_slice(log_p .+ w̃, dims=3)
    log_p   = dropdims(log_p, dims=3)

    # Outlier probability (here uniform)
    # and outlier map
    if typeof(outlier) <: AbstractArray
        outlier = reshape(outlier, Base.fill(1, ndims(log_p))..., length(outlier))
    end

    log_out = - log.(zmax)
    outl = nothing
    if return_outliermap
        outl = log.(1 .- outlier) .+ log_p .< log.(outlier) .+ log_out
    end

    # Mixture of inlier and outlier probability
    log_p = log.(
        (1 .- outlier).*exp.(log_p) .+ outlier.*exp.(log_out)
    )

    # Pointwise log-probabilities
    ptw = nothing
    if return_pointwise
        ptw = log_p
    end

    log_p = sum(log_p, dims=2)
    log_p = dropdims(log_p, dims=2)

    return log_p, ptw, outl
end;

using Distributions: TruncatedNormal

struct DepthDistribution2DP3 <: Distribution{Vector{Float64}}
end

"""
    z::Vector{Float64} = depthdist_2dp3(ỹ, w̃, sig, outlier, zmax)

Restricted distribution from the 2dp3-likelihood.
Takes 1d-mixture components `ỹ` and their weights `w̃`,  and
samples a vector `z` of depth values.

Arguments:
 - `ỹ`: 1d-mixture components `(n,m)`
 - `w̃`: 1d-mixture weights `(n,m)`
 - ...
Returns:
 - `z`: Observation vector of depth values `(n,)`
"""
const depthdist_2dp3 = DepthDistribution2DP3()


function Gen.logpdf(::DepthDistribution2DP3, z, ỹ, d̃, sig, outlier, zmax, scale_noise=false, noise_anchor=1.0)
    n = size(ỹ, 1)
    m = size(ỹ, 2)

    ỹ = reshape(ỹ, 1, n, m)
    d̃ = reshape(d̃, 1, n, m)

    log_p, = depthdist_logpdf(z, ỹ, d̃, sig, outlier, zmax, scale_noise, noise_anchor; return_pointwise=false, return_outliermap=false)

    if _cuda[]
        log_p = CUDA.@allowscalar log_p[1]
    else
        log_p = log_p[1]
    end
    return log_p
end

function Gen.random(::DepthDistribution2DP3, ỹ, d̃, sig, outlier, zmax, scale_noise=false, noise_anchor=1.0)
    n = size(ỹ,1)
    m = size(ỹ,2)

    # Sample a depth values `z`
    z = Float64[]
    for i=1:n
        if bernoulli(outlier)
            # Todo: Change that to a uniform distribution using `zmax`.
            z_i = uniform(0.0, zmax)
        else
            # @assert sum(exp.(w̃[i,:])) == 1.0
            # Scale noise level proportional to depth of the mixture component.
            # Todo: Is this the right way to do it? Should we clamp `sig`?
            ỹ = Array(ỹ)
            d̃ = Array(d̃)

            if scale_noise
                # At distance `noise_anchor` the noise is `sig`
                sig′ = ỹ[i,:]./noise_anchor .* sig
            end

            # Compute normalized mixture weights
            w̃ = gaussian_logpdf(d̃[i,:], 0.0, sig)
            w̃ = w̃ .- logsumexp(w̃)

            probs = exp.(w̃)/sum(exp.(w̃))
            j   = categorical(probs)
            z_i = rand(TruncatedNormal(ỹ[i,j], scale_noise ? sig′[j] : sig, 0.0, zmax))
        end
        push!(z, z_i)
    end

    return z
end

(D::DepthDistribution2DP3)(args...)          = Gen.random(D, args...)
Gen.has_output_grad(::DepthDistribution2DP3) = false
Gen.has_argument_grads(::DepthDistribution2DP3) = (false, false);
