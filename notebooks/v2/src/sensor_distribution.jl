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
using CUDA
using Gen
using MyUtils # ../src
using GenDistributionZoo: diagnormal
using Test

"""
    log_p = gaussian_logpdf(x, mu, sig)

Benchmarked in `33 - CUDA Accelerated Gen Distributions`.
"""
function gaussian_logpdf(x, mu, sig)
    d = (x .- mu).^2 ./ sig.^2
    log_p = - log.(sig) .- log(sqrt(2π)) .- 1/2 * d
    return log_p
end;

"""
    logsumexp(x; dims)

Logsumexp along dimenstions.
"""
function Gen.logsumexp(x; dims)
    c = maximum(x)
    return c .+ log.(sum(exp.(x .- c), dims=dims))
end;

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

# Todo: handle wrap around and padding smarter?
"""
    slw_kernel!(x, y, w, pad, fill_val)

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
            j = mod(j - 1 , n) + 1
            val = x[j_pose, j]
        else
            if 1 <= j <= n
                val = x[j_pose, j]
            else
                if fill
                    val = fill_val
                else
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
    y = slw_cu!(x::CuArray, w::Int; blockdims=(8,8,4), wrap=false, pad_val=nothing)
```
CUDA-accelerated function computing sliding windows.
Takes a CuArray of shape `(k,n)` and returns a CuArray
of shape `(k,n,m)`, where `m = 2w+1`....
"""
function slw_cu!(x::CuArray, w::Int; blockdims=(8,8,4), wrap=false, fill=false, fill_val=Inf)

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

polar_inv(z::CuArray, a::CuArray) = cat(z.*cos.(a), z.*sin.(a), dims=ndims(a)+1);

"""
    ys_tilde_ = get_ys_tilde_cu(zs_::CuArray, w::Int)

Computes the 2d mixture components for the "2dp3" likelihood from
    depth measurements `zs_` along angles `as_`, and with a filter radius of `w`.

    Arguments:
            zs_: Range measurements `(k,n)`
            as_: Angles of measuremnts `(n,)`
            w:   Filter window radius

    Returns:
        CuArray of shape `(k, n, 2w+1, 2)`
"""
function get_ys_tilde_cu(zs_::CuArray, as_::CuArray, w::Int; wrap=false, fill=false, fill_val=0.0)

    zs_tilde_ = slw_cu!(zs_, w; blockdims=(8,8,4), wrap=wrap)
    as_tilde_ = slw_cu!(reshape(as_,1,:), w; blockdims=(8,8,4), wrap=wrap)
    ys_tilde_ = polar_inv(zs_tilde_, as_tilde_)

    return ys_tilde_
end;

# Same as above but want to switch
# to different name going forward
"""
```julia
    mcs_::CuArray = get_2d_mixture_components(zs_::CuArray, as_::CuArray, w::Int;
                                     wrap=false, fill=true, fill_val=Inf)
```
Computes the 2d mixture components for the "2dp3" likelihood from
depth measurements `zs_` along angles `as_`, and with a filter radius of `w`.

Arguments:
     zs_: Range measurements `(k,n)`
     as_: Angles of measuremnts `(n,)`
     w:   Filter window radius

Returns:
    CuArray of shape `(k, n, 2w+1, 2)`
"""
function get_2d_mixture_components(zs_::CuArray, as_::CuArray, w::Int; wrap=false, fill=true, fill_val=Inf)
    zs_tilde_ = slw_cu!(zs_, w; blockdims=(8,8,4),  wrap=wrap, fill=fill, fill_val=fill_val)
    as_tilde_ = slw_cu!(reshape(as_,1,:), w; blockdims=(8,8,4), wrap=false, fill=true, fill_val=Inf)
    ys_tilde_ = polar_inv(zs_tilde_, as_tilde_)

    return ys_tilde_
end

"""
```julia
    log_ps, ptw = sensor_smc_logpdf_cu(x, ys, sig, outlier, outlier_vol=1.0; return_pointwise=false)
```
Evaluates an observation `x` under the 2dp3 likelihood with
a family of mixture components `ys` and parameters `sig`, `outlier`, and `outlier_vol`.

Arguments:
    x:  Observation point cloud `(n,2)`
    ys: Family of mixture components `(k,n,m,2)`
    ...

Returns:
    log_ps: Logprobs `(k,)`
    ptw:    Logprobs of each observation point `(k,n)`
"""
function sensor_smc_logpdf_cu(x, y, sig, outlier, outlier_vol=1.0; return_pointwise=false)
    @assert size(x,1) == size(y,2)

    n = size(y,2)
    m = size(y,3)
    x = reshape(x, 1, n, 1, 2)

    # Line by line...
    # 1. Compute 1D Gaussians (n,m,2)
    # 2. Convert to 2D gausians (n,m)
    # 3. Convert to mixture of m 2D gausians `GM` (n,)
    # 4. Convert to mixture of `GM` and `anywhere` (n,)
    log_p = gaussian_logpdf(x, y, sig)
    log_p = sum(log_p, dims=4)
    log_p = logsumexp(log_p .- log(m), dims=3)
    log_p = log.((1 .- outlier).*exp.(log_p) .+ outlier./outlier_vol)

    # If we don't need pointwise logprobs
    # we can save us the time and space to copy
    pointwise = nothing
    if return_pointwise
        pointwise = dropdims(log_p, dims=(3,4))
    end

    # Convert to Product of mixtures
    log_p = sum(log_p, dims=2)
    log_p  = dropdims(log_p, dims=(2,3,4))

    return log_p, pointwise
end;

"""
```julia
    log_ps, ptw = sensor_logpdf(x, ys, sig, outlier, outlier_vol=1.0; return_pointwise=false)
```
Evaluates an observation `x` under the 2dp3 likelihood with
a family of mixture components `ys` and parameters `sig`, `outlier`, and `outlier_vol`.

Arguments:
- x:  Observation point cloud `(n,2)`
- ys: Family of mixture components `(k,n,m,2)`
- ...

Returns:
- log_ps: Logprobs `(k,)`
- ptw:    Logprobs of each observation point `(k,n)`
"""
sensor_logpdf(x, y, sig, outlier, outlier_vol=1.0; return_pointwise=false) = sensor_smc_logpdf_cu(x, y, sig, outlier, outlier_vol; return_pointwise=return_pointwise)

struct SensorDistribution_CUDA <: Distribution{Vector{Vector{Float64}}}
end

"""
    sensordist_cu(mcs_::CuArray, sig, outlier, outlier_vol=1.0)
"""
const sensordist_cu = SensorDistribution_CUDA()

function Gen.logpdf(::SensorDistribution_CUDA, x, y_tilde_::CuArray, sig, outlier, outlier_vol=1.0)

    n = size(y_tilde_, 1)
    m = size(y_tilde_, 2)

    x_        = CuArray(stack(x))
    ys_tilde_ = reshape(y_tilde_, 1, n, m, 2)

    log_p, = sensor_logpdf(x_, ys_tilde_, sig, outlier, outlier_vol) # CuArray of length 1
    return CUDA.@allowscalar log_p[1]
end

function Gen.random(::SensorDistribution_CUDA, y_tilde_::CuArray, sig, outlier, outlier_vol=1.0)
    n = size(y_tilde_,1)
    m = size(y_tilde_,2)

    # Sample an observation point cloud `x`
    x = Vector{Float64}[]
    for i=1:n
        if bernoulli(outlier)
            # Todo: Change that to a uniform distribution, e.g. over a
            #       circular area with radius `zmax`.
            x_i = [Inf;Inf]
        else
            j   = rand(1:m)
            y   = Array(y_tilde_[i,j,:])
            x_i = diagnormal(y, [sig;sig])

        end
        push!(x, x_i)
    end

    return x
end

(D::SensorDistribution_CUDA)(args...)             = Gen.random(D, args...)
Gen.has_output_grad(::SensorDistribution_CUDA)    = false
Gen.has_argument_grads(::SensorDistribution_CUDA) = (false, false);
