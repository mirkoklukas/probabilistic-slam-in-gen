# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``13 - Sensor Distribution - CUDA.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

using CUDA
using Gen
using GenDistributionZoo: diagnormal


# Todo: handle wrap around smarter?
"""
    slw_kernel!(x, w::Int, y)

CUDA kernel to compute sliding windows...
Takes CuArrays of shape `(k,n)` and `(k,n,m=2w+1)`
and fills the latter with ...
"""
function slw_kernel!(x, w::Int, y, wrap=false)

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
                j = max(min(j,n),1)
                val = x[j_pose, j]
            end
        end

        # Fill entries of `y`
        @inbounds y[j_pose, j_obs, j_mix] = val
    end
    return
end

"""
```julia
    y = slw_cu!(x::CuArray, w::Int; blockdims=(8,8,4))
```
CUDA-accelerated function computing sliding windows.
Takes a CuArray of shape `(k,n)` and returns a CuArray
of shape `(k,n,m)`, where `m = 2w+1`....
"""
function slw_cu!(x::CuArray, w::Int; blockdims=(8,8,4), wrap=false)

    k = size(x, 1)
    n = size(x, 2)
    m = 2*w+1

    y = CUDA.ones(k,n,m)

    griddims = cuda_grid((k,n,m), blockdims)
    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims slw_kernel!(x, w, y, wrap)
    end

    return y
end;

polar_inv_cu(z::CuArray, a::CuArray) = cat(z.*cos.(a), z.*sin.(a), dims=ndims(a)+1);

"""
    ys_tilde_ = get_ys_tilde_cu(zs_::CuArray, w::Int)

Takes depth measurements and returns
the point clouds for the gaussian mixtures ...
Returns array of shape `(k, n, m, 2)` ...
"""
function get_ys_tilde_cu(zs_::CuArray, as_::CuArray, w::Int; wrap=false)

    zs_tilde_ = slw_cu!(zs_, w; blockdims=(8,8,4), wrap=wrap)
    as_tilde_ = slw_cu!(reshape(as_,1,:), w; blockdims=(8,8,4), wrap=wrap)
    ys_tilde_ = polar_inv_cu(zs_tilde_, as_tilde_)

    return ys_tilde_
end;

function logsumexp_cu(x; dim)
    c = maximum(x)
    return c .+ log.(sum(exp.(x .- c), dims=dim))
end


"""
    log_p = gaussian_logpdf(x, mu, sig)

Benchmarks in `33 - CUDA Accelerated Gen Distributions` ...
"""
function gaussian_logpdf(x, mu, sig)
    d = (x .- mu).^2 ./ sig.^2
    log_p = - log.(sig .* sqrt(2π)) .- 1/2 * d
    return log_p
end;

"""
    log_p = sensor_smc_logpdf_cu(x::CuArray, ys_tilde::CuArray, sig, outlier)

Evaluates the logpdf of an observation `x` (shape: `(n,2)`)
with respect to a number of different gaussian mixtures (e.g. from family
of different poses) stacked along the first dim of `y` (shape: `(k,n,m,2)`).
Returns an array of log probabilities (shape: `(k,)`).
"""
function sensor_smc_logpdf_cu(x, y, sig, outlier, outlier_vol=1.0; return_pointwise=false)
    @assert size(x,1) == size(y,2)

    n = size(y,2)
    m = size(y,3)
    x = reshape(x, 1, n, 1, 2)

    # Line by line...
    # 1. Compute 1D Gaussians - (n,m,2)
    # 2. Convert to 2D gausians - (n,m)
    # 3. Convert to mixture of m 2D gausians (GM) - (n,)
    # 4. Convert to mixture of `GM` and `anywhere` (D) - (n,)
    # 5. Convert to Product of D's - ()
    log_p = gaussian_logpdf(x, y, sig)
    log_p = sum(log_p, dims=4)
    log_p = logsumexp_cu(log_p .- log(m), dim=3)
    log_p = log.((1 .- outlier).*exp.(log_p) .+ outlier./outlier_vol)

    pointwise = nothing
    if return_pointwise
        pointwise = dropdims(log_p, dims=(3,4))
    end

    log_p = sum(log_p, dims=2)
    log_p  = dropdims(log_p, dims=(2,3,4))

    return log_p, pointwise
end;

struct SensorDistribution_CUDA <: Distribution{Vector{Vector{Float64}}}
end

const sensordist_cu = SensorDistribution_CUDA()

function Gen.logpdf(::SensorDistribution_CUDA, x, y_tilde_::CuArray, sig, outlier, outlier_vol=1.0)

    n = size(y_tilde_, 1)
    m = size(y_tilde_, 2)

    x_        = CuArray(stack(x))
    ys_tilde_ = reshape(y_tilde_, 1, n, m, 2)

    log_p, = sensor_smc_logpdf_cu(x_, ys_tilde_, sig, outlier, outlier_vol) # CuArray of length 1
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

(D::SensorDistribution_CUDA)(args...) = Gen.random(D, args...)
Gen.has_output_grad(::SensorDistribution_CUDA)    = false
Gen.has_argument_grads(::SensorDistribution_CUDA) = (false, false);
