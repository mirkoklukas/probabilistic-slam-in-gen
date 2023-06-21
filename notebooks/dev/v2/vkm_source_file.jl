# Sensor distribution

```julia
function cast_cu(ps_::CuArray, ys_::CuArray; fov=2π, zmax::Float64=Inf, cell_rad=0.01)
    z_ = zmax*CUDA.ones(size(ps_, 1), 361)
    cast_cu!(z_, ys_, ps_; fov=fov, cell_rad=cell_rad)
    return z_
end

"""
    slw_kernel!(x, w::Int, y)

CUDA kernel to compute sliding windows...
Takes CuArrays of shape `(k,n)` and `(k,n,m=2w+1)` 
and fills the latter with ...
"""
function slw_kernel!(x, w::Int, y)
    
    m = 2*w + 1
    
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
        # Transform `1:m` to `-w:w`
        offset = j_mix-1-w
        
        # Wrap around, e.g. map `n+1` to `1`
        # Note that we have `size(x,2) == size(y,2)`
        j_mix_adj = mod(j_obs + offset - 1 , size(x,2)) + 1
        
        # Fill y's
        @inbounds y[j_pose, j_obs, j_mix] = x[j_pose, j_mix_adj]
    end
    return
end

"""

    y = slw_cu!(x::CuArray, w::Int; blockdims=(8,8,4))

CUDA-accelerated function computing sliding windows. 
Takes a CuArray of shape `(k,n)` and returns a CuArray
of shape `(k,n,m)`, where `m = 2w+1`....
"""
function slw_cu!(x::CuArray, w::Int; blockdims=(8,8,4))
    
    k = size(x, 1)
    n = size(x, 2)
    m = 2*w+1
    
    y = CUDA.ones(k,n,m)
    
    griddims = cuda_grid((k,n,m), blockdims)
    CUDA.@sync begin
        @cuda threads=blockdims blocks=griddims slw_kernel!(x, w, y)
    end
    
    return y
end;

"""
    ys_tilde_ = get_ys_tilde_cu(zs_::CuArray, w::Int)    

Takes depth measurements and returns 
the point clouds for the gaussian mixtures ...
Returns array of shape `(k, n, m, 2)` ...
"""
function get_ys_tilde_cu(zs_::CuArray, as_::CuArray, w::Int)   

    zs_tilde_ = slw_cu!(zs_, w; blockdims=(8,8,4))
    as_tilde_ = slw_cu!(reshape(as_,1,:), w; blockdims=(8,8,4))
    ys_tilde_ = polar_inv_cu(zs_tilde_, as_tilde_)
    
    return ys_tilde_
end;

"""
    log_p = sensor_smc_logpdf_cu(x::CuArray, ys_tilde::CuArray, sig, outlier)

Evaluates the logpdf of an observation `x` (shape: `(n,2)`)
with respect to a number of different gaussian mixtures (e.g. from family 
of different poses) stacked along the first dim of `ys_tilde` (k,n,m,2) ...
"""
function sensor_smc_logpdf_cu(x, ys_tilde, sig, outlier, outlier_vol=1.0)

    n = size(ys_tilde,2)
    m = size(ys_tilde,3)
    
    xs = reshape(x, 1, n, 1, 2)
    
    # Line by line...
    # 1. Compute 1D Gaussians - (n,m,2)
    # 2. Convert to 2D gausians - (n,m)
    # 3. Convert to mixture of m 2D gausians (GM) - (n,)
    # 4. Convert to mixture of `GM` and `anywhere` (D) - (n,)
    # 5. Convert to Product of D's - ()
    log_p = gaussian_logpdf(xs, ys_tilde, sig)
    log_p = sum(log_p, dims=4)[:,:,:,1] 
    log_p = logsumexp_cu(log_p .- log(m), 3)[:,:,1] 
    log_p_or_any = log.((1-outlier)*exp.(log_p) .+ outlier/outlier_vol)
    log_p = sum(log_p_or_any ,dims=2)[:,1]
    
    return log_p
end;

struct SensorDistribution_CUDA <: Distribution{Vector{Vector{Float64}}} 
end

const sensordist_cu = SensorDistribution_CUDA()

function Gen.logpdf(::SensorDistribution_CUDA, x, y_tilde_::CuArray, sig, outlier, outlier_vol=1.0)
    
    n = size(y_tilde_, 1)
    m = size(y_tilde_, 2)
    
    x_        = CuArray(stack(x))
    ys_tilde_ = reshape(y_tilde_, 1, n, m, 2) 
    
    log_p = sensor_smc_logpdf_cu(x_, ys_tilde_, sig, outlier, outlier_vol) # CuArray of length 1
    return sum(log_p)
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

```

# Model

```julia


#  ------------------
#   SLAM Model Parts
#  ------------------
@gen function pose_prior_model(p=nothing, x_noise=0.25, hd_noise=45/360*2π, bb=_bb)
    if p == nothing
        x  ~ mvuniform(bb...)
        hd ~ uniform(0,2π)
    else
        x  ~ diagnormal(p.x, [x_noise, x_noise])
        hd ~ normal(p.hd, hd_noise)
    end
    return Pose(x, hd)
end;


@gen function motion_model(p, u, x_noise=0.25, hd_noise=45/360*2π)
    dx, dhd = u.dx, u.dhd
    x   ~ diagnormal(p.x  + dx , [x_noise, x_noise])
    hd  ~ normal(p.hd + dhd, hd_noise)
    return p = Pose(x, hd)
end;


@gen function sensor_model_cu(p, segs_, w, s_noise, outlier, outliver_vol=1.0)
    p_  = CuArray(Vector(p))
    ps_ = reshape(p_, 1, 3)
    zs_ = cast_cu(ps_, segs_; fov=_fov)
    
    y_tilde_ = get_ys_tilde_cu(zs_, as_, w)[1,:,:,:]
    x ~ sensordist_cu(y_tilde_, s_noise, outlier, outliver_vol)
    
    return x
end

#  ------------
#   SLAM Model
#  ------------
@gen function slam_kernel(t, state, m, us, x_noise, hd_noise, w, s_noise, outlier, outliver_vol)
    
    p,_ = state
    u = us[t]

    p  = {:pose}   ~ motion_model(p, u, x_noise, hd_noise)
    x  = {:sensor} ~ sensor_model_cu(p, m, w, s_noise, outlier, outliver_vol)
    
return (p, x)
end
slam_chain = Gen.Unfold(slam_kernel)


@gen (static) function static_slam_model(T, 
    m,
    us, 
    p0, x0_noise, hd0_noise,  
    x_noise, hd_noise, 
    w, s_noise, outlier, outliver_vol)

# Start the Markov chain; 
# No motion, just the prior
p  = { :pose   } ~ pose_prior_model(p0, x0_noise, hd0_noise)
x  = { :sensor } ~ sensor_model_cu(p, m, w, s_noise, outlier, outliver_vol) # GPU accelerated

# Unfold the Markov chain
chain ~ slam_chain(T, (p, nothing), m, us, 
    x_noise, hd_noise, 
    w, s_noise, outlier, outliver_vol)

return [(p,x);chain]
end

Gen.@load_generated_functions

```

# Inference

## SMC Proposal

```julia
"""
    ps_ = pose_grid(p, k, dx, dhd)

Returns CuArray of pose vectors (k^3, 3).
"""
function pose_grid(p, k, dx, dhd)
    dx1_ = CUDA.collect(CUDA.range(- dx,  dx, k))
    dx2_ = CUDA.collect(CUDA.range(- dx,  dx, k))
    dhd_ = CUDA.collect(CUDA.range(-dhd, dhd, k))
    dx1_ = CuArray(dx1_)
    dx2_ = CuArray(dx2_)
    dhd_ = CuArray(dhd_)
    dx1_ = CUDA.repeat(reshape(dx1_, k,1,1), 1,k,k)
    dx2_ = CUDA.repeat(reshape(dx2_, 1,k,1), k,1,k)
    dhd_ = CUDA.repeat(reshape(dhd_, 1,1,k), k,k,1)

    ps_ = CUDA.cat(dx1_, dx2_, dhd_, dims=4)
    ps_ = reshape(ps_, :,3)
    ps_ = ps_ .+ reshape(CuArray(Vector(p)),1,3)
    
    return ps_
end


function eval_poses(
            ps_::CuArray,
            x_::CuArray, 
            ys_, as_, w::Int, 
            s_noise::Float64, outlier::Float64, outlier_vol::Float64=1.0, 
            zmax::Float64=100.0; sorted=false)
    
    # Compute sensor measurements and 
    # Gaussian mixture components
    zs_       = cast_cu(ps_, ys_; fov=π, zmax=zmax, cell_rad=0.01)
    ys_tilde_ = get_ys_tilde_cu(zs_, as_, w)
        
    # Evaluate the the observations with respect to the 
    # different Gaussian mixtures computed above
    log_ps_ = sensor_smc_logpdf_cu(x_, ys_tilde_, s_noise, outlier, outlier_vol);
    
    # Move everyting back to CPU
    ps     = Vector.(eachrow(Array(ps_)))
    log_ps = Array(log_ps_)

    
    # Sort by log prob
    # and return 
    if sorted
        perm  = sortperm(log_ps)
        log_ps = log_ps[perm]
        ps    = ps[perm]
    end
    
    return ps, log_ps
end;


@gen function grid_proposal(p::Pose, x::Matrix{Float64}, 
            k, dx, dhd,
            ys_, as_, w::Int, 
            s_noise::Float64, outlier::Float64, outlier_vol::Float64=1.0)
    
    ps_ = pose_grid(p, k, dx, dhd)
    x_  = CuArray(x)
    ps, log_ps = eval_poses(ps_, x_, ys_, as_, w, s_noise, outlier, outlier_vol)
        
    probs = exp.(log_ps .- logsumexp(log_ps))
    probs = probs/sum(probs)
    
    j = {:j} ~ categorical(probs)
    
    return Pose(ps[j]), log_ps[j], (ps, log_ps)
end;


@gen function iterated_proposal(p::Pose, x::Matrix, 
                                grid_k, grid_dx, grid_dhd,
                                ys_, as_, 
                                w, s_noise, outlier, outlier_vol)
    
    ps     = Pose[p]
    log_ps = [0.0] 
    
    n = length(grid_k)
    for i=1:n
        args = (
            p, x, 
            grid_k[i], grid_dx[i], grid_dhd[i],
            ys_, as_, 
            w[i], s_noise[i], outlier[i], outlier_vol[i]
        )
        
        p, log_p, = {i} ~ grid_proposal(args...)
        push!(ps, p)
        push!(log_ps, log_p)
    end
    
    return ps[end], sum(log_ps), (ps, log_ps)
end;
```

## Inference Loop

```julia

#  ------------
#   SMC update 
#  ------------
function extend(tr, u, obs, grid_args)

    args  = get_args(tr)    
    diffs = argdiffs([1;fill(0,length(args))])
    t = args[1]    

    # Current observation 
    # and pose estimate
    x = stack(obs[:chain => t+1 => :sensor => :x])
    p = get_last_pose(tr)
    p = Pose(p.x + u.dx, p.hd + u.dhd)

    # SMC proposal
    proposal_tr = simulate(iterated_proposal, (p, x, grid_args...));
    proposal_sc = get_score(proposal_tr)
    p′, ps′, log_ps′,  = proposal_tr[];
    
    # Update trace and return 
    # adjusted importance weights
    ch = choicemap()
    ch[:chain => t+1 => :pose => :x]  = p′.x
    ch[:chain => t+1 => :pose => :hd] = p′.hd
        
    tr′,w′,_,_ = Gen.update(tr,(t+1,args[2:end]...),diffs,merge(obs,ch))
    
    return tr′, w′ - proposal_sc, ps′, log_ps′
    
end

#  ----------------
#   Inference Loop
#  ----------------

# Grid args
n   = 3
k   = fill(5, n);
dx  = 1. ./ (k*0.75) .^ collect(range(0,n-1));
dhd = 35/360*2π ./ (k*0.75) .^ collect(range(0,n-1));
s_noise     = [range(0.5,0.1,n)...];
outlier     = fill(0.1, n);
outlier_vol = fill(1., n);
w           = fill(20, n);

grid_args = (k, dx, dhd, m_, as_, w, s_noise, outlier, outlier_vol)

# Run localization
ch = constraints(0,_zs,_as);
tr, w = generate(static_slam_model, (0, args...), ch);

for t=1:_T-1
    ch = constraints(t,_zs,_as);
    tr, w, ps, log_ps = extend(tr, us_noisy[t], ch, grid_args)
end
```

