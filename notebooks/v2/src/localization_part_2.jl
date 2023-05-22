# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``52 - Localization Tutorial - Part 2.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

using GenParticleFilters

function perturb(u::Control, x_noise, hd_noise)
    dx  = u.dx  + diagnormal([0.,0.], [x_noise, x_noise])
    dhd = u.dhd + normal(0, hd_noise)
    return Control(dx,dhd)
end

@gen function slam_kernel(t, state, m, us, x_noise, hd_noise, w, s_noise, outlier, outlier_vol)

    p,_ = state
    u = us[t]

    p  = {:pose}   ~ motion_model(p, u, x_noise, hd_noise)
    x  = {:sensor} ~ sensor_model(p, m, w, s_noise, outlier, outlier_vol)

    state = (p, x)
return state
end

slam_chain = Gen.Unfold(slam_kernel)
Gen.@load_generated_functions

@gen (static) function static_slam_model(T,
        m,
        us,
        p0, x0_noise, hd0_noise,
        x_noise, hd_noise,
        w, s_noise, outlier, outlier_vol)

    # Start the Markov chain;
    # No motion, just the prior
    p  = { :pose   } ~ pose_prior_model(p0, x0_noise, hd0_noise)
    x  = { :sensor } ~ sensor_model(p, m, w, s_noise, outlier, outlier_vol) # GPU accelerated

    # Unfold the MArkov chain
    chain ~ slam_chain(T, (p, nothing), m, us,
        x_noise, hd_noise,
        w, s_noise, outlier, outlier_vol)

    return [(p,x);chain]
end

Gen.@load_generated_functions

add_addr_prefix(t, addr) = t==0 ? addr : :chain => t => addr

function constraints(t::Int, _zs, _as)
    ch = choicemap()
    # if t==0
    #     addr  = :sensor => :x
    # else
    #     addr  = :chain => t => :sensor => :x
    # end
    n = length(_zs[t+1])
    x = polar_inv(_zs[t+1],_as)
    ch[add_addr_prefix(t, :sensor => :x)] = x
    return ch
end

get_pose(tr,t)     = tr[][t][1]
get_last_pose(tr)  = tr[][end][1]
get_first_pose(tr) = get_pose(tr,1)
get_obs(tr,t)      = tr[][t][2]
get_first_obs(tr)  = get_obs(tr,1)

function plot_slam_trace!(tr; show_obs=true)
    T,m, = get_args(tr)
    ps = get_pose.([tr],1:T+1)
    xs = get_obs.([tr],1:T+1)

    myplot = plot(size=(300,300),
        title="A Trace",
        aspect_ratio=1., legend=nothing)

    plot!(_ps, c=:red)
    plot!(ps, c=col[1])
    plot!(_segs,  c=:black)
    plot!(_boxes, c=:magenta)
    if show_obs
        for (x,p) in zip(xs, ps)
            myplot = scatter!(x .* p, c=col[1], markersize=2)
        end
    end
    return myplot
end

argdiffs(bs::Array{T,1}) where T <: Real = Tuple(map(b -> Bool(b) ? UnknownChange() : NoChange(), bs));
argdiffs([0,0.0,1.0, 1])

@gen function pose_drift_proposal(tr, x_noise, hd_noise, vars=[:x,:hd])

    T, = get_args(tr)
    p  = get_pose(tr,T+1)

    if :x in vars
        x  = {add_addr_prefix(T, :pose => :x)}  ~ diagnormal(p.x, [x_noise, x_noise])
    end

    if :hd in vars
        hd = {add_addr_prefix(T, :pose => :hd)} ~ normal(p.hd, hd_noise)
    end

    tr
end;
