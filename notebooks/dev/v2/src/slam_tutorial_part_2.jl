# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``42 - SLAM Tutorial - Part 2.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

by_not_infs(x) = all(x.!=Inf) && all(x.!=-Inf)

@gen function first_slam_model(T, t0, step, ys, us, x_noise, hd_noise, w, s_noise, outlier, outliver_vol, zmax)

    ps = [Pose()]
    xs = []
    zs = []

    ms = [ys[t0]]

    for i = 1:T
        t1 = t0 + (i-1) * step
        t2 = t0 +     i * step

        p = ps[i]
        m = ms[i]
        ũ = sum(us[t1:t2-1])
        u = Control(rot(-_hds[t1]+p.hd)*ũ.dx, ũ.dhd)

        p′    = {i => :pose}   ~ motion_model(p, u, x_noise, hd_noise)
        x′,z′ = {i => :sensor} ~ sensor_model(p′, m, w, s_noise, outlier, outliver_vol, zmax)

        # Need to filter Inf's otherwise
        # creates NaN's when pose gets applied.
        x′ = filter(by_not_infs, x′)


        push!(ps,p′)
        push!(xs,x′)
        push!(zs,z′)
        push!(ms, vcat(m, x′.*p′) )

    end

    return Dict(:ps=>ps,:xs=>xs,:ms=>ms, :zs=>zs)
end

argdiffs(bs::Array{T,1}) where T <: Real = Tuple(map(b -> Bool(b) ? UnknownChange() : NoChange(), bs));
argdiffs([0,0.0,1.0, 1])

@gen function slam_kernel(i, state, t0, step, us, x_noise, hd_noise, w, s_noise, outlier, outliver_vol, zmax)

    # The map could be sample and set during updates
    p,_,m = state

    t1 = t0 + (i-1) * step
    t2 = t0 +     i * step
    ũ = sum(us[t1:t2-1])
    u = Control(rot(-_hds[t1]+p.hd)*ũ.dx, ũ.dhd)

    p′    = {:pose}   ~ motion_model(p, u, x_noise, hd_noise)
    x′,z′ = {:sensor} ~ sensor_model(p′, m, w, s_noise, outlier, outliver_vol, zmax)

    # Need to filter Inf's otherwise
    # creates NaN's when pose gets applied.
    x′ = filter(by_not_infs, x′)

    # Add the current observation to the map
    m′ = vcat(m, x′.*p′)

    state = (p′, x′, m′)
    return state
end

slam_chain = Gen.Unfold(slam_kernel)
Gen.@load_generated_functions
