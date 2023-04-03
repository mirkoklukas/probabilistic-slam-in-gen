# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``24 - Slam model - Localization - SMC Move.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #


@gen function slam_kernel(t, state, us, x_noise, hd_noise, w, s_noise, dropout)

        p,_ = state
        u = us[t]

        p  = {:pose}   ~ motion_model(p, u, x_noise, hd_noise)
        x, = {:sensor} ~ sensor_model_GPU(p, w, s_noise, dropout) # GPU accelerated

    return (p, x)
end

slam_chain = Gen.Unfold(slam_kernel)
Gen.@load_generated_functions

"""
    [(p,z),...] = static_slam_model(T, segs_, a_, us, motion_noise, sensor_noise, dropout, inds)

Static SLAM model ...
"""
@gen (static) function static_slam_model(T, us,
        p0, x0_noise, hd0_noise,
        x_noise, hd_noise,
        w, s_noise, dropout)

    # Start the Markov chain;
    # No motion, just the prior
    p  = { :pose   } ~ pose_prior_model(p0, x0_noise, hd0_noise)
    x, = { :sensor } ~ sensor_model_GPU(p, w, s_noise, dropout) # GPU accelerated

    # Unfold the MArkov chain
    chain ~ slam_chain(T, (p, nothing), us,
        x_noise, hd_noise,
        w, s_noise, dropout)

    return [(p,x);chain]
end

Gen.@load_generated_functions
