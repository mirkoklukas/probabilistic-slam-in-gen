# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``24 - Slam model - Localization - SMC Move.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

_bb = bounding_box(_segs);

@doc """
    p = pose_prior_model(p=nothing, x_noise=0.25, hd_noise=22.5)

Model depends on global variable `_segs`.
"""
@gen function pose_prior_model(p=nothing, x_noise=0.25, hd_noise=22.5, bb=_bb)
    if p == nothing
        x  ~ mvuniform([bb[1]...],[bb[2]...])
        hd ~ uniform(0,2π)
    else
        x  ~ diagnormal(p.x, [x_noise,x_noise])
        hd ~ normal(p.hd, hd_noise/360*π)
    end
    return Pose(x,hd)
end;
