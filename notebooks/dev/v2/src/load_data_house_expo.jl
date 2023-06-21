# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``01 - HouseExpo Data.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

using JLD2
using StatsBase: mean
using Geometry: bounding_box, Segment

fname = "../data/data_4.jld2"
d = load_object(fname)

# Environment
_segs   = env_segs = d[:env][:segs];
_boxes  = env_segs = d[:env][:clutter];
_center = mean(bounding_box(_segs))

# Poses
_xs   = d[:motion][:x];
_hds  = d[:motion][:hd];
_ps   = [Pose(x,hd) for (x,hd) in zip(_xs, _hds)];

# Controls
_dxs  = d[:motion][:dx]
_dhds = d[:motion][:dhd]
_us   = [Control(dx,dhd) for (dx,dhd) in zip(_dxs, _dhds)]

_T = length(_xs);

println("Loading `$(fname)` ...\n")
for x in [:_segs, :_boxes, :_xs, :_hds, :_ps, :_dxs, :_dhds, :_us, :_T]
    local y = getproperty(Main,x)
    println("\t$(x) \t$(typeof(y))")
end
