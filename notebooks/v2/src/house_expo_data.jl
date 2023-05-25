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
using Fmt
using MyUtils: summarize_vars

fname = "../data/data_4.jld2"
d = load_object(fname)

# Environment
_segs   = env_segs = d[:env][:segs];
_boxes  = env_segs = d[:env][:clutter];
_bb     = bounding_box(_segs)
_center = mean(_bb);

# Poses
_xs   = d[:motion][:x];
_hds  = d[:motion][:hd];
_ps   = [Pose(x,hd) for (x,hd) in zip(_xs, _hds)];

# Controls
_dxs  = d[:motion][:dx]
_dhds = d[:motion][:dhd]
_us   = [Control(dx,dhd) for (dx,dhd) in zip(_dxs, _dhds)]

_T = length(_xs);

println("Loading data from `$(fname)`, and\npopulating Main namespace with variables:\n")
summarize_vars(:(_segs, _boxes, _center, _xs, _hds, _ps, _dxs, _dhds, _us, _T))
