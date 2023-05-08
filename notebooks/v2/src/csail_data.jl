# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``02 - CSAIL Data.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

import JSON
using MyUtils: polar_inv
SLAM_DATA_KEYS = ["z", "a", "x", "hd", "dx", "dhd"]

"""
    d::Dict = load_sensor_data(fname)

Loads sensor data json. Assumes/requires that the json
contains keys: `z`, `a`, `x`, `hd`, `dx`, and `dhd`.
"""
function load_sensor_data(fname)
    d′  = JSON.parsefile(fname)
    d = Dict()
    for k in SLAM_DATA_KEYS
        d[k] = [Float64.(x) for x in d′[k]]
    end
    return d
end

fname = "../data/mit-csail.json"
d     = load_sensor_data(fname);

_zs   = d["z"]
_zmax = maximum(maximum.(_zs))

_zs_inf = d["z"]
for i=1:length(_zs)
    z = _zs[i]
    _zs_inf[i][z.==_zmax] .= Inf
end

_as     = d["a"]
_ys     = [polar_inv(z,_as) for z in _zs];
_ys_inf = [polar_inv(z,_as) for z in _zs_inf];

_xs    = d["x"]
_hds   = d["hd"]
_ps    = [Pose(x,hd) for (x,hd) in zip(_xs, _hds)]

_dxs   = d["dx"]
_dhds  = d["dhd"]
_us    = [Control(dx,dhd) for (dx,dhd) in zip(_dxs, _dhds)]

_T     = length(_zs)
_num_a = length(_as)

println("Loading `$(fname)` ...\n")
for x in [:_zs, :_zs_inf, :_as, :_zmax, :_ys, :_ys_inf, :_num_a, :_xs, :_hds, :_ps, :_dxs, :_dhds, :_us, :_T]
    local y = getproperty(Main,x)
    println("\t$(x) \t$(typeof(y))")
end
