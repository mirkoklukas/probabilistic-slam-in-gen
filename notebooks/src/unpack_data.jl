# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``02 - Observation data.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

push!(LOAD_PATH, "src");
using JLD2
using StatsBase: mean
using Geometry: bounding_box, Segment
using GridSlam: Pose, Measurement

fname = "data/data_3.jld2"
d = load_object(fname)

# Environment
_segs   = env_segs = d[:env][:segs];
_center = mean(bounding_box(_segs))

# Poses
_xs   = d[:motion][:x];
_hds  = d[:motion][:hd];
_ps   = [Pose(x,hd) for (x,hd) in zip(_xs, _hds)];

# Controls
_dxs  = d[:motion][:dx]
_dhds = d[:motion][:dhd]
_us   = [(dx,dhd) for (dx,dhd) in zip(_dxs, _dhds)]

# Sensor and measurements
_as    = d[:sensor][:a];
_num_a = _na = length(_as)
_zs   = d[:sensor][:z];
_ss   = [Measurement.(z,_as) for z in _zs]
_fov  = d[:sensor][:fov]
_zmax = d[:sensor][:zmax]

_T = length(_zs);

println("Loading `$(fname)` ...\n")
for x in [:_segs, :_xs, :_hds, :_ps, :_dxs, :_dhds, :_us, :_as, :_zs, :_ss, :_fov, :_zmax, :_T]
    local y = getproperty(Main,x)
    println("\t$(x) \t$(typeof(y))")
end
println("\nTo take a look call `glimpse_at_data(_segs, _ps, _ss)`")

using Colors, Plots
col = palette(:default);

function glimpse_at_data(_segs, _ps, _ss, t=nothing)
    t = t == nothing ? rand(1:_T) : t;
    # -----------------
    myplot = plot(size=(350,350), aspect_ratio=:equal, legend=false)
    plot!(_segs, c=:black)
    plot!([p.x for p in _ps], marker=:o, label=nothing, markersize=2, c=col[1], alpha=0.5)
    scatter!([_ps[t].x], markersize=4, marker=:^, c=:red)
    scatter!(_ps[t], _ss[t], markersize=2, alpha=1, markerstrokewidth=0., marker=:o, c=col[2])
    display(myplot)
end;

