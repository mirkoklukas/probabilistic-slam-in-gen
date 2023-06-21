# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``02 - Data Helper.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #


import JSON
using Colors, Plots
col = palette(:default);

"""
    _segs, _boxes, _paths = load_env(fname)
"""
function load_env(fname)
    d = JSON.parsefile(fname)
    verts    = Vector{Vector{Float64}}(d["verts"]);
    clutter  = Vector{Vector{Vector{Float64}}}(d["clutter_verts"]);
    _paths  = Vector{Vector{Vector{Float64}}}(d["paths"]);

    _segs   = segments(verts);
    _boxes  = vcat(segments.(clutter)...);

    return _segs, _boxes, _paths
end

"""
    _ps, _us = unpack_path(path)
"""
function unpack_path(path)
    xs = path

    # Unpack path into
    # poses and controls
    _dxs  = xs[2:end] - xs[1:end-1]
    _hds  = angle.(_dxs)
    _dhds = _hds[2:end] - _hds[1:end-1];
    _xs   = xs[1:end-2];

    _ps = [Pose(x,hd) for (x,hd) in zip(_xs, _hds)];
    _us = [Control(dx,dhd) for (dx,dhd) in zip(_dxs, _dhds)]

    _T  = length(_xs);

    return _ps, _us
end
