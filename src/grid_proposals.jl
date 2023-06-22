# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``32 - Grid Proposals.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

using Revise
using BenchmarkTools
using Colors, Plots, Images;
col = palette(:default);
(cgrad::PlotUtils.ContinuousColorGradient)(xs::Vector{Vector{Float64}}) = [cgrad[x] for x in xs];
(cgrad::PlotUtils.ContinuousColorGradient)(m::Matrix{Float64}) = reshape(cgrad[m[:]], size(m));
using Gen
using Fmt: @f_str, format # Python-style f-strings
using CUDA

push!(LOAD_PATH, "../../../src");
using Pose2D: Pose
using Geometry: Segment
using MyUtils
using Raycaster
using SensorDistributions;

"""
Discretize into bins of diameter r, bin-centers lie
at `z - k*r` for intergers `k`.
"""
quantize(x, r; zero=0) = Int.(floor.((x .+ r./2 .- zero)./r))

"""
    get_offset(v0, k, r)

Computes the offset to move the center
of the grid to `v0`. The offset is a vector
pointing to the lower-left corner of the grid.
"""
function get_offset(v0, k, r)
    center = (r + k.*r)/2
    return v0 - center
end

""""
   v1 = first_grid_vec(v0, k, r)

Returns the first vector in the grid, ie. for a given
grid `vs` it computes vs[1].
"""
function first_grid_vec(v0::Vector{T}, k::Vector{Int}, r::Vector{T}) where T <: Real
    return r + get_offset(v0, k, r)
end

"""
    vs, ls = vector_grid(v0, k, r)

Returns grid of vectors and their linear indices, given
a grid center, numnber of grid points along each dimension and
the resolution along each dimension.
"""
function vector_grid(v0::Vector{Float64}, k::Vector{Int}, r::Vector{Float64})
    # Todo: Does it make sense to get a CUDA version of this?
    offset = get_offset(v0, k, r)

    shape = Tuple(k)
    cs = CartesianIndices(shape)
    vs = map(I -> [Tuple(I)...].*r + offset, cs);
    return vs
end


function grid_index(x, v0, k, r; linear=false)
    I = quantize(x, r, zero=get_offset(v0, k, r));
    I = CartesianIndex(I...)
    if linear
        shape = Tuple(k)
        I = LinearIndices(shape)[I]
    end
    return I
end;


function grid_index(x, vs::AbstractArray; linear=false)
    r = vs[fill(2, ndims(vs))...] - vs[fill(1, ndims(vs))...]
    offset = vs[1] - r
    I = quantize(x, r, zero=offset);
    I = CartesianIndex(I...)
    if linear
        shape = size(vs)
        I = LinearIndices(shape)[I]
    end
    return I
end;

"""
    logps, outl = eval_pose_vectors(
                    vs   :: Array{Vector{Float64}},
                    z    :: Vector{Float64},
                    segs :: Vector{Segment},
                    fov, num_a,
                    w::Int, sig, outlier, zmax::Float64=50.0;
                    sorted = false)

Evaluates a collection of poses
with respect to different Gaussian mixtures...
"""
function eval_pose_vectors(
            vs   :: Array{Vector{Float64}},
            z    :: Vector{Float64},
            segs :: Vector{Segment},
            fov, num_a::Int, w::Int,
            sig, outlier,
            zmax::Float64=50.0;
            sorted=false, return_outliermap=false)

    ps   = stack(vs[:])
    segs = stack(Vector.(segs))
    as   = create_angles(fov, num_a)

    if _cuda[]
        ps   = CuArray(ps)
        z    = CuArray(z)
        segs = CuArray(segs)
        as   = CuArray(as)
    end

    zs = cast(ps, segs; fov=fov, num_a=num_a, zmax=zmax)
    ỹ, d̃ = get_1d_mixture_components(zs, as, w);


    # Evaluate the the observations with respect to the
    # different Gaussian mixtures computed above
    logps,_, outl = depthdist_logpdf(z, ỹ, d̃, sig, outlier, zmax;
                              scale_noise=false,
                              return_pointwise=false,
                              return_outliermap=return_outliermap);

    # Move everyting back to CPU
    # if is not already there
    logps = Array(logps)
    if return_outliermap
        outl = Array(outl)
    end

    return logps, outl
end;

"""
    logps, outl = eval_pose_vectors(
                    vs   :: Array{Vector{Float64}},
                    z    :: Vector{Float64},
                    segs :: Vector{Segment},
                    fov, num_a::Int, w::Int,
                    sig     :: AbstractVector,
                    outlier :: AbstractVector,
                    zmax::Float64=50.0
                    ;sorted=false, return_outliermap=false)

Evaluates a collection of poses
with respect to different Gaussian mixtures...
"""
function eval_pose_vectors(
    vs   :: Array{Vector{Float64}},
    z    :: Vector{Float64},
    segs :: Vector{Segment},
    fov, num_a::Int, w::Int,
    sig::AbstractVector,
    outlier::AbstractVector,
    zmax::Float64=50.0
    ;sorted=false, return_outliermap=false)

    if _cuda[]
        sig     = CuArray(sig)
        outlier = CuArray(outlier)
    end

    return eval_pose_vectors(vs, z, segs, fov, num_a, w, sig, outlier, zmax;
            return_outliermap=return_outliermap)
end;
