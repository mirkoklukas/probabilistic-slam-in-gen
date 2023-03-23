# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``14 - Grid_Map.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

using MyUtils
using Colors, Plots
col = palette(:default);
(cgrad::PlotUtils.ContinuousColorGradient)(xs::Vector{Vector{Float64}}) = [cgrad[x] for x in xs];
(cgrad::PlotUtils.ContinuousColorGradient)(m::Matrix{Float64}) = reshape(cgrad[m[:]], size(m));

logit(p::Real)   = log(p/(1 -p));
ell(p::Real)     = logit(p);
expit(ell::Real) = 1 - 1/(1 + exp(ell));

export logit, ell, expit

abstract type AbstractGridMap{C} end

function Base.getindex(m::AbstractGridMap{C}, cell::C) where C error("Not implemented") end
function Base.setindex!(m::AbstractGridMap{C}, v::Float64, cell::C) where C error("Not implemented") end
function on_map(cell::C, m::AbstractGridMap{C}) where C error("Not implemented") end

export AbstractGridMap

function update!(m::AbstractGridMap, p::Pose, s::Measurement; args...)
    for c in affected_cells(m, p, s)
        if !on_map(c, m);
            continue
        end;

        # Cell occupancy from beam model ...
        w = cell_occupancy(c, m, p, s; args...)

        # Bayesian cell update
        # Todo: cite reference
        m[c] = clamp(m[c] + ell(w) - ell(m.prior), ell(1e-6), ell(1.0 - 1e-6))
    end
    return m
end

function update!(m::AbstractGridMap, p::Pose, ss::Vector{Measurement}; args...)
    for s in ss
            update!(m, p, s; args...)
    end
    return m
end;
export update!

"""
Discretize into bins of diameter r, bin-centers lie
at `z - k*r` for intergers `k`.
"""
quantize(x, r; zero=0) = Int(floor((x + r/2 - zero)/r))
export quantize

grid_index(x::Vector{Float64}, m::AbstractGridMap{CartesianIndex}) = CartesianIndex((quantize.(x - m.xs[1,1], m.res) .+ 1)...)
on_map(i::CartesianIndex,  m::AbstractGridMap{CartesianIndex}) = prod(1 .<= Tuple(i) .<= m.shape)
on_map(x::Vector{Float64}, m::AbstractGridMap{CartesianIndex}) = on_map(grid_index(x,m), m::AbstractGridMap)
Base.getindex( m::AbstractGridMap{CartesianIndex}, i::CartesianIndex) = m.vs[i]
Base.getindex( m::AbstractGridMap{CartesianIndex}, x::Vector{Float64}) = m.vs[grid_index(x,m)]
Base.setindex!(m::AbstractGridMap{CartesianIndex}, v::Float64, i::CartesianIndex) = m.vs[i] = v
export grid_index, on_map

function Plots.plot!(m::AbstractGridMap{CartesianIndex}, repr=:prob; ticks=false)
    vs = Matrix(m.vs')
    if repr == :raw
        mat = Matrix(vs)
    elseif repr == :prob
        mat = expit.(vs)
    elseif repr == :log
        mat = log.(expit.(vs))
    end

    x = m.xs[1]

    xticks = m.xs[1][1]-m.res/2:m.res:m.xs[end][1]+m.res/2
    yticks = m.xs[1][2]-m.res/2:m.res:m.xs[end][2]+m.res/2
    if ticks
        plot!(xticks, yticks, cgrad(:viridis)(mat),  yflip = false, xticks=xticks, yticks=yticks)
    else
        plot!(xticks, yticks, cgrad(:viridis)(mat),  yflip = false)
    end
end;

mutable struct SimpleGridMap <: AbstractGridMap{CartesianIndex}
    shape::Tuple{Int, Int}
    res::Float64
    prior::Float64
    xs::Matrix{Vector{Float64}}
    vs::Matrix{Float64}
    # Normalizing constant, i.e. `sum(expit.(vs))`
    # We need that later for quicker sampling and evaluating
    # Todo: keep that updated in `update!`
    weight::Float64
end

function SimpleGridMap(shape::Tuple{Int,Int}; res::Float64=1.0, prior::Float64=0.5)
    center = res*([shape[1];shape[2]]/2 + [1,1]/2)
    xs     = [collect.(Iterators.product(1.0:shape[1], 1.0:shape[2]))...]
    xs     = res.*xs .- [center]
    xs     = reshape(xs, shape)
    vs     = ell.(zeros(shape) .+ prior)
    w = sum(expit.(vs[:]))
    SimpleGridMap(shape, res, prior, xs, vs, w)
end

function SimpleGridMap(shape::Tuple{Int,Int}, center::Vector{Float64}; res::Float64=1.0, prior::Float64=0.5)
    mid = res*([shape[1];shape[2]]/2 + [1,1]/2)
    xs     = [collect.(Iterators.product(1.0:shape[1], 1.0:shape[2]))...]
    xs     = res.*xs .- [mid - center]
    xs     = reshape(xs, shape)
    vs     = ell.(zeros(shape) .+ prior)
    w = sum(expit.(vs[:]))
    SimpleGridMap(shape, res, prior, xs, vs, w)
end

function weigh!(m::SimpleGridMap)
    m.weight = sum(expit.(m.vs))
end

function grid_probs(m::SimpleGridMap)
    probs = expit.(m.vs)
    probs = probs ./ sum(probs)
    return probs
end

export SimpleGridMap, weigh!

using Bresenham: line
using LinearAlgebra: dot, norm
using BeamModels: beam_model

function affected_cells(m::SimpleGridMap, p::Pose, s::Measurement)
    x = p.x
    z = s.z
    # Todo: find a better endpoint for the line,
    # maybe let the measurement know about `zmax`
    # so we can use it here ...
    y = (1 + 10*m.res/z)*vector(s)*p
    return line(grid_index(x, m), grid_index(y, m), m.shape)
end


function cell_occupancy(i::CartesianIndex, m::SimpleGridMap, p::Pose, s::Measurement; free=0.4, args...)
    ip = grid_index(p.x, m)
    is = grid_index(vector(s)*p, m)
    if !on_map(is, m)
        return free
    end
    c = norm(m.xs[ip] - m.xs[i])
    z = norm(m.xs[ip] - m.xs[is])

    return beam_model(c, z; zmax=100, sig=m.res, res=m.res,
                            free=free, prior=m.prior, occ=.8, args...)
end
export affected_cells, cell_occupancy

using Gen

struct GridMapDistribution <: Gen.Distribution{Vector{Float64}} end
const gridmapdist = GridMapDistribution()

function Gen.logpdf(::GridMapDistribution, x::Vector{Float64}, p::Pose, m::AbstractGridMap{CartesianIndex})
    # Transfroms x (assumed in pose coordinates) to map coordinates.
    # (syntactic sugar for poses)
    y = x*p

    if !on_map(y, m) return -Inf end

    # Todo: make sure `m.weights` is up to date
    # Note: assumes an "ell"-map, i.e. logit values.
    return log(expit(m[y])) - log(1/m.res^2) - log(m.weight)
end

function Gen.random(::GridMapDistribution, p::Pose, m::AbstractGridMap{CartesianIndex})
    # Sample from mixture of all grid cells,
    # weighted by their occupancy value.
    # Todo: make sure `m.weights` is up to date
    probs = expit.(m.vs[:])
    probs ./= m.weight
    i = categorical(probs)
    x = m.xs[i] + m.res*(rand(2) .- 0.5)

    # Transforms from map coordinates into pose's coordinates
    y = x/p
    return y
end

(::GridMapDistribution)(p::Pose, m::AbstractGridMap{CartesianIndex}) = Gen.random(GridMapDistribution(), p, m)
# `false` for now, if `true` I assume I have to define something else so...
Gen.has_output_grad(::GridMapDistribution)    = false;
Gen.has_argument_grads(::GridMapDistribution) = (false,false);
export gridmapdist, GridMapDistribution
