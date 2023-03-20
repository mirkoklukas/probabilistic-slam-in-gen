# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``12 - Measurements.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

using Colors, Plots
col = palette(:default);
using MyUtils
using LinearAlgebra: norm

"""
    Measurement(z,a)

Depth measurement in a specified direction.
"""
mutable struct Measurement
    z::Float64
    a::Float64
end
Measurement(x::Vector{Float64}) = Measurement(norm(x), atan(x[2],x[1]));
tuple(s::Measurement) = (z,a);
depth(s::Measurement) = s.z;
angle(s::Measurement) = s.a;
euclidean(s::Measurement) = [s.z*cos(s.a); s.z*sin(s.a)];
polar(s::Measurement) = [s.z;s.a]
vector(s::Measurement) = [s.z*cos(s.a); s.z*sin(s.a)];
# Vector(s::Measurement) = [s.z*cos(s.a); s.z*sin(s.a)];
vec(s::Measurement)    = [s.z*cos(s.a); s.z*sin(s.a)];
Base.:(*)(s::Measurement, p::Pose) = vector(s) * p;

export Measurement, depth, dir, vector, vec, depth, angle

function Plots.scatter!(p::Pose, ss::Vector{Measurement}; args...)
    xs = vector.(ss) .* p
    xs = stack(xs)
    scatter!(xs[:,1], xs[:,2]; label=nothing, args...)
end

Plots.scatter!(ss::Vector{Measurement}, p::Pose; args...) = scatter!(p,ss; args...)
Plots.scatter!(ss::Vector{Measurement}; args...) = scatter!(Pose(), ss;args)
