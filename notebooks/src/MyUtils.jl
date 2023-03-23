# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``00 - My Utils.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

##################################### 
module MyUtils  
#####################################

using LinearAlgebra

unit_vec(a::Float64)    = [cos(a);sin(a)];
polar(x::Vector{Float64}) = [norm(x);atan(x[2],x[1])];
euclidean(r_and_phi::Vector{Float64}) = [r_and_phi[1]*cos(r_and_phi[2]);r_and_phi[1]*sin(r_and_phi[2])]
euclidean(r::Float64, phi::Float64) = [r*cos(phi);r*sin(phi)]
LinearAlgebra.angle(x::Vector{Float64}) = atan(x[2],x[1]);
stack(xs::AbstractVector) = reduce(vcat, transpose.(xs));
peak_to_peak(xs) = (xs .- minimum(xs))./(maximum(xs) - minimum(xs))

export unit_vec, polar, angle, stack, peak_to_peak, euclidean

rot(hd) = [[cos(hd) -sin(hd)]; [sin(hd) cos(hd)]]

export rot

using Colors, Plots
col = palette(:default);

Plots.scatter!(xs::Vector{Vector{Float64}}; args...) = scatter!([x[1] for x in xs], [x[2] for x in xs]; args...)
Plots.plot!(xs::Vector{Vector{Float64}}; args...)    = plot!([x[1] for x in xs], [x[2] for x in xs]; args...)

#####################################
end  
#####################################
