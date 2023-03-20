# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``13 - Beam_Models.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

##################################### 
module BeamModels  
#####################################

using Colors, Plots
col = palette(:default);
using LinearAlgebra: norm

using SpecialFunctions: erf
normalcdf(x, mu=0.0, sig=1.0) = (1+erf((x-mu)/sig*sqrt(2)))/2

raw"""
    p_occ = beam_model(x, z; zmax=Inf, sig=0.5, res=0.1,
                     free=0.0, prior=0.5, occ=1.)

Implementation of the function $p(\text{occ} \mid 𝑥,𝑧)$.

Given a measurement $z$ from one of the simulated range sensors, we would like to compute
the probability of a cell at postion $x$ along the beam to be occupied:

$$
    p( \text{occ} \mid x, z) = \int p( \text{occ} \mid x, z') \cdot p( z' \mid z)  dz'.
$$

The first term of the integrand is the desired probability but knowing the ground truth $z'$, i.e.
assuming a noise-free scenario.
"""
function beam_model(x, z; zmax=Inf, sig=0.5, res=0.1,
                          free=0.0, prior=0.5, occ=1.)

    if zmax <= z; return x < zmax ? free : prior; end;
    if zmax <= x; return prior; end;

    x = clamp(x, res/2, zmax-res/2)

    x1 = x - res/2
    x2 = x + res/2

    c = normalcdf(zmax, z, sig) - normalcdf(0.0, z, sig)

    v  = 0.0
    v += prior*(normalcdf(  x1, z, sig) - normalcdf(0.0, z, sig))
    v += occ  *(normalcdf(  x2, z, sig) - normalcdf( x1, z, sig))
    v += free *(normalcdf(zmax, z, sig) - normalcdf( x2, z, sig))

    return v/c
end

bump(x, w) = (cos(clamp(x/w,-1,1)*π) + 1)/2

raw"""
    beam_model_cone(x, z; zmax=Inf, sig=0.5, res=0.1,
                     free=0.0, prior=0.5, occ=1.)

...
"""
function beam_model_cone(x, z; amax=10/180*π ,zmax=Inf, sig=1.5, res=1.,
                              free=0.0, prior=0.5, occ=1.)

    a = atan(x[2], x[1])
    if abs(a) >= amax
        return prior;
    end

    da = bump(a, amax)
    return beam_model(norm(x), z; zmax=zmax, sig=sig, res=res,
                            free=da*free + (1-da)*prior, prior=prior, occ=da*occ + (1-da)*prior)
end

#####################################
end  
#####################################
