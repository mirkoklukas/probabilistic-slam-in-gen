# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``24 - Slam model - Localization - SMC Move.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

yxlims(x::Vector{Float64},w::Number) = Dict(
    :xlim=>(x[1]-w, x[1]+w),
    :ylim=>(x[2]-w, x[2]+w)
);

push!(LOAD_PATH, ENV["probcomp"]*"/Gen-Distribution-Zoo/src")
using GenDistributionZoo: ProductDistribution, diagnormal

mvuniform = ProductDistribution(uniform);
mvuniform(zeros(5), ones(5))


# We use `anywhere` to model a sensor failure, in which case
# we want to ignore its measurement. This is a bit of a hack,
# but works
struct Anywhere <: Distribution{Vector{Float64}} end
const anywhere = Anywhere()

Gen.logpdf(::Anywhere, x::Vector{Float64}) = 0.0
Gen.random(::Anywhere) = [Inf;Inf]
(::Anywhere)() = Gen.random(Anywhere())
Gen.has_output_grad(::Anywhere)    = false
Gen.has_argument_grads(::Anywhere) = ();
