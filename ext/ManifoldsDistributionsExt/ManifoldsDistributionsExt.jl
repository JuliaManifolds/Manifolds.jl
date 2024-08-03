module ManifoldsDistributionsExt

if isdefined(Base, :get_extension)
    using Manifolds
    using Distributions
    using Random
    using LinearAlgebra

    import Manifolds:
        normal_rotation_distribution,
        normal_tvector_distribution,
        projected_distribution,
        uniform_distribution

    using Manifolds: get_parameter

    using RecursiveArrayTools: ArrayPartition
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..Manifolds
    using ..Distributions
    using ..Random
    using ..LinearAlgebra

    import ..Manifolds:
        normal_rotation_distribution,
        normal_tvector_distribution,
        projected_distribution,
        uniform_distribution

    using ..Manifolds: get_parameter

    using ..RecursiveArrayTools: ArrayPartition
end

include("distributions.jl")
include("distributions_for_manifolds.jl")
include("projected_distribution.jl")

end
