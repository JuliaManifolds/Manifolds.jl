module ManifoldsTestExt

if isdefined(Base, :get_extension)
    using Manifolds
    using ManifoldsBase

    import Manifolds:
        test_manifold, test_group, test_action, test_parallel_transport, find_eps
    using Manifolds: RieszRepresenterCotangentVector, get_chart_index
    using Manifolds: ManifoldFeatures, ManifoldExpectations, has_feature_expectations
    using Random: MersenneTwister, rand!

    using Test: Test
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..Manifolds
    using ..ManifoldsBase

    import ..Manifolds:
        test_manifold, test_group, test_action, test_parallel_transport, find_eps
    using ..Manifolds: RieszRepresenterCotangentVector, get_chart_index
    using ..Manifolds: ManifoldFeatures, ManifoldExpectations, has_feature_expectations

    using ..Random: MersenneTwister, rand!

    using ..Test: Test
end

include("tests_general.jl")
include("tests_group.jl")
include("tests_manifold.jl")
end
