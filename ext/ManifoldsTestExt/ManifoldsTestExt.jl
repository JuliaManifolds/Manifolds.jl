module ManifoldsTestExt

using Manifolds
using ManifoldsBase

import Manifolds: test_manifold, test_group, test_action
using Manifolds: get_chart_index

using Random: MersenneTwister, rand!

isdefined(Base, :get_extension) ? (using Test: Test) : (using ..Test: Test)

include("tests_general.jl")
include("tests_group.jl")

end
