module ManifoldsTestExt

using Manifolds
using ManifoldsBase

import Manifolds: test_manifold, test_group, test_action, test_parallel_transport, find_eps
using Manifolds: RieszRepresenterCotangentVector, get_chart_index

using Random: MersenneTwister, rand!

using Test: Test

include("tests_general.jl")
include("tests_group.jl")

end
