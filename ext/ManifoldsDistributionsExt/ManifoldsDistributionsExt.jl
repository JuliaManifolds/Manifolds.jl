module ManifoldsDistributionsExt

using Manifolds
using Distributions
using Random
using LinearAlgebra

import Manifolds:
    normal_rotation_distribution,
    normal_tvector_distribution,
    projected_distribution,
    uniform_distribution

using Manifolds: get_iterator, get_parameter, _read, _write

using RecursiveArrayTools: ArrayPartition

include("distributions.jl")
include("distributions_for_manifolds.jl")
include("projected_distribution.jl")

end
