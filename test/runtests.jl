include("autodiff.jl")
include("utils.jl")
include("groups/group_utils.jl")
include("sized_abstract_array.jl")

# starting with tests of simple manifolds
include("euclidean.jl")
include("fixedRank.jl")
include("grassmann.jl")
include("sphere.jl")
include("stiefel.jl")
include("symmetric.jl")
include("rotations.jl")
include("symmetric_positive_definite.jl")
include("cholesky_space.jl")
include("product_manifold.jl")
include("power_manifold.jl")
include("vector_bundle.jl")

include("metric.jl")
include("statistics.jl")

# Lie groups and actions
include("groups/rotation_action.jl")
