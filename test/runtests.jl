include("autodiff.jl")
include("utils.jl")
include("numbers.jl")
include("groups/group_utils.jl")
include("sized_abstract_array.jl")

# starting with tests of simple manifolds
include("circle.jl")
include("euclidean.jl")
include("fixed_rank.jl")
include("grassmann.jl")
include("hyperbolic.jl")
include("sphere.jl")
include("stiefel.jl")
include("symmetric.jl")
include("rotations.jl")
include("symmetric_positive_definite.jl")
include("cholesky_space.jl")

#meta manifolds
include("product_manifold.jl")
include("power_manifold.jl")
include("vector_bundle.jl")

include("metric.jl")
include("statistics.jl")

# Lie groups and actions
include("groups/groups_general.jl")
include("groups/rotation_action.jl")
include("groups/translation_action.jl")
