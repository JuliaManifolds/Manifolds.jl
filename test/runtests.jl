include("autodiff.jl")
include("utils.jl")
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
