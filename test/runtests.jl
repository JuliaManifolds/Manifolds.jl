
include("utils.jl")

include("sized_abstract_array.jl")
include("hybrid_abstract_array.jl")

# starting with tests of simple manifolds
include("euclidean.jl")
include("sphere.jl")
include("rotations.jl")
include("product_manifold.jl")
include("vector_bundle.jl")

include("array_manifold.jl")
include("metric_test.jl")
