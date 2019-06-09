module ManifoldMuseum

import Base: isapprox, exp, log, convert
import LinearAlgebra: dot, norm, I


abstract type Manifold end

isapprox(m::Manifold, x, y; kwargs...) = isapprox(x, y; kwargs...)

retract!(M::Manifold, y, x, v) = exp!(M, y, x, v)
retract(M::Manifold, x, v) = retract!(M, copy(x), x, v)

project_tangent!(M::Manifold, w, x, v) = error("Not implemented")
project_tangent(M::Manifold, x, v) = project_tangent!(M, copy(x), x, v)

dimension(M::Manifold) = error("Not implemented")

vector_transport!(M::Manifold, vto, x, v, y) = project_tangent!(M, vto, x, v)
vector_transport(M::Manifold, x, v, y) = vector_transport!(M, copy(v), x, y, v)

random_point(M::Manifold) = error("Not implemented")
random_tangent_vector(M::Manifold, x) = error("Not implemented")

zero_tangent_vector(M::Manifold, x) = log(M, x, x)
zero_tangent_vector!(M::Manifold, v, x) = log!(M, v, x, x)

include("Metric.jl")
include("Euclidean.jl")
include("Sphere.jl")

export Manifold
export dimension,
    distance,
    dot,
    exp,
    exp!,
    geodesic,
    isapprox,
    log,
    log!,
    norm,
    angle,
    injectivity_radius,
    zero_tangent_vector,
    zero_tangent_vector!
export Metric, RiemannianMetric, LorentzianMetric
export manifold,
    local_matrix,
    inverse_local_matrix
export Euclidean, EuclideanMetric, TransformedEuclideanMetric

end # module
