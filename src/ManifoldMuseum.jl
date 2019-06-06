module ManifoldMuseum

import Base: isapprox, exp, log
import LinearAlgebra: dot, norm


abstract type Manifold end

isapprox(m::Manifold, x, y; kwargs...) = error("Not implemented")

retract!(M::Manifold, y, x, v) = exp!(M, y, x, v)
retract(M::Manifold, x, v) = retract!(M, copy(x), x, v)

project_tangent!(M::Manifold, w, x, v) = error("Not implemented")
project_tangent(M::Manifold, x, v) = project_tangent!(M, copy(x), x, v)

distance(M::Manifold, x, y) = error("Not implemented")

dot(M::Manifold, x, v, w) = dot(v, w)
norm(M::Manifold, x, v) = sqrt(dot(M, x, v, v))

exp!(M::Manifold, y, x, v) = error("Not implemented")
exp(M::Manifold, x, v) = exp!(M,copy(x), x, v)

log!(M::Manifold, v, x, y) = error("Not implemented")

function log(M::Manifold, x, y)
    v = zero_tangent_vector(M, x)
    log!(M, v, x, y)
    return v
end

dimension(M::Manifold) = error("Not implemented")

vector_transport!(M::Manifold, vto, x, v, y) = project_tangent!(M, vto, x, v)
vector_transport(M::Manifold, x, v, y) = vector_transport!(M, copy(v), x, y, v)

random_point(M::Manifold) = error("Not implemented")
random_tangent_vector(M::Manifold, x) = error("Not implemented")

typical_distance(M::Manifold) = 1.0
zero_tangent_vector(M::Manifold, x) = log(M, x, x)
zero_tangent_vector!(M::Manifold, v, x) = log!(M, v, x, x)

geodesic(M::Manifold, x, y, t) = exp(M, x, log(M, x, y), t)

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
    typical_distance,
    zero_tangent_vector,
    zero_tangent_vector!


end # module
