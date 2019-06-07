module ManifoldMuseum

import Base: isapprox, exp, log
import LinearAlgebra: dot, norm

export Manifold
export dimension,
    distance,
    dot,
    exp!,
    geodesic,
    log!,
    norm,
    retract,
    retract!,
    typical_distance,
    zero_tangent_vector,
    zero_tangent_vector!

abstract type Manifold end

"""
    isapprox(M::Manifold, x, y; kwargs...)

Check if points `x` and `y` from manifold `M` are approximately equal.

Keyword arguments can be used to specify tolerances.
"""
isapprox(M::Manifold, x, y; kwargs...) = isapprox(x, y; kwargs...)

"""
    isapprox(M::Manifold, x, v, w; kwargs...)

Check if vectors `v` and `w` tangent at `x` from manifold `M` are
approximately equal.

Keyword arguments can be used to specify tolerances.
"""
isapprox(M::Manifold, x, v, w; kwargs...) = isapprox(v, w; kwargs...)

"""
    retract!(M::Manifold, y, x, v, t=1)

Retraction (cheaper, approximate version of exponential map) of tangent
vector `t*v` at point `x` from manifold `M`.
Result is saved to `y`.
"""
retract!(M::Manifold, y, x, v) = exp!(M, y, x, v)

retract!(M::Manifold, y, x, v, t) = retract!(M, y, x, t*v)

"""
    retract(M::Manifold, x, v, t=1)

Retraction (cheaper, approximate version of exponential map) of tangent
vector `t*v` at point `x` from manifold `M`.
"""
retract(M::Manifold, x, v) = retract!(M, similar(x), x, v)

retract(M::Manifold, x, v, t) = retract(M, x, t*v)

project_tangent!(M::Manifold, w, x, v) = error("Not implemented")
project_tangent(M::Manifold, x, v) = project_tangent!(M, copy(x), x, v)

distance(M::Manifold, x, y) = norm(M, x, log(M, x, y))

"""
    dot(M::Manifold, x, v, w)

Inner product of tangent vectors `v` and `w` at point `x` from manifold `M`.
"""
dot(M::Manifold, x, v, w) = error("Not implemented")

"""
    norm(M::Manifold, x, v)

Norm of tangent vector `v` at point `x` from manifold `M`.
"""
norm(M::Manifold, x, v) = sqrt(dot(M, x, v, v))

"""
    exp!(M::Manifold, y, x, v, t=1)

Exponential map of tangent vector `t*v` at point `x` from manifold `M`.
Result is saved to `y`.
"""
exp!(M::Manifold, y, x, v, t) = exp!(M::Manifold, y, x, t*v)

exp!(M::Manifold, y, x, v) = error("Not implemented")

"""
    exp(M::Manifold, x, v, t=1)

Exponential map of tangent vector `t*v` at point `x` from manifold `M`.
"""
exp(M::Manifold, x, v) = exp!(M, similar(x), x, v)

exp(M::Manifold, x, v, t) = exp(M, x, t*v)

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


end # module
