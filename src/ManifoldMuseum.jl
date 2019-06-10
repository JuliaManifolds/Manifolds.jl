module ManifoldMuseum

import Base: isapprox, exp, log
import LinearAlgebra: dot, norm
import Markdown: @doc_str

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
    injectivity_radius,
    zero_tangent_vector,
    zero_tangent_vector!

"""
    Manifold

A manifold type. The `Manifold` is used to dispatch to different exponential
and logarithmic maps as well as other function on manifold.
"""
abstract type Manifold end

"""
    MPoint

Type for a point on a manifold. While a [`Manifold`](@ref) not necessarily
requires this type, for example when it is implemented for `Vector`s or
`Matrix` type elements, this type can be used for more complicated
representations, semantic verification or even dispatch for different
representations of points on a manifold.
"""
abstract type MPoint end

"""
    TVector

Type for a tangent vector of a manifold. While a [`Manifold`](@ref) not
necessarily requires this type, for example when it is implemented for `Vector`s
or `Matrix` type elements, this type can be used for more complicated
representations, semantic verification or even dispatch for different
representations of tangent vectors and their types on a manifold.
"""
abstract type TVector end

"""
    CoTVector

Type for a cotangent vector of a manifold. While a [`Manifold`](@ref) not
necessarily requires this type, for example when it is implemented for `Vector`s
or `Matrix` type elements, this type can be used for more complicated
representations, semantic verification or even dispatch for different
representations of cotangent vectors and their types on a manifold.
"""
abstract type CoTVector end

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

manifold_dimension(M::Manifold) = error("Not implemented")

vector_transport!(M::Manifold, vto, x, v, y) = project_tangent!(M, vto, x, v)
vector_transport(M::Manifold, x, v, y) = vector_transport!(M, copy(v), x, y, v)

random_point(M::Manifold) = error("Not implemented")
random_tangent_vector(M::Manifold, x) = error("Not implemented")

"""
    injectivity_radius(M::Manifold, x)

Distance such that `log(M, x, y)` is defined for all points within this radius.
"""
injectivity_radius(M::Manifold, x) = 1.0

zero_tangent_vector(M::Manifold, x) = log(M, x, x)
zero_tangent_vector!(M::Manifold, v, x) = log!(M, v, x, x)

geodesic(M::Manifold, x, y, t) = exp(M, x, log(M, x, y), t)

"""
    is_manifold_point(M,x)

check, whether `x` is a valid point on the [`Manifold`](@ref) `M`. If it is not,
an error is thrown.
The default is to return `true`, i.e. if no checks are implmented,
the assumption is to be optimistic.
"""
is_manifold_point(M::Manifold,x; kwargs...) = true

"""
    is_tangent_vector(M,x,v)

check, whether `v` is a valid tangnt vector in the tangent plane of `x` on the
[`Manifold`](@ref) `M`. An implementation should first check
[`is_manifold_point`](@ref)`(M,x)` and then validate `v`. If it is not a tangent
vector an error should be thrown.
The default is to return `true`, i.e. if no checks are implmented,
the assumption is to be optimistic.
"""
is_tangent_vector(M::Manifold,x,v; kwargs...) = true

include("ArrayManifold.jl")
include("Sphere.jl")

export Manifold,
    MPoint,
    TVector,
    CoTVector
export manifold_dimension,
    distance,
    dot,
    exp,
    exp!,
    geodesic,
    isapprox,
    is_manifold_point,
    is_tangent_vector,
    log,
    log!,
    norm,
    injectivity_radius,
    zero_tangent_vector,
    zero_tangent_vector!

end # module