module ManifoldMuseum

import Base: isapprox, exp, log
import LinearAlgebra: dot, norm


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
abstract type TVector end


isapprox(m::Manifold, x, y; kwargs...) = isapprox(x, y; kwargs...)

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

include("MatrixManifold.jl")
include("Sphere.jl")

export Manifold,
    MPoint,
    TVector 
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
