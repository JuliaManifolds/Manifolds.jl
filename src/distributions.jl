"""
    FVectorvariate

Structure that subtypes `VariateForm`, indicating that a single sample
is a vector from a fiber of a vector bundle.
"""
struct FVectorvariate <: VariateForm end

"""
    FVectorSupport(space::AbstractManifold, VectorBundleFibers)

Value support for vector bundle fiber-valued distributions (values from a fiber of a vector
bundle at a `point` from the given manifold).
For example used for tangent vector-valued distributions.
"""
struct FVectorSupport{TSpace<:VectorBundleFibers,T} <: ValueSupport
    space::TSpace
    point::T
end

"""
    FVectorDistribution{TSpace<:VectorBundleFibers, T}

An abstract distribution for vector bundle fiber-valued distributions (values from a fiber
of a vector bundle at point `x` from the given manifold).
For example used for tangent vector-valued distributions.
"""
abstract type FVectorDistribution{TSpace<:VectorBundleFibers,T} <:
              Distribution{FVectorvariate,FVectorSupport{TSpace,T}} end

"""
    MPointvariate

Structure that subtypes `VariateForm`, indicating that a single sample
is a point on a manifold.
"""
struct MPointvariate <: VariateForm end

"""
    MPointSupport(M::AbstractManifold)

Value support for manifold-valued distributions (values from given
[`AbstractManifold`](@ref) `M`).
"""
struct MPointSupport{TM<:AbstractManifold} <: ValueSupport
    manifold::TM
end

"""
    MPointDistribution{TM<:AbstractManifold}

An abstract distribution for points on manifold of type `TM`.
"""
abstract type MPointDistribution{TM<:AbstractManifold} <:
              Distribution{MPointvariate,MPointSupport{TM}} end

"""
    support(d::FVectorDistribution)

Get the object of type `FVectorSupport` for the distribution `d`.
"""
function Distributions.support(::T) where {T<:FVectorDistribution}
    return error("support not implemented for type $T")
end

function uniform_distribution(M::AbstractManifold)
    return uniform_distribution(M, allocate_result(M, uniform_distribution))
end

"""
    Random.rand(M::AbstractManifold, [d::Integer]; vector_at=nothing)
    Random.rand(rng::AbstractRNG, M::AbstractManifold, [d::Integer]; vector_at=nothing)


Generate a random point on manifold `M` (when `vector_at` is `nothing`) or a tangent
vector at point `vector_at` (when it is not `nothing`).

Optionally a random number generator `rng` to be used can be specified. An optional integer
`d` indicates that a vector of `d` points or tangent vectors is to be generated.

!!! note

    Usually a uniform distribution should be expected for compact manifolds and a
    Gaussian-like distribution for non-compact manifolds and tangent vectors, although it is
    not guaranteed. The distribution may change between releases. 

    `rand` methods for specific manifolds may take additional keyword arguments.

"""
Random.rand(M::AbstractManifold)
function Random.rand(M::AbstractManifold, d::Integer; kwargs...)
    return [rand(M; kwargs...) for _ in 1:d]
end
function Random.rand(rng::AbstractRNG, M::AbstractManifold, d::Integer; kwargs...)
    return [rand(rng, M; kwargs...) for _ in 1:d]
end
function Random.rand(M::AbstractManifold; kwargs...)
    p = allocate_result(M, rand)
    rand!(M, p; kwargs...)
    return p
end
function Random.rand(rng::AbstractRNG, M::AbstractManifold; kwargs...)
    p = allocate_result(M, rand)
    rand!(rng, M, p; kwargs...)
    return p
end
