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
