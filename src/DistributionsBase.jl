
using Distributions

"""
    MPointvariate

Structure that subtypes `VariateForm`, indicating that a single sample
is a point on a manifold.
"""
struct MPointvariate <: VariateForm
end

"""
    FVectorvariate

Structure that subtypes `VariateForm`, indicating that a single sample
is a vector from a fiber of a vector bundle.
"""
struct FVectorvariate <: VariateForm
end

"""
    MPointSupport(manifold::Manifold)

Value support for manifold-valued distributions (values from given
manifold `M`).
"""
struct MPointSupport{TM<:Manifold} <: ValueSupport
    manifold::TM
end

"""
    MPointDistribution{TM<:Manifold}

An abstract distribution for points on manifold of type `TM`.
"""
abstract type MPointDistribution{TM<:Manifold} <: Distribution{MPointvariate, MPointSupport{TM}}
end

"""
    FVectorSupport(space::Manifold, VectorBundleFibers)

Value support for tangent vector-valued distributions (values from tangent
space at point `x` from given manifold).
"""
struct FVectorSupport{TSpace<:VectorBundleFibers, T} <: ValueSupport
    space::TSpace
    x::T
end

"""
    FVectorDistribution{TSpace<:VectorBundleFibers, T}

An abstract distribution for tangent vectors at point of type `T`
from vector space of type `TSpace`.
"""
abstract type FVectorDistribution{TSpace<:VectorBundleFibers, T} <: Distribution{FVectorvariate, FVectorSupport{TSpace, T}}
end


"""
    support(d::FVectorDistribution)

Get the object of type `FVectorSupport` for the distribution `d`.
"""
function Distributions.support(::T) where T<:FVectorDistribution
    error("support not implemented for type $T")
end
