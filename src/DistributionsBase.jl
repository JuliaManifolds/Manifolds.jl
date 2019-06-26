
using Distributions

struct MPointvariate <: VariateForm
end

struct TVectorvariate <: VariateForm
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
    TVectorSupport(manifold::Manifold, x)

Value support for tangent vector-valued distributions (values from tangent
space at point `x` from given manifold).
"""
struct TVectorSupport{TM<:Manifold, T} <: ValueSupport
    manifold::TM
    x::T
end

"""
    TVectorDistribution{TM<:Manifold, T}

An abstract distribution for tangent vectors at point of type `T`
from manifold of type `TM`.
"""
abstract type TVectorDistribution{TM<:Manifold, T} <: Distribution{TVectorvariate, TVectorSupport{TM, T}}
end


"""
    support(d::TVectorDistribution)

Get the object of type `TVectorSupport` for the distribution `d`.
"""
function Distributions.support(::T) where T<:TVectorDistribution
    error("support not implemented for type $T")
end
