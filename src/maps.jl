
"""
    AbstractMap{D,CoD}

An abstract type that represents maps between elements of sets.
Every map has a domain of type `D`, which is the set to which inputs
belong and a codomain of type `CoD`, which is the set to which outputs belong. Note
that maps are not required to be total. That is, the true (co)domain may be
a subset of the provided (co)domain.

Every new map type must implement [`domain`](@ref) and [`codomain`](@ref) and be
callable.
"""
abstract type AbstractMap{D,CoD} end

"""
    Map(domain, codomain)

A generic [`AbstractMap`](@ref) between `domain` and `codomain`.
"""
struct Map{D,CoD} <: AbstractMap{D,CoD}
    domain::D
    codomain::CoD
end

"""
    domain(m::AbstractMap)

Get the manifold to which inputs to the map `m` belong. By default, this is
assumed to be stored in the field `m.domain`.
"""
domain(m::AbstractMap) = m.domain

"""
    codomain(m::AbstractMap)

Get the manifold to which outputs to the map `m` belong. By default, this is
assumed to be stored in the field `m.codomain`.
"""
codomain(m::AbstractMap) = m.codomain

@doc doc"""
    Curve(M::Manifold)

A type for a curve, a map from 1-D real space to a manifold, i.e.
$\phi: ℝ \to M$
"""
struct Curve{TM<:Manifold} <: AbstractMap{ℝ,TM}
    codomain::TM
end

domain(::Curve) = ℝ

"""
    RealField(M::Manifold)

A type for a generic field, a map from a point on a manifold `M` to a real number.
"""
struct RealField{TM<:Manifold} <: AbstractMap{TM,ℝ}
    domain::TM
end

codomain(::RealField) = ℝ
