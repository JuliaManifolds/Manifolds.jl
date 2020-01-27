"""
    AbstractMap{D,CoD}

Abstract type for maps between elements of sets.
Every map has a domain of type `D`, which is the set to which inputs
belong and a codomain `CoD`, which is the set to which outputs belong. Note
that maps are not required to be total. That is, the true (co)domain may be
a subset of the provided (co)domain.

Every new map type must implement [`domain`](@ref) and [`codomain`](@ref) and be
callable.
"""
abstract type AbstractMap{D,CoD} end

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
    AbstractCurve{M} = AbstractMap{ℝ,M}

An alias for a curve, a map from 1-D real space to a manifold, i.e.
$\phi: ℝ \to M$
"""
const AbstractCurve{M} = AbstractMap{ℝ,M}

domain(::AbstractCurve) = ℝ

"""
    AbstractRealField{M} = AbstractMap{M,ℝ}

An alias for a generic field, a map from a point on a manifold `M` to a real number.
"""
const AbstractRealField{M} = AbstractMap{M,ℝ}

codomain(::AbstractRealField) = ℝ

"""
    FunctionMap{F,D,CoD} <: AbstractMap{D,CoD}

A map that wraps a generic callable, annotating it with a [`domain`](@ref) and
[`codomain`](@ref).

# Constructor

    FunctionMap(f, domain, codomain)
"""
struct FunctionMap{F,D,CoD} <: AbstractMap{D,CoD}
    f::F
    domain::D
    codomain::CoD
end

(f::FunctionMap)(args...; kwargs...) = f.f(args...; kwargs...)

"""
    FunctionCurve{F,M} <: AbstractCurve{M}

A curve on `manifold` that wraps a generic callable

# Constructor

    FunctionCurve(f, manifold)
"""
struct FunctionCurve{F,M} <: AbstractCurve{M}
    f::F
    codomain::M
end

(f::FunctionCurve)(args...; kwargs...) = f.f(args...; kwargs...)

"""
    FunctionRealField{M} = AbstractRealField{M}

A real-valued field defined by a generic callable.

# Constructor

    FunctionRealField(f, manifold)
"""
struct FunctionRealField{F,M} <: AbstractRealField{M}
    f::F
    domain::M
end

(f::FunctionRealField)(args...; kwargs...) = f.f(args...; kwargs...)
