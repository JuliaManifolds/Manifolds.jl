"""
    AbstractMap{D,CoD}

Abstract type for maps between elements of sets that are [`Manifold`](@ref)s.
Every map has a domain of type `D`, which is the manifold to which inputs
belong and a codomain `CoD`, which is the manifold to which outputs belong.
Every new map type must implement [`domain`](@ref) and [`codomain`](@ref) and
be callable.

Maps can be composed in the same way as functions using the ∘ command. See
[`CompositeMap`](@ref).
"""
abstract type AbstractMap{D,CoD} end

function show(io::IO, mime::MIME"text/plain", m::AbstractMap)
    print(io, "$(typeof(m).name): 𝑀 ⟶ 𝑁\n",
              "𝑀 = $(repr(mime, domain(m)))\n",
              "𝑁 = $(repr(mime, codomain(m)))")
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
    AbstractCurve{M} = AbstractMap{Euclidean{Tuple{1}},M}

An alias for a curve, a map from 1-D real space to a manifold, i.e.
$\phi: \mathbb R \to M$
"""
AbstractCurve{M} = AbstractMap{Euclidean{Tuple{1}},M}

@doc doc"""
    AbstractField{M,T<:Tuple} = AbstractMap{M,Euclidean{T}}

An alias for a generic field, a map from a point on a manifold `M` to real
space of some dimension stored in `T`,
i.e. $\phi: M \to \mathbb R^{m \times \dots \times n}$
"""
AbstractField{M,T<:Tuple} = AbstractMap{M,Euclidean{T}}

@doc doc"""
    AbstractScalarField{M} = AbstractField{M,Tuple{1}}

An alias for a scalar field, a map from a point on a manifold `M` to 1-D real
space, i.e. $\phi: M \to \mathbb R$
"""
AbstractScalarField{M} = AbstractField{M,Tuple{1}}

@doc doc"""
    AbstractVectorField{M,N} = AbstractField{M,Tuple{N}}

An alias for a vector field, a map from a point on a manifold `M` to
$n$-dimensional vectors, i.e. $\phi: M \to \mathbb R^n$
"""
AbstractVectorField{M,N} = AbstractField{M,Tuple{N}}

@doc doc"""
    AbstractMatrixField{M,P,Q} = AbstractField{M,Tuple{P,Q}}

An alias for a field that maps from a point on a manifold `M` to a vector space
represented as $p \times q$-dimensional matrices,
i.e. $\phi: M \to \mathbb R^{p \times q}$
"""
AbstractMatrixField{M,P,Q} = AbstractField{M,Tuple{P,Q}}

@doc doc"""
    CompositeMap{D,CoD,F,G} <: AbstractMap{D,CoD}

A map that is a composition of two maps. Given two maps $f: B \to C$ and
$g: A \to B$, this implements the map

```math
\begin{aligned}
f \circ g =\ &h: A \to C\\
             &h: x \mapsto f(g(x))
\end{aligned}
```

# Constructor

    CompositeMap(f::F,
                 g::G) where {D,I,CoD,F<:AbstractMap{I,CoD},G<:AbstractMap{D,I}}
"""
struct CompositeMap{D,CoD,F,G} <: AbstractMap{D,CoD}
    f::F
    g::G
end

function CompositeMap(f::F, g::G) where {D,
                                         I,
                                         CoD,
                                         F<:AbstractMap{I,CoD},
                                         G<:AbstractMap{D,I}}
    return CompositeMap{D,CoD,F,G}(f, g)
end

∘(f::AbstractMap, g::AbstractMap) = CompositeMap(f, g)

(m::CompositeMap)(x...) = m.f(m.g(x...))

domain(m::CompositeMap) = domain(m.g)

codomain(m::CompositeMap) = codomain(m.f)

function show(io::IO, mime::MIME"text/plain", m::CompositeMap)
    print(io, "𝑓∘𝑔: 𝑀 ⟶ 𝑁 ⟶ 𝑃\n",
              "𝑀 = $(repr(mime, domain(m)))\n",
              "𝑁 = $(repr(mime, codomain(m.g)))\n",
              "𝑃 = $(repr(mime, codomain(m)))\n",
              "𝑔 = $(m.g)\n",
              "𝑓 = $(m.f)")
end

"""
    FunctionMap{D,CoD,F} <: AbstractMap{D,CoD}

A map that wraps a generic callable, annotating it with a domain and codomain.

# Constructor

    FunctionMap(domain, codomain, f)
"""
struct FunctionMap{D,CoD,F} <: AbstractMap{D,CoD}
    domain::D
    codomain::CoD
    f::F
end

FunctionMap(domain::D, codomain::CoD, m::AbstractMap{D,CoD}) where {D,CoD} = m

(m::FunctionMap)(x...) = m.f(x...)

"""
    PseudoInvertibleFunctionMap{D,CoD,F,PIF} <: AbstractMap{D,CoD}

An abstract type for a map that wraps a callable `f` and its pseudo-inverse
`g`, which is either a left-, right-, or true inverse of `f`. `f` is also a
pseudo-inverse of `g`.
"""
abstract type PseudoInvertibleFunctionMap{D,CoD,F,PIF} <: AbstractMap{D,CoD} end

"""
    Bijection{D,CoD,F,IF} <: PseudoInvertibleFunctionMap{D,CoD,F,IF}

A one-to-one and onto (injective and surjective) map with a known true inverse.
The inverse map is also a bijection and can be retrieved with `inv`.

# Constructor

    Bijection(domain, codomain, f, invf)
"""
struct Bijection{D,CoD,F,IF} <: PseudoInvertibleFunctionMap{D,CoD,F,IF}
    domain::D
    codomain::CoD
    f::F
    invf::IF
end

function Bijection(f::F, invf::IF) where {D,CoD,F<:AbstractMap{D,CoD},IF}
    return Bijection(domain(f), codomain(f), f, invf)
end

function ∘(f::Bijection, g::Bijection)
    return Bijection(domain(g), codomain(f), f.f ∘ g.f, g.invf ∘ f.invf)
end

(m::Bijection)(x...) = m.f(x...)

inv(m::Bijection) = Bijection(codomain(m), domain(m), m.invf, m.f)

pinv(m::Bijection) = inv(m)

@doc doc"""
    Injection{D,CoD,F,IF} <: PseudoInvertibleFunctionMap{D,CoD,F,IF}

An injective (one-to-one) map with a known left inverse such that an injection
$f$ composed with its left inverse $g$ produces the identity map:
$g ∘ f: x \mapsto x$.

# Constructor

    Injection(domain, codomain, f, linvf)
"""
struct Injection{D,CoD,F,LIF} <: PseudoInvertibleFunctionMap{D,CoD,F,LIF}
    domain::D
    codomain::CoD
    f::F
    linvf::LIF
end

function Injection(f::F, linvf::LIF) where {D,CoD,F<:AbstractMap{D,CoD},LIF}
    return Injection(domain(f), codomain(f), f, linvf)
end

function ∘(f::Injection, g::Injection)
    return Injection(domain(g), codomain(f), f.f ∘ g.f, g.linvf ∘ f.linvf)
end

(m::Injection)(x...) = m.f(x...)

@doc doc"""
    Surjection{D,CoD,F,RIF} <: PseudoInvertibleFunctionMap{D,CoD,F,RIF}

A surjective map (onto) with a known right inverse such that the right inverse
$g$ of a surjection $f$ composed with $f$ produces the identity map:
$f ∘ g: x \mapsto x$.

The pseudo-inverse of a surjection is an injection and can be retrieved with
`pinv`.

# Constructor

    Surjection(domain, codomain, f, rinvf)
"""
struct Surjection{D,CoD,F,RIF} <: PseudoInvertibleFunctionMap{D,CoD,F,RIF}
    domain::D
    codomain::CoD
    f::F
    rinvf::RIF
end

function Surjection(f::F, rinvf::RIF) where {D,CoD,F<:AbstractMap{D,CoD},RIF}
    return Surjection(domain(f), codomain(f), f, rinvf)
end

function ∘(f::Surjection, g::Surjection)
    return Surjection(domain(g), codomain(f), f.f ∘ g.f, g.rinvf ∘ f.rinvf)
end

(m::Surjection)(x...) = m.f(x...)

pinv(m::Surjection) = Injection(codomain(m), domain(m), m.rinvf, m.f)

# Composition rules for combinations of injections, surjections, and bijections

function ∘(f::Bijection, g::Injection)
    return Injection(domain(g), codomain(f), f.f ∘ g.f, g.linvf ∘ f.invf)
end

function ∘(f::Injection, g::Bijection)
    return Injection(domain(g), codomain(f), f.f ∘ g.f, g.invf ∘ f.linvf)
end

function ∘(f::Surjection, g::Bijection)
    return Surjection(domain(g), codomain(f), f.f ∘ g.f, g.invf ∘ f.rinvf)
end

function ∘(f::Bijection, g::Surjection)
    return Surjection(domain(g), codomain(f), f.f ∘ g.f, g.rinvf ∘ f.invf)
end

@doc doc"""
    Exponential{TMT,MT,PT} <: AbstractMap{TMT,MT}

Riemannian exponential map from the tangent space $T_x M$ at a point $x \in M$
to $M$:

$\mathrm{Exp}_x: T_x M \to M$

# Constructor

    Exponential(M::Manifold, x)
"""
struct Exponential{TMT,MT,PT} <: AbstractMap{TMT,MT}
    domain::TMT
    codomain::MT
    point::PT
end

const Exp = Exponential

function Exponential(M::MT, x::PT) where {MT,PT}
    shape = representation_size(M, MPoint)
    TM = Euclidean(shape...)
    return Exponential(TM, M, x)
end

(m::Exponential)(v) = exp(m.codomain, m.point, v)

function show(io::IO, mime::MIME"text/plain", m::Exponential)
    print(io, "Expₓ: 𝑇ₓ𝑀 ⟶ 𝑀,  𝑥 ∈ 𝑀\n",
              "𝑀 = $(repr(mime, codomain(m)))\n",
              "𝑥 = $(repr(mime, m.point))")
end

@doc doc"""
    Logarithm{MT,TMT,PT} <: AbstractMap{MT,TMT}

Riemannian logarithm map, which is the right inverse of the Riemannian
exponential, defined as

```math
\begin{aligned}
\mathrm{Log}_x:& M \to T_x M\\
\mathrm{Exp}_x \circ \mathrm{Log}_x:& y \mapsto y
\end{aligned}
```

# Constructor

    Logarithm(M::Manifold, x)
"""
struct Logarithm{MT,TMT,PT} <: AbstractMap{MT,TMT}
    domain::MT
    codomain::TMT
    point::PT
end

const Log = Logarithm

function Logarithm(M::MT, x::PT) where {MT,PT}
    shape = representation_size(M, TVector)
    TM = Euclidean(shape...)
    return Logarithm(M, TM, x)
end

(m::Logarithm)(x) = log(m.domain, m.point, x)

function show(io::IO, mime::MIME"text/plain", m::Logarithm)
    print(io, "Logₓ: 𝑀 ⟶ 𝑇ₓ𝑀,  𝑥 ∈ 𝑀\n",
              "𝑀 = $(repr(mime, domain(m)))\n",
              "𝑥 = $(repr(mime, m.point))")
end

@doc doc"""
    Geodesic{MT,ET<:Exp,PT,VT} <: AbstractCurve{MT}

Geodesic curve $\gamma(t)$ on manifold $M$:

```math
\begin{aligned}
&\gamma: \mathbb R \to M\\
&\gamma(0) = x \in M\\
&\gamma\dot(0) = v \in T_x M
\end{aligned}
```

# Constructor

    Geodesic(M::Manifold, x, v)
"""
struct Geodesic{MT,ET<:Exp,PT,VT} <: AbstractCurve{MT}
    domain::MT
    Exp::ET
    x₀::PT
    v₀::VT
end

Geodesic(M, x, v) = Geodesic(M, Exp(M, x), x, v)

(g::Geodesic)(t::Real) = g.Exp(t*g.v₀)
(g::Geodesic)(T::AbstractVector) = map(t -> g.Exp(t*g.v₀), T)

codomain(g::Geodesic) = codomain(g.Exp)

function show(io::IO, mime::MIME"text/plain", m::Geodesic)
    print(io, "𝛾: ℝ ⟶ 𝑀\n",
              "𝑀 = $(repr(mime, codomain(m)))\n",
              "𝛾(0) = $(repr(mime, m.x₀))\n",
              "𝛾̇(0) = $(repr(mime, m.v₀))")
end

@doc doc"""
    ShortestGeodesic{MT,GT<:Geodesic,IT,FT} <: AbstractCurve{MT}

Geodesic curve $\gamma(t)$ on manifold $M$:

```math
\begin{aligned}
&\gamma: \mathbb R \to M\\
&\gamma(0) = x \in M\\
&\gamma(1) = y \in M,
\end{aligned}
```

such that the length of the curve between $x$ and $y$ on the interval
$t=[0,1]$ is the shortest distance on the manifold between those two points.

# Constructor

    ShortestGeodesic(M::Manifold, x, y)
"""
struct ShortestGeodesic{MT,GT<:Geodesic,IT,FT} <: AbstractCurve{MT}
    domain::MT
    geodesic::GT
    x₀::IT
    x₁::FT
end

function ShortestGeodesic(M, x, y)
    v = Log(M, x)(y)
    return ShortestGeodesic(M, Geodesic(M, x, v), x, y)
end

(g::ShortestGeodesic)(t) = g.geodesic(t)

codomain(g::ShortestGeodesic) = codomain(g.geodesic)

function show(io::IO, mime::MIME"text/plain", m::ShortestGeodesic)
    print(io, "𝛾: ℝ ⟶ 𝑀\n",
              "𝑀 = $(repr(mime, codomain(m)))\n",
              "𝛾(0) = $(repr(mime, m.x₀))\n",
              "𝛾(1) = $(repr(mime, m.x₁))")
end
