"""
    AbstractMap{D,CoD}

Abstract type for maps between elements of sets.
Every map has a domain of type `D`, which is the set to which inputs
belong and a codomain `CoD`, which is the set to which outputs belong. Note
that maps are not required to be total. That is, the true (co)domain may be
a subset of the provided (co)domain.

Every new map type must implement [`domain`](@ref) and [`codomain`](@ref) and
be callable. Optionally, `inv` and [`pinv`](@ref) may be used to specify
a map that is a true inverse or pseudo-inverse, respectively.

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
domain(::AbstractMap{M}) where {M<:Euclidean} = M()

"""
    codomain(m::AbstractMap)

Get the manifold to which outputs to the map `m` belong. By default, this is
assumed to be stored in the field `m.codomain`.
"""
codomain(m::AbstractMap) = m.codomain
codomain(::AbstractMap{M,N}) where {M,N<:Euclidean} = N()

"""
    pinv(m::AbstractMap)

Get the pseudo-inverse of the map `m`. The pseudo-inverse is generally a left-
or right-inverse of the original map.
"""
pinv(m::AbstractMap) = inv(m)

@doc doc"""
    AbstractCurve{M} = AbstractMap{Euclidean{Tuple{1}},M}

An alias for a curve, a map from 1-D real space to a manifold, i.e.
$\phi \colon \mathbb R \to M$
"""
AbstractCurve{M} = AbstractMap{RealScalars,M}

@doc doc"""
    AbstractField{M,T<:Tuple} = AbstractMap{M,Euclidean{T}}

An alias for a generic field, a map from a point on a manifold `M` to real
space of some dimension stored in `T`,
i.e. $\phi \colon M \to \mathbb R^{m \times \dots \times n}$
"""
AbstractField{M,T<:Tuple} = AbstractMap{M,Euclidean{T}}

@doc doc"""
    AbstractScalarField{M} = AbstractField{M,Tuple{1}}

An alias for a scalar field, a map from a point on a manifold `M` to 1-D real
space, i.e. $\phi \colon M \to \mathbb R$
"""
AbstractScalarField{M} = AbstractField{M,Tuple{1}}

@doc doc"""
    AbstractVectorField{M,N} = AbstractField{M,Tuple{N}}

An alias for a vector field, a map from a point on a manifold `M` to
$n$-dimensional vectors, i.e. $\phi \colon M \to \mathbb R^n$
"""
AbstractVectorField{M,N} = AbstractField{M,Tuple{N}}

@doc doc"""
    AbstractMatrixField{M,P,Q} = AbstractField{M,Tuple{P,Q}}

An alias for a field that maps from a point on a manifold `M` to a vector space
represented as $p \times q$-dimensional matrices,
i.e. $\phi \colon M \to \mathbb R^{p \times q}$
"""
AbstractMatrixField{M,P,Q} = AbstractField{M,Tuple{P,Q}}

@doc doc"""
    CompositeMap{D,CoD,F,G} <: AbstractMap{D,CoD}

A map that is a composition of two maps. Given two maps $f \colon B \to C$ and
$g \colon A \to B$, this implements the map

```math
\begin{aligned}
h = f \circ g & \colon A \to C\\
              & \colon x \mapsto f(g(x))
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

inv(m::CompositeMap) = inv(m.g) ∘ inv(m.f)

pinv(m::CompositeMap) = pinv(m.g) ∘ pinv(m.f)

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

A map that wraps a generic callable, annotating it with a [`domain`](@ref) and
[`codomain`](@ref).

# Constructor

    FunctionMap(domain, codomain, f)
"""
struct FunctionMap{D,CoD,F} <: AbstractMap{D,CoD}
    domain::D
    codomain::CoD
    f::F
end

FunctionMap(domain::D, codomain::CoD, m::AbstractMap{D,CoD}) where {D,CoD} = m
FunctionMap(domain::D, ::D, ::typeof(identity)) where {D} = Identity(domain)

(m::FunctionMap)(x...) = m.f(x...)

@doc doc"""
    ProductMap{D,CoD,TP<:Tuple} <: AbstractMap{D,CoD}

Given maps $f_i$ with respective domains $M_i$ and codomains $N_i$, construct
the product map $g$, such that

```math
\begin{aligned}
g &\colon \times_i M_i \to \times_i N_i\\
  &\colon (x_1, x_2, \dots) \mapsto (f_1(x_1), f_2(x_2), \dots)
\end{aligned}
```

Such a map may also be created using `cross` or `×`.

# Constructor

    ProductMap(maps::AbstractMap...)
"""
struct ProductMap{D,CoD,TP<:Tuple} <: AbstractMap{D,CoD}
    domain::D
    codomain::CoD
    maps::TP
end

function ProductMap(maps::AbstractMap...)
    d = mapreduce(domain, cross, maps)
    cod = mapreduce(codomain, cross, maps)
    ProductMap(d, cod, maps)
end

function cross(M1::AbstractMap, M2::AbstractMap)
    return ProductMap(M1, M2)
end

function cross(M1::ProductMap, M2::AbstractMap)
    return ProductMap(M1.maps..., M2)
end

function cross(M1::AbstractMap, M2::ProductMap)
    return ProductMap(M1, M2.maps...)
end

function cross(M1::ProductMap, M2::ProductMap)
    return ProductMap(M1.maps..., M2.maps...)
end

(m::ProductMap)(xs...) = map((f, x) -> f(x), m.maps, xs)
(m::ProductMap)(x::Tuple) = m(x...)
(m::ProductMap)(x) = m(x.parts...)

pinv(m::ProductMap) = ProductMap(pinv.(m.maps)...)

inv(m::ProductMap) = ProductMap(codomain(m), domain(m), inv.(m.maps))

"""
    IdentityMap{D} <: AbstractMap{D,D}

Construct the identity map on `D`, which returns its inputs.

# Constructor

    IdentityMap(M)
"""
struct IdentityMap{D} <: AbstractMap{D,D}
    domain::D
end

codomain(id::IdentityMap) = domain(id)

(::IdentityMap)(x) = identity(x)
(::IdentityMap)(x...) = identity(x)

inv(id::IdentityMap) = id

∘(f::AbstractMap{D}, ::IdentityMap{D}) where {D} = f
∘(::IdentityMap{CoD}, f::AbstractMap{D,CoD}) where {D,CoD} = f
∘(id::IdentityMap{D}, ::IdentityMap{D}) where {D} = id

cross(id1::IdentityMap, id2::IdentityMap) = IdentityMap(domain(id1) × domain(id2))

function show(io::IO, mime::MIME"text/plain", id::IdentityMap)
    print(io, "idₘ: 𝑀 ⟶ 𝑀\n",
              "𝑀 = $(repr(mime, domain(id)))")
end

@doc doc"""
    Inclusion{D,CoD} <: AbstractMap{D,CoD}

Construct the inclusion map $\iota$ from `D` to `CoD`, which identifies the
input in the domain as belonging to the codomain. By default, this is just the
identity map. However, it may be specialized to convert types of points where
sensible.

# Constructor

    Inclusion(domain, codomain)
"""
struct Inclusion{D,CoD} <: AbstractMap{D,CoD}
    domain::D
    codomain::CoD
end

Inclusion(domain::D, ::D) where {D} = Identity(domain)

(::Inclusion)(x) = identity(x)

pinv(𝜄::Inclusion) = Inclusion(codomain(𝜄), domain(𝜄))

function show(io::IO, mime::MIME"text/plain", 𝜄::Inclusion)
    print(io, "𝜄: 𝑀 ↪ 𝑁\n",
              "𝑀 = $(repr(mime, domain(𝜄)))\n",
              "𝑁 = $(repr(mime, codomain(𝜄)))")
end

@doc doc"""
    RiemannianExponential{MT,TMT,MMT} <: AbstractMap{TMT,MMT}

Riemannian exponential map from a point on the tangent bundle $T M$ to
to $M \times M$:

```math
\begin{aligned}
\mathrm{Exp} &\colon T M \to M \times M \\
             &\colon (x, v_x) \mapsto (x, y)\\
\mathrm{where\ }& x, y \in M, v_x \in T_x M
\end{aligned}
```

The abbreviated map name `Exp` is also available.

# Constructor

    RiemannianExponential(M::Manifold)
"""
struct RiemannianExponential{MT,TMT,MMT} <: AbstractMap{TMT,MMT}
    manifold::MT
    domain::TMT
    codomain::MMT
end

const Exp = RiemannianExponential

function RiemannianExponential(M::Manifold)
    shape = representation_size(M, TVector)
    TₓM = Euclidean(shape...)
    TM = M × TₓM
    M² = M × M
    return RiemannianExponential(M, TM, M²)
end

(m::RiemannianExponential)(x, vₓ) = (x, exp(m.manifold, x, vₓ))
(m::RiemannianExponential)(x::Tuple) = m(x...)
(m::RiemannianExponential)(x) = ProductMPoint(x.parts[1], m(x.parts...))

pinv(m::Exp) = Log(m.manifold)

function show(io::IO, mime::MIME"text/plain", m::RiemannianExponential)
    print(io, "Exp: 𝑇𝑀 ⟶ 𝑀 × 𝑀\n",
              "𝑀 = $(repr(mime, m.manifold))")
end

@doc doc"""
    RiemannianLogarithm{MT,MMT,TMT} <: AbstractMap{MMT,TMT}

Riemannian logarithm map, which is the right inverse of the Riemannian
exponential, defined as

```math
\begin{aligned}
\mathrm{Log} &\colon M \times M \to T M\\
             &\colon (x, y) \mapsto (x, v_x)\\
\mathrm{Exp} \circ \mathrm{Log} &\colon (x, y) \mapsto (x, y)\\
\mathrm{where\ }& x,y \in M, v_x \in T_x M
\end{aligned}
```

The abbreviated map name `Log` is also available.

# Constructor

    RiemannianLogarithm(M::Manifold)
"""
struct RiemannianLogarithm{MT,MMT,TMT} <: AbstractMap{MMT,TMT}
    manifold::MT
    domain::MMT
    codomain::TMT
end

const Log = RiemannianLogarithm

function RiemannianLogarithm(M)
    shape = representation_size(M, TVector)
    TₓM = Euclidean(shape...)
    TM = M × TₓM
    M² = M × M
    return RiemannianLogarithm(M, M², TM)
end

(m::RiemannianLogarithm)(x, y) = (x, log(m.manifold, x, y))
(m::RiemannianLogarithm)(x::Tuple) = m(x...)
(m::RiemannianLogarithm)(x) = ProductMPoint(x.parts[1], m(x.parts...))

inv(m::Log) = Exp(m.manifold)

∘(e::Exp{M}, l::Log{M}) where {M} = IdentityMap(domain(l))

function show(io::IO, mime::MIME"text/plain", m::RiemannianLogarithm)
    print(io, "Log: 𝑀 × 𝑀 ⟶ 𝑇𝑀\n",
              "𝑀 = $(repr(mime, m.manifold))")
end

@doc doc"""
    Geodesic{MT,ET<:Exp,PT,VT} <: AbstractCurve{MT}

Geodesic curve $\gamma(t)$ on manifold $M$:

```math
\begin{aligned}
\gamma &\colon \mathbb R \to M\\
\gamma(0) &= x \in M\\
\dot\gamma(0) &= v \in T_x M
\end{aligned}
```

# Constructor

    Geodesic(M::Manifold, x, v)
"""
struct Geodesic{MT,PT,VT} <: AbstractCurve{MT}
    codomain::MT
    x₀::PT
    v₀::VT
end

(g::Geodesic)(t::Real) = Exp(codomain(g))(g.x₀, t * g.v₀)[2]

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
\gamma &\colon \mathbb R \to M\\
\gamma(0) &= x \in M\\
\gamma(1) &= y \in M,
\end{aligned}
```

such that the length of the curve between $x$ and $y$ on the interval
$t=[0,1]$ is the shortest distance on the manifold between those two points.

# Constructor

    ShortestGeodesic(M::Manifold, x, y)
"""
struct ShortestGeodesic{MT,GT,IT,FT} <: AbstractCurve{MT}
    codomain::MT
    geodesic::GT
    x₀::IT
    x₁::FT
end

function ShortestGeodesic(M, x, y)
    v = Log(M)(x, y)[2]
    return ShortestGeodesic(M, Geodesic(M, x, v), x, y)
end

(g::ShortestGeodesic)(t) = g.geodesic(t)

function show(io::IO, mime::MIME"text/plain", m::ShortestGeodesic)
    print(io, "𝛾: ℝ ⟶ 𝑀\n",
              "𝑀 = $(repr(mime, codomain(m)))\n",
              "𝛾(0) = $(repr(mime, m.x₀))\n",
              "𝛾(1) = $(repr(mime, m.x₁))")
end
