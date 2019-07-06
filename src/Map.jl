abstract type AbstractMap{D,CoD} end

domain(m::AbstractMap) = m.domain

codomain(m::AbstractMap) = m.codomain

struct CompositeMap{D,CoD,F,G}
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




struct FunctionMap{D,CoD,F} <: AbstractMap{D,CoD}
    domain::D
    codomain::CoD
    f::F

    function FunctionMap(domain::D, codomain::CoD, f::F) where {D,CoD,F<:Function}
        return new{D,CoD,F}(domain, codomain, f)
    end
end

function FunctionMap(f::F) where {D,CoD,F<:AbstractMap{D,CoD}}
    return FunctionMap(domain(F), codomain(F), f)
end




struct Bijection{D,CoD,F,IF} <: AbstractMap{D,CoD}
    domain::D
    codomain::CoD
    f::F
    invf::IF

    function Bijection(domain::D, codomain::CoD, f::F, invf::IF) where {D,CoD,F<:Function,IF}
        return new{D,CoD,F,IF}(domain, codomain, f, invf)
    end
end

function Bijection(f::F, invf::IF) where {D,CoD,F<:AbstractMap{D,CoD},IF}
    return Bijection(domain(F), codomain(F), f, invf)
end

function ∘(f::Bijection, g::Bijection)
    return Bijection(domain(g), codomain(f), f.f ∘ g.f, g.invf ∘ f.invf)
end

inv(m::Bijection) = Bijection(codomain(m), domain(m), m.invf, m.f)

pinv(m::Bijection) = inv(m)



struct Injection{D,CoD,F,LIF} <: AbstractMap{D,CoD}
    domain::D
    codomain::CoD
    f::F
    linvf::LIF

    function Injection(domain::D, codomain::CoD, f::F, linvf::LIF) where {D,CoD,F<:Function,LIF}
        return new{D,CoD,F,LIF}(domain, codomain, f, linvf)
    end
end

function Injection(f::F, linvf::LIF) where {D,CoD,F<:AbstractMap{D,CoD},LIF}
    return Injection(domain(f), codomain(f), f, linvf)
end

pinv(m::Injection) = Surjection(codomain(m), domain(m), m.linvf, m.f)



struct Surjection{D,CoD,F,RIF} <: AbstractMap{D,CoD}
    domain::D
    codomain::CoD
    f::F
    rinvf::RIF

    function Surjection(domain::D, codomain::CoD, f::F, rinvf::RIF) where {D,CoD,F<:Function,RIF}
        return new{D,CoD,F,RIF}(domain, codomain, f, rinvf)
    end
end

function Surjection(f::F, rinvf::RIF) where {D,CoD,F<:AbstractMap{D,CoD},RIF}
    return Surjection(domain(f), codomain(f), f, rinvf)
end

pinv(m::Surjection) = Injection(codomain(m), domain(m), m.rinvf, m.f)

function ∘(f::Injection, g::Injection)
    return Injection(domain(g), codomain(f), f.f ∘ g.f, g.linvf ∘ f.linvf)
end

function ∘(f::Bijection, g::Injection)
    return Injection(domain(g), codomain(f), f.f ∘ g.f, g.linvf ∘ f.invf)
end

function ∘(f::Injection, g::Bijection)
    return Injection(domain(g), codomain(f), f.f ∘ g.f, g.invf ∘ f.linvf)
end

function ∘(f::Surjection, g::Surjection)
    return Surjection(domain(g), codomain(f), f.f ∘ g.f, g.rinvf ∘ f.rinvf)
end

function ∘(f::Surjection, g::Bijection)
    return Surjection(domain(g), codomain(f), f.f ∘ g.f, g.invf ∘ f.rinvf)
end

function ∘(f::Bijection, g::Surjection)
    return Surjection(domain(g), codomain(f), f.f ∘ g.f, g.rinvf ∘ f.invf)
end

(m::FunctionMap)(x...) = m.f(x...)
(m::Injection)(x...) = m.f(x...)
(m::Bijection)(x...) = m.f(x...)
(m::Surjection)(x...) = m.f(x...)

AbstractCurve{CoD} = AbstractMap{Euclidean{Tuple{1}},CoD}

AbstractField{D,T<:Tuple} = AbstractMap{D,Euclidean{T}}

AbstractScalarField{D} = AbstractField{D,Tuple{1}}

AbstractVectorField{D,N} = AbstractField{D,Tuple{N}}

AbstractMatrixField{D,M,N} = AbstractField{D,Tuple{M,N}}

struct ExponentialMap{MT,TMT,PT} <: AbstractMap{TMT,MT}
    domain::MT
    codomain::TMT
    point::PT

    function ExponentialMap(M::MT, x::PT) where {MT,PT}
        # shape = representation_size(M, TVector)
        shape = size(x)
        TM = Euclidean(shape...)
        return new{MT,typeof(TM),PT}(M, TM, x)
    end
end

(m::ExponentialMap)(v) = exp(m.domain, m.point, v)

show(io::IO, m::ExponentialMap) = print(io, "ExponentialMap(", m.domain, ", ", m.point, ")")




struct LogarithmMap{TMT,MT,PT} <: AbstractMap{TMT,MT}
    domain::TMT
    codomain::MT
    point::PT

    function LogarithmMap(M::MT, x::PT) where {MT,PT}
        # shape = representation_size(M, TVector)
        shape = size(x)
        TM = Euclidean(shape...)
        return new{typeof(TM),MT,PT}(TM, M, x)
    end
end

(m::LogarithmMap)(x) = log(m.codomain, m.point, x)

show(io::IO, m::LogarithmMap) = print(io, "LogarithmMap(", m.codomain, ", ", m.point, ")")


struct Geodesic{MT,ET<:ExponentialMap,VT} <: AbstractCurve{MT}
    domain::MT
    Exp::ET
    tvector::VT
end

Geodesic(M, x, v) = Geodesic(M, ExponentialMap(M, x), v)
(g::Geodesic)(t::Real) = g.Exp(t*g.tvector)
(g::Geodesic)(T::AbstractVector) = map(t -> g.Exp(t*g.tvector), T)

show(io::IO, m::Geodesic) = print(io, "Geodesic(", m.domain, ", ", m.Exp.point, ", ", m.tvector, ")")

codomain(g::Geodesic) = codomain(g.Exp)


struct ShortestGeodesic{MT,GT<:Geodesic} <: AbstractCurve{MT}
    domain::MT
    geodesic::GT
end

function ShortestGeodesic(M, x, y)
    v = LogarithmMap(M, x)(y)
    return ShortestGeodesic(M, Geodesic(M, x, v))
end

(g::ShortestGeodesic)(t) = g.geodesic(t)

show(io::IO, m::ShortestGeodesic) = print(io, "ShortestGeodesic(", m.domain, ", ", m.geodesic.Exp.point, ", ", m.geodesic.tvector, ")")

codomain(g::ShortestGeodesic) = codomain(g.geodesic)

function derivation(γ::AbstractCurve{MT}) where {MT}
    dim = manifold_dimension(domain(γ))
    TM = Euclidean(dim)
    return FunctionMap(domain(γ), TM, t -> ForwardDiff.derivative(γ, t))
end
