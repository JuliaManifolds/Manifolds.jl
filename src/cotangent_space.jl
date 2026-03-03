"""
    RieszRepresenterCotangentVector(M::AbstractManifold, p, X)

Cotangent vector in Riesz representer form on manifold `M` at point `p` with Riesz
representer `X`.
"""
struct RieszRepresenterCotangentVector{TM <: AbstractManifold, TP, TX}
    manifold::TM
    p::TP
    X::TX
end

function allocate(ξ::RieszRepresenterCotangentVector)
    return RieszRepresenterCotangentVector(ξ.manifold, copy(ξ.manifold, ξ.p), allocate(ξ.X))
end

function (ξ::RieszRepresenterCotangentVector)(Y)
    return inner(ξ.manifold, ξ.p, ξ.X, Y)
end

function Base.copy(M::AbstractManifold, p, ξ::RieszRepresenterCotangentVector)
    return RieszRepresenterCotangentVector(ξ.manifold, copy(ξ.manifold, ξ.p), copy(ξ.manifold, ξ.X))
end

@trait_function flat!(M::AbstractDecoratorManifold, ξ::CoTFVector, p, X::TFVector)

@doc raw"""
    flat(M::AbstractManifold, p, X)

Compute the flat isomorphism (one of the musical isomorphisms) of tangent vector `X`
from the vector space of type `M` at point `p` from the underlying `AbstractManifold`.

The function can be used for example to transform vectors
from the tangent bundle to vectors from the cotangent bundle
``♭ : T\mathcal M → T^{*}\mathcal M``
"""
flat(M::AbstractManifold, p, X) = RieszRepresenterCotangentVector(M, p, X)
function flat(M::AbstractManifold, p, X::TFVector{<:Any, <:AbstractBasis})
    return CoTFVector(X.data, dual_basis(M, p, X.basis))
end

is_metric_function(::typeof(flat)) = true

function flat!(::AbstractManifold, ξ::RieszRepresenterCotangentVector, p, X)
    # TODO: maybe assert that ξ.p is equal to p? Allowing for varying p in ξ leads to
    # issues with power manifold.
    copyto!(ξ.X, X)
    return ξ
end

function get_coordinates(
        M::AbstractManifold,
        p,
        ξ::RieszRepresenterCotangentVector,
        ::DefaultOrthonormalBasis{𝔽, CotangentSpaceType},
    ) where {𝔽}
    return get_coordinates(M, p, ξ.X, DefaultOrthonormalBasis{𝔽}())
end
for TM in filter(x -> x != AbstractManifold, Manifolds.METAMANIFOLDS)
    eval(
        quote
            @invoke_maker 1 AbstractManifold get_coordinates(
                M::$TM,
                p,
                ξ::RieszRepresenterCotangentVector,
                b::DefaultOrthonormalBasis{𝔽, CotangentSpaceType},
            ) where {𝔽}
        end,
    )
end
# define also for all decorators and explicit defining subtypes

function get_coordinates!(
        M::AbstractManifold,
        v,
        p,
        ξ::RieszRepresenterCotangentVector,
        ::DefaultOrthonormalBasis{𝔽, CotangentSpaceType},
    ) where {𝔽}
    get_coordinates!(M, v, p, ξ.X, DefaultOrthonormalBasis{𝔽}())
    return v
end
for TM in filter(x -> x != AbstractManifold, Manifolds.METAMANIFOLDS)
    eval(
        quote
            @invoke_maker 1 AbstractManifold get_coordinates!(
                M::$TM,
                v,
                p,
                ξ::RieszRepresenterCotangentVector,
                b::DefaultOrthonormalBasis{𝔽, CotangentSpaceType},
            ) where {𝔽}
        end,
    )
end

function get_vector(
        M::AbstractManifold,
        p,
        v,
        ::DefaultOrthonormalBasis{𝔽, CotangentSpaceType},
    ) where {𝔽}
    X = get_vector(M, p, v, DefaultOrthonormalBasis{𝔽}())
    return RieszRepresenterCotangentVector(M, p, X)
end
for TM in filter(x -> x != AbstractManifold, Manifolds.METAMANIFOLDS)
    eval(
        quote
            @invoke_maker 1 AbstractManifold get_vector(
                M::$TM,
                p,
                v,
                b::DefaultOrthonormalBasis{𝔽, CotangentSpaceType},
            ) where {𝔽}
        end,
    )
end

function get_vector!(
        M::AbstractManifold,
        ξr::RieszRepresenterCotangentVector,
        p,
        v,
        ::DefaultOrthonormalBasis{𝔽, CotangentSpaceType},
    ) where {𝔽}
    get_vector!(M, ξr.X, p, v, DefaultOrthonormalBasis{𝔽}())
    return ξr
end
for TM in filter(x -> x != AbstractManifold, Manifolds.METAMANIFOLDS)
    eval(
        quote
            @invoke_maker 1 AbstractManifold get_vector!(
                M::$TM,
                ξr::RieszRepresenterCotangentVector,
                p,
                v,
                b::DefaultOrthonormalBasis{𝔽, CotangentSpaceType},
            ) where {𝔽}
        end,
    )
end

@doc raw"""
    sharp(M::AbstractManifold, p, ξ)

Compute the sharp isomorphism (one of the musical isomorphisms) of vector `ξ`
from the vector space `M` at point `p` from the underlying `AbstractManifold`.

The function can be used for example to transform vectors
from the cotangent bundle to vectors from the tangent bundle
``♯ : T^{*}\mathcal M → T\mathcal M``
"""
sharp(::AbstractManifold, p, ξ)

@trait_function sharp(M::AbstractDecoratorManifold, X::TFVector, p, ξ::CoTFVector)

sharp(::AbstractManifold, p, ξ::RieszRepresenterCotangentVector) = ξ.X
function sharp(M::AbstractManifold, p, X::CoTFVector{<:Any, <:AbstractBasis})
    return TFVector(X.data, dual_basis(M, p, X.basis))
end

is_metric_function(::typeof(sharp)) = true

@trait_function sharp!(M::AbstractDecoratorManifold, X::TFVector, p, ξ::CoTFVector)

function sharp!(::AbstractManifold, X, p, ξ::RieszRepresenterCotangentVector)
    copyto!(X, ξ.X)
    return X
end

function Base.isapprox(M::AbstractManifold, ξ1::RieszRepresenterCotangentVector, ξ2::RieszRepresenterCotangentVector; kwargs...)
    return isapprox(M, ξ1.p, ξ2.p; kwargs...) && isapprox(M, ξ1.p, ξ1.X, ξ2.X; kwargs...)
end
