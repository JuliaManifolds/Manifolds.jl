
"""
    RieszRepresenterCotangentVector(M::Manifold, p, X)

Cotangent vector in Riesz representer form on manifold `M` at point `p` with Riesz
representer `X`.
"""
struct RieszRepresenterCotangentVector{TM<:Manifold,TP,TX}
    manifold::TM
    p::TP
    X::TX
end

function (ξ::RieszRepresenterCotangentVector)(Y)
    return inner(ξ.manifold, ξ.p, ξ.X, Y)
end


flat(M::Manifold, p, X) = RieszRepresenterCotangentVector(M, p, X)
flat(M::Manifold, p, X::TFVector{<:Any,Nothing}) = CoTFVector(flat(M, p, X.data))
function flat(M::Manifold, p, X::TFVector{<:Any,<:AbstractBasis})
    return CoTFVector(X.data, dual_basis(M, p, X.basis))
end

function flat!(M::Manifold, ξ::CoTFVector{<:Any,Nothing}, p, X::TFVector{<:Any,Nothing})
    return flat!(M, ξ.data, p, X.data)
end
function flat!(
    M::Manifold,
    ξ::CoTFVector{<:Any,<:AbstractBasis},
    p,
    X::TFVector{<:Any,<:AbstractBasis},
)
    Xv = get_vector(M, p, X.data, X.basis)
    ξv = flat(M, p, Xv)
    get_coordinates!(M, ξ.data, p, ξv, ξ.basis)
    return ξ
end

function get_coordinates(
    M::Manifold,
    p,
    ξ::RieszRepresenterCotangentVector,
    ::DefaultOrthonormalBasis{𝔽,CotangentSpaceType},
) where {𝔽}
    return get_coordinates(M, p, ξ.X, DefaultOrthonormalBasis{𝔽}())
end

function get_coordinates!(
    M::Manifold,
    v,
    p,
    ξ::RieszRepresenterCotangentVector,
    ::DefaultOrthonormalBasis{𝔽,CotangentSpaceType},
) where {𝔽}
    get_coordinates!(M, v, p, ξ.X, DefaultOrthonormalBasis{𝔽}())
    return v
end

function get_vector(M::Manifold, p, v, ::DefaultOrthonormalBasis{𝔽,CotangentSpaceType}) where {𝔽}
    X = get_vector(M, p, v, DefaultOrthonormalBasis{𝔽}())
    return RieszRepresenterCotangentVector(M, p, X)
end

function get_vector!(
    M::Manifold,
    ξr::RieszRepresenterCotangentVector,
    p,
    v,
    ::DefaultOrthonormalBasis{𝔽,CotangentSpaceType},
) where {𝔽}
    get_vector!(M, ξr.X, p, v, DefaultOrthonormalBasis{𝔽}())
    return ξr
end

sharp(::Manifold, p, ξ::RieszRepresenterCotangentVector) = ξ.X
sharp(M::Manifold, p, X::CoTFVector{<:Any,Nothing}) = TFVector(sharp(M, p, X.data))
function sharp(M::Manifold, p, X::TFVector{<:Any,<:AbstractBasis})
    return TFVector(X.data, dual_basis(M, p, X.basis))
end

function sharp!(M::Manifold, X::TFVector{<:Any,Nothing}, p, ξ::CoTFVector{<:Any,Nothing})
    return sharp!(M, X.data, p, ξ.data)
end
function sharp!(
    M::Manifold,
    X::TFVector{<:Any,<:AbstractBasis},
    p,
    ξ::CoTFVector{<:Any,<:AbstractBasis},
)
    ξv = get_vector(M, p, ξ.data, ξ.basis)
    Xv = sharp(M, p, ξv)
    get_coordinates!(M, X.data, p, Xv, X.basis)
    return X
end
