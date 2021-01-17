
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

function get_coordinates(M::Manifold, p, ξ, B::DefaultOrthonormalBasis)
    return get_coordinates(M, p, ξ.X, B)
end

function get_vector(M::Manifold, p, v, B::DefaultOrthonormalBasis)
    X = get_vector(M, p, v, B)
    return RieszRepresenterCotangentVector(M, p, X)
end
