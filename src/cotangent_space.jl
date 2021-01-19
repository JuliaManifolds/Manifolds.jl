
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
    # TODO: check handle basis change here? Right now it assumes dual basis.
    copyto!(ξ.data, X.data)
    return ξ
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
    # TODO: check handle basis change here? Right now it assumes dual basis.
    copyto!(X.data, ξ.data)
    return X
end
