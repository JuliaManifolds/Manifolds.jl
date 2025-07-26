function lie_bracket(G::SpecialEuclidean, X::ArrayPartition, Y::ArrayPartition)
    nX, hX = submanifold_components(G, X)
    nY, hY = submanifold_components(G, Y)
    return ArrayPartition(hX * nY - hY * nX, lie_bracket(G.manifold.manifolds[2], hX, hY))
end

"""
    project(M::SpecialEuclideanInGeneralLinear, p)

Project point `p` in [`GeneralLinear`](@ref) to the [`SpecialEuclidean`](@ref) group.
This is performed by extracting the rotation and translation part as in [`affine_matrix`](@ref).
"""
function project(M::SpecialEuclideanInGeneralLinear, p)
    G = M.manifold
    np, hp = submanifold_components(G, p)
    return ArrayPartition(np, hp)
end
"""
    project(M::SpecialEuclideanInGeneralLinear, p, X)

Project tangent vector `X` at point `p` in [`GeneralLinear`](@ref) to the
[`SpecialEuclidean`](@ref) Lie algebra.
This reverses the transformation performed by [`embed`](@ref embed(M::SpecialEuclideanInGeneralLinear, p, X))
"""
function project(M::SpecialEuclideanInGeneralLinear, p, X)
    G = M.manifold
    np, hp = submanifold_components(G, p)
    nX, hX = submanifold_components(G, X)
    if vector_representation(M.manifold) isa LeftInvariantRepresentation
        return ArrayPartition(nX, hX)
    else
        return ArrayPartition(hp * nX, hX)
    end
end

### Special methods for better performance of selected operations

function Base.exp(
        M::SpecialEuclidean{T, <:HybridTangentRepresentation}, p::ArrayPartition, X::ArrayPartition,
    ) where {T}
    M1, M2 = M.manifold.manifolds
    return ArrayPartition(
        exp(M1.manifold, p.x[1], X.x[1]),
        exp(M2.manifold, p.x[2], X.x[2]),
    )
end
function Base.log(
        M::SpecialEuclidean{T, <:HybridTangentRepresentation}, p::ArrayPartition, q::ArrayPartition,
    ) where {T}
    M1, M2 = M.manifold.manifolds
    return ArrayPartition(
        log(M1.manifold, p.x[1], q.x[1]),
        log(M2.manifold, p.x[2], q.x[2]),
    )
end
function vee(
        M::SpecialEuclidean{T, <:HybridTangentRepresentation}, p::ArrayPartition, X::ArrayPartition,
    ) where {T}
    M1, M2 = M.manifold.manifolds
    return vcat(vee(M1.manifold, p.x[1], X.x[1]), vee(M2.manifold, p.x[2], X.x[2]))
end
function get_coordinates(
        M::SpecialEuclidean{T, <:HybridTangentRepresentation}, p::ArrayPartition,
        X::ArrayPartition, basis::DefaultOrthogonalBasis,
    ) where {T}
    M1, M2 = M.manifold.manifolds
    return vcat(
        get_coordinates(M1.manifold, p.x[1], X.x[1], basis),
        get_coordinates(M2.manifold, p.x[2], X.x[2], basis),
    )
end
function hat(
        M::SpecialEuclidean{TypeParameter{Tuple{2}}, <:HybridTangentRepresentation},
        p::ArrayPartition, c::AbstractVector,
    )
    M1, M2 = M.manifold.manifolds
    return ArrayPartition(
        get_vector_orthogonal(M1.manifold, p.x[1], c[SOneTo(2)], ℝ),
        get_vector_orthogonal(M2.manifold, p.x[2], c[SA[3]], ℝ),
    )
end
function get_vector(
        M::SpecialEuclidean{TypeParameter{Tuple{2}}, <:HybridTangentRepresentation},
        p::ArrayPartition, c::AbstractVector, basis::DefaultOrthogonalBasis,
    )
    return ArrayPartition(
        get_vector(M.manifold.manifolds[1].manifold, p.x[1], c[SOneTo(2)], basis),
        get_vector(M.manifold.manifolds[2].manifold, p.x[2], c[SA[3]], basis),
    )
end

function hat(
        M::SpecialEuclidean{TypeParameter{Tuple{3}}, <:HybridTangentRepresentation},
        p::ArrayPartition, c::AbstractVector,
    )
    M1, M2 = M.manifold.manifolds
    return ArrayPartition(
        get_vector_orthogonal(M1.manifold, p.x[1], c[SOneTo(3)], ℝ),
        get_vector_orthogonal(M2.manifold, p.x[2], c[SA[4, 5, 6]], ℝ),
    )
end
function get_vector(
        M::SpecialEuclidean{TypeParameter{Tuple{3}}, <:HybridTangentRepresentation},
        p::ArrayPartition, c::AbstractVector, basis::DefaultOrthogonalBasis,
    )
    return ArrayPartition(
        get_vector(M.manifold.manifolds[1].manifold, p.x[1], c[SOneTo(3)], basis),
        get_vector(M.manifold.manifolds[2].manifold, p.x[2], c[SA[4, 5, 6]], basis),
    )
end
function compose(
        ::SpecialEuclidean{T, <:HybridTangentRepresentation},
        p::ArrayPartition, q::ArrayPartition,
    ) where {T}
    return ArrayPartition(p.x[2] * q.x[1] + p.x[1], p.x[2] * q.x[2])
end
