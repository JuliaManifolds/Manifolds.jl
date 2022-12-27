
function inner(
    M::Flag{N,dp1},
    p::AbstractMatrix,
    X::AbstractMatrix,
    Y::AbstractMatrix,
) where {N,dp1}
    inner_prod = zero(eltype(X))
    for i in 1:(dp1 - 1)
        for j in i:dp1
            block_X = _extract_flag(M, X, j, i)
            block_Y = _extract_flag(M, Y, j, i)
            inner_prod += dot(block_X, block_Y)
        end
    end
    return inner_prod
end

function _extract_flag_stiefel(M::Flag, pX::AbstractMatrix, i::Int)
    range = (M.subspace_dimensions[i - 1] + 1):M.subspace_dimensions[i]
    return view(pX, :, range)
end

function check_point(M::Flag, p::AbstractMatrix)
    return check_point(get_embedding(M), p)
end

function check_vector(
    M::Flag{N,dp1},
    p::AbstractMatrix,
    X::AbstractMatrix;
    atol=eps(eltype(X)),
) where {N,dp1}
    p_ortho = stiefel_point_to_orthogonal(M, p)
    pX = p_ortho.value * X
    for i in 1:(dp1 - 1)
        p_i = _extract_flag_stiefel(M, p, i)
        X_i = _extract_flag_stiefel(M, pX, i)

        pTX_norm = norm(p_i' * X_i)
        if pTX_norm > atol
            return DomainError(
                pTX_norm,
                "Orthogonality condition check failed at subspace $i.",
            )
        end

        for j in i:(dp1 - 1)
            p_j = _extract_flag_stiefel(M, p, j)
            X_j = _extract_flag_stiefel(M, pX, j)
            sum_norm = norm(p_i' * X_j + X_i' * p_j)
            if sum_norm > atol
                return DomainError(
                    sum_norm,
                    "Tangent vector condition check failed at subspaces ($i, $j)",
                )
            end
        end
    end
    return nothing
end

function project!(
    M::Flag{N,dp1},
    Y::AbstractMatrix,
    p::AbstractMatrix,
    X::AbstractMatrix,
) where {N,dp1}
    (_, k) = representation_size(M)
    project!(SkewHermitianMatrices(k), view(Y, 1:k, :), view(X, 1:k, :))
    for i in 1:(dp1 - 1)
        Bi = _extract_flag(M, Y, i)
        fill!(Bi, 0)
    end
    return Y
end

function Random.rand!(M::Flag{N,dp1}, pX::AbstractMatrix; vector_at=nothing) where {N,dp1}
    EM = get_embedding(M)
    if vector_at === nothing
        rand!(EM, pX)
    else
        rand!(EM, pX; vector_at=vector_at)
        project!(M, pX, vector_at, pX)
    end
    return pX
end
function Random.rand!(
    rng::AbstractRNG,
    M::Flag{N,dp1},
    pX::AbstractMatrix;
    vector_at=nothing,
) where {N,dp1}
    EM = get_embedding(M)
    if vector_at === nothing
        rand!(rng, EM, pX)
    else
        rand!(rng, EM, pX; vector_at=vector_at)
        project!(M, pX, vector_at, pX)
    end
    return pX
end
