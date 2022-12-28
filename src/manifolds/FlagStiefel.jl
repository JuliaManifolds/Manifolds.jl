
function inner(
    M::Flag{N,dp1},
    p::AbstractMatrix,
    X::AbstractMatrix,
    Y::AbstractMatrix,
) where {N,dp1}
    inner_prod = zero(eltype(X))
    pX = p' * X
    pY = p' * Y
    X_perp = X - p * pX
    Y_perp = Y - p * pY
    for i in 1:(dp1 - 1)
        for j in i:(dp1 - 1)
            block_X = _extract_flag(M, pX, j, i)
            block_Y = _extract_flag(M, pY, j, i)
            inner_prod += dot(block_X, block_Y)
        end
    end
    inner_prod += dot(X_perp, Y_perp)
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
    atol=sqrt(eps(eltype(X))),
) where {N,dp1}
    for i in 1:(dp1 - 1)
        p_i = _extract_flag_stiefel(M, p, i)
        X_i = _extract_flag_stiefel(M, X, i)

        pTX_norm = norm(p_i' * X_i)
        if pTX_norm > atol
            return DomainError(
                pTX_norm,
                "Orthogonality condition check failed at subspace $i.",
            )
        end

        for j in i:(dp1 - 1)
            p_j = _extract_flag_stiefel(M, p, j)
            X_j = _extract_flag_stiefel(M, X, j)
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
    pX = p' * X
    X_perp = X - p * pX
    project!(SkewHermitianMatrices(k), pX, pX)
    for i in 1:(dp1 - 1)
        Bi = _extract_flag(M, pX, i)
        fill!(Bi, 0)
    end
    Y .+= X_perp .+ p * pX
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

@doc raw"""
    retract(M::Flag, p, X, ::PolarRetraction)

Compute the SVD-based retraction [`PolarRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.PolarRetraction) on the
[`Flag`](@ref) `M`. With $USV = p + X$ the retraction reads
````math
\operatorname{retr}_p X = UV^\mathrm{H},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
retract(::Flag, ::Any, ::Any, ::PolarRetraction)

function retract_polar!(::Flag, q, p, X)
    s = svd(p + p * X)
    return mul!(q, s.U, s.Vt)
end
