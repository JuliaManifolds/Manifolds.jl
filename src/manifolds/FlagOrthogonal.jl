
function check_vector(
    M::Flag{N,dp1},
    p::AbstractMatrix,
    X::AbstractMatrix;
    kwargs...,
) where {N,dp1}
    for i in 1:dp1
        for j in i:dp1
            if i == j
                Bi = _extract_flag(M, X, i)
                if !iszero(Bi)
                    return DomainError(
                        norm(Bi),
                        "All diagonal blocks of matrix X must be zero; block $i has norm $(norm(Bi)).",
                    )
                end
            else
                Bij = _extract_flag(M, X, i, j)
                Bji = _extract_flag(M, X, j, i)
                Bdiff = Bij + Bji'
                if !iszero(Bdiff)
                    return DomainError(
                        norm(Bdiff),
                        "Matrix X must be block skew-symmetric; block ($i, $j) violates this with norm of sum equal to $(norm(Bdiff)).",
                    )
                end
            end
        end
    end
    return nothing
end

@doc raw"""
    distance(::Flag, p::AbstractMatrix, q::AbstractMatrix)

Distance between two points `p`, `q` on [`Flag`](@ref) manifold. The formula reads
```math
d(p, q) = \sqrt{\sum_{i=1}^r λ_i^2}
```
where ``λ_1, λ_2, …, λ_r`` are real numbers corresponding to positive angles of pairs of
complex eigenvalues of matrix `p' * q`.
"""
function distance(::Flag, p::AbstractMatrix, q::AbstractMatrix)
    eigval_angles = map(angle, eigvals(p' * q))
    positive_angles = filter(x -> x > 0, eigval_angles)
    return norm(positive_angles)
end

function exp!(::Flag, q::AbstractMatrix, p::AbstractMatrix, X::AbstractMatrix)
    return q .= p * exp(X)
end

function _extract_flag(M::Flag, p::AbstractMatrix, i::Int)
    range = (M.subspace_dimensions[i - 1] + 1):M.subspace_dimensions[i]
    return view(p, range, range)
end

function _extract_flag(M::Flag, p::AbstractMatrix, i::Int, j::Int)
    range_i = (M.subspace_dimensions[i - 1] + 1):M.subspace_dimensions[i]
    range_j = (M.subspace_dimensions[j - 1] + 1):M.subspace_dimensions[j]
    return view(p, range_i, range_j)
end

function project!(
    M::Flag{N,dp1},
    Y::AbstractMatrix,
    ::AbstractMatrix,
    X::AbstractMatrix,
) where {N,dp1}
    project!(SkewHermitianMatrices(N), Y, X)
    for i in 1:dp1
        Bi = _extract_flag(M, Y, i)
        fill!(Bi, 0)
    end
    return Y
end

function Random.rand!(M::Flag{N,dp1}, pX::AbstractMatrix; vector_at=nothing) where {N,dp1}
    if vector_at === nothing
        RN = Rotations(N)
        rand!(RN, pX)
    else
        for i in 1:dp1
            for j in i:dp1
                Bij = _extract_flag(M, pX, i, j)
                if i == j
                    fill!(Bij, 0)
                else
                    Bij .= randn(size(Bij))
                    Bji = _extract_flag(M, pX, j, i)
                    Bji .= -Bij'
                end
            end
        end
    end
    return pX
end
function Random.rand!(
    rng::AbstractRNG,
    ::Flag{N,dp1},
    pX::AbstractMatrix;
    vector_at=nothing,
) where {N,dp1}
    if vector_at === nothing
        RN = Rotations(N)
        rand!(rng, RN, pX)
    else
        for i in 1:dp1
            for j in i:dp1
                Bij = _extract_flag(M, pX, i, j)
                if i == j
                    fill!(Bij, 0)
                else
                    Bij .= randn(rng, size(Bij))
                    Bji = _extract_flag(M, pX, j, i)
                    Bji .= -Bij'
                end
            end
        end
    end
    return pX
end

representation_size(::Flag{N}) where {N} = (N, N)

function retract_qr!(::Flag, q::AbstractMatrix{T}, p, X) where {T}
    A = p + p * X
    qr_decomp = qr(A)
    d = diag(qr_decomp.R)
    D = Diagonal(sign.(d .+ convert(T, 0.5)))
    return copyto!(q, qr_decomp.Q * D)
end
