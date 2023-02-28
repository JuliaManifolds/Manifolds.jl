
function check_vector(
    M::Flag{N,dp1},
    p::OrthogonalPoint,
    X::OrthogonalTVector;
    kwargs...,
) where {N,dp1}
    for i in 1:dp1
        for j in i:dp1
            if i == j
                Bi = _extract_flag(M, X.value, i)
                if !iszero(Bi)
                    return DomainError(
                        norm(Bi),
                        "All diagonal blocks of matrix X must be zero; block $i has norm $(norm(Bi)).",
                    )
                end
            else
                Bij = _extract_flag(M, X.value, i, j)
                Bji = _extract_flag(M, X.value, j, i)
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

function exp!(::Flag, q::OrthogonalPoint, p::OrthogonalPoint, X::OrthogonalTVector)
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

function inner(::Flag, p::OrthogonalPoint, X::OrthogonalTVector, Y::OrthogonalTVector)
    return dot(X.value, Y.value) / 2
end

function project!(
    M::Flag{N,dp1},
    Y::OrthogonalTVector,
    ::OrthogonalPoint,
    X::OrthogonalTVector,
) where {N,dp1}
    project!(SkewHermitianMatrices(N), Y.value, X.value)
    for i in 1:dp1
        Bi = _extract_flag(M, Y.value, i)
        fill!(Bi, 0)
    end
    return Y
end

function project(M::Flag{N,dp1}, ::OrthogonalPoint, X::OrthogonalTVector) where {N,dp1}
    Y = project(SkewHermitianMatrices(N), X.value)
    for i in 1:dp1
        Bi = _extract_flag(M, Y, i)
        fill!(Bi, 0)
    end
    return OrthogonalTVector(Y)
end

function Random.rand!(
    M::Flag{N,dp1},
    pX::Union{OrthogonalPoint,OrthogonalTVector};
    vector_at=nothing,
) where {N,dp1}
    if vector_at === nothing
        RN = Rotations(N)
        rand!(RN, pX)
    else
        for i in 1:dp1
            for j in i:dp1
                Bij = _extract_flag(M, pX.value, i, j)
                if i == j
                    fill!(Bij, 0)
                else
                    Bij .= randn(size(Bij))
                    Bji = _extract_flag(M, pX.value, j, i)
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
    pX::Union{OrthogonalPoint,OrthogonalTVector};
    vector_at=nothing,
) where {N,dp1}
    if vector_at === nothing
        RN = Rotations(N)
        rand!(rng, RN, pX)
    else
        for i in 1:dp1
            for j in i:dp1
                Bij = _extract_flag(M, pX.value, i, j)
                if i == j
                    fill!(Bij, 0)
                else
                    Bij .= randn(rng, size(Bij))
                    Bji = _extract_flag(M, pX.value, j, i)
                    Bji .= -Bij'
                end
            end
        end
    end
    return pX
end

function _phi_B_map!(M::Flag, Y, p, B, X)
    Y .= (B * X .- X * B) ./ 2
    return project!(M, Y, p, Y)
end

function parallel_transport_direction!(
    M::Flag,
    Y::OrthogonalTVector,
    p::OrthogonalPoint,
    X::OrthogonalTVector,
    d::OrthogonalTVector,
)
    Y.value .= X.value
    Z = copy(X.value)
    k_factor = -1
    # TODO: check more carefully the series cutoff.
    for k in 1:10
        _phi_B_map!(M, Z, p, d, Z)
        k_factor *= -k
        Y .+= Z ./ k_factor
    end
    return Y
end

function retract_qr!(
    ::Flag,
    q::OrthogonalPoint{AbstractMatrix{T}},
    p::OrthogonalPoint,
    X::OrthogonalTVector,
) where {T}
    A = p + p * X
    qr_decomp = qr(A)
    d = diag(qr_decomp.R)
    D = Diagonal(sign.(d .+ convert(T, 0.5)))
    return copyto!(q, qr_decomp.Q * D)
end
