@doc raw"""
    OrthogonalPoint <: AbstractManifoldPoint

A type to represent points on a manifold [`Flag`](@ref) in the orthogonal coordinates
representation, i.e. a rotation matrix.
"""
struct OrthogonalPoint{T<:AbstractMatrix} <: AbstractManifoldPoint
    value::T
end

@doc raw"""
    OrthogonalTVector <: TVector

A type to represent tangent vectors to points on a [`Flag`](@ref) manifold  in the
orthogonal coordinates representation.
"""
struct OrthogonalTVector{T<:AbstractMatrix} <: TVector
    value::T
end

ManifoldsBase.@manifold_vector_forwards OrthogonalTVector value
ManifoldsBase.@manifold_element_forwards OrthogonalPoint value

Base.eltype(p::OrthogonalPoint) = eltype(p.value)
Base.eltype(X::OrthogonalTVector) = eltype(X.value)

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

@doc raw"""
    distance(::Flag, p::OrthogonalPoint, q::OrthogonalPoint)

Distance between two points `p`, `q` on [`Flag`](@ref) manifold. The formula reads
```math
d(p, q) = \sqrt{\sum_{i=1}^r λ_i^2}
```
where ``λ_1, λ_2, …, λ_r`` are real numbers corresponding to positive angles of pairs of
complex eigenvalues of matrix `p' * q`.
"""
function distance(::Flag, p::OrthogonalPoint, q::OrthogonalPoint)
    eigval_angles = map(angle, eigvals(p' * q))
    positive_angles = filter(x -> x > 0, eigval_angles)
    return norm(positive_angles)
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

"""
    orthogonal_tv_to_stiefel(M::Flag, p::OrthogonalPoint, X::OrthogonalTVector)

Convert tangent vector from [`Flag`](@ref) manifold `M` from orthogonal representation to
Stiefel representation.
"""
function orthogonal_tv_to_stiefel(M::Flag, p::OrthogonalPoint, X::OrthogonalTVector)
    (N, k) = representation_size(M)
    return p.value * X.value[:, 1:k]
end

"""
    stiefel_tv_to_orthogonal(M::Flag, p::AbstractMatrix, X::AbstractMatrix)

Convert tangent vector from [`Flag`](@ref) manifold `M` from Stiefel representation to
orthogonal representation.
"""
function stiefel_tv_to_orthogonal(M::Flag, p::AbstractMatrix, X::AbstractMatrix)
    (N, k) = representation_size(M)
    out = similar(X, N, N)
    fill!(out, 0)
    out[:, 1:k] = X
    out[1:k, (k + 1):N] = -transpose(view(X, (k + 1):N, 1:k))
    p_ortho = stiefel_point_to_orthogonal(M, p)
    return OrthogonalTVector(p_ortho.value' * out)
end

"""
    orthogonal_point_to_stiefel(M::Flag, p::OrthogonalTVector)

Convert point `p` from [`Flag`](@ref) manifold `M` from orthogonal representation to
Stiefel representation.
"""
function orthogonal_point_to_stiefel(M::Flag, p::OrthogonalTVector)
    (N, k) = representation_size(M)
    return p.value[:, 1:k]
end

"""
    stiefel_point_to_orthogonal(M::Flag, p::AbstractMatrix)

Convert point `p` from [`Flag`](@ref) manifold `M` from Stiefel representation to
orthogonal representation.
"""
function stiefel_point_to_orthogonal(M::Flag, p::AbstractMatrix)
    (N, k) = representation_size(M)
    out = similar(p, N, N)
    fill!(out, 0)
    out[:, 1:k] = p
    out[:, (k + 1):N] = nullspace(p')
    return OrthogonalPoint(out)
end
