@doc raw"""
    check_vector(M::Flag, p::OrthogonalPoint, X::OrthogonalTangentVector; kwargs... )

Check whether `X` is a tangent vector to point `p` on the [`Flag`](@ref) manifold `M`
``\operatorname{Flag}(n_1, n_2, ..., n_d; N)`` in the orthogonal matrix representation,
i.e. that `X` is block-skew-symmetric with zero diagonal:
````math
X = \begin{bmatrix}
0                     & B_{1,2}               & ⋯ & B_{1,d+1} \\
-B_{1,2}^\mathrm{T}   & 0                     & ⋯ & B_{2,d+1} \\
\vdots                & \vdots                & ⋱ & \vdots    \\
-B_{1,d+1}^\mathrm{T} & -B_{2,d+1}^\mathrm{T} & ⋯ & 0
\end{bmatrix}
````
where ``B_{i,j} ∈ ℝ^{(n_i - n_{i-1}) × (n_j - n_{j-1})}``, for  ``1 ≤ i < j ≤ d+1``.
"""
function check_vector(
        M::Flag{<:Any, dp1},
        p::OrthogonalPoint,
        X::OrthogonalTangentVector;
        kwargs...,
    ) where {dp1}
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

embed(::Flag, p::OrthogonalPoint) = p.value
embed!(::Flag, q, p::OrthogonalPoint) = copyto!(q, p.value)
embed(::Flag, p::OrthogonalPoint, X::OrthogonalTangentVector) = X.value
function embed!(::Flag, Y, p::OrthogonalPoint, X::OrthogonalTangentVector)
    return copyto!(Y, X.value)
end

"""
    get_embedding(M::Flag, p::OrthogonalPoint)

Get embedding of [`Flag`](@ref) manifold `M`, i.e. the manifold [`OrthogonalMatrices`](@ref).
"""
function get_embedding(::Flag{TypeParameter{Tuple{N}}}, p::OrthogonalPoint) where {N}
    return OrthogonalMatrices(N)
end
function get_embedding(M::Flag{Tuple{Int}}, p::OrthogonalPoint)
    return OrthogonalMatrices(M.size[1]; parameter = :field)
end
function ManifoldsBase.get_embedding_type(::Flag, ::OrthogonalPoint)
    return ManifoldsBase.IsometricallyEmbeddedManifoldType(ManifoldsBase.DirectEmbedding())
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

function inner(
        ::Flag,
        p::OrthogonalPoint,
        X::OrthogonalTangentVector,
        Y::OrthogonalTangentVector,
    )
    return dot(X.value, Y.value) / 2
end

function project!(
        M::Flag{<:Any, dp1},
        Y::OrthogonalTangentVector,
        ::OrthogonalPoint,
        X::OrthogonalTangentVector,
    ) where {dp1}
    N = get_parameter(M.size)[1]
    project!(SkewHermitianMatrices(N), Y.value, X.value)
    for i in 1:dp1
        Bi = _extract_flag(M, Y.value, i)
        fill!(Bi, 0)
    end
    return Y
end

@doc raw"""
    project(M::Flag, p::OrthogonalPoint, X::OrthogonalTangentVector)

Project vector `X` to tangent space at point `p` from [`Flag`](@ref) manifold `M`
``\operatorname{Flag}(n_1, n_2, ..., n_d; N)``, in the orthogonal matrix representation.
It works by first projecting `X` to the space of [`SkewHermitianMatrices`](@ref) and then
setting diagonal blocks to 0:
````math
X = \begin{bmatrix}
0                     & B_{1,2}               & ⋯ & B_{1,d+1} \\
-B_{1,2}^\mathrm{T}   & 0                     & ⋯ & B_{2,d+1} \\
\vdots                & \vdots                & ⋱ & \vdots    \\
-B_{1,d+1}^\mathrm{T} & -B_{2,d+1}^\mathrm{T} & ⋯ & 0
\end{bmatrix}
````
where ``B_{i,j} ∈ ℝ^{(n_i - n_{i-1}) × (n_j - n_{j-1})}``, for  ``1 ≤ i < j ≤ d+1``.
"""
function project(
        M::Flag{<:Any, dp1},
        ::OrthogonalPoint,
        X::OrthogonalTangentVector,
    ) where {dp1}
    N = get_parameter(M.size)[1]
    Y = project(SkewHermitianMatrices(N), X.value)
    for i in 1:dp1
        Bi = _extract_flag(M, Y, i)
        fill!(Bi, 0)
    end
    return OrthogonalTangentVector(Y)
end

function Random.rand!(
        rng::AbstractRNG,
        M::Flag{<:Any, dp1},
        pX::Union{OrthogonalPoint, OrthogonalTangentVector};
        vector_at = nothing,
    ) where {dp1}
    if vector_at === nothing
        N = get_parameter(M.size)[1]
        RN = Rotations(N)
        rand!(rng, RN, pX.value)
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

@doc raw"""
    retract(M::Flag, p::OrthogonalPoint, X::OrthogonalTangentVector, ::QRRetraction)

Compute the QR retraction on the [`Flag`](@ref) in the orthogonal matrix representation
as the first order approximation to the exponential map. Similar to QR retraction for
[`GeneralUnitaryMatrices`].
"""
retract(M::Flag, p::OrthogonalPoint, X::OrthogonalTangentVector, ::QRRetraction)

function ManifoldsBase.retract_qr!(
        M::Flag,
        q::OrthogonalPoint,
        p::OrthogonalPoint,
        X::OrthogonalTangentVector,
    )
    return ManifoldsBase.retract_qr_fused!(M, q, p, X, one(eltype(p)))
end
function ManifoldsBase.retract_qr_fused!(
        ::Flag,
        q::OrthogonalPoint{<:AbstractMatrix{T}},
        p::OrthogonalPoint,
        X::OrthogonalTangentVector,
        t::Number,
    ) where {T}
    A = p.value + p.value * (t * X.value)
    qr_decomp = qr(A)
    d = diag(qr_decomp.R)
    D = Diagonal(sign.(d .+ convert(T, 0.5)))
    copyto!(q.value, qr_decomp.Q * D)
    return q
end
