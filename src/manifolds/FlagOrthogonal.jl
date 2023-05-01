
@doc raw"""
    check_vector(M::Flag, p::OrthogonalPoint, X::OrthogonalTVector; kwargs... )

Check whether `X` is a tangent vector to point `p` on the [`Flag`](@ref) manifold `M`
``\operatorname{Flag}(n_1, n_2, ..., n_d; N)`` in the orthogonal matrix representation,
i.e. that `X` is block-skew-symmetric with zero diagonal:
````math
X = \begin{bmatrix}
0                     & B_{1,2}               & \cdots & B_{1,d+1} \\
-B_{1,2}^\mathrm{T}   & 0                     & \cdots & B_{2,d+1} \\
\vdots                & \vdots                & \ddots & \vdots    \\
-B_{1,d+1}^\mathrm{T} & -B_{2,d+1}^\mathrm{T} & \cdots & 0            
\end{bmatrix}
````
where ``B_{i,j} ∈ ℝ^{(n_i - n_{i-1}) × (n_j - n_{j-1})}``, for  ``1 ≤ i < j ≤ d+1``.
"""
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

embed(::Flag, p::OrthogonalPoint) = p.value
embed!(::Flag, q, p::OrthogonalPoint) = copyto!(q, p.value)
embed(::Flag, p::OrthogonalPoint, X::OrthogonalTVector) = X.value
function embed!(::Flag, Y, p::OrthogonalPoint, X::OrthogonalTVector)
    return copyto!(Y, X.value)
end

"""
    get_embedding(M::Flag, p::OrthogonalPoint)

Get embedding of [`Flag`](@ref) manifold `M`, i.e. the manifold [`OrthogonalMatrices`](@ref).
"""
get_embedding(::Flag{N}, p::OrthogonalPoint) where {N} = OrthogonalMatrices(N)

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

@doc raw"""
    project(M::Flag, p::OrthogonalPoint, X::OrthogonalTVector)

Project vector `X` to tangent space at point `p` from [`Flag`](@ref) manifold `M`
``\operatorname{Flag}(n_1, n_2, ..., n_d; N)``, in the orthogonal matrix representation.
It works by first projecting `X` to the space of [`SkewHermitianMatrices`](@ref) and then
setting diagonal blocks to 0:
````math
X = \begin{bmatrix}
0                     & B_{1,2}               & \cdots & B_{1,d+1} \\
-B_{1,2}^\mathrm{T}   & 0                     & \cdots & B_{2,d+1} \\
\vdots                & \vdots                & \ddots & \vdots    \\
-B_{1,d+1}^\mathrm{T} & -B_{2,d+1}^\mathrm{T} & \cdots & 0            
\end{bmatrix}
````
where ``B_{i,j} ∈ ℝ^{(n_i - n_{i-1}) × (n_j - n_{j-1})}``, for  ``1 ≤ i < j ≤ d+1``.
"""
function project(M::Flag{N,dp1}, ::OrthogonalPoint, X::OrthogonalTVector) where {N,dp1}
    Y = project(SkewHermitianMatrices(N), X.value)
    for i in 1:dp1
        Bi = _extract_flag(M, Y, i)
        fill!(Bi, 0)
    end
    return OrthogonalTVector(Y)
end

function Random.rand!(
    M::Flag,
    pX::Union{OrthogonalPoint,OrthogonalTVector};
    vector_at=nothing,
)
    return rand!(Random.default_rng(), M, pX; vector_at=vector_at)
end
function Random.rand!(
    rng::AbstractRNG,
    M::Flag{N,dp1},
    pX::Union{OrthogonalPoint,OrthogonalTVector};
    vector_at=nothing,
) where {N,dp1}
    if vector_at === nothing
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
    retract(M::Flag, p::OrthogonalPoint, X::OrthogonalTVector, ::QRRetraction)

Compute the QR retraction on the [`Flag`](@ref) in the orthogonal matrix representation
as the first order approximation to the exponential map. Similar to QR retraction for
[`GeneralUnitaryMatrices`].
"""
retract(M::Flag, p::OrthogonalPoint, X::OrthogonalTVector, ::QRRetraction)

function retract_qr!(
    ::Flag,
    q::OrthogonalPoint{<:AbstractMatrix{T}},
    p::OrthogonalPoint,
    X::OrthogonalTVector,
    t::Number,
) where {T}
    A = p.value + p.value * (t * X.value)
    qr_decomp = qr(A)
    d = diag(qr_decomp.R)
    D = Diagonal(sign.(d .+ convert(T, 0.5)))
    copyto!(q.value, qr_decomp.Q * D)
    return q
end
