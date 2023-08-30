
"""
    default_inverse_retraction_method(M::Flag)

Return [`PolarInverseRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.PolarInverseRetraction)
as the default inverse retraction for the [`Flag`](@ref) manifold.
"""
default_inverse_retraction_method(::Flag) = PolarInverseRetraction()

"""
    default_retraction_method(M::Flag)

Return [`PolarRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.PolarRetraction)
as the default retraction for the [`Flag`](@ref) manifold.
"""
default_retraction_method(::Flag) = PolarRetraction()

"""
    default_vector_transport_method(M::Flag)

Return the [`ProjectionTransport`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/vector_transports.html#ManifoldsBase.ProjectionTransport)
as the default vector transport method for the [`Flag`](@ref) manifold.
"""
default_vector_transport_method(::Flag) = ProjectionTransport()

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

@doc raw"""
    inverse_retract(M::Flag, p, q, ::PolarInverseRetraction)

Compute the inverse retraction for the [`PolarRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.PolarRetraction), on the
[`Flag`](@ref) manifold `M`.
"""
inverse_retract(::Flag, ::Any, ::Any, ::PolarInverseRetraction)

function inverse_retract_polar!(::Flag, X, p, q)
    return copyto!(X, q / (p' * q) - p)
end

function _extract_flag_stiefel(M::Flag, pX::AbstractMatrix, i::Int)
    range = (M.subspace_dimensions[i - 1] + 1):M.subspace_dimensions[i]
    return view(pX, :, range)
end

function check_point(M::Flag, p::AbstractMatrix)
    return check_point(get_embedding(M), p)
end

@doc raw"""
    check_vector(M::Flag, p::AbstractMatrix, X::AbstractMatrix; kwargs... )

Check whether `X` is a tangent vector to point `p` on the [`Flag`](@ref) manifold `M`
``\operatorname{Flag}(n_1, n_2, ..., n_d; N)`` in the Stiefel representation,
i.e. that `X` is a matrix of the form
````math
X = \begin{bmatrix}
0                     & B_{1,2}               & \cdots & B_{1,d} \\
-B_{1,2}^\mathrm{T}   & 0                     & \cdots & B_{2,d} \\
\vdots                & \vdots                & \ddots & \vdots  \\
-B_{1,d}^\mathrm{T}   & -B_{2,d}^\mathrm{T}   & \cdots & 0       \\
-B_{1,d+1}^\mathrm{T} & -B_{2,d+1}^\mathrm{T} & \cdots & -B_{d,d+1}^\mathrm{T}
\end{bmatrix}
````
where ``B_{i,j} ∈ ℝ^{(n_i - n_{i-1}) × (n_j - n_{j-1})}``, for  ``1 ≤ i < j ≤ d+1``.
"""
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

@doc raw"""
    project(::Flag, p, X)

Project vector `X` in the Euclidean embedding to the tangent space at point `p` on
[`Flag`](@ref) manifold. The formula reads [YeWongLim:2021](@cite):

```math
Y_i = X_i - (p_i p_i^{\mathrm{T}}) X_i + \sum_{j \neq i} p_j X_j^{\mathrm{T}} p_i
```
for ``i`` from 1 to ``d`` where the resulting vector is ``Y = [Y_1, Y_2, …, Y_d]`` and
``X = [X_1, X_2, …, X_d]``, ``p = [p_1, p_2, …, p_d]`` are decompositions into basis vector
matrices for consecutive subspaces of the flag.
"""
project(::Flag, p, X)

function project!(
    M::Flag{N,dp1},
    Y::AbstractMatrix,
    p::AbstractMatrix,
    X::AbstractMatrix,
) where {N,dp1}
    Xc = X .- p * (p' * X) ./ 2
    for i in 1:(dp1 - 1)
        Y_i = _extract_flag_stiefel(M, Y, i)
        p_i = _extract_flag_stiefel(M, p, i)
        Xc_i = _extract_flag_stiefel(M, Xc, i)

        Y_i .= Xc_i .- p_i * (p_i' * Xc_i)
        for j in 1:(dp1 - 1)
            i == j && continue
            p_j = _extract_flag_stiefel(M, p, j)
            Xc_j = _extract_flag_stiefel(M, Xc, j)
            Y_i .-= p_j * Xc_j' * p_i
        end
    end

    return Y
end
function project!(M::Flag, q::AbstractMatrix, p::AbstractMatrix)
    return project!(get_embedding(M), q, p)
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

function retract_polar!(::Flag, q, p, X, t::Number)
    s = svd(p .+ t .* X)
    return mul!(q, s.U, s.Vt)
end
