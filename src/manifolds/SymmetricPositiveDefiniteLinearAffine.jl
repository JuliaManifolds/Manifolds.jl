@doc raw"""
    LinearAffineMetric <: Metric

The linear affine metric is the metric for symmetric positive definite matrices, that employs
matrix logarithms and exponentials, which yields a linear and affine metric.
"""
struct LinearAffineMetric <: RiemannianMetric end

val_is_default_metric(::SymmetricPositiveDefinite, ::LinearAffineMetric) = Val(true)

@doc raw"""
    distance(M::SymmetricPositiveDefinite, p, q)
    distance(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}, p, q)

Compute the distance on the [`SymmetricPositiveDefinite`](@ref) manifold between `p` and `q`,
as a [`MetricManifold`](@ref) with [`LinearAffineMetric`](@ref). The formula reads

```math
d_{\mathcal P(n)}(p,q)
= \lVert \operatorname{Log}(p^{-\frac{1}{2}}qp^{-\frac{1}{2}})\rVert_{\mathrm{F}}.,
```
where $\operatorname{Log}$ denotes the matrix logarithm and
$\lVert\cdot\rVert_{\mathrm{F}}$ denotes the matrix Frobenius norm.
"""
function distance(M::SymmetricPositiveDefinite{N}, p, q) where {N}
    s = real.(eigvals(p, q))
    return any(s .<= eps()) ? 0 : sqrt(sum(abs.(log.(s)) .^ 2))
end

@doc raw"""
    exp(M::SymmetricPositiveDefinite, p, X)
    exp(M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, p, X)

Compute the exponential map from `p` with tangent vector `X` on the
[`SymmetricPositiveDefinite`](@ref) `M` with its default [`MetricManifold`](@ref) having the
[`LinearAffineMetric`](@ref). The formula reads

```math
\exp_p X = p^{\frac{1}{2}}\operatorname{Exp}(p^{-\frac{1}{2}} X p^{-\frac{1}{2}})p^{\frac{1}{2}},
```

where $\operatorname{Exp}$ denotes to the matrix exponential.
"""
exp(::SymmetricPositiveDefinite, ::Any...)

function exp!(M::SymmetricPositiveDefinite{N}, q, p, X) where {N}
    e = eigen(Symmetric(p))
    U = e.vectors
    S = e.values
    Ssqrt = Diagonal(sqrt.(S))
    SsqrtInv = Diagonal(1 ./ sqrt.(S))
    pSqrt = Symmetric(U * Ssqrt * transpose(U))
    pSqrtInv = Symmetric(U * SsqrtInv * transpose(U))
    T = Symmetric(pSqrtInv * X * pSqrtInv)
    eig1 = eigen(T) # numerical stabilization
    Se = Diagonal(exp.(eig1.values))
    Ue = eig1.vectors
    pUe = pSqrt * Ue
    return copyto!(q, pUe * Se * transpose(pUe))
end

@doc raw"""
    [Ξ,κ] = get_basis(M::SymmetricPositiveDefinite, p, B::DiagonalizingOrthonormalBasis)
    [Ξ,κ] = get_basis(M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, p, B::DiagonalizingOrthonormalBasis)

Return a orthonormal basis `Ξ` as a vector of tangent vectors (of length
[`manifold_dimension`](@ref) of `M`) in the tangent space of `p` on the
[`MetricManifold`](@ref) of [`SymmetricPositiveDefinite`](@ref) manifold `M` with
[`LinearAffineMetric`](@ref) that diagonalizes the curvature tensor $R(u,v)w$
with eigenvalues `κ` and where the direction `B.frame_direction` has curvature `0`.
"""
function get_basis(
    M::SymmetricPositiveDefinite{N},
    p,
    B::DiagonalizingOrthonormalBasis,
) where {N}
    pSqrt = sqrt(p)
    eigv = eigen(B.frame_direction)
    V = eigv.vectors
    Ξ = [
        (i == j ? 1 / 2 :
             1 / sqrt(2)) * (V[:, i] * transpose(V[:, j]) + V[:, j] * transpose(V[:, i]))
        for i = 1:N
        for j = i:N
    ]
    λ = eigv.values
    κ = [-1 / 4 * (λ[i] - λ[j])^2 for i = 1:N for j = i:N]
    return PrecomputedDiagonalizingOrthonormalBasis(Ξ, κ)
end
function get_basis(
    M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},
    p,
    B::DiagonalizingOrthonormalBasis,
) where {N}
    return get_basis(base_manifold(M), p, B)
end

function get_coordinates(
    M::SymmetricPositiveDefinite{N},
    p,
    X,
    B::ArbitraryOrthonormalBasis,
) where {N}
    dim = manifold_dimension(M)
    Y = similar(X, dim)
    @assert size(X) == (N, N)
    @assert dim == div(N * (N + 1), 2)
    k = 1
    for i = 1:N, j = i:N
        scale = ifelse(i == j, 1, sqrt(2))
        @inbounds Y[k] = X[i, j] * scale
        k += 1
    end
    return Y
end
function get_coordinates(
    M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},
    p,
    X,
    B::ArbitraryOrthonormalBasis,
) where {N}
    return get_coordinates(base_manifold(M), p, X, B)
end

function get_vector(
    M::SymmetricPositiveDefinite{N},
    p,
    X,
    B::ArbitraryOrthonormalBasis,
) where {N}
    dim = manifold_dimension(M)
    Y = allocate_result(M, get_vector, p)
    @assert size(X) == (div(N * (N + 1), 2),)
    @assert size(Y) == (N, N)
    k = 1
    for i = 1:N, j = i:N
        scale = ifelse(i == j, 1, 1 / sqrt(2))
        @inbounds Y[i, j] = X[k] * scale
        @inbounds Y[j, i] = X[k] * scale
        k += 1
    end
    return Y
end
function get_vector(
    M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},
    p,
    X,
    B::ArbitraryOrthonormalBasis,
) where {N}
    return get_vector(base_manifold(M), p, X, B)
end

@doc raw"""
    inner(M::SymmetricPositiveDefinite, p, X, Y)
    inner(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}, p, X, Y)

Compute the inner product of `X`, `Y` in the tangent space of `p` on
the [`SymmetricPositiveDefinite`](@ref) manifold `M`, as
a [`MetricManifold`](@ref) with [`LinearAffineMetric`](@ref). The formula reads

````math
g_p(X,Y) = \operatorname{tr}(p^{-1} X p^{-1} Y),
````
"""
function inner(M::SymmetricPositiveDefinite, p, X, Y)
    F = cholesky(Symmetric(p))
    return tr((F \ Symmetric(X)) * (F \ Symmetric(Y)))
end

@doc raw"""
    log(M::SymmetricPositiveDefinite, p, q)
    log(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}, p, q)

Compute the logarithmic map from `p` to `q` on the [`SymmetricPositiveDefinite`](@ref)
as a [`MetricManifold`](@ref) with [`LinearAffineMetric`](@ref). The formula reads

```math
\log_p q =
p^{\frac{1}{2}}\operatorname{Log}(p^{-\frac{1}{2}}qp^{-\frac{1}{2}})p^{\frac{1}{2}},
```
where $\operatorname{Log}$ denotes to the matrix logarithm.
"""
log(::SymmetricPositiveDefinite, ::Any...)

function log!(M::SymmetricPositiveDefinite{N}, X, p, q) where {N}
    e = eigen(Symmetric(p))
    U = e.vectors
    S = e.values
    Ssqrt = Diagonal(sqrt.(S))
    SsqrtInv = Diagonal(1 ./ sqrt.(S))
    pSqrt = Symmetric(U * Ssqrt * transpose(U))
    pSqrtInv = Symmetric(U * SsqrtInv * transpose(U))
    T = Symmetric(pSqrtInv * q * pSqrtInv)
    e2 = eigen(T)
    Se = Diagonal(log.(max.(e2.values, eps())))
    pUe = pSqrt * e2.vectors
    return mul!(X, pUe, Se * transpose(pUe))
end

@doc raw"""
    vector_transport_to(M::SymmetricPositiveDefinite, p, X, q, ::ParallelTransport)
    vector_transport_to(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}, p, X, y, ::ParallelTransport)

Compute the parallel transport of `X` from the tangent space at `p` to the
tangent space at `q` on the [`SymmetricPositiveDefinite`](@ref) as a
[`MetricManifold`](@ref) with the [`LinearAffineMetric`](@ref).
The formula reads

```math
\mathcal P_{q←p}X = p^{\frac{1}{2}}
\operatorname{Exp}\bigl(
\frac{1}{2}p^{-\frac{1}{2}}\log_p(q)p^{-\frac{1}{2}}
\bigr)
p^{-\frac{1}{2}}X p^{-\frac{1}{2}}
\operatorname{Exp}\bigl(
\frac{1}{2}p^{-\frac{1}{2}}\log_p(q)p^{-\frac{1}{2}}
\bigr)
p^{\frac{1}{2}},
```

where $\operatorname{Exp}$ denotes the matrix exponential
and `log` the logarithmic map on [`SymmetricPositiveDefinite`](@ref)
(again with respect to the [`LinearAffineMetric`](@ref)).
"""
vector_transport_to(::SymmetricPositiveDefinite, ::Any, ::Any, ::Any, ::ParallelTransport)

function vector_transport_to!(
    M::SymmetricPositiveDefinite{N},
    Y,
    p,
    X,
    q,
    ::ParallelTransport,
) where {N}
    distance(M, p, q) < 2 * eps(eltype(p)) && copyto!(Y, X)
    e = eigen(Symmetric(p))
    U = e.vectors
    S = e.values
    Ssqrt = sqrt.(S)
    SsqrtInv = Diagonal(1 ./ Ssqrt)
    Ssqrt = Diagonal(Ssqrt)
    pSqrt = Symmetric(U * Ssqrt * transpose(U))
    pSqrtInv = Symmetric(U * SsqrtInv * transpose(U))
    tv = Symmetric(pSqrtInv * X * pSqrtInv)
    ty = Symmetric(pSqrtInv * q * pSqrtInv)
    e2 = eigen(ty)
    Se = Diagonal(log.(e2.values))
    Ue = e2.vectors
    ty2 = Symmetric(Ue * Se * transpose(Ue))
    e3 = eigen(ty2)
    Sf = Diagonal(exp.(e3.values))
    Uf = e3.vectors
    pUe = pSqrt * Uf * Sf * transpose(Uf)
    vtp = Symmetric(pUe * tv * transpose(pUe))
    return copyto!(Y, vtp)
end
