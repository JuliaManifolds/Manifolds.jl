@doc raw"""
    LinearAffineMetric <: AbstractMetric

The linear affine metric is the metric for symmetric positive definite matrices, that employs
matrix logarithms and exponentials, which yields a linear and affine metric.
"""
struct LinearAffineMetric <: RiemannianMetric end

@doc raw"""
    change_representer(M::SymmetricPositiveDefinite, E::EuclideanMetric, p, X)

Given a tangent vector ``X ∈ T_p\mathcal M`` representing a linear function on the tangent
space at `p` with respect to the [`EuclideanMetric`](@ref) `g_E`,
this is turned into the representer with respect to the (default) metric,
the [`LinearAffineMetric`](@ref) on the [`SymmetricPositiveDefinite`](@ref) `M`.

To be precise we are looking for ``Z∈T_p\mathcal P(n)`` such that for all ``Y∈T_p\mathcal P(n)```
it holds

```math
⟨X,Y⟩ = \operatorname{tr}(XY) = \operatorname{tr}(p^{-1}Zp^{-1}Y) = g_p(Z,Y)
```

and hence ``Z = pXp``.
"""
change_representer(::SymmetricPositiveDefinite, ::EuclideanMetric, ::Any, ::Any)

function change_representer!(::SymmetricPositiveDefinite, Y, ::EuclideanMetric, p, X)
    Y .= p * X * p
    return Y
end

@doc raw"""
    change_metric(M::SymmetricPositiveDefinite{n}, E::EuclideanMetric, p, X)

Given a tangent vector ``X ∈ T_p\mathcal P(n)`` with respect to the [`EuclideanMetric`](@ref) `g_E`,
this function changes into the [`LinearAffineMetric`](@ref) (default) metric on the
[`SymmetricPositiveDefinite`](@ref) `M`.

To be precise we are looking for ``c\colon T_p\mathcal P(n) \to T_p\mathcal P(n) ``
such that for all ``Y,Z ∈ T_p\mathcal P(n)``` it holds

```math
⟨Y,Z⟩ = \operatorname{tr}(YZ) = \operatorname{tr}(p^{-1}c(Y)p^{-1}c(Z)) = g_p(c(Z),c(Y))
```

and hence ``c(X) = pX`` is computed.
"""
change_metric(::SymmetricPositiveDefinite, ::EuclideanMetric, ::Any, ::Any)

function change_metric!(::SymmetricPositiveDefinite, Y, ::EuclideanMetric, p, X)
    Y .= p * X
    return Y
end

default_metric_dispatch(::SymmetricPositiveDefinite, ::LinearAffineMetric) = Val(true)

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
function distance(::SymmetricPositiveDefinite{N}, p, q) where {N}
    # avoid numerical instabilities in cholesky
    norm(p - q) < eps(eltype(p + q)) && return zero(eltype(p + q))
    cq = cholesky(Symmetric(q)) # to avoid numerical inaccuracies
    s = eigvals(Symmetric(cq.L \ p / cq.U))
    return any(s .<= eps()) ? zero(eltype(p)) : sqrt(sum(abs.(log.(s)) .^ 2))
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

function exp!(::SymmetricPositiveDefinite{N}, q, p, X) where {N}
    e = eigen(Symmetric(p))
    U = e.vectors
    S = max.(e.values, floatmin(eltype(e.values)))
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
with eigenvalues `κ` and where the direction `B.frame_direction` ``V`` has curvature `0`.

The construction is based on an ONB for the symmetric matrices similar to [`get_basis(::SymmetricPositiveDefinite, p, ::DefaultOrthonormalBasis`](@ref  get_basis(M::SymmetricPositiveDefinite,p,B::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType}))
just that the ONB here is build from the eigen vectors of ``p^{\frac{1}{2}}Vp^{\frac{1}{2}}``.
"""
function get_basis(
    ::SymmetricPositiveDefinite{N},
    p,
    B::DiagonalizingOrthonormalBasis,
) where {N}
    e = eigen(Symmetric(p))
    U = e.vectors
    S = max.(e.values, floatmin(eltype(e.values)))
    Ssqrt = Diagonal(sqrt.(S))
    pSqrt = Symmetric(U * Ssqrt * transpose(U))
    SsqrtInv = Diagonal(1 ./ sqrt.(S))
    pSqrtInv = Symmetric(U * SsqrtInv * transpose(U))
    eigv = eigen(Symmetric(pSqrtInv * B.frame_direction * pSqrtInv))
    V = eigv.vectors
    Ξ = [
        (i == j ? 1 / 2 : 1 / sqrt(2)) *
        pSqrt *
        (V[:, i] * transpose(V[:, j]) + V[:, j] * transpose(V[:, i])) *
        pSqrt for i in 1:N for j in i:N
    ]
    λ = eigv.values
    κ = [-1 / 4 * (λ[i] - λ[j])^2 for i in 1:N for j in i:N]
    return CachedBasis(B, κ, Ξ)
end

@doc raw"""
    [Ξ,κ] = get_basis(M::SymmetricPositiveDefinite, p, B::DefaultOrthonormalBasis)
    [Ξ,κ] = get_basis(M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, p, B::DefaultOrthonormalBasis)

Return a default ONB for the tangent space ``T_p\mathcal P(n)`` of the [`SymmetricPositiveDefinite`](@ref) with respect to the [`LinearAffineMetric`]

    ```math
        g_p(X,Y) = \operatorname{tr}(p^{-1} X p^{-1} Y),
    ```

    The basis constructed here is based on the ONB for symmetric matrices constructed as follows.
    Let ``\Delta_{i,j} = (a_{kl}=_{k,l=1}^n`` with ``a_{kl} =
    \begin{cases}
      1 & \mbox{ for } k=l \text{ if } i=j\\
      \frac{1}{\sqrt{2}}} & \mbox{ for } k=i, l=j \text{ or } k=j, l=i\\
      0 & \text{ else.}
    \end{cases}$
    which forms an ONB for the space of symmetric matrices.

    We then form the ONB by
    ```math
    \Xi_{i,j} = p^{\frac{1}{2}}\Delta_{i,j}p^{\frac{1}{2}},\qquad i=1,\ldots,n, j=i,\ldots,n
    ```
"""
function get_basis(
    ::SymmetricPositiveDefinite{N},
    p,
    B::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType},
) where {N}
    e = eigen(Symmetric(p))
    U = e.vectors
    S = max.(e.values, floatmin(eltype(e.values)))
    Ssqrt = Diagonal(sqrt.(S))
    pSqrt = Symmetric(U * Ssqrt * transpose(U))
    V = Matrix{Float64}(I, N, N)
    Ξ = [
        (i == j ? 1 / 2 : 1 / sqrt(2)) *
        pSqrt *
        (V[:, i] * transpose(V[:, j]) + V[:, j] * transpose(V[:, i])) *
        pSqrt for i in 1:N for j in i:N
    ]
    return CachedBasis(B, Ξ)
end

@doc raw"""
    get_coordinates(::SymmetricPositiveDefinite, p, X, ::DefaultOrthonormalBasis)

Using the basis from [`get_basis(::SymmetricPositiveDefinite, p, ::DefaultOrthonormalBasis`](@ref  get_basis(M::SymmetricPositiveDefinite,p,B::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType}))
the coordinates with respect to this ONB can be simplified to

```math
   c_k = \mathrm{tr}(p^{-\frac{1}{2}}\Delta_{i,j} X)
```
where $k$ is trhe linearized index of the $i=1,\ldots,n, j=i,\ldots,n$.
"""
get_coordinates(::SymmetricPositiveDefinite, c, p, X, ::DefaultOrthonormalBasis)

function get_coordinates!(
    M::SymmetricPositiveDefinite{N},
    c,
    p,
    X,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
) where {N}
    dim = manifold_dimension(M)
    @assert size(c) == (dim,)
    @assert size(X) == (N, N)
    @assert dim == div(N * (N + 1), 2)
    e = eigen(Symmetric(p))
    U = e.vectors
    S = max.(e.values, floatmin(eltype(e.values)))
    SInvsqrt = Diagonal(1 ./ sqrt.(S))
    pInvSqrt = Symmetric(U * SInvsqrt * transpose(U))
    V = Matrix{Float64}(I, N, N)
    k = 1
    for i in 1:N, j in i:N
        s = i == j ? 1 / 2 : 1 / sqrt(2)
        @inbounds c[k] =
            s *
            tr(pInvSqrt * (V[:, i] * transpose(V[:, j]) + V[:, j] * transpose(V[:, i])) * X)
        k += 1
    end
    return c
end

@doc raw"""
    get_vector(::SymmetricPositiveDefinite, p, c, ::DefaultOrthonormalBasis)

Using the basis from [`get_basis(::SymmetricPositiveDefinite, p, ::DefaultOrthonormalBasis`](@ref  get_basis(M::SymmetricPositiveDefinite,p,B::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType}))
the vector reconstruction with respect to this ONB can be simplified to

```math
   X = p^{\frac{1}{2}} \Biggl( \sum_{i=1,j=i}^n c_k \Delta_{i,j} \Biggr) p^{\frac{1}{2}}
```
where $k$ is trhe linearized index of the $i=1,\ldots,n, j=i,\ldots,n$.
"""
get_vector(::SymmetricPositiveDefinite, X, p, c, ::DefaultOrthonormalBasis)

function get_vector!(
    M::SymmetricPositiveDefinite{N},
    X,
    p,
    c,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
) where {N}
    dim = manifold_dimension(M)
    @assert size(c) == (div(N * (N + 1), 2),)
    @assert size(X) == (N, N)
    e = eigen(Symmetric(p))
    U = e.vectors
    S = max.(e.values, floatmin(eltype(e.values)))
    Ssqrt = Diagonal(sqrt.(S))
    pSqrt = Symmetric(U * Ssqrt * transpose(U))
    V = Matrix{Float64}(I, N, N)
    X .= 0
    k = 1
    for i in 1:N, j in i:N
        s = i == j ? 1 / 2 : 1 / sqrt(2)
        X .+= (s * c[k]) .* (V[:, i] * transpose(V[:, j]) + V[:, j] * transpose(V[:, i]))
        k += 1
    end
    X .= pSqrt * X * pSqrt
    return X
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
function inner(::SymmetricPositiveDefinite, p, X, Y)
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
    S = max.(e.values, floatmin(eltype(e.values)))
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
    S = max.(e.values, floatmin(eltype(e.values)))
    Ssqrt = sqrt.(S)
    SsqrtInv = Diagonal(1 ./ Ssqrt)
    Ssqrt = Diagonal(Ssqrt)
    pSqrt = Symmetric(U * Ssqrt * transpose(U)) # p^1/2
    pSqrtInv = Symmetric(U * SsqrtInv * transpose(U)) # p^(-1/2)
    tv = Symmetric(pSqrtInv * X * pSqrtInv) # p^(-1/2)Xp^{-1/2}
    ty = Symmetric(pSqrtInv * q * pSqrtInv) # p^(-1/2)qp^(-1/2)
    e2 = eigen(ty)
    Se = Diagonal(log.(max.(e2.values, floatmin(eltype(e2.values)))))
    Ue = e2.vectors
    logty = Symmetric(Ue * Se * transpose(Ue)) # nearly log_pq without the outer p^1/2
    e3 = eigen(logty) # since they cancel with the pInvSqrt in the next line
    Sf = Diagonal(exp.(e3.values / 2)) # Uf * Sf * Uf' is the Exp
    Uf = e3.vectors
    pUe = pSqrt * Uf * Sf * transpose(Uf) # factors left of tv (and transposed right)
    vtp = Symmetric(pUe * tv * transpose(pUe)) # so this is the documented formula
    return copyto!(Y, vtp)
end
