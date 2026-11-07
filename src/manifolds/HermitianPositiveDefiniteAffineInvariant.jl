@doc raw"""
    AffineInvariantMetric <: AbstractMetric

The linear affine metric is the metric for Hermitian, and in particular symmetric, positive definite matrices,
that employs matrix logarithms and exponentials, which yields a linear and affine metric.
"""
struct AffineInvariantMetric <: RiemannianMetric end

@doc raw"""
    change_representer(M::HermitianPositiveDefinite, E::EuclideanMetric, p, X)

Compute the representer of the linear function given by ``X ∈ T_p\mathcal M`` with respect to
the (default) [`AffineInvariantMetric`](@ref) on the [`HermitianPositiveDefinite`](@ref) `M`.
Here `X` represents the linear function on the tangent space at `p` with respect to the
[`EuclideanMetric`](@extref `ManifoldsBase.EuclideanMetric`) `g_E`.

To be precise we are looking for ``Z∈T_p\mathcal H(n)`` such that for all ``Y∈T_p\mathcal H(n)``
it holds

```math
⟨X,Y⟩ = \operatorname{Re}\operatorname{tr}(X^{\mathrm{H}}Y) = \operatorname{tr}(XY) = \operatorname{tr}(p^{-1}Zp^{-1}Y) = g_p(Z,Y)
```

and hence ``Z = pXp``.
"""
change_representer(::HermitianPositiveDefinite, ::EuclideanMetric, ::Any, ::Any)

function change_representer!(::HermitianPositiveDefinite, Y, ::EuclideanMetric, p, X)
    Y .= p * X * p
    return Y
end

@doc raw"""
    change_metric(M::HermitianPositiveDefinite, E::EuclideanMetric, p, X)

Given a tangent vector ``X ∈ T_p\mathcal H(n)`` with respect to the [`EuclideanMetric`](@extref `ManifoldsBase.EuclideanMetric`)
`g_E`, this function changes into the [`AffineInvariantMetric`](@ref) (default) metric on the
[`HermitianPositiveDefinite`](@ref) `M`.

To be precise we are looking for ``c\colon T_p\mathcal H(n) → T_p\mathcal H(n) ``
such that for all ``Y,Z ∈ T_p\mathcal H(n)`` it holds

```math
⟨Y,Z⟩ = \operatorname{Re}\operatorname{tr}(Y^{\mathrm{H}}Z) = \operatorname{tr}(YZ) = \operatorname{tr}(p^{-1}c(Y)p^{-1}c(Z)) = g_p(c(Z),c(Y))
```

and hence ``c(X) = pX`` is computed.
"""
change_metric(::HermitianPositiveDefinite, ::EuclideanMetric, ::Any, ::Any)

function change_metric!(::HermitianPositiveDefinite, Y, ::EuclideanMetric, p, X)
    Y .= p * X
    return Y
end

@doc raw"""
    distance(M::HermitianPositiveDefinite, p, q)
    distance(M::MetricManifold{HermitianPositiveDefinite,AffineInvariantMetric}, p, q)

Compute the distance on the [`HermitianPositiveDefinite`](@ref) manifold between `p` and `q`,
as a [`MetricManifold`](@extref ManifoldsBase.MetricManifold) with [`AffineInvariantMetric`](@ref). The formula reads

```math
d_{\mathcal H(n)}(p,q)
= \lVert \operatorname{Log}(p^{-\frac{1}{2}}qp^{-\frac{1}{2}})\rVert_{\mathrm{F}},
```
where ``\operatorname{Log}`` denotes the matrix logarithm and
``\lVert⋅\rVert_{\mathrm{F}}`` denotes the matrix Frobenius norm,
see [BhatiaHolbrook:2006](@cite), Proposition 3, Eq. (17).
"""
function distance(::HermitianPositiveDefinite, p, q)
    # avoid numerical instabilities in cholesky
    p == q && return zero(real(eltype(p + q)))
    cq = cholesky(Hermitian(q)) # to avoid numerical inaccuracies
    s = eigvals(Hermitian(cq.L \ p / cq.U))
    return sqrt(sum(abs.(log.(max.(s, floatmin(eltype(s))))) .^ 2))
end
function distance(M::HermitianPositiveDefinite, p::MatrixSqrtManifoldPoint, q::MatrixSqrtManifoldPoint)
    return distance(M, convert(AbstractMatrix, p), convert(AbstractMatrix, q))
end

@doc raw"""
    exp(M::HermitianPositiveDefinite, p, X)
    exp(M::MetricManifold{<:HermitianPositiveDefinite,AffineInvariantMetric}, p, X)

Compute the exponential map from `p` with tangent vector `X` on the
[`HermitianPositiveDefinite`](@ref) `M` with its default [`MetricManifold`](@extref ManifoldsBase.MetricManifold) having the
[`AffineInvariantMetric`](@ref). The formula reads

```math
\exp_p X = p^{\frac{1}{2}}\operatorname{Exp}(p^{-\frac{1}{2}} X p^{-\frac{1}{2}})p^{\frac{1}{2}},
```

where ``\operatorname{Exp}`` denotes the matrix exponential,
see [Smith:2005](@cite), Eq. (67).
"""
exp(::HermitianPositiveDefinite, ::Any...)

function exp(::HermitianPositiveDefinite, p::MatrixSqrtManifoldPoint, X)
    (p_sqrt, p_sqrt_inv) = hpd_sqrt_and_sqrt_inv(p)
    T = Hermitian(p_sqrt_inv * X * p_sqrt_inv)
    eig1 = eigen(T) # numerical stabilization
    Se = Diagonal(exp.(eig1.values))
    U_e = eig1.vectors
    pU_e = p_sqrt * U_e
    q = MatrixSqrtManifoldPoint(
        pU_e * Se * pU_e',
        store_p = !ismissing(p.p),
        store_sqrt = !ismissing(p.sqrt),
        store_sqrt_inv = !ismissing(p.sqrt_inv),
    )
    return q
end

function exp_fused!(M::HermitianPositiveDefinite, q, p, X, t::Number)
    return exp!(M, q, p, t * X)
end
function exp!(::HermitianPositiveDefinite, q, p, X)
    (p_sqrt, p_sqrt_inv) = hpd_sqrt_and_sqrt_inv(p)
    T = Hermitian(p_sqrt_inv * X * p_sqrt_inv)
    eig1 = eigen(T) # numerical stabilization
    Se = Diagonal(exp.(eig1.values))
    U_e = eig1.vectors
    pU_e = p_sqrt * U_e
    return copyto!(q, pU_e * Se * pU_e')
end
function exp!(::HermitianPositiveDefinite, q::MatrixSqrtManifoldPoint, p, X)
    (p_sqrt, p_sqrt_inv) = hpd_sqrt_and_sqrt_inv(p)
    T = Hermitian(p_sqrt_inv * X * p_sqrt_inv)
    eig1 = eigen(T) # numerical stabilization
    Se = Diagonal(exp.(eig1.values))
    U_e = eig1.vectors
    pU_e = p_sqrt * U_e
    Q = pU_e * Se * pU_e'
    !ismissing(q.p) && copyto!(q.p, Q)
    Q_e = eigen(Hermitian(Q))
    copyto!(q.eigen.values, Q_e.values)
    copyto!(q.eigen.vectors, Q_e.vectors)
    if !ismissing(q.sqrt) && !ismissing(q.sqrt_inv)
        copyto!.([q.sqrt, q.sqrt_inv], hpd_sqrt_and_sqrt_inv(Q))
    else
        !ismissing(q.sqrt) && copyto!(q.sqrt, hpd_sqrt(Q))
        !ismissing(q.sqrt_inv) && copyto!(q.sqrt_inv, hpd_sqrt_inv(Q))
    end
    return q
end

@doc raw"""
    [Ξ,κ] = get_basis(M::HermitianPositiveDefinite{ℂ}, p, B::DefaultOrthonormalBasis)
    [Ξ,κ] = get_basis(M::MetricManifold{ℂ,<:HermitianPositiveDefinite{ℂ},AffineInvariantMetric}, p, B::DefaultOrthonormalBasis)

Return a default ONB for the tangent space ``T_p\mathcal H(n)`` of the complex [`HermitianPositiveDefinite`](@ref) with respect to the [`AffineInvariantMetric`](@ref).

```math
    g_p(X,Y) = \operatorname{tr}(p^{-1} X p^{-1} Y)
    = \operatorname{Re}\operatorname{tr}\bigl( (p^{-\frac{1}{2}} X p^{-\frac{1}{2}})^{\mathrm{H}} (p^{-\frac{1}{2}} Y p^{-\frac{1}{2}}) \bigr),
```

that is, the Frobenius inner product of ``p^{-\frac{1}{2}} X p^{-\frac{1}{2}}`` and ``p^{-\frac{1}{2}} Y p^{-\frac{1}{2}}``,
since the trace is cyclic and both are Hermitian.
The basis constructed here is hence based on the following ONB for the Hermitian matrices with respect to the Frobenius inner product.
Let

```math
\Delta_{i,j} = (a_{k,l})_{k,l=1}^n \quad \text{ with }
a_{k,l} =
\begin{cases}
  1 & \mbox{ for } k=l \text{ if } i=j\\
  \frac{1}{\sqrt{2}} & \mbox{ for } k=i, l=j \text{ or } k=j, l=i\\
  0 & \text{ else,}
\end{cases}
\qquad i=1,\ldots,n, j=i,\ldots,n,
```

and

```math
\Delta^{\mathrm{i}}_{i,j} = (b_{k,l})_{k,l=1}^n \quad \text{ with }
b_{k,l} =
\begin{cases}
  \frac{\mathrm{i}}{\sqrt{2}} & \mbox{ for } k=i, l=j\\
  -\frac{\mathrm{i}}{\sqrt{2}} & \mbox{ for } k=j, l=i\\
  0 & \text{ else,}
\end{cases}
\qquad i=1,\ldots,n, j=i+1,\ldots,n,
```

where ``\Delta^{\mathrm{i}}_{i,j}`` directly follows ``\Delta_{i,j}`` in the ordering of the basis.

We then form the ONB by
```math
   \Xi_{i,j} = p^{\frac{1}{2}}\Delta_{i,j}p^{\frac{1}{2}}
   \qquad\text{ and }\qquad
   \Xi^{\mathrm{i}}_{i,j} = p^{\frac{1}{2}}\Delta^{\mathrm{i}}_{i,j}p^{\frac{1}{2}}.
```
"""
get_basis(::HermitianPositiveDefinite{ℂ}, p, B::DefaultOrthonormalBasis)

function get_basis_orthonormal(M::HermitianPositiveDefinite{ℂ}, p, Ns::RealNumbers)
    N = get_parameter(M.size)[1]
    p_sqrt = hpd_sqrt(p)
    pM = convert(AbstractMatrix, p)
    T = complex(eltype(pM))
    Ξ = [similar(pM, T) for _ in 1:manifold_dimension(M)]
    k = 1
    for i in 1:N, j in i:N
        s = i == j ? 1 / 2 : 1 / sqrt(2)
        fill!(Ξ[k], zero(T))
        @inbounds Ξ[k][i, j] += 1
        @inbounds Ξ[k][j, i] += 1
        @inbounds Ξ[k] .= s * p_sqrt * Ξ[k] * p_sqrt
        k += 1
        i == j && continue
        fill!(Ξ[k], zero(T))
        @inbounds Ξ[k][i, j] += im
        @inbounds Ξ[k][j, i] -= im
        @inbounds Ξ[k] .= s * p_sqrt * Ξ[k] * p_sqrt
        k += 1
    end
    return CachedBasis(DefaultOrthonormalBasis(Ns), Ξ)
end

@doc raw"""
    get_coordinates(::HermitianPositiveDefinite{ℂ}, p, X, ::DefaultOrthonormalBasis)

Using the basis from [`get_basis`](@ref get_basis(M::HermitianPositiveDefinite{ℂ},p,B::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType}))
the coordinates with respect to this ONB are the inner products with its basis vectors ``\Xi_{i,j}`` and ``\Xi^{\mathrm{i}}_{i,j}``,

```math
   c_k = g_p(X, \Xi_{i,j}) = \operatorname{tr}(p^{-1} X p^{-1} \Xi_{i,j}),
   \qquad
   c_{k+1} = g_p(X, \Xi^{\mathrm{i}}_{i,j}) = \operatorname{tr}(p^{-1} X p^{-1} \Xi^{\mathrm{i}}_{i,j}),
```
where ``k`` is the linearized index of the ``i=1,\ldots,n, j=i,\ldots,n`` in the ordering of the basis,
and the second coordinate only exists for ``i<j``.
"""
get_coordinates(::HermitianPositiveDefinite{ℂ}, c, p, X, ::DefaultOrthonormalBasis)

function get_coordinates_orthonormal!(M::HermitianPositiveDefinite{ℂ}, c, p, X, ::RealNumbers)
    N = get_parameter(M.size)[1]
    dim = manifold_dimension(M)
    @assert size(c) == (dim,)
    @assert size(X) == (N, N)
    @assert dim == N^2
    p_sqrt = hpd_sqrt(p)
    pM = convert(AbstractMatrix, p)
    k = 1
    V = similar(pM, complex(eltype(pM)))
    fill!(V, zero(eltype(V)))
    F = cholesky(Hermitian(pM))
    for i in 1:N, j in i:N
        s = i == j ? 1 / 2 : 1 / sqrt(2)
        @inbounds V[i, j] += 1
        @inbounds V[j, i] += 1
        Yij = p_sqrt * V * p_sqrt
        @inbounds c[k] = s * real(dot(F \ Hermitian(X), (Hermitian(Yij) / F)))
        k += 1
        @inbounds V[i, j] = 0
        @inbounds V[j, i] = 0
        i == j && continue
        @inbounds V[i, j] += im
        @inbounds V[j, i] -= im
        Yij = p_sqrt * V * p_sqrt
        @inbounds c[k] = s * real(dot(F \ Hermitian(X), (Hermitian(Yij) / F)))
        k += 1
        @inbounds V[i, j] = 0
        @inbounds V[j, i] = 0
    end
    return c
end

@doc raw"""
    get_vector(::HermitianPositiveDefinite{ℂ}, p, c, ::DefaultOrthonormalBasis)

Using the basis from [`get_basis`](@ref  get_basis(M::HermitianPositiveDefinite{ℂ},p,B::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType}))
the vector reconstruction with respect to this ONB can be simplified to

```math
   X = p^{\frac{1}{2}} \Biggl( \sum_{i=1,j=i}^n c_k \Delta_{i,j} + \sum_{i=1,j=i+1}^n c_{k+1} \Delta^{\mathrm{i}}_{i,j} \Biggr) p^{\frac{1}{2}}
```
where ``k`` is the linearized index of the ``i=1,\ldots,n, j=i,\ldots,n`` in the ordering of the basis.
"""
get_vector(::HermitianPositiveDefinite{ℂ}, X, p, c, ::DefaultOrthonormalBasis)

function get_vector_orthonormal!(M::HermitianPositiveDefinite{ℂ}, X, p, c, ::RealNumbers)
    N = get_parameter(M.size)[1]
    @assert size(c) == (N^2,)
    @assert size(X) == (N, N)
    p_sqrt = hpd_sqrt(p)
    X .= 0
    k = 1
    V = similar(X)
    fill!(V, zero(eltype(V)))
    for i in 1:N, j in i:N
        s = i == j ? 1 / 2 : 1 / sqrt(2)
        @inbounds V[i, j] += 1
        @inbounds V[j, i] += 1
        Vij = p_sqrt * V * p_sqrt
        @. X += (s * c[k]) * Vij
        k += 1
        @inbounds V[i, j] = 0
        @inbounds V[j, i] = 0
        i == j && continue
        @inbounds V[i, j] += im
        @inbounds V[j, i] -= im
        Vij = p_sqrt * V * p_sqrt
        @. X += (s * c[k]) * Vij
        k += 1
        @inbounds V[i, j] = 0
        @inbounds V[j, i] = 0
    end
    return X
end

@doc raw"""
    inner(M::HermitianPositiveDefinite, p, X, Y)
    inner(M::MetricManifold{HermitianPositiveDefinite,AffineInvariantMetric}, p, X, Y)

Compute the inner product of `X`, `Y` in the tangent space of `p` on
the [`HermitianPositiveDefinite`](@ref) manifold `M`, as
a [`MetricManifold`](@extref ManifoldsBase.MetricManifold) with [`AffineInvariantMetric`](@ref). The formula reads

````math
g_p(X,Y) = \operatorname{tr}(p^{-1} X p^{-1} Y),
````

see [Smith:2005](@cite), Eq. (4). The trace is real for Hermitian ``X`` and ``Y``, since by the cyclic property
of the trace it equals ``\operatorname{tr}\bigl((p^{-\frac{1}{2}}Xp^{-\frac{1}{2}})(p^{-\frac{1}{2}}Yp^{-\frac{1}{2}})\bigr)``,
the trace of a product of two Hermitian matrices.
"""
function inner(::HermitianPositiveDefinite, p, X, Y)
    F = cholesky(Hermitian(convert(AbstractMatrix, p)))
    return real(dot((F \ Hermitian(X)), (Hermitian(Y) / F)))
end

"""
    is_flat(M::MetricManifold{𝔽,<:HermitianPositiveDefinite{𝔽},AffineInvariantMetric})

Return true if the [`HermitianPositiveDefinite`](@ref) with [`AffineInvariantMetric`](@ref)
is one-dimensional, i.e. for ``n = 1``, and false otherwise.
"""
function is_flat(
        M::MetricManifold{𝔽, <:HermitianPositiveDefinite{𝔽}, AffineInvariantMetric}
    ) where {𝔽}
    return is_flat(M.manifold)
end

@doc raw"""
    log(M::HermitianPositiveDefinite, p, q)
    log(M::MetricManifold{HermitianPositiveDefinite,AffineInvariantMetric}, p, q)

Compute the logarithmic map from `p` to `q` on the [`HermitianPositiveDefinite`](@ref)
as a [`MetricManifold`](@extref ManifoldsBase.MetricManifold) with [`AffineInvariantMetric`](@ref). The formula reads

```math
\log_p q =
p^{\frac{1}{2}}\operatorname{Log}(p^{-\frac{1}{2}}qp^{-\frac{1}{2}})p^{\frac{1}{2}},
```
where ``\operatorname{Log}`` denotes the matrix logarithm, see [Smith:2005](@cite), Eq. (69); it is the inverse of the exponential map.
"""
log(::HermitianPositiveDefinite, ::Any...)

function allocate_result(
        M::HermitianPositiveDefinite, ::typeof(log), q::MatrixSqrtManifoldPoint, p::MatrixSqrtManifoldPoint
    )
    return allocate_result(M, log, convert(AbstractMatrix, q), convert(AbstractMatrix, p))
end

function log!(::HermitianPositiveDefinite, X, p, q)
    (p_sqrt, p_sqrt_inv) = hpd_sqrt_and_sqrt_inv(p)
    T = Hermitian(p_sqrt_inv * convert(AbstractMatrix, q) * p_sqrt_inv)
    e2 = eigen(T)
    R = eltype(e2.values)
    a = norm(e2.values, Inf)
    # below eps() clamp at the LAPACK eigenvalue error bound (Anderson et al. 1999, Sec. 4.7)
    c = a < eps() ? max(eps(R) / 2 * a, floatmin(R)) : eps()
    Se = Diagonal(log.(max.(e2.values, c)))
    pU_e = p_sqrt * e2.vectors
    return mul!(X, pU_e, Se * pU_e')
end

"""
    manifold_volume(::HermitianPositiveDefinite)

Return volume of the [`HermitianPositiveDefinite`](@ref) manifold, i.e. infinity.
"""
manifold_volume(::HermitianPositiveDefinite) = Inf

@doc raw"""
    parallel_transport_to(M::HermitianPositiveDefinite, p, X, q)
    parallel_transport_to(M::MetricManifold{HermitianPositiveDefinite,AffineInvariantMetric}, p, X, y)

Compute the parallel transport of `X` from the tangent space at `p` to the
tangent space at `q` on the [`HermitianPositiveDefinite`](@ref) as a
[`MetricManifold`](@extref ManifoldsBase.MetricManifold) with the [`AffineInvariantMetric`](@ref).
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

where ``\operatorname{Exp}`` denotes the matrix exponential
and `log` the logarithmic map on [`HermitianPositiveDefinite`](@ref)
(again with respect to the [`AffineInvariantMetric`](@ref)),
see [Smith:2005](@cite), Eq. (104), the parallel translation along the geodesic from ``p`` in direction ``D``,
evaluated at ``t = 1`` with ``D = \log_p q``.
"""
parallel_transport_to(::HermitianPositiveDefinite, ::Any, ::Any, ::Any)

function parallel_transport_to!(M::HermitianPositiveDefinite, Y, p, X, q)
    distance(M, p, q) < 2 * eps(real(eltype(p))) && return copyto!(Y, X)
    (p_sqrt, p_sqrt_inv) = hpd_sqrt_and_sqrt_inv(p)
    tv = Hermitian(p_sqrt_inv * X * p_sqrt_inv) # p^(-1/2)Xp^{-1/2}
    ty = Hermitian(p_sqrt_inv * convert(AbstractMatrix, q) * p_sqrt_inv) # p^(-1/2)qp^(-1/2)
    e2 = eigen(ty)
    Se = Diagonal(log.(max.(e2.values, floatmin(eltype(e2.values)))))
    U_e = e2.vectors
    logty = Hermitian(U_e * Se * U_e') # nearly log_pq without the outer p^1/2
    e3 = eigen(logty) # since they cancel with the pInvSqrt in the next line
    Sf = Diagonal(exp.(e3.values / 2)) # Uf * Sf * Uf' is the Exp
    Uf = e3.vectors
    pU_e = p_sqrt * Uf * Sf * Uf' # factors left of tv (and adjoint right)
    vtp = Hermitian(pU_e * tv * pU_e') # so this is the documented formula
    return copyto!(Y, vtp)
end

@doc raw"""
    riemannian_Hessian(M::HermitianPositiveDefinite, p, G, H, X)

The Riemannian Hessian can be computed as stated in Eq. (7.3) [Nguyen:2023](@cite),
which covers the real and the complex case.
Let ``\nabla f(p)`` denote the Euclidean gradient `G`,
``\nabla^2 f(p)[X]`` the Euclidean Hessian `H`, and
``\operatorname{sym}(X) = \frac{1}{2}\bigl(X^{\mathrm{H}}+X\bigr)``
the Hermitian part of ``X``, which is its symmetric part for ``𝔽=ℝ``. Then the formula reads

```math
    \operatorname{Hess}f(p)[X]
    =
    p\operatorname{sym}(∇^2 f(p)[X])p
    + \operatorname{sym}\bigl( X\operatorname{sym}\bigl(∇ f(p)\bigr)p)
```
"""
riemannian_Hessian(M::HermitianPositiveDefinite, p, G, H, X)

function riemannian_Hessian!(::HermitianPositiveDefinite, Y, p, G, H, X)
    symmetrize!(Y, G)
    symmetrize!(Y, X * Y * p)
    Z = symmetrize(H)
    Y .+= p * Z * p
    return Y
end

"""
    sectional_curvature_max(M::HermitianPositiveDefinite)

Return the upper bound 0 of the sectional curvature of the [`HermitianPositiveDefinite`](@ref) `M`.

With the [`AffineInvariantMetric`](@ref) the sectional curvature is non-positive,
see [Smith:2005](@cite), Eq. (28).
"""
function sectional_curvature_max(::HermitianPositiveDefinite)
    return 0.0
end
