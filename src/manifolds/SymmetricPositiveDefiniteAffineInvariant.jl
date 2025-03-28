@doc raw"""
    AffineInvariantMetric <: AbstractMetric

The linear affine metric is the metric for symmetric positive definite matrices, that employs
matrix logarithms and exponentials, which yields a linear and affine metric.
"""
struct AffineInvariantMetric <: RiemannianMetric end

@doc raw"""
    change_representer(M::SymmetricPositiveDefinite, E::EuclideanMetric, p, X)

Given a tangent vector ``X ∈ T_p\mathcal M`` representing a linear function on the tangent
space at `p` with respect to the [`EuclideanMetric`](@extref `ManifoldsBase.EuclideanMetric`)
`g_E`, this is turned into the representer with respect to the (default) metric,
the [`AffineInvariantMetric`](@ref) on the [`SymmetricPositiveDefinite`](@ref) `M`.

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
    change_metric(M::SymmetricPositiveDefinite, E::EuclideanMetric, p, X)

Given a tangent vector ``X ∈ T_p\mathcal P(n)`` with respect to the [`EuclideanMetric`](@extref `ManifoldsBase.EuclideanMetric`)
`g_E`, this function changes into the [`AffineInvariantMetric`](@ref) (default) metric on the
[`SymmetricPositiveDefinite`](@ref) `M`.

To be precise we are looking for ``c\colon T_p\mathcal P(n) → T_p\mathcal P(n) ``
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

@doc raw"""
    distance(M::SymmetricPositiveDefinite, p, q)
    distance(M::MetricManifold{SymmetricPositiveDefinite,AffineInvariantMetric}, p, q)

Compute the distance on the [`SymmetricPositiveDefinite`](@ref) manifold between `p` and `q`,
as a [`MetricManifold`](@ref) with [`AffineInvariantMetric`](@ref). The formula reads

```math
d_{\mathcal P(n)}(p,q)
= \lVert \operatorname{Log}(p^{-\frac{1}{2}}qp^{-\frac{1}{2}})\rVert_{\mathrm{F}}.,
```
where ``\operatorname{Log}`` denotes the matrix logarithm and
``\lVert⋅\rVert_{\mathrm{F}}`` denotes the matrix Frobenius norm.
"""
function distance(::SymmetricPositiveDefinite, p, q)
    # avoid numerical instabilities in cholesky
    norm(p - q) < eps(eltype(p + q)) && return zero(eltype(p + q))
    cq = cholesky(Symmetric(q)) # to avoid numerical inaccuracies
    s = eigvals(Symmetric(cq.L \ p / cq.U))
    return any(s .<= eps()) ? zero(eltype(p)) : sqrt(sum(abs.(log.(s)) .^ 2))
end
function distance(M::SymmetricPositiveDefinite, p::SPDPoint, q::SPDPoint)
    return distance(M, convert(AbstractMatrix, p), convert(AbstractMatrix, q))
end

@doc raw"""
    exp(M::SymmetricPositiveDefinite, p, X)
    exp(M::MetricManifold{<:SymmetricPositiveDefinite,AffineInvariantMetric}, p, X)

Compute the exponential map from `p` with tangent vector `X` on the
[`SymmetricPositiveDefinite`](@ref) `M` with its default [`MetricManifold`](@ref) having the
[`AffineInvariantMetric`](@ref). The formula reads

```math
\exp_p X = p^{\frac{1}{2}}\operatorname{Exp}(p^{-\frac{1}{2}} X p^{-\frac{1}{2}})p^{\frac{1}{2}},
```

where ``\operatorname{Exp}`` denotes to the matrix exponential.
"""
exp(::SymmetricPositiveDefinite, ::Any...)

function exp(::SymmetricPositiveDefinite, p::SPDPoint, X)
    (p_sqrt, p_sqrt_inv) = spd_sqrt_and_sqrt_inv(p)
    T = Symmetric(p_sqrt_inv * X * p_sqrt_inv)
    eig1 = eigen(T) # numerical stabilization
    Se = Diagonal(exp.(eig1.values))
    Ue = eig1.vectors
    pUe = p_sqrt * Ue
    q = SPDPoint(
        pUe * Se * transpose(pUe),
        store_p=!ismissing(p.p),
        store_sqrt=!ismissing(p.sqrt),
        store_sqrt_inv=!ismissing(p.sqrt_inv),
    )
    return q
end

function exp_fused!(M::SymmetricPositiveDefinite, q, p, X, t::Number)
    return exp!(M, q, p, t * X)
end
function exp!(::SymmetricPositiveDefinite, q, p, X)
    (p_sqrt, p_sqrt_inv) = spd_sqrt_and_sqrt_inv(p)
    T = Symmetric(p_sqrt_inv * X * p_sqrt_inv)
    eig1 = eigen(T) # numerical stabilization
    Se = Diagonal(exp.(eig1.values))
    Ue = eig1.vectors
    pUe = p_sqrt * Ue
    return copyto!(q, pUe * Se * transpose(pUe))
end
function exp!(::SymmetricPositiveDefinite, q::SPDPoint, p, X)
    (p_sqrt, p_sqrt_inv) = spd_sqrt_and_sqrt_inv(p)
    T = Symmetric(p_sqrt_inv * X * p_sqrt_inv)
    eig1 = eigen(T) # numerical stabilization
    Se = Diagonal(exp.(eig1.values))
    Ue = eig1.vectors
    pUe = p_sqrt * Ue
    Q = pUe * Se * transpose(pUe)
    !ismissing(q.p) && copyto!(q.p, Q)
    Q_e = eigen(Q)
    copyto!(q.eigen.values, Q_e.values)
    copyto!(q.eigen.vectors, Q_e.vectors)
    if !ismissing(q.sqrt) && !ismissing(q.sqrt_inv)
        copyto!.([q.sqrt, q.sqrt_inv], spd_sqrt_and_sqrt_inv(Q))
    else
        !ismissing(q.sqrt) && copyto!(q.sqrt, spd_sqrt(Q))
        !ismissing(q.sqrt_inv) && copyto!(q.sqrt_inv, spd_sqrt_inv(Q))
    end
    return q
end

@doc raw"""
    [Ξ,κ] = get_basis_diagonalizing(M::SymmetricPositiveDefinite, p, B::DiagonalizingOrthonormalBasis)
    [Ξ,κ] = get_basis_diagonalizing(M::MetricManifold{<:SymmetricPositiveDefinite,AffineInvariantMetric}, p, B::DiagonalizingOrthonormalBasis)

Return a orthonormal basis `Ξ` as a vector of tangent vectors (of length
[`manifold_dimension`](@ref) of `M`) in the tangent space of `p` on the
[`MetricManifold`](@ref) of [`SymmetricPositiveDefinite`](@ref) manifold `M` with
[`AffineInvariantMetric`](@ref) that diagonalizes the curvature tensor ``R(u,v)w``
with eigenvalues `κ` and where the direction `B.frame_direction` ``V`` has curvature `0`.

The construction is based on an ONB for the symmetric matrices similar to [`get_basis(::SymmetricPositiveDefinite, p, ::DefaultOrthonormalBasis`](@ref  get_basis(M::SymmetricPositiveDefinite,p,B::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType}))
just that the ONB here is build from the eigen vectors of ``p^{\frac{1}{2}}Vp^{\frac{1}{2}}``.
"""
function get_basis_diagonalizing(
    M::SymmetricPositiveDefinite,
    p,
    B::DiagonalizingOrthonormalBasis,
)
    N = get_parameter(M.size)[1]
    (p_sqrt, p_sqrt_inv) = spd_sqrt_and_sqrt_inv(p)
    eigv = eigen(Symmetric(p_sqrt_inv * B.frame_direction * p_sqrt_inv))
    V = eigv.vectors
    Ξ = [
        (i == j ? 1 / 2 : 1 / sqrt(2)) *
        p_sqrt *
        (V[:, i] * transpose(V[:, j]) + V[:, j] * transpose(V[:, i])) *
        p_sqrt for i in 1:N for j in i:N
    ]
    λ = eigv.values
    κ = [-1 / 4 * (λ[i] - λ[j])^2 for i in 1:N for j in i:N]
    return CachedBasis(B, κ, Ξ)
end

@doc raw"""
    [Ξ,κ] = get_basis(M::SymmetricPositiveDefinite, p, B::DefaultOrthonormalBasis)
    [Ξ,κ] = get_basis(M::MetricManifold{<:SymmetricPositiveDefinite,AffineInvariantMetric}, p, B::DefaultOrthonormalBasis)

Return a default ONB for the tangent space ``T_p\mathcal P(n)`` of the [`SymmetricPositiveDefinite`](@ref) with respect to the [`AffineInvariantMetric`](@ref).

```math
    g_p(X,Y) = \operatorname{tr}(p^{-1} X p^{-1} Y),
```

The basis constructed here is based on the ONB for symmetric matrices constructed as follows.
Let

```math
\Delta_{i,j} = (a_{k,l})_{k,l=1}^n \quad \text{ with }
a_{k,l} =
\begin{cases}
  1 & \mbox{ for } k=l \text{ if } i=j\\
  \frac{1}{\sqrt{2}} & \mbox{ for } k=i, l=j \text{ or } k=j, l=i\\
  0 & \text{ else.}
\end{cases}
```

which forms an ONB for the space of symmetric matrices.

We then form the ONB by
```math
   \Xi_{i,j} = p^{\frac{1}{2}}\Delta_{i,j}p^{\frac{1}{2}},\qquad i=1,\ldots,n, j=i,\ldots,n.
```
"""
get_basis(::SymmetricPositiveDefinite, p, B::DefaultOrthonormalBasis)

function get_basis_orthonormal(M::SymmetricPositiveDefinite, p, Ns::RealNumbers)
    N = get_parameter(M.size)[1]
    p_sqrt = spd_sqrt(p)
    Ξ = [similar(convert(AbstractMatrix, p)) for _ in 1:manifold_dimension(M)]
    k = 1
    for i in 1:N, j in i:N
        fill!(Ξ[k], zero(eltype(Ξ[k])))
        s = i == j ? 1 / 2 : 1 / sqrt(2)
        @inbounds Ξ[k][i, j] += 1
        @inbounds Ξ[k][j, i] += 1
        @inbounds Ξ[k] .= s * p_sqrt * Ξ[k] * p_sqrt
        k += 1
    end
    return CachedBasis(DefaultOrthonormalBasis(Ns), Ξ)
end

@doc raw"""
    get_coordinates(::SymmetricPositiveDefinite, p, X, ::DefaultOrthonormalBasis)

Using the basis from [`get_basis`](@ref get_basis(M::SymmetricPositiveDefinite,p,B::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType}))
the coordinates with respect to this ONB can be simplified to

```math
   c_k = \mathrm{tr}(p^{-\frac{1}{2}}\Delta_{i,j} X)
```
where ``k`` is trhe linearized index of the ``i=1,\ldots,n, j=i,\ldots,n``.
"""
get_coordinates(::SymmetricPositiveDefinite, c, p, X, ::DefaultOrthonormalBasis)

function get_coordinates_orthonormal!(M::SymmetricPositiveDefinite, c, p, X, ::RealNumbers)
    N = get_parameter(M.size)[1]
    dim = manifold_dimension(M)
    @assert size(c) == (dim,)
    @assert size(X) == (N, N)
    @assert dim == div(N * (N + 1), 2)
    p_sqrt = spd_sqrt(p)
    pM = convert(AbstractMatrix, p)
    k = 1
    V = similar(pM)
    fill!(V, zero(eltype(V)))
    F = cholesky(Symmetric(pM))
    for i in 1:N, j in i:N
        s = i == j ? 1 / 2 : 1 / sqrt(2)
        @inbounds V[i, j] += 1
        @inbounds V[j, i] += 1
        Yij = p_sqrt * V * p_sqrt
        @inbounds c[k] = s * dot(F \ Symmetric(X), (Symmetric(Yij) / F))
        k += 1
        @inbounds V[i, j] = 0
        @inbounds V[j, i] = 0
    end
    return c
end

@doc raw"""
    get_vector(::SymmetricPositiveDefinite, p, c, ::DefaultOrthonormalBasis)

Using the basis from [`get_basis`](@ref  get_basis(M::SymmetricPositiveDefinite,p,B::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType}))
the vector reconstruction with respect to this ONB can be simplified to

```math
   X = p^{\frac{1}{2}} \Biggl( \sum_{i=1,j=i}^n c_k \Delta_{i,j} \Biggr) p^{\frac{1}{2}}
```
where ``k`` is the linearized index of the ``i=1,\ldots,n, j=i,\ldots,n``.
"""
get_vector(::SymmetricPositiveDefinite, X, p, c, ::DefaultOrthonormalBasis)

function get_vector_orthonormal!(M::SymmetricPositiveDefinite, X, p, c, ::RealNumbers)
    N = get_parameter(M.size)[1]
    @assert size(c) == (div(N * (N + 1), 2),)
    @assert size(X) == (N, N)
    p_sqrt = spd_sqrt(p)
    X .= 0
    k = 1
    V = similar(convert(AbstractMatrix, p))
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
    end
    return X
end

@doc raw"""
    inner(M::SymmetricPositiveDefinite, p, X, Y)
    inner(M::MetricManifold{SymmetricPositiveDefinite,AffineInvariantMetric}, p, X, Y)

Compute the inner product of `X`, `Y` in the tangent space of `p` on
the [`SymmetricPositiveDefinite`](@ref) manifold `M`, as
a [`MetricManifold`](@ref) with [`AffineInvariantMetric`](@ref). The formula reads

````math
g_p(X,Y) = \operatorname{tr}(p^{-1} X p^{-1} Y),
````
"""
function inner(::SymmetricPositiveDefinite, p, X, Y)
    F = cholesky(Symmetric(convert(AbstractMatrix, p)))
    return dot((F \ Symmetric(X)), (Symmetric(Y) / F))
end

"""
    is_flat(::MetricManifold{ℝ,<:SymmetricPositiveDefinite,AffineInvariantMetric})

Return false. [`SymmetricPositiveDefinite`](@ref) with [`AffineInvariantMetric`](@ref)
is not a flat manifold.
"""
is_flat(M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,AffineInvariantMetric}) = false

@doc raw"""
    log(M::SymmetricPositiveDefinite, p, q)
    log(M::MetricManifold{SymmetricPositiveDefinite,AffineInvariantMetric}, p, q)

Compute the logarithmic map from `p` to `q` on the [`SymmetricPositiveDefinite`](@ref)
as a [`MetricManifold`](@ref) with [`AffineInvariantMetric`](@ref). The formula reads

```math
\log_p q =
p^{\frac{1}{2}}\operatorname{Log}(p^{-\frac{1}{2}}qp^{-\frac{1}{2}})p^{\frac{1}{2}},
```
where ``\operatorname{Log}`` denotes to the matrix logarithm.
"""
log(::SymmetricPositiveDefinite, ::Any...)

function allocate_result(
    M::SymmetricPositiveDefinite,
    ::typeof(log),
    q::SPDPoint,
    p::SPDPoint,
)
    return allocate_result(M, log, convert(AbstractMatrix, q), convert(AbstractMatrix, p))
end

function log!(::SymmetricPositiveDefinite, X, p, q)
    (p_sqrt, p_sqrt_inv) = spd_sqrt_and_sqrt_inv(p)
    T = Symmetric(p_sqrt_inv * convert(AbstractMatrix, q) * p_sqrt_inv)
    e2 = eigen(T)
    Se = Diagonal(log.(max.(e2.values, eps())))
    pUe = p_sqrt * e2.vectors
    return mul!(X, pUe, Se * transpose(pUe))
end

"""
    manifold_volume(::SymmetricPositiveDefinite)

Return volume of the [`SymmetricPositiveDefinite`](@ref) manifold, i.e. infinity.
"""
manifold_volume(::SymmetricPositiveDefinite) = Inf

@doc raw"""
    parallel_transport_to(M::SymmetricPositiveDefinite, p, X, q)
    parallel_transport_to(M::MetricManifold{SymmetricPositiveDefinite,AffineInvariantMetric}, p, X, y)

Compute the parallel transport of `X` from the tangent space at `p` to the
tangent space at `q` on the [`SymmetricPositiveDefinite`](@ref) as a
[`MetricManifold`](@ref) with the [`AffineInvariantMetric`](@ref).
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
and `log` the logarithmic map on [`SymmetricPositiveDefinite`](@ref)
(again with respect to the [`AffineInvariantMetric`](@ref)).
"""
parallel_transport_to(::SymmetricPositiveDefinite, ::Any, ::Any, ::Any)

function parallel_transport_to!(M::SymmetricPositiveDefinite, Y, p, X, q)
    distance(M, p, q) < 2 * eps(eltype(p)) && copyto!(Y, X)
    (p_sqrt, p_sqrt_inv) = spd_sqrt_and_sqrt_inv(p)
    tv = Symmetric(p_sqrt_inv * X * p_sqrt_inv) # p^(-1/2)Xp^{-1/2}
    ty = Symmetric(p_sqrt_inv * convert(AbstractMatrix, q) * p_sqrt_inv) # p^(-1/2)qp^(-1/2)
    e2 = eigen(ty)
    Se = Diagonal(log.(max.(e2.values, floatmin(eltype(e2.values)))))
    Ue = e2.vectors
    logty = Symmetric(Ue * Se * transpose(Ue)) # nearly log_pq without the outer p^1/2
    e3 = eigen(logty) # since they cancel with the pInvSqrt in the next line
    Sf = Diagonal(exp.(e3.values / 2)) # Uf * Sf * Uf' is the Exp
    Uf = e3.vectors
    pUe = p_sqrt * Uf * Sf * transpose(Uf) # factors left of tv (and transposed right)
    vtp = Symmetric(pUe * tv * transpose(pUe)) # so this is the documented formula
    return copyto!(Y, vtp)
end

@doc raw"""
    riemannian_Hessian(M::SymmetricPositiveDefinite, p, G, H, X)

The Riemannian Hessian can be computed as stated in Eq. (7.3) [Nguyen:2023](@cite).
Let ``\nabla f(p)`` denote the Euclidean gradient `G`,
``\nabla^2 f(p)[X]`` the Euclidean Hessian `H`, and
``\operatorname{sym}(X) = \frac{1}{2}\bigl(X^{\mathrm{T}}+X\bigr)``
the symmetrization operator. Then the formula reads

```math
    \operatorname{Hess}f(p)[X]
    =
    p\operatorname{sym}(∇^2 f(p)[X])p
    + \operatorname{sym}\bigl( X\operatorname{sym}\bigl(∇ f(p)\bigr)p)
```
"""
riemannian_Hessian(M::SymmetricPositiveDefinite, p, G, H, X)

function riemannian_Hessian!(::SymmetricPositiveDefinite, Y, p, G, H, X)
    symmetrize!(Y, G)
    symmetrize!(Y, X * Y * p)
    Z = symmetrize(H)
    Y .+= p * Z * p
    return Y
end

@doc raw"""
    riemann_tensor(::SymmetricPositiveDefinite, p, X, Y, Z)

Compute the value of Riemann tensor on the [`SymmetricPositiveDefinite`](@ref) manifold.
The formula reads [Rentmeesters:2011](@cite) ``R(X,Y)Z=p^{1/2}R(X_I, Y_I)Z_Ip^{1/2}``, where
``R_I(X_I, Y_I)Z_I=\frac{1}{4}[Z_I, [X_I, Y_I]]``,  ``X_I=p^{-1/2}Xp^{-1/2}``,
``Y_I=p^{-1/2}Yp^{-1/2}`` and ``Z_I=p^{-1/2}Zp^{-1/2}``.
"""
riemann_tensor(::SymmetricPositiveDefinite, p, X, Y, Z)

function riemann_tensor!(::SymmetricPositiveDefinite, Xresult, p, X, Y, Z)
    ps = sqrt(p)
    ips = inv(ps)
    XI = ips * X * ips
    YI = ips * Y * ips
    ZI = ips * Z * ips
    Xtmp = XI * YI - YI * XI
    Xresult .= ps * (1 // 4 .* (ZI * Xtmp .- Xtmp * ZI)) * ps
    return Xresult
end

"""
    sectional_curvature_min(M::SymmetricPositiveDefinite)

Return minimum sectional curvature of [`SymmetricPositiveDefinite`](@ref) manifold,
that is 0 for SPD(1) and SPD(2) and -0.25 otherwise.
"""
function sectional_curvature_min(M::SymmetricPositiveDefinite)
    if manifold_dimension(M) < 2
        return 0.0
    else
        return -0.25
    end
end

"""
    sectional_curvature_max(M::SymmetricPositiveDefinite)

Return minimum sectional curvature of [`SymmetricPositiveDefinite`](@ref) manifold,
that is 0.
"""
function sectional_curvature_max(::SymmetricPositiveDefinite)
    return 0.0
end

"""
    volume_density(::SymmetricPositiveDefinite, p, X)

Compute the volume density of the [`SymmetricPositiveDefinite`](@ref) manifold at `p`
in direction `X`. See [ChevallierKalungaAngulo:2017](@cite), Section 6.2 for details.
Note that metric in Manifolds.jl has a different scaling factor than the reference.
"""
function volume_density(::SymmetricPositiveDefinite, p, X)
    eig = eigvals(X)
    dens = 1.0
    for i in 1:length(eig)
        for j in (i + 1):length(eig)
            absdiff = abs(eig[i] - eig[j])
            if absdiff > eps(absdiff)
                dens *= sinh(absdiff) / absdiff
            end
        end
    end
    return dens
end
