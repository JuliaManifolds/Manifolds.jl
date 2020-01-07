@doc doc"""
    LinearAffineMetric <: Metric

The linear affine metric is the metric for symmetric positive definite matrices, that employs
matrix logarithms and exponentials, which yields a linear and affine metric.
"""
struct LinearAffineMetric <: RiemannianMetric end

@doc doc"""
    distance(M::SymmetricPositiveDefinite, x, y)
    distance(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric})

Compute the distance on the [`SymmetricPositiveDefinite`](@ref) manifold between `x` and `y`,
as a [`MetricManifold`](@ref) with [`LinearAffineMetric`](@ref). The formula reads

```math
d_{\mathcal P(n)}(x,y)
= \lVert \operatorname{Log}(x^{-\frac{1}{2}}yx^{-\frac{1}{2}})\rVert_{\mathrm{F}}.,
```
where $\operatorname{Log}$ denotes the matrix logarithm and
$\lVert\cdot\rVert_{\mathrm{F}}$ denotes the matrix Frobenius norm.
"""
function distance(M::SymmetricPositiveDefinite{N},x,y) where N
    s = real.( eigvals( x,y ) )
    return any(s .<= eps() ) ? 0 : sqrt(  sum( abs.(log.(s)).^2 )  )
end

@doc doc"""
    exp(M::SymmetricPositiveDefinite, x, v)
    exp(M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, x, v)

Compute the exponential map from `x` with tangent vector `v` on the
[`SymmetricPositiveDefinite`](@ref) `M` with its default [`MetricManifold`](@ref) having the
[`LinearAffineMetric`](@ref). The formula reads

```math
\exp_x v = x^{\frac{1}{2}}\operatorname{Exp}(x^{-\frac{1}{2}} v x^{-\frac{1}{2}})x^{\frac{1}{2}},
```

where $\operatorname{Exp}$ denotes to the matrix exponential.
"""
exp(::SymmetricPositiveDefinite, ::Any...)
function exp!(M::SymmetricPositiveDefinite{N}, y, x, v) where N
    e = eigen(Symmetric(x))
    U = e.vectors
    S = e.values
    Ssqrt = Diagonal( sqrt.(S) )
    SsqrtInv = Diagonal( 1 ./ sqrt.(S) )
    xSqrt = Symmetric(U*Ssqrt*transpose(U))
    xSqrtInv = Symmetric(U*SsqrtInv*transpose(U))
    T = Symmetric(xSqrtInv * v * xSqrtInv)
    eig1 = eigen( T ) # numerical stabilization
    Se = Diagonal( exp.(eig1.values) )
    Ue = eig1.vectors
    xue = xSqrt*Ue
    copyto!(y, xue*Se*transpose(xue) )
    return y
end

@doc doc"""
    inner(M::SymmetricPositiveDefinite, x, v, w)
    inner(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}, x, v, w)

Compute the inner product of `v`, `w` in the tangent space of `x` on
the [`SymmetricPositiveDefinite`](@ref) manifold `M`, as
a [`MetricManifold`](@ref) with [`LinearAffineMetric`](@ref). The formula reads

````math
(v, w)_x = \operatorname{tr}(x^{-1} v x^{-1} w),
````
"""
function inner(M::SymmetricPositiveDefinite, x, v, w)
    F = cholesky(Symmetric(x))
    return tr((F \ Symmetric(v)) * (F \ Symmetric(w)))
end

@doc doc"""
    log(M::SymmetricPositiveDefinite, x, y)
    log(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}, x, y)

Compute the logarithmic map from `x` to `y` on the [`SymmetricPositiveDefinite`](@ref)
as a [`MetricManifold`](@ref) with [`LinearAffineMetric`](@ref). The formula reads

```math
\log_x y =
x^{\frac{1}{2}}\operatorname{Log}(x^{-\frac{1}{2}}yx^{-\frac{1}{2}})x^{\frac{1}{2}},
```
where $\operatorname{Log}$ denotes to the matrix logarithm.
"""
log(::SymmetricPositiveDefinite, ::Any...)
function log!(M::SymmetricPositiveDefinite{N}, v, x, y) where N
    e = eigen(Symmetric(x))
    U = e.vectors
    S = e.values
    Ssqrt = Diagonal( sqrt.(S) )
    SsqrtInv = Diagonal( 1 ./ sqrt.(S) )
    xSqrt = Symmetric( U*Ssqrt*transpose(U) )
    xSqrtInv = Symmetric( U*SsqrtInv*transpose(U) )
    T = Symmetric( xSqrtInv * y * xSqrtInv )
    e2 = eigen( T )
    Se = Diagonal( log.(max.(e2.values,eps()) ) )
    xue = xSqrt*e2.vectors
    mul!(v,xue,Se*transpose(xue))
    return v
end

@doc doc"""
    [Ξ,κ] = tangent_orthonormal_basis(M::SymmetricPositiveDefinite, x, v)
    [Ξ,κ] = tangent_orthonormal_basis(M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, x, v)

Return a orthonormal basis `Ξ` as a vector of tangent vectors (of length
[`manifold_dimension`](@ref) of `M`) in the tangent space of `x` on the
[`MetricManifold`](@ref) of [`SymmetricPositiveDefinite`](@ref) manifold `M` with
[`LinearAffineMetric`](@ref) that diagonalizes the curvature tensor $R(u,v)w$
with eigenvalues `κ` and where the direction `v` has curvature `0`.
"""
function tangent_orthonormal_basis(M::SymmetricPositiveDefinite{N},x,v) where N
    xSqrt = sqrt(x)
    V = eigvecs(v)
    Ξ = [ (i==j ? 1/2 : 1/sqrt(2))*( V[:,i] * transpose(V[:,j])  +  V[:,j] * transpose(V[:,i]) )
        for i=1:N for j= i:N
    ]
    λ = eigvals(v)
    κ = [ -1/4 * (λ[i]-λ[j])^2 for i=1:N for j= i:N ]
  return Ξ,κ
end
tangent_orthonormal_basis(M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},args...) where {N} = tangent_orthonormal_basis(base_manifold(M), args...)

@doc doc"""
    vector_transport_to(M::SymmetricPositiveDefinite, x, v, y, ::ParallelTransport)
    vector_transport_to(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}, x, v, y, ::ParallelTransport)

Compute the parallel transport on the [`SymmetricPositiveDefinite`](@ref) as a
[`MetricManifold`](@ref) with the [`LinearAffineMetric`](@ref).
The formula reads

```math
P_{y\gets x}(v) = x^{\frac{1}{2}}
\operatorname{Exp}\bigl(
\frac{1}{2}x^{-\frac{1}{2}}\log_x(y)x^{-\frac{1}{2}}
\bigr)
x^{-\frac{1}{2}}v x^{-\frac{1}{2}}
\operatorname{Exp}\bigl(
\frac{1}{2}x^{-\frac{1}{2}}\log_x(y)x^{-\frac{1}{2}}
\bigr)
x^{\frac{1}{2}},
```

where $\operatorname{Exp}$ denotes the matrix exponential
and `log` the logarithmic map on [`SymmetricPositiveDefinite`](@ref)
(again with respect to the metric mentioned).
"""
vector_transport_to(::SymmetricPositiveDefinite, ::Any, ::Any, ::Any, ::ParallelTransport)
function vector_transport_to!(M::SymmetricPositiveDefinite{N}, vto, x, v, y, ::ParallelTransport) where N
    if distance(M,x,y)<2*eps(eltype(x))
        copyto!(vto, v)
        return vto
    end
    e = eigen(Symmetric(x))
    U = e.vectors
    S = e.values
    Ssqrt = sqrt.(S)
    SsqrtInv = Diagonal( 1 ./ Ssqrt )
    Ssqrt = Diagonal( Ssqrt )
    xSqrt = Symmetric(U*Ssqrt*transpose(U))
    xSqrtInv = Symmetric(U*SsqrtInv*transpose(U))
    tv = Symmetric(xSqrtInv * v * xSqrtInv)
    ty = Symmetric(xSqrtInv * y * xSqrtInv)
    e2 = eigen( ty )
    Se = Diagonal( log.(e2.values) )
    Ue = e2.vectors
    ty2 = Symmetric(Ue*Se*transpose(Ue))
    e3 = eigen( ty2 )
    Sf = Diagonal( exp.(e3.values) )
    Uf = e3.vectors
    xue = xSqrt*Uf*Sf*transpose(Uf)
    vtp = Symmetric(xue*tv*transpose(xue))
    copyto!(vto, vtp)
    return vto
end
