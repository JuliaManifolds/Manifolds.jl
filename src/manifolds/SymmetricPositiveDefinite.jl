using LinearAlgebra: diag, eigen, eigvals, eigvecs, Symmetric, Diagonal, tr, cholesky, LowerTriangular
using ManifoldsBase: ParallelTransport
@doc doc"""
    SymmetricPositiveDefinite{N} <: Manifold

The manifold of symmetric positive definite matrices, i.e.

```math
\mathcal P(n) =
\bigl\{
x \in \mathbb R^{n\times n} :
\xi^\mathrm{T}x\xi > 0 \text{ for all } \xi \in \mathbb R^{n}\backslash\{0\}
\bigr\}
```

# Constructor

    SymmetricPositiveDefinite(n)

generates the manifold $\mathcal P(n) \subset \mathbb R^{n\times n}$
"""
struct SymmetricPositiveDefinite{N} <: Manifold end
SymmetricPositiveDefinite(n::Int) = SymmetricPositiveDefinite{n}()

@doc doc"""
    LinearAffineMetric <: Metric

The linear affine metric is the metric for symmetric positive definite matrices, that employs
matrix logarithms and exponentials, which yields a linear and affine metric.
"""
struct LinearAffineMetric <: RiemannianMetric end
is_default_metric(::SymmetricPositiveDefinite,::LinearAffineMetric) = Val(true)

@doc doc"""
    LogEuclideanMetric <: Metric

The LogEuclidean Metric consists of the Euclidean metric applied to all elements after mapping them
into the Lie Algebra, i.e. performing a matrix logarithm beforehand.
"""
struct LogEuclideanMetric <: RiemannianMetric end

@doc doc"""
    LogCholeskyMetric <: Metric

The Log-Cholesky metric imposes a metric based on the Cholesky decomposition as
introduced by [^Lin2019].

[^Lin2019]:
    > Lin, Zenhua: "Riemannian Geometry of Symmetric Positive Definite Matrices via
    > Cholesky Decomposition", arXiv: [1908.09326](https://arxiv.org/abs/1908.09326).
"""
struct LogCholeskyMetric <: RiemannianMetric end

cholesky_to_spd(l,w) = (l*l', w*l' + l*w')
tangent_cholesky_to_tangent_spd!(l,w) = (w .= w*l' + l*w')
spd_to_cholesky(x,v) = spd_to_cholesky(x,cholesky(x).L,v)
function spd_to_cholesky(x,l,v)
    w = inv(l)*v*inv(transpose(l))
    # strictly lower triangular plus half diagonal
    return (l, l*(LowerTriangular(w) - Diagonal(w)/2) )
end


@doc doc"""
    check_manifold_point(M,x; kwargs...)

checks, whether `x` is a valid point on the [`SymmetricPositiveDefinite`](@ref) `M`, i.e. is a matrix
of size `(N,N)`, symmetric and positive definite.
The tolerance for the second to last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::SymmetricPositiveDefinite{N},x; kwargs...) where N
    if size(x) != representation_size(M)
        return DomainError(size(x),"The point $(x) does not lie on $(M), since its size is not $(representation_size(M)).")
    end
    if !isapprox(norm(x-transpose(x)), 0.; kwargs...)
        return DomainError(norm(x), "The point $(x) does not lie on $(M) since its not a symmetric matrix:")
    end
    if ! all( eigvals(x) .> 0 )
        return DomainError(norm(x), "The point $x does not lie on $(M) since its not a positive definite matrix.")
    end
    return nothing
end
check_manifold_point(M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric},x; kwargs...) where N = check_manifold_point(M.manifold,x;kwargs...)

"""
    check_tangent_vector(M::SymmetricPositiveDefinite, x, v; kwargs... )

Check whether `v` is a tangent vector to `x` on the [`SymmetricPositiveDefinite`](@ref) `M`,
i.e. atfer [`check_manifold_point`](@ref)`(M,x)`, `v` has to be of same dimension as `x`
and a symmetric matrix, i.e. this stores tangent vetors as elements of the corresponding
Lie group. The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(M::SymmetricPositiveDefinite{N},x,v; kwargs...) where N
    mpe = check_manifold_point(M,x)
    mpe === nothing || return mpe
    if size(v) != representation_size(M)
        return DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $(M) since its size does not match $(representation_size(M)).")
    end
    if !isapprox(norm(v-transpose(v)), 0.; kwargs...)
        return DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $(M) (represented as an element of the Lie algebra) since its not symmetric.")
    end
    return nothing
end
function check_tangent_vector(M::MetricManifold{SymmetricPositiveDefinite{N},T},x,v; kwargs...) where {N, T <: Metric}
    return check_tangent_vector(base_manifold(M), x, v;kwargs...)
end

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
    distance(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, x, y)

Compute the distance on the manifold of [`SymmetricPositiveDefinite`](@ref)
nmatrices, i.e. between two symmetric positive definite matrices `x` and `y`
with respect to the [`LogCholeskyMetric`](@ref). The formula reads

````math
d_{\mathcal P(n)}(x,y) = \sqrt{
 \lVert \lfloor l \rfloor - \lfloor k \rfloor \rVert_{\mathrm{F}}^2
 + \lVert \log(\operatorname{diag}(l)) - \log(\operatorname{diag}(k))\rVert_{\mathrm{F}}^2 }\ \ ,
````

where $l$ and $k$ are the cholesky factors of $x$ and $y$, respectively,
$\lfloor\cdot\rfloor$ denbotes the strictly lower triangular matrix of its argument,
and $\lVert\cdot\rVert_{\mathrm{F}}$ denotes the Frobenius norm.
"""
distance(M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric},x,y) where N = distance(CholeskySpace{N}(), cholesky(x).L, cholesky(y).L)

@doc doc"""
    distance(M::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric}, x, y)

Compute the distance on the [`SymmetricPositiveDefinite`](@ref) manifold between
`x` and `y` as a [`MetricManifold`](@ref) with [`LogEuclideanMetric`](@ref).
The formula reads

```math
    d_{\mathcal P(n)}(x,y) = \lVert \Log x - \Log y \rVert_{\mathrm{F}}
```

where $\operatorname{Log}$ denotes the matrix logarithm and
$\lVert\cdot\rVert_{\mathrm{F}}$ denotes the matrix Frobenius norm.
"""
function distance(M::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric},x,y) where N
    return norm(log(Symmetric(x)) - log(Symmetric(y)))
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
exp!(M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, args...) where {N} = exp!(base_manifold(M), args...)

@doc doc"""
    exp(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, x, v)

Compute the exponential map on the [`SymmetricPositiveDefinite`](@ref) `M` with
[`LogCholeskyMetric`](@ref) from `x` into direction `v`. The formula reads

````math
\exp_x v = (\exp_l w)(\exp_l w)^\mathrm{T}
````

where $\exp_lw$ is the exponential map on [`CholeskySpace`](@ref), $l$ is the cholesky
decomposition of $x$, $w = l(l^{-1}vl^{-\mathrm{T}})_\frac{1}{2}$,
and $(\cdot)_\frac{1}{2}$
denotes the lower triangular matrix with the diagonal multiplied by $\frac{1}{2}$.
"""
exp(::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, ::Any...)
function exp!(M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric}, y, x, v) where {N}
    (l,w) = spd_to_cholesky(x,v)
    z = exp(CholeskySpace{N}(),l,w)
    y .= z*z'
    return y
end

@doc doc"""
    injectivity_radius(M::SymmetricPositiveDefinite[, x])
    injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}[, x])
    injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}[, x])

Return the injectivity radius of the [`SymmetricPositiveDefinite`](@ref).
Since `M` is a Hadamard manifold with respect to the [`LinearAffineMetric`](@ref) and the
[`LogCholeskyMetric`](@ref), the injectivity radius is globally $\infty$.
"""
injectivity_radius(M::SymmetricPositiveDefinite{N}, args...) where N = Inf
injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, args...) where N = Inf
injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric}, args...) where N = Inf

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
inner(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}, x,v,w) = inner(base_manifold(M),x,v,w)

@doc doc"""
    inner(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, x, v, w)

Compute the inner product of two matrices `v`, `w` in the tangent space of `x`
on the [`SymmetricPositiveDefinite`](@ref) manifold `M`, as
a [`MetricManifold`](@ref) with [`LogCholeskyMetric`](@ref). The formula reads

````math
    (v,w)_x = (p_l(w),p_l(v))_l,
````

where the right hand side is the inner product on the [`CholeskySpace`](@ref),
$l$ is the cholesky factor of $x$,
$p_l(w) = l (l^{-1}wl^{-\mathrm{T}})_{\frac{1}{2}}$, and $(\cdot)_\frac{1}{2}$
denotes the lower triangular matrix with the diagonal multiplied by $\frac{1}{2}$
"""
function inner(M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric},x,v,w) where N
    (l,vl) = spd_to_cholesky(x,v)
    (l,wl) = spd_to_cholesky(x,l,w)
    return inner(CholeskySpace{N}(), l, vl, wl)
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
log!(M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, args...) where {N} = log!(base_manifold(M), args...)

@doc doc"""
    log(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, x, y)

Compute the logarithmic map on [`SymmetricPositiveDefinite`](@ref) `M` with
respect to the [`LogCholeskyMetric`](@ref) eminating from `x` to `y`.
The formula can be adapted from the [`CholeskySpace`](@ref) as
````math
\log_x y = lw^{\mathrm{T}} + wl^{\mathrm{T}},
````
where $l$ is the colesky factor of $x$ and $w=\log_lk$ for $k$ the cholesky factor
of $y$ and the just mentioned logarithmic map is the one on [`CholeskySpace`](@ref).
"""
log(::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, ::Any...)
function log!(M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric}, v, x, y) where N
    l = cholesky(x).L
    k = cholesky(y).L
    log!(CholeskySpace{N}(), v, l, k)
    tangent_cholesky_to_tangent_spd!(l, v)
    return v
end

@doc doc"""
    manifold_dimension(M::SymmetricPositiveDefinite)
    manifold_dimension(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric})
    manifold_dimension(M::MetricManifold{SymmetricPositiveDefinite,LogEuclideanMetric})

returns the dimension of
[`SymmetricPositiveDefinite`](@ref) `M`$=\mathcal P(n), n\in \mathbb N$, i.e.
````math
\operatorname{dim}_{\mathcal P(n)} \frac{n(n+1)}{2}
````
"""
@generated manifold_dimension(M::SymmetricPositiveDefinite{N}) where {N} = div(N*(N+1), 2)
@generated manifold_dimension(M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric}) where {N} = div(N*(N+1), 2)
@generated manifold_dimension(M::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric}) where {N} = div(N*(N+1), 2)

"""
    mean(
        M::SymmetricPositiveDefinite,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolation();
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` using
[`GeodesicInterpolation`](@ref).
"""
mean(::SymmetricPositiveDefinite, ::Any)
mean!(M::SymmetricPositiveDefinite, y, x::AbstractVector, w::AbstractVector; kwargs...) =
    mean!(M, y, x, w, GeodesicInterpolation(); kwargs...)

norm(M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric}, x,v) where {N} = norm(base_manifold(M), x,v)
norm(M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric}, x,v) where {N} = sqrt(inner(M,x,v,v))

@doc doc"""
    representation_size(M::SymmetricPositiveDefinite)
    representation_size(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric})
    representation_size(M::MetricManifold{SymmetricPositiveDefinite,LogEuclideanMetric})

Return the size of an array representing an element on the
[`SymmetricPositiveDefinite`](@ref) manifold `M`, i.e. $n\times n$, the size of such a
symmetric positive definite matrix on $\mathcal M = \mathcal P(n)$.
"""
representation_size(::SymmetricPositiveDefinite{N}) where N = (N,N)
representation_size(::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric}) where N = (N,N)
representation_size(::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric}) where N = (N,N)

@doc doc"""
    [Ξ,κ] = tangent_orthonormal_basis(M::SymmetricPositiveDefinite, x, v)

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

@doc doc"""
    vector_transport_to(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, x, v, y, ::ParallelTransport)

Parallely transport the tangent vector `v` at `x` along the geodesic to `y` with respect to
the [`SymmetricPositiveDefinite`](@ref) manifold `M` and [`LogCholeskyMetric`](@ref).
The parallel transport is based on the parallel transport on [`CholeskySpace`](@ref):
Let $l$ and $k$ denote the cholesky factors of `x` and `y`, respectively and
$w = l(l^{-1}vl^{-\mathrm{T}})_\frac{1}{2}$, where $(\cdot)_\frac{1}{2}$ denotes the lower
triangular matrix with the diagonal multiplied by $\frac{1}{2}$. With $u$ the parallel
transport on [`CholeskySpace`](@ref) from $l$ to $k$. The formula hear reads

````math
    \mathcal P_{y\gets x}(v) = ku^{\mathrm{T}} + uk^{\mathrm{T}}.
````
"""
vector_transport_to(::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, ::Any, ::Any, ::Any, ::ParallelTransport)
function vector_transport_to!(M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric}, vto, x, v, y, ::ParallelTransport) where N
    k = cholesky(y).L
    (l,w) = spd_to_cholesky(x,v)
    vector_transport_to!(CholeskySpace{N}(),vto,l , w , k, ParallelTransport())
    tangent_cholesky_to_tangent_spd!(k,vto)
    return vto
end

@doc doc"""
    zero_tangent_vector(M,x)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `x` on the [`SymmetricPositiveDefinite`](@ref) manifold `M`.
"""
zero_tangent_vector(M::SymmetricPositiveDefinite, x) = zero(x)
zero_tangent_vector(M::MetricManifold{SymmetricPositiveDefinite{N},T}, x) where {N, T<:Metric} = zero(x)
zero_tangent_vector!(M::SymmetricPositiveDefinite{N}, v, x) where N = fill!(v, 0)
zero_tangent_vector!(M::MetricManifold{SymmetricPositiveDefinite{N},T}, v, x) where {N, T<:Metric} = fill!(v, 0)
