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
