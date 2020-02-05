@doc raw"""
    LogCholeskyMetric <: Metric

The Log-Cholesky metric imposes a metric based on the Cholesky decomposition as
introduced by [^Lin2019].

[^Lin2019]:
    > Lin, Zenhua: "Riemannian Geometry of Symmetric Positive Definite Matrices via
    > Cholesky Decomposition", arXiv: [1908.09326](https://arxiv.org/abs/1908.09326).
"""
struct LogCholeskyMetric <: RiemannianMetric end

cholesky_to_spd(x, W) = (x * x', W * x' + x * W')

tangent_cholesky_to_tangent_spd!(x, W) = (W .= W * x' + x * W')

spd_to_cholesky(p, X) = spd_to_cholesky(p, cholesky(p).L, X)

function spd_to_cholesky(p, x, X)
    w = inv(x) * X * inv(transpose(x))
    # strictly lower triangular plus half diagonal
    return (x, x * (LowerTriangular(w) - Diagonal(w) / 2))
end

@doc raw"""
    distance(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, p, q)

Compute the distance on the manifold of [`SymmetricPositiveDefinite`](@ref)
nmatrices, i.e. between two symmetric positive definite matrices `p` and `q`
with respect to the [`LogCholeskyMetric`](@ref). The formula reads

````math
d_{\mathcal P(n)}(p,q) = \sqrt{
 \lVert ⌊ x ⌋ - ⌊ y ⌋ \rVert_{\mathrm{F}}^2
 + \lVert \log(\operatorname{diag}(x)) - \log(\operatorname{diag}(y))\rVert_{\mathrm{F}}^2 }\ \ ,
````

where $x$ and $y$ are the cholesky factors of $p$ and $q$, respectively,
$⌊\cdot⌋$ denbotes the strictly lower triangular matrix of its argument,
and $\lVert\cdot\rVert_{\mathrm{F}}$ the Frobenius norm.
"""
function distance(
    M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric},
    p,
    q,
) where {N}
    return distance(CholeskySpace{N}(), cholesky(p).L, cholesky(q).L)
end

@doc raw"""
    exp(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, p, X)

Compute the exponential map on the [`SymmetricPositiveDefinite`](@ref) `M` with
[`LogCholeskyMetric`](@ref) from `p` into direction `X`. The formula reads

````math
\exp_p X = (\exp_y W)(\exp_y W)^\mathrm{T}
````

where $\exp_xW$ is the exponential map on [`CholeskySpace`](@ref), $y$ is the cholesky
decomposition of $p$, $W = y(y^{-1}Xy^{-\mathrm{T}})_\frac{1}{2}$,
and $(\cdot)_\frac{1}{2}$
denotes the lower triangular matrix with the diagonal multiplied by $\frac{1}{2}$.
"""
exp(::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, ::Any...)

function exp!(
    M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric},
    q,
    p,
    X,
) where {N}
    (y, W) = spd_to_cholesky(p, X)
    z = exp(CholeskySpace{N}(), y, W)
    return copyto!(q, z * z')
end

@doc raw"""
    inner(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, p, X, Y)

Compute the inner product of two matrices `X`, `Y` in the tangent space of `p`
on the [`SymmetricPositiveDefinite`](@ref) manifold `M`, as
a [`MetricManifold`](@ref) with [`LogCholeskyMetric`](@ref). The formula reads

````math
    g_p(X,Y) = ⟨a_z(X),a_z(Y)⟩_z,
````

where $⟨\cdot,\cdot⟩_x$ denotes inner product on the [`CholeskySpace`](@ref),
$z$ is the cholesky factor of $p$,
$a_z(W) = z (z^{-1}Wz^{-\mathrm{T}})_{\frac{1}{2}}$, and $(\cdot)_\frac{1}{2}$
denotes the lower triangular matrix with the diagonal multiplied by $\frac{1}{2}$
"""
function inner(
    M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric},
    p,
    X,
    Y,
) where {N}
    (z, Xz) = spd_to_cholesky(p, X)
    (z, Yz) = spd_to_cholesky(p, z, Y)
    return inner(CholeskySpace{N}(), z, Xz, Yz)
end

@doc raw"""
    log(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, p, q)

Compute the logarithmic map on [`SymmetricPositiveDefinite`](@ref) `M` with
respect to the [`LogCholeskyMetric`](@ref) emanating from `p` to `q`.
The formula can be adapted from the [`CholeskySpace`](@ref) as
````math
\log_p q = xW^{\mathrm{T}} + Wx^{\mathrm{T}},
````
where $x$ is the colesky factor of $p$ and $W=\log_xy$ for $y$ the cholesky factor
of $q$ and the just mentioned logarithmic map is the one on [`CholeskySpace`](@ref).
"""
log(::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}, ::Any...)

function log!(
    M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric},
    X,
    p,
    q,
) where {N}
    x = cholesky(p).L
    y = cholesky(q).L
    log!(CholeskySpace{N}(), X, x, y)
    return tangent_cholesky_to_tangent_spd!(x, X)
end

@doc raw"""
    vector_transport_to(
        M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric},
        p,
        X,
        q,
        ::ParallelTransport,
    )

Parallel transport the tangent vector `X` at `p` along the geodesic to `q` with respect to
the [`SymmetricPositiveDefinite`](@ref) manifold `M` and [`LogCholeskyMetric`](@ref).
The parallel transport is based on the parallel transport on [`CholeskySpace`](@ref):
Let $x$ and $y$ denote the cholesky factors of `p` and `q`, respectively and
$W = x(x^{-1}Xx^{-\mathrm{T}})_\frac{1}{2}$, where $(\cdot)_\frac{1}{2}$ denotes the lower
triangular matrix with the diagonal multiplied by $\frac{1}{2}$. With $V$ the parallel
transport on [`CholeskySpace`](@ref) from $x$ to $y$. The formula hear reads

````math
\mathcal P_{q←p}X = yV^{\mathrm{T}} + Vy^{\mathrm{T}}.
````
"""
vector_transport_to(
    ::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric},
    ::Any,
    ::Any,
    ::Any,
    ::ParallelTransport,
)

function vector_transport_to!(
    M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric},
    Y,
    p,
    X,
    q,
    ::ParallelTransport,
) where {N}
    y = cholesky(q).L
    (x, W) = spd_to_cholesky(p, X)
    vector_transport_to!(CholeskySpace{N}(), Y, x, W, y, ParallelTransport())
    return tangent_cholesky_to_tangent_spd!(y, Y)
end
