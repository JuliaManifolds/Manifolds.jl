@doc raw"""
    CanonicalMetric <: Metric

The Canonical Metric is name used on several manifolds, for example for the [`Stiefel`](@ref)
manifold, see for example[^EdelmanAriasSmith1998].

[^EdelmanAriasSmith1998]:
    > Edelman, A., Ariar, T. A., Smith, S. T.:
    > _The Geometry of Algorihthms with Orthogonality Constraints_,
    > SIAM Journal on Matrix Analysis and Applications (20(2), pp. 303–353, 1998.
    > doi: [10.1137/S0895479895290954](https://doi.org/10.1137/S0895479895290954)
    > arxiv: [9806030](https://arxiv.org/abs/physics/9806030)
"""
struct CanonicalMetric <: RiemannianMetric end

"""
    q = exp(MetricManifold{ℝ, Stiefel{n,k,ℝ}, CanonicalMetric}, p, X)
    exp!(MetricManifold{ℝ, Stiefel{n,k,ℝ}, q, CanonicalMetric}, p, X)

Compute the exponential map on the [`Stiefel`](@ref)`(n,k)` manifold with respect to the [`CanonicalMetric`](@ref).

First, decompose The tangent vector ``X`` into its horizontal and vertical component with
respect to ``p``, i.e.

```math
X = pp^{\mathrm{T}}X + (I_n-pp^{\mathrm{T}})X,
```
where ``I_n`` is the ``n\times n`` identity matrix.
We introduce ``A=p^{\mathrm{T}}X`` and ``QR = (I_n-pp^{\mathrm{T}})X``` the qr decomposition
of the vertical component. Then using the matrix exponential ``operatorname{Exp}`` we introduce ``B`` and ``C`` as

```math
\begin{pmatrix}
B\\C
\end{pmatrix}
\coloneqq
\operatorname{Exp}\left(
\begin{pmatrix}
A & -R^{\mathrm{T}}\\ R & 0
\end{pmatrix}
\right)
\begin{pmatrix}I_k\\0\end{pmatrix}
```

the exponential map reads

```math
q = \exp_p X = pC + QB.
```
For more details, see [^EdelmanAriasSmith1998][^Zimmermann2017].
"""
exp(::MetricManifold{ℝ, Stiefel{n,k,ℝ}, CanonicalMetric}, ::Any...) where {n,k}

function exp!(::MetricManifold{ℝ, Stiefel{n,k,ℝ}, CanonicalMetric}, Q, p, X) where{n,k}
    A = p'*X
    QR = qr(X-p*A)
    BC_ext = exp([A -QR.R'; QR.R, zeros(k,k)])
    q .= [p Matrix(QR.Q)] * BC_ext[:,1:k]
    return q
end

"""
    X = log(MetricManifold{ℝ, Stiefel{n,k,ℝ}, CanonicalMetric}, p, q)
    log!(MetricManifold{ℝ, Stiefel{n,k,ℝ}, X, CanonicalMetric}, p, q)

Compute the logarithmic map on the [`Stiefel`](@ref)`(n,k)` manifold with respect to the [`CanonicalMetric`](@ref)
using a matrix-algebraic based approach to an iterative inversion of the formula of the
[`exp`](@ref exp(::MetricManifold{ℝ, Stiefel{n,k,ℝ}, CanonicalMetric}, ::Any...)).

The algorithm is derived in[^Zimmermann2017].

[^Zimmermann2017]:
    > Zimmermann, R.: _A matrix-algebraic algorithm for the Riemannian logarithm on the Stiefel manifold under the canoncial metric.
    > SIAM Journal on Matrix Analysis and Applications 28(2), pp. 322-342, 2017.
    > doi: [10.1137/16M1074485](https://doi.org/10.1137/16M1074485),
    > arXiv: [1604.05054](https://arxiv.org/abs/1604.05054).
"""
log(::MetricManifold{ℝ, Stiefel{n,k,ℝ}, CanonicalMetric}, ::Any...) where {n,k}

function log!(::MetricManifold{ℝ, Stiefel{n,k,ℝ}, CanonicalMetric}, p, q; tolerance=1e-9, maxiter=1e5) where{n,k}
    M = p'*q
    QR = qr(q-p*M)
    V = qr([M;Matrix(QR.R)]).Q*Matrix(I, 2*k,2*k)
    S = svd(V[(k+1):2*k, (k+1):2k]) #bottom right corner
    V[:,(k+1):2*k] = V[:,(k+1):2*k]*(S.V*S.U')
    LV = log(V)
    C = view(LV, (k+1):2*k, (k+1):2*k)
    i=0
    while (i < maxiter) && (norm(C) > tolerance)
        LV = log(V)
        V[:, (k+1):2*k] *= exp(-C)
    end
    X .= p*LV[1:k,1:k] + QR.Q*LV[(k+1):2*k, 1:k]
    return X
end
