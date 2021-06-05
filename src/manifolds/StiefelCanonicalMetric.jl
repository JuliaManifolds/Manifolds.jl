@doc raw"""
    CanonicalMetric <: AbstractMetric

The Canonical Metric refers to a metric for the [`Stiefel`](@ref)
manifold, see[^EdelmanAriasSmith1998].

[^EdelmanAriasSmith1998]:
    > Edelman, A., Ariar, T. A., Smith, S. T.:
    > _The Geometry of Algorihthms with Orthogonality Constraints_,
    > SIAM Journal on Matrix Analysis and Applications (20(2), pp. 303–353, 1998.
    > doi: [10.1137/S0895479895290954](https://doi.org/10.1137/S0895479895290954)
    > arxiv: [9806030](https://arxiv.org/abs/physics/9806030)
"""
struct CanonicalMetric <: RiemannianMetric end

"""
    ApproximateLogarithmicMap <: ApproximateInverseRetraction

An approximate implementation of the logarithmic map, which is an [`inverse_retract`](@ref)ion.
See [`inverse_retract(::MetricManifold{ℝ,Stiefel{n,k,ℝ},CanonicalMetric}, ::Any, ::Any, ::ApproximateLogarithmicMap) where {n,k}`](@ref) for a use case.
"""
struct ApproximateLogarithmicMap{T} <: ApproximateInverseRetraction
    max_iterations::Int
    tolerance::T
end

function distance(M::MetricManifold{ℝ,Stiefel{n,k,ℝ},CanonicalMetric}, q, p) where {n,k}
    return norm(M, p, log(M, p, q))
end

@doc raw"""
    q = exp(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, CanonicalMetric}, p, X)
    exp!(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, q, CanonicalMetric}, p, X)

Compute the exponential map on the [`Stiefel`](@ref)`(n,k)` manifold with respect to the [`CanonicalMetric`](@ref).

First, decompose The tangent vector ``X`` into its horizontal and vertical component with
respect to ``p``, i.e.

```math
X = pp^{\mathrm{T}}X + (I_n-pp^{\mathrm{T}})X,
```
where ``I_n`` is the ``n\times n`` identity matrix.
We introduce ``A=p^{\mathrm{T}}X`` and ``QR = (I_n-pp^{\mathrm{T}})X`` the `qr` decomposition
of the vertical component. Then using the matrix exponential ``\operatorname{Exp}`` we introduce ``B`` and ``C`` as

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
exp(::MetricManifold{ℝ,Stiefel{n,k,ℝ},CanonicalMetric}, ::Any...) where {n,k}

function exp!(::MetricManifold{ℝ,Stiefel{n,k,ℝ},CanonicalMetric}, q, p, X) where {n,k}
    A = p' * X
    n == k && return mul!(q, p, exp(A))
    QR = qr(X - p * A)
    BC_ext = exp([A -QR.R'; QR.R 0*I])
    @views begin
        mul!(q, p, BC_ext[1:k, 1:k])
        mul!(q, Matrix(QR.Q), BC_ext[(k + 1):(2 * k), 1:k], true, true)
    end
    return q
end

@doc raw"""
    inner(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, X, CanonicalMetric}, p, X, Y)

compute the inner procuct on the [`Stiefel`](@ref) manifold with respect to the
[`CanonicalMetric`](@ref). The formula reads

```math
g_p(X,Y) = \operatorname{tr}\bigl( X^{\mathrm{T}}(I_n - \frac{1}{2}pp^{\mathrm{T}})Y \bigr).
```
"""
function inner(::MetricManifold{ℝ,Stiefel{n,k,ℝ},CanonicalMetric}, p, X, Y) where {n,k}
    T = Base.promote_eltype(p, X, Y)
    s = T(dot(X, Y))
    if n != k
        s -= dot(p'X, p'Y) / 2
    end
    return s
end

@doc raw"""
    X = inverse_retract(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, CanonicalMetric}, p, q, a::ApproximateLogarithmicMap)
    inverse_retract!(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, X, CanonicalMetric}, p, q, a::ApproximateLogarithmicMap)

Compute an approximation to the logarithmic map on the [`Stiefel`](@ref)`(n,k)` manifold with respect to the [`CanonicalMetric`](@ref)
using a matrix-algebraic based approach to an iterative inversion of the formula of the
[`exp`](@ref exp(::MetricManifold{ℝ, Stiefel{n,k,ℝ}, CanonicalMetric}, ::Any...) where {n,k}).

The algorithm is derived in[^Zimmermann2017] and it uses the `max_iterations` and the `tolerance` field
from the [`ApproximateLogarithmicMap`](@ref).

[^Zimmermann2017]:
    > Zimmermann, R.: _A matrix-algebraic algorithm for the Riemannian logarithm on the Stiefel manifold under the canoncial metric.
    > SIAM Journal on Matrix Analysis and Applications 28(2), pp. 322-342, 2017.
    > doi: [10.1137/16M1074485](https://doi.org/10.1137/16M1074485),
    > arXiv: [1604.05054](https://arxiv.org/abs/1604.05054).
"""
inverse_retract(
    ::MetricManifold{ℝ,Stiefel{n,k,ℝ},CanonicalMetric},
    ::Any,
    ::Any,
    ::ApproximateLogarithmicMap,
) where {n,k}

function log(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},CanonicalMetric},
    p,
    q;
    maxiter=1e5,
    tolerance=1e-9,
) where {n,k}
    X = allocate_result(M, log, p, q)
    inverse_retract!(M, X, p, q, ApproximateLogarithmicMap(maxiter, tolerance))
    return X
end

function log!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},CanonicalMetric},
    X,
    p,
    q;
    maxiter=1e5,
    tolerance=1e-9,
) where {n,k}
    inverse_retract!(M, X, p, q, ApproximateLogarithmicMap(maxiter, tolerance))
    return X
end

function inverse_retract!(
    ::MetricManifold{ℝ,Stiefel{n,k,ℝ},CanonicalMetric},
    X,
    p,
    q,
    a::ApproximateInverseRetraction,
) where {n,k}
    M = p' * q
    QR = qr(q - p * M)
    V = convert(Matrix, qr([M; QR.R]).Q * I)
    Vcorner = @view V[(k + 1):(2 * k), (k + 1):(2k)] #bottom right corner
    Vpcols = @view V[:, (k + 1):(2 * k)] #second half of the columns
    S = svd(Vcorner) # preprocessing: Procrustes
    mul!(Vpcols, Vpcols * S.U, S.V')
    V[1:k, 1:k] .= M
    V[(k + 1):(2 * k), 1:k] .= QR.R

    LV = log_safe!(allocate(V), V)
    C = view(LV, (k + 1):(2 * k), (k + 1):(2 * k))
    expnC = exp(-C)
    i = 0
    new_Vpcols = Vpcols * expnC # allocate once
    while (i < a.max_iterations) && (norm(C) > a.tolerance)
        i = i + 1
        log_safe!(LV, V)
        expnC = exp(-C)
        mul!(new_Vpcols, Vpcols, expnC)
        copyto!(Vpcols, new_Vpcols)
    end
    mul!(X, p, @view(LV[1:k, 1:k]))
    # force the first - Q - to be the reduced form, not the full matrix
    mul!(X, @view(QR.Q[:, 1:k]), @view(LV[(k + 1):(2 * k), 1:k]), true, true)
    return X
end
