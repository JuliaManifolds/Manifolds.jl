@doc raw"""
    LorentzMetric <: Metric

Abstract type for Lorentz metrics, which have a single time dimension. These
metrics assume the spacelike convention with the time dimension being last,
giving the signature $(++...+-)$.
"""
abstract type LorentzMetric <: Metric end

@doc raw"""
    MinkowskiMetric <: LorentzMetric

As a special metric of signature  $(++...+-)$, i.e. a [`LorentzMetric`](@ref)
the minkowski metric on $\mathbb R^n$ is given by
````math
⟨a,b⟩_{\mathrm{M}} = -a_{n}b_{n} +
\displaystyle\sum_{k=1}^{n-1} a_kb_k.
````
"""
struct MinkowskiMetric <: LorentzMetric end

@doc doc"""
    Lorentz{N} = MetricManifold{Euclidean{N},LorentzMetric}

The Lorentz manifold (or Lorentzian) is a pseudo-Riemannian manifold.

# Constructor

    Lorentz(n[, metric=MinkowskiMetric()])

Generate the Lorentz manifold of dimension `n` with the [`LorentzMetric`](@ref) `m`,
which is by default set to the [`MinkowskiMetric`](@ref).
"""
const Lorentz = MetricManifold{Euclidean{Tuple{N,},ℝ},<:LorentzMetric} where N

function Lorentz(n, m::MT = MinkowskiMetric()) where {MT<:LorentzMetric}
    return Lorentz{n,typeof(m)}(Euclidean(n),m)
end

function local_metric(::MetricManifold{Euclidean{Tuple{N,},ℝ},MinkowskiMetric}, p) where {N}
    return Diagonal([ones(N-1)...,-1])
end

function inner(::MetricManifold{Euclidean{Tuple{N,},ℝ},MinkowskiMetric}, p, X, Y) where {N}
    return -X[end] * Y[end] + sum(X[1:end-1] .* Y[1:end-1])
end
