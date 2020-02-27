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

The Lorenzt manifold (or Lorentzian) is a pseudo-Riemannian manifold
"""
Lorentz{N} = MetricManifold{Euclidean{N},LorentzMetric}

local_metric(Lorentz{N}, p) = Diagonal([ones(N-1)...,-1])