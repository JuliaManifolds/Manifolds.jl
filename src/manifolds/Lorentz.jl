@doc raw"""
    LorentzMetric <: AbstractMetric

Abstract type for Lorentz metrics, which have a single time dimension. These
metrics assume the spacelike convention with the time dimension being last,
giving the signature $(++...+-)$.
"""
abstract type LorentzMetric <: AbstractMetric end

@doc raw"""
    MinkowskiMetric <: LorentzMetric

As a special metric of signature  $(++...+-)$, i.e. a [`LorentzMetric`](@ref),
see [`minkowski_metric`](@ref) for the formula.
"""
struct MinkowskiMetric <: LorentzMetric end

@doc raw"""
    Lorentz{T} = MetricManifold{Euclidean{T,ℝ},LorentzMetric}

The Lorentz manifold (or Lorentzian) is a pseudo-Riemannian manifold.

# Constructor

    Lorentz(n[, metric=MinkowskiMetric()])

Generate the Lorentz manifold of dimension `n` with the [`LorentzMetric`](@ref) `m`,
which is by default set to the [`MinkowskiMetric`](@ref).
"""
const Lorentz = MetricManifold{ℝ,Euclidean{T,ℝ},<:LorentzMetric} where {T}

function Lorentz(n::Int, m::LorentzMetric=MinkowskiMetric(); parameter::Symbol=:type)
    E = Euclidean(n; parameter=parameter)
    return Lorentz(E, m)
end
function Lorentz(E::Euclidean{T}, m::LorentzMetric=MinkowskiMetric()) where {T}
    return Lorentz{T,typeof(m)}(E, m)
end

function local_metric(M::Lorentz{<:Any,MinkowskiMetric}, p)
    n = get_parameter(M.manifold.size)[1]
    return Diagonal([ones(n - 1)..., -1])
end

function inner(::Lorentz{<:Any,MinkowskiMetric}, p, X, Y)
    return minkowski_metric(X, Y)
end
@doc raw"""
    minkowski_metric(a, b)

Compute the minkowski metric on $\mathbb R^n$ is given by
````math
⟨a,b⟩_{\mathrm{M}} = -a_{n}b_{n} +
\displaystyle\sum_{k=1}^{n-1} a_kb_k.
````
"""
function minkowski_metric(a, b)
    a_part = @view a[1:(end - 1)]
    b_part = @view b[1:(end - 1)]
    return -a[end] * b[end] + dot(a_part, b_part)
end
function minkowski_metric(a::StaticVector{N}, b::StaticVector{N}) where {N}
    return -a[N] * b[N] + dot(a[SOneTo(N - 1)], b[SOneTo(N - 1)])
end
