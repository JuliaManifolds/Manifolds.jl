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
    Lorentz{T} = MetricManifold{ℝ,Euclidean{ℝ,T},<:LorentzMetric}

The Lorentz manifold (or Lorentzian) is a pseudo-Riemannian manifold.

# Constructor

    Lorentz(n[, m=MinkowskiMetric()]; parameter::Symbol=:type)

Generate the Lorentz manifold of dimension `n` with the [`LorentzMetric`](@ref) `m`,
which is by default set to the [`MinkowskiMetric`](@ref).
"""
const Lorentz = MetricManifold{ℝ, Euclidean{ℝ, T}, <:LorentzMetric} where {T}

function Lorentz(n::Int, m::LorentzMetric = MinkowskiMetric(); parameter::Symbol = :type)
    E = Euclidean(n; parameter = parameter)
    return Lorentz(E, m)
end
function Lorentz(E::Euclidean{𝔽, T}, m::LorentzMetric = MinkowskiMetric()) where {𝔽, T}
    return Lorentz{T, typeof(m)}(E, m)
end

"""
    is_flat(M::Lorentz{<:Any, MinkowskiMetric})

Return whether the [`Lorentz`](@ref) manifold `M` is flat. This is the case for the
[`MinkowskiMetric`](@ref), since it is the same matrix at every point.
"""
is_flat(M::Lorentz{<:Any, MinkowskiMetric}) = true

function local_metric(M::Lorentz{<:Any, MinkowskiMetric}, p)
    n = get_parameter(M.manifold.size)[1]
    return Diagonal([ones(n - 1)..., -1])
end

function inner(::Lorentz{<:Any, MinkowskiMetric}, p, X, Y)
    return minkowski_metric(X, Y)
end

@tfvector_inner_via_get_vector Lorentz{<:Any, MinkowskiMetric}

@doc raw"""
    metric_signature(M::Lorentz)

Return the signature ``(n-1, 1, 0)`` of the [`LorentzMetric`](@ref) of the ``n``-dimensional
[`Lorentz`](@ref) manifold `M`, which has a single time dimension.
"""
metric_signature(M::Lorentz) = (manifold_dimension(M) - 1, 1, 0)

@doc raw"""
    exp(M::Lorentz{<:Any, MinkowskiMetric}, p, X)
    log(M::Lorentz{<:Any, MinkowskiMetric}, p, q)
    parallel_transport_to(M::Lorentz{<:Any, MinkowskiMetric}, p, X, q)

Compute the exponential map, the logarithmic map and the parallel transport on the
[`Lorentz`](@ref) manifold `M` with the [`MinkowskiMetric`](@ref).
Since the metric is the same matrix at every point, its Levi-Civita connection is the one
of the [`Euclidean`](@ref) space, so these are ``p + X``, ``q - p`` and ``X``, respectively.
"""
exp(::Lorentz{<:Any, MinkowskiMetric}, ::Any, ::Any)

exp!(M::Lorentz{<:Any, MinkowskiMetric}, q, p, X) = exp!(M.manifold, q, p, X)
function exp_fused!(M::Lorentz{<:Any, MinkowskiMetric}, q, p, X, t::Number)
    return exp_fused!(M.manifold, q, p, X, t)
end

log!(M::Lorentz{<:Any, MinkowskiMetric}, X, p, q) = log!(M.manifold, X, p, q)

function parallel_transport_direction!(M::Lorentz{<:Any, MinkowskiMetric}, Y, p, X, d)
    return parallel_transport_direction!(M.manifold, Y, p, X, d)
end
function parallel_transport_to!(M::Lorentz{<:Any, MinkowskiMetric}, Y, p, X, q)
    return parallel_transport_to!(M.manifold, Y, p, X, q)
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
