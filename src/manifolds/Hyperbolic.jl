@doc raw"""
    Hyperbolic{N} <: AbstractEmbeddedManifold{DefaultEmbeddingType}

The hyperbolic space $ℍ^n$ represented by $n+1$-Tuples, i.e. embedded in the
[`Lorentz`](@ref)ian manifold equipped with the [`MinkowskiMetric`](@ref)
$⟨\cdot,\cdot⟩_{\mathrm{M}}$. The space is defined as

```math
ℍ^n = \Bigl\{p ∈ ℝ^{n+1} : ⟨p,p⟩_{\mathrm{M}}= -p_{n+1}^2
  + \displaystyle\sum_{k=1}^n p_k^2 = -1, p_{n+1} > 0\Bigr\},.
```

The tangent space $T_p ℍ^n$ is given by

````math
T_p ℍ^n := \bigl\{
X ∈ ℝ^{n+1} : ⟨p,X⟩_{\mathrm{M}} = 0
\bigr\}.
````
Note that while the [`MinkowskiMetric`](@ref) renders the [`Lorentz`](@ref) manifold (only)
pseudo-Riemannian, on the tangent bundle of the Hyperbolic space it induces a Riemannian
metric. The corresponding sectional curvature is $-1$.

# Constructor

    Hyperbolic(n)

Generate the $ℍ^{n} ⊂ ℝ^{n+1}$
"""
struct Hyperbolic{N} <: AbstractEmbeddedManifold{DefaultIsometricEmbeddingType} end

Hyperbolic(n::Int) = Hyperbolic{n}()

base_manifold(M::Hyperbolic) = M
decorated_manifold(M::Hyperbolic{N}) where {N} = Lorentz(N+1, MinkowskiMetric())
default_metric_dispatch(::Hyperbolic, ::MinkowskiMetric) = Val(true)

"""
    check_manifold_point(M::Hyperbolic, p; kwargs...)

Check whether `p` is a valid point on the [`Hyperbolic`](@ref) `M`, i.e. is a `p` vector with
inner product in the embedding of -1, see [`MinkowskiMetric`](@ref).
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::Hyperbolic, p; kwargs...)
    if size(p) != representation_size(M)
        return DomainError(
            size(p),
            "The point $(p) does not lie on $(M), since its size is not $(representation_size(M)).",
        )
    end
    if !isapprox(minkowski_metric(p, p), -1.0; kwargs...)
        return DomainError(
            minkowski_metric(p, p),
            "The point $(p) does not lie on $(M) since its Minkowski inner product is not -1.",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M::Hyperbolic, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to `p` on the [`Hyperbolic`](@ref) `M`, i.e.
after [`check_manifold_point`](@ref)`(M,p)`, `X` has to be of the same dimension as `p`
and orthogonal to `p` with respect to the inner product from the embedding, see
[`MinkowskiMetric`](@ref). The optional parameter `check_base_point` indicates whether to
call [`check_manifold_point`](@ref)  for `p`. The tolerance for the last test can be set
using the `kwargs...`.
"""
function check_tangent_vector(M::Hyperbolic, p, X; check_base_point = true, kwargs...)
    if check_base_point
        perr = check_manifold_point(M, p; kwargs...)
        perr === nothing || return perr
    end
    if size(X) != representation_size(M)
        return DomainError(
            size(X),
            "The vector $(X) is not a tangent to a point on $M since its size does not match $(representation_size(M)).",
        )
    end
    if !isapprox(minkowski_metric(p, X), 0.0; kwargs...)
        return DomainError(
            abs(minkowski_metric(p, X)),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not orthogonal (with respect to the Minkowski inner product) in the embedding.",
        )
    end
    return nothing
end

@doc raw"""
    distance(M::Hyperbolic, p, q)

Compute the distance on the [`Hyperbolic`](@ref) `M`, which reads

````math
d_{ℍ^n}(p,q) = \operatorname{acosh}( - ⟨p, q⟩_{\mathrm{M}}),
````

where $⟨\cdot,\cdot⟩_{\mathrm{M}}$ denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold.
"""
distance(M::Hyperbolic, p, q) = acosh(max(-minkowski_metric(p, q), 1.0))

embed!(::Hyperbolic, q, p) = (q .= p)

embed!(::Hyperbolic, Y, p, X) = (Y .= X)

@doc raw"""
    exp(M::Hyperbolic, p, X)

Compute the exponential map on the [`Hyperbolic`](@ref) space $ℍ^n$ emanating
from `p` towards `X`. The formula reads

````math
\exp_p X = \cosh(\sqrt{⟨X,X⟩_{\mathrm{M}}})p
+ \sinh(\sqrt{⟨X,X⟩_{\mathrm{M}}})\frac{X}{\sqrt{⟨X,X⟩_{\mathrm{M}}}},
````

where $⟨\cdot,\cdot⟩_{\mathrm{M}}$ denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold.
"""
exp(::Hyperbolic, ::Any...)

function exp!(M::Hyperbolic, q, p, X)
    vn = sqrt(max(inner(M, p, X, X), 0.0))
    vn < eps(eltype(p)) && return copyto!(q, p)
    return copyto!(q, cosh(vn) * p + sinh(vn) / vn * X)
end

@doc raw"""
    injectivity_radius(M::Hyperbolic)
    injectivity_radius(M::Hyperbolic, p)

Return the injectivity radius on the [`Hyperbolic`](@ref), which is $∞$.
"""
injectivity_radius(H::Hyperbolic, ::Any...) = Inf

@doc raw"""
    log(M::Hyperbolic, p, q)

Compute the logarithmic map on the [`Hyperbolic`](@ref) space $ℍ^n$, the tangent
vector representing the [`geodesic`](@ref) starting from `p`
reaches `q` after time 1. The formula reads for $x ≠ y$

```math
\log_p q = d_{ℍ^n}(p,q)
\frac{q-⟨p,q⟩_{\mathrm{M}} p}{\lVert q-⟨p,q⟩_{\mathrm{M}} p \rVert_2}
```

and is zero otherwise.
"""
log(::Hyperbolic, ::Any...)

function log!(M::Hyperbolic, X, p, q)
    scp = minkowski_metric(p, q)
    w = q + scp * p
    wn = sqrt(max(scp .^ 2 - 1, 0.0))
    wn < eps(eltype(p)) && return zero_tangent_vector!(M, X, p)
    X .= acosh(max(1.0, -scp)) / wn .* w
    return X
end

@doc raw"""
    manifold_dimension(H::Hyperbolic)

Return the dimension of the hyperbolic space manifold $ℍ^n$, i.e. $\dim(ℍ^n) = n$.
"""
manifold_dimension(::Hyperbolic{N}) where {N} = N

"""
    mean(
        M::Hyperbolic,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = CyclicProximalPointEstimationMethod();
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` on the
[`Hyperbolic`](@ref) space using [`CyclicProximalPointEstimation`](@ref).
"""
mean(::Hyperbolic, ::Any...)

function mean!(M::Hyperbolic, p, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(M, p, x, w, CyclicProximalPointEstimation(); kwargs...)
end

@doc raw"""
    project_tangent(M::Hyperbolic, p, X)

Perform an orthogonal projection with respect to the Minkowski inner product of `X` onto
the tangent space at `p` of the [`Hyperbolic`](@ref) space `M`.

The formula reads
````math
Y = X + ⟨p,X⟩_{\mathrm{M}} p,
````
where $⟨\cdot, \cdot⟩_{\mathrm{M}}$ denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold.
"""
project_tangent(::Hyperbolic, ::Any...)

project_tangent!(M::Hyperbolic, Y, p, X) = (Y .= X .+ minkowski_metric(p, X) .* p)

show(io::IO, ::Hyperbolic{N}) where {N} = print(io, "Hyperbolic($(N))")

@doc raw"""
    vector_transport_to(M::Hyperbolic, p, X, q, ::ParallelTransport)

Compute the paralllel transport of the `X` from the tangent space at `p` on the
[`Hyperbolic`](@ref) space $ℍ^n$ to the tangent at `q` along the [`geodesic`](@ref)
connecting `p` and `q`. The formula reads

````math
\mathcal P_{q←p}X = X - \frac{⟨\log_p q,X⟩_x}{d^2_{ℍ^n}(p,q)}
\bigl(\log_p q + \log_qp \bigr).
````
"""
vector_transport_to(::Hyperbolic, ::Any, ::Any, ::Any, ::ParallelTransport)

function vector_transport_to!(M::Hyperbolic, Y, p, X, q, ::ParallelTransport)
    w = log(M, p, q)
    wn = norm(M, p, w)
    wn < eps(eltype(p + q)) && return copyto!(Y, X)
    return copyto!(Y, X - (inner(M, p, w, X) * (w + log(M, q, p)) / wn^2))
end
