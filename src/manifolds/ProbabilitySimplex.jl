@doc raw"""
    ProbabilitySimplex{n} <: AbstractDecoratorManifold{𝔽}

The (relative interior of) the probability simplex is the set
````math
Δ^n := \biggl\{ p ∈ ℝ^{n+1}\ \big|\ p_i > 0 \text{ for all } i=1,…,n+1,
\text{ and } ⟨\mathbb{1},p⟩ = \sum_{i=1}^{n+1} p_i = 1\biggr\},
````
where $\mathbb{1}=(1,…,1)^{\mathrm{T}}∈ ℝ^{n+1}$ denotes the vector containing only ones.

This set is also called the unit simplex or standard simplex.

The tangent space is given by
````math
T_pΔ^n = \biggl\{ X ∈ ℝ^{n+1}\ \big|\ ⟨\mathbb{1},X⟩ = \sum_{i=1}^{n+1} X_i = 0 \biggr\}
````

The manifold is implemented assuming the Fisher-Rao metric for the multinomial distribution,
which is equivalent to the induced metric from isometrically embedding the probability
simplex in the $n$-sphere of radius 2.
The corresponding diffeomorphism $\varphi: \mathbb Δ^n → \mathcal N$,
where $\mathcal N \subset 2𝕊^n$ is given by $\varphi(p) = 2\sqrt{p}$.

This implementation follows the notation in [^ÅströmPetraSchmitzerSchnörr2017].

[^ÅströmPetraSchmitzerSchnörr2017]:
    > F. Åström, S. Petra, B. Schmitzer, C. Schnörr: “Image Labeling by Assignment”,
    > Journal of Mathematical Imaging and Vision, 58(2), pp. 221–238, 2017.
    > doi: [10.1007/s10851-016-0702-4](https://doi.org/10.1007/s10851-016-0702-4)
    > arxiv: [1603.05285](https://arxiv.org/abs/1603.05285).
"""
struct ProbabilitySimplex{n} <: AbstractDecoratorManifold{ℝ} end

ProbabilitySimplex(n::Int) = ProbabilitySimplex{n}()

"""
    FisherRaoMetric <: AbstractMetric

The Fisher-Rao metric or Fisher information metric is a particular Riemannian metric which
can be defined on a smooth statistical manifold, i.e., a smooth manifold whose points are
probability measures defined on a common probability space.

See for example the [`ProbabilitySimplex`](@ref).
"""
struct FisherRaoMetric <: AbstractMetric end

active_traits(f, ::ProbabilitySimplex, args...) = merge_traits(IsEmbeddedManifold())

@doc raw"""
    change_representer(M::ProbabilitySimplex, ::EuclideanMetric, p, X)

Given a tangent vector with respect to the metric from the embedding, the [`EuclideanMetric`](@ref),
the representer of a linear functional on the tangent space is adapted as ``Z = p .* X``, since
this “compensates” for the divsion by ``p`` in the Riemannian metric on the [`ProbabilitySimplex`](@ref).

To be precise for any ``Y ∈ T_pΔ^n`` we are looking for ``Z ∈ T_pΔ^n`` such that

```math
    ⟨X,Y⟩ = X^\mathrm{T}Y = \sum_{i=1}^{n+1}\frac{Z_iY_i}{p_i} = g_p(Z,Y)
```

and hence ``Z_i = X_ip_i, i=1,…,n+1``.
"""
change_representer(::ProbabilitySimplex, ::EuclideanMetric, ::Any, ::Any)

function change_representer!(::ProbabilitySimplex, Y, ::EuclideanMetric, p, X)
    return Y .= p .* X
end

@doc raw"""
    change_metric(M::ProbabilitySimplex, ::EuclideanMetric, p, X)

To change the metric, we are looking for a function ``c\colon T_pΔ^n \to T_pΔ^n`` such that for all ``X,Y ∈ T_pΔ^n``

```math
    ⟨X,Y⟩ = X^\mathrm{T}Y = \sum_{i=1}^{n+1}\frac{c(X)_ic(Y)_i}{p_i} = g_p(X,Y)
```

and hence ``C(X)_i = X_i\sqrt{p_i}, i=1,…,n+1``.
"""
change_metric(::ProbabilitySimplex, ::EuclideanMetric, ::Any, ::Any)

function change_metric!(::ProbabilitySimplex, Y, ::EuclideanMetric, p, X)
    return Y .= sqrt.(p) .* X
end

"""
    check_point(M::ProbabilitySimplex, p; kwargs...)

Check whether `p` is a valid point on the [`ProbabilitySimplex`](@ref) `M`, i.e. is a point in
the embedding with positive entries that sum to one
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_point(M::ProbabilitySimplex, p; kwargs...)
    if minimum(p) <= 0
        return DomainError(
            minimum(p),
            "The point $(p) does not lie on the $(M) since it has nonpositive entries.",
        )
    end
    if !isapprox(sum(p), 1.0; kwargs...)
        return DomainError(
            sum(p),
            "The point $(p) does not lie on the $(M) since its sum is not 1.",
        )
    end
    return nothing
end

"""
    check_vector(M::ProbabilitySimplex, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`ProbabilitySimplex`](@ref) `M`, i.e.
after [`check_point`](@ref check_point(::ProbabilitySimplex, ::Any))`(M,p)`,
`X` has to be of same dimension as `p` and its elements have to sum to one.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_vector(M::ProbabilitySimplex, p, X; kwargs...)
    if !isapprox(sum(X), 0.0; kwargs...)
        return DomainError(
            sum(X),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since its elements to not sum up to 0.",
        )
    end
    return nothing
end

get_embedding(M::ProbabilitySimplex) = Euclidean(representation_size(M)...; field=ℝ)

default_metric_dispatch(::ProbabilitySimplex, ::FisherRaoMetric) = Val(true)

@doc raw"""
    distance(M,p,q)

Compute the distance between two points on the [`ProbabilitySimplex`](@ref) `M`.
The formula reads
````math
d_{Δ^n}(p,q) = 2\arccos \biggl( \sum_{i=1}^{n+1} \sqrt{p_i q_i} \biggr)
````
"""
function distance(::ProbabilitySimplex, p, q)
    sumsqrt = zero(Base.promote_eltype(p, q))
    @inbounds for i in eachindex(p, q)
        sumsqrt += sqrt(p[i] * q[i])
    end
    return 2 * acos(sumsqrt)
end

@doc raw"""
    exp(M::ProbabilitySimplex,p,X)

Compute the exponential map on the probability simplex.

````math
\exp_pX = \frac{1}{2}\Bigl(p+\frac{X_p^2}{\lVert X_p \rVert^2}\Bigr)
+ \frac{1}{2}\Bigl(p - \frac{X_p^2}{\lVert X_p \rVert^2}\Bigr)\cos(\lVert X_p\rVert)
+ \frac{1}{\lVert X_p \rVert}\sqrt{p}\sin(\lVert X_p\rVert),
````

where $X_p = \frac{X}{\sqrt{p}}$, with its division meant elementwise, as well as for the
operations $X_p^2$ and $\sqrt{p}$.
"""
exp(::ProbabilitySimplex, ::Any...)

function exp!(::ProbabilitySimplex, q, p, X)
    s = sqrt.(p)
    Xs = X ./ s ./ 2
    θ = norm(Xs)
    q .= (cos(θ) .* s .+ usinc(θ) .* Xs) .^ 2
    return q
end

@doc raw"""
    injectivity_radius(M,p)

compute the injectivity radius on the [`ProbabilitySimplex`](@ref) `M` at the point `p`,
i.e. the distanceradius to a point near/on the boundary, that could be reached by following the
geodesic.
"""
function injectivity_radius(::ProbabilitySimplex{n}, p) where {n}
    i = argmin(p)
    s = sum(p) - p[i]
    return 2 * acos(sqrt(s))
end
_injectivity_radius(M::ProbabilitySimplex, p, ::SoftmaxRetraction) = injectivity_radius(M, p)
injectivity_radius(M::ProbabilitySimplex) = 0
injectivity_radius(M::ProbabilitySimplex, ::AbstractRetractionMethod) = 0

@doc raw"""
    inner(M::ProbabilitySimplex,p,X,Y)

Compute the inner product of two tangent vectors `X`, `Y` from the tangent space $T_pΔ^n$ at
`p`. The formula reads
````math
g_p(X,Y) = \sum_{i=1}^{n+1}\frac{X_iY_i}{p_i}
````
"""
function inner(::ProbabilitySimplex, p, X, Y)
    d = zero(Base.promote_eltype(p, X, Y))
    @inbounds for i in eachindex(p, X, Y)
        d += X[i] * Y[i] / p[i]
    end
    return d
end

@doc raw"""
    inverse_retract(M::ProbabilitySimplex, p, q, ::SoftmaxInverseRetraction)

Compute a first order approximation by projection. The formula reads
````math
\operatorname{retr}^{-1}_p q = \bigl( I_{n+1} - \frac{1}{n}\mathbb{1}^{n+1,n+1} \bigr)(\log(q)-\log(p))
````
where $\mathbb{1}^{m,n}$ is the size `(m,n)` matrix containing ones, and $\log$ is applied elementwise.
"""
inverse_retract(::ProbabilitySimplex, ::Any, ::Any, ::SoftmaxInverseRetraction)

function inverse_retract_softmax!(::ProbabilitySimplex{n}, X, p, q) where {n}
    X .= log.(q) .- log.(p)
    meanlogdiff = mean(X)
    X .-= meanlogdiff
    return X
end

@doc raw"""
    log(M::ProbabilitySimplex, p, q)

Compute the logarithmic map of `p` and `q` on the [`ProbabilitySimplex`](@ref) `M`.

````math
\log_pq = \frac{d_{Δ^n}(p,q)}{\sqrt{1-⟨\sqrt{p},\sqrt{q}⟩}}(\sqrt{pq} - ⟨\sqrt{p},\sqrt{q}⟩p),
````

where $pq$ and $\sqrt{p}$ is meant elementwise.
"""
log(::ProbabilitySimplex, ::Any...)

function log!(::ProbabilitySimplex, X, p, q)
    if p ≈ q
        fill!(X, 0)
    else
        z = sqrt.(p .* q)
        s = sum(z)
        X .= 2 * acos(s) / sqrt(1 - s^2) .* (z .- s .* p)
    end
    return X
end

@doc raw"""
    manifold_dimension(M::ProbabilitySimplex{n})

Returns the manifold dimension of the probability simplex in $ℝ^{n+1}$, i.e.
````math
    \dim_{Δ^n} = n.
````
"""
manifold_dimension(::ProbabilitySimplex{n}) where {n} = n

@doc raw"""
    mean(
        M::ProbabilitySimplex,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolation();
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` using
[`GeodesicInterpolation`](@ref).
"""
mean(::ProbabilitySimplex, ::Any...)

function Statistics.mean!(
    M::ProbabilitySimplex,
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)
    return mean!(M, p, x, w, GeodesicInterpolation(); kwargs...)
end

@doc raw"""
    project(M::ProbabilitySimplex, p, Y)

project `Y` from the embedding onto the tangent space at `p` on
the [`ProbabilitySimplex`](@ref) `M`. The formula reads

````math
\operatorname{proj}_{Δ^n}(p,Y) = Y - ⟨\mathbb 1,Y⟩p,
````
where ``\mathbb 1 ∈ ℝ`` denotes the vector of ones.
"""
project(::ProbabilitySimplex, ::Any, ::Any)

function project!(::ProbabilitySimplex, X, p, Y)
    X .= Y .- sum(Y) .* p
    return X
end

@doc raw"""
    project(M::ProbabilitySimplex, p)

project `p` from the embedding onto the [`ProbabilitySimplex`](@ref) `M`.
The formula reads

````math
\operatorname{proj}_{Δ^n}(p) = \frac{1}{⟨\mathbb 1,p⟩}p,
````
where ``\mathbb 1 ∈ ℝ`` denotes the vector of ones.
Not that this projection is only well-defined if ``p`` has positive entries.
"""
project(::ProbabilitySimplex, ::Any)

function project!(::ProbabilitySimplex, q, p)
    if any(x -> x <= 0, p)
        throw(
            DomainError(
                p,
                "All coordinates of point from the embedding, that should be projected, must be positive, otherwise the projection is not well defined.",
            ),
        )
    end
    q .= p ./ sum(p)
    return q
end

@doc raw"""
    representation_size(::ProbabilitySimplex{n})

return the representation size of points in the $n$-dimensional probability simplex,
i.e. an array size of `(n+1,)`.
"""
representation_size(::ProbabilitySimplex{n}) where {n} = (n + 1,)

@doc raw"""
    retract(M::ProbabilitySimplex, p, X, ::SoftmaxRetraction)

Compute a first order approximation by applying the softmax function. The formula reads

````math
\operatorname{retr}_p X = \frac{p\mathrm{e}^X}{⟨p,\mathrm{e}^X⟩},
````

where multiplication, exponentiation and division are meant elementwise.
"""
retract(::ProbabilitySimplex, ::Any, ::Any, ::SoftmaxRetraction)

function retract_softmax!(::ProbabilitySimplex, q, p, X)
    s = zero(eltype(q))
    @inbounds for i in eachindex(q, p, X)
        q[i] = p[i] * exp(X[i])
        s += q[i]
    end
    q ./= s
    return q
end

function Base.show(io::IO, ::ProbabilitySimplex{n}) where {n}
    return print(io, "ProbabilitySimplex($(n))")
end

@doc raw"""
    zero_vector(M::ProbabilitySimplex,p)

returns the zero tangent vector in the tangent space of the point `p`  from the
[`ProbabilitySimplex`](@ref) `M`, i.e. its representation by the zero vector in the embedding.
"""
zero_vector(::ProbabilitySimplex, ::Any)

zero_vector!(M::ProbabilitySimplex, v, p) = fill!(v, 0)
