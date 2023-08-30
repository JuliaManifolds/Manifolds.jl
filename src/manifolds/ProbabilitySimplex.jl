@doc raw"""
    ProbabilitySimplex{n,boundary} <: AbstractDecoratorManifold{𝔽}

The (relative interior of) the probability simplex is the set
````math
Δ^n := \biggl\{ p ∈ ℝ^{n+1}\ \big|\ p_i > 0 \text{ for all } i=1,…,n+1,
\text{ and } ⟨\mathbb{1},p⟩ = \sum_{i=1}^{n+1} p_i = 1\biggr\},
````
where $\mathbb{1}=(1,…,1)^{\mathrm{T}}∈ ℝ^{n+1}$ denotes the vector containing only ones.

If `boundary` is set to `:open`, then the object represents an open simplex. Otherwise,
that is when `boundary` is set to `:closed`, the boundary is also included:
````math
\hat{Δ}^n := \biggl\{ p ∈ ℝ^{n+1}\ \big|\ p_i \geq 0 \text{ for all } i=1,…,n+1,
\text{ and } ⟨\mathbb{1},p⟩ = \sum_{i=1}^{n+1} p_i = 1\biggr\},
````

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

This implementation follows the notation in [AastroemPetraSchmitzerSchnoerr:2017](@cite).

# Constructor

    ProbabilitySimplex(n::Int; boundary::Symbol=:open)
"""
struct ProbabilitySimplex{n,boundary} <: AbstractDecoratorManifold{ℝ} end

function ProbabilitySimplex(n::Int; boundary::Symbol=:open)
    if boundary !== :open && boundary !== :closed
        throw(
            ArgumentError(
                "boundary can only be set to :open or :closed; received $boundary",
            ),
        )
    end
    return ProbabilitySimplex{n,boundary}()
end

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

Given a tangent vector with respect to the metric from the embedding, the [`EuclideanMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.EuclideanMetric),
the representer of a linear functional on the tangent space is adapted as ``Z = p .* X .- p .* dot(p, X)``.
The first part “compensates” for the divsion by ``p`` in the Riemannian metric on the [`ProbabilitySimplex`](@ref)
and the second part performs appropriate projection to keep the vector tangent.

For details see Proposition 2.3 in [AastroemPetraSchmitzerSchnoerr:2017](@cite).
"""
change_representer(::ProbabilitySimplex, ::EuclideanMetric, ::Any, ::Any)

function change_representer!(::ProbabilitySimplex, Y, ::EuclideanMetric, p, X)
    return Y .= p .* X .- p .* dot(p, X)
end

@doc raw"""
    change_metric(M::ProbabilitySimplex, ::EuclideanMetric, p, X)

To change the metric, we are looking for a function ``c\colon T_pΔ^n \to T_pΔ^n`` such that for all ``X,Y ∈ T_pΔ^n``
This can be achieved by rewriting representer change in matrix form as `(Diagonal(p) - p * p') * X`
and taking square root of the matrix
"""
change_metric(::ProbabilitySimplex, ::EuclideanMetric, ::Any, ::Any)

function change_metric!(::ProbabilitySimplex, Y, ::EuclideanMetric, p, X)
    return Y .= sqrt(Diagonal(p) - p * p') * X
end

"""
    check_point(M::ProbabilitySimplex, p; kwargs...)

Check whether `p` is a valid point on the [`ProbabilitySimplex`](@ref) `M`, i.e. is a point in
the embedding with positive entries that sum to one
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_point(M::ProbabilitySimplex{n,boundary}, p; kwargs...) where {n,boundary}
    if boundary === :closed && minimum(p) < 0
        return DomainError(
            minimum(p),
            "The point $(p) does not lie on the $(M) since it has negative entries.",
        )
    end
    if boundary === :open && minimum(p) <= 0
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
            "The vector $(X) is not a tangent vector to $(p) on $(M), since its elements do not sum up to 0.",
        )
    end
    return nothing
end

@doc raw"""
    distance(M, p, q)

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

embed(::ProbabilitySimplex, p) = p
embed(::ProbabilitySimplex, p, X) = X

@doc raw"""
    exp(M::ProbabilitySimplex, p, X)

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

function get_coordinates_orthonormal!(
    M::ProbabilitySimplex{N},
    Xc,
    p,
    X,
    R::RealNumbers,
) where {N}
    get_coordinates_orthonormal!(
        Sphere(N),
        Xc,
        simplex_to_amplitude(M, p),
        simplex_to_amplitude_diff(M, p, X),
        R,
    )
    return Xc
end

get_embedding(M::ProbabilitySimplex) = Euclidean(representation_size(M)...; field=ℝ)

function get_vector_orthonormal!(
    M::ProbabilitySimplex{N},
    Y,
    p,
    Xc,
    R::RealNumbers,
) where {N}
    ps = simplex_to_amplitude(M, p)
    X = get_vector_orthonormal(Sphere(N), ps, Xc, R)
    return amplitude_to_simplex_diff!(M, Y, ps, X)
end

@doc raw"""
    injectivity_radius(M, p)

Compute the injectivity radius on the [`ProbabilitySimplex`](@ref) `M` at the point `p`,
i.e. the distanceradius to a point near/on the boundary, that could be reached by following the
geodesic.
"""
function injectivity_radius(::ProbabilitySimplex{n}, p) where {n}
    i = argmin(p)
    s = sum(p) - p[i]
    return 2 * acos(sqrt(s))
end
function injectivity_radius(M::ProbabilitySimplex, p, ::AbstractRetractionMethod)
    return injectivity_radius(M, p)
end
injectivity_radius(M::ProbabilitySimplex) = 0
injectivity_radius(M::ProbabilitySimplex, ::AbstractRetractionMethod) = 0

@doc raw"""
    inner(M::ProbabilitySimplex, p, X, Y)

Compute the inner product of two tangent vectors `X`, `Y` from the tangent space $T_pΔ^n$ at
`p`. The formula reads
````math
g_p(X,Y) = \sum_{i=1}^{n+1}\frac{X_iY_i}{p_i}
````
When `M` includes boundary, we can just skip coordinates where ``p_i`` is equal to 0, see
Proposition 2.1 in [AyJostLeSchwachhoefer:2017](@cite).
"""
function inner(::ProbabilitySimplex{n,boundary}, p, X, Y) where {n,boundary}
    d = zero(Base.promote_eltype(p, X, Y))
    if boundary === :closed
        @inbounds for i in eachindex(p, X, Y)
            if p[i] > 0
                d += X[i] * Y[i] / p[i]
            end
        end
    else
        @inbounds for i in eachindex(p, X, Y)
            d += X[i] * Y[i] / p[i]
        end
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

"""
    is_flat(::ProbabilitySimplex)

Return false. [`ProbabilitySimplex`](@ref) is not a flat manifold.
"""
is_flat(M::ProbabilitySimplex) = false

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
    manifold_volume(::ProbabilitySimplex{n}) where {n}

Return the volume of the [`ProbabilitySimplex`](@ref), i.e. volume of the `n`-dimensional
[`Sphere`](@ref) divided by ``2^{n+1}``, corresponding to the volume of its positive
orthant.
"""
manifold_volume(::ProbabilitySimplex{n}) where {n} = manifold_volume(Sphere(n)) / 2^(n + 1)

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

default_estimation_method(::ProbabilitySimplex, ::typeof(mean)) = GeodesicInterpolation()

function parallel_transport_to!(M::ProbabilitySimplex{N}, Y, p, X, q) where {N}
    q_s = simplex_to_amplitude(M, q)
    Ys = parallel_transport_to(
        Sphere(N),
        simplex_to_amplitude(M, p),
        simplex_to_amplitude_diff(M, p, X),
        q_s,
    )
    return amplitude_to_simplex_diff!(M, Y, q_s, Ys)
end

@doc raw"""
    rand(::ProbabilitySimplex; vector_at=nothing, σ::Real=1.0)


When `vector_at` is `nothing`, return a random (uniform over the Fisher-Rao metric; that is, uniform with respect to the `n`-sphere whose positive orthant is mapped to the simplex).
point `x` on the [`ProbabilitySimplex`](@ref) manifold `M` according to the isometric embedding into
the `n`-sphere by normalizing the vector length of a sample from a multivariate Gaussian. See [Marsaglia:1972](@cite).

When `vector_at` is not `nothing`, return a (Gaussian) random vector from the tangent space
``T_{p}\mathrm{\Delta}^n``by shifting a multivariate Gaussian with standard deviation `σ`
to have a zero component sum.
"""
rand(::ProbabilitySimplex; σ::Real=1.0)

function Random.rand!(
    rng::AbstractRNG,
    M::ProbabilitySimplex,
    pX;
    vector_at=nothing,
    σ=one(eltype(pX)),
)
    if isnothing(vector_at)
        Random.randn!(rng, pX)
        LinearAlgebra.normalize!(pX, 2)
        pX .= abs2.(pX)
    else
        Random.randn!(rng, pX)
        pX .= (pX .- mean(pX)) .* σ
        change_metric!(M, pX, EuclideanMetric(), vector_at, pX)
    end
    return pX
end

@doc raw"""
    project(M::ProbabilitySimplex, p, Y)

Project `Y` from the embedding onto the tangent space at `p` on
the [`ProbabilitySimplex`](@ref) `M`. The formula reads

````math
\operatorname{proj}_{Δ^n}(p,Y) = Y - \bar{Y}
```
where ``\bar{Y}`` denotes mean of ``Y``.
"""
project(::ProbabilitySimplex, ::Any, ::Any)

function project!(::ProbabilitySimplex, X, p, Y)
    X .= Y .- mean(Y)
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

Return the representation size of points in the $n$-dimensional probability simplex,
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

function retract_softmax!(::ProbabilitySimplex, q, p, X, t::Number)
    s = zero(eltype(q))
    @inbounds for i in eachindex(q, p, X)
        q[i] = p[i] * exp(t * X[i])
        s += q[i]
    end
    q ./= s
    return q
end

@doc raw"""
    X = riemannian_gradient(M::ProbabilitySimplex{n}, p, Y)
    riemannian_gradient!(M::ProbabilitySimplex{n}, X, p, Y)

Given a gradient ``Y = \operatorname{grad} \tilde f(p)`` in the embedding ``ℝ^{n+1}`` of the
[`ProbabilitySimplex`](@ref) ``Δ^n``, this function computes the Riemannian gradient
``X = \operatorname{grad} f(p)`` where ``f`` is the function ``\tilde f`` restricted to the manifold.

The formula reads

```math
    X = p ⊙ Y - ⟨p, Y⟩p,
```
where ``⊙`` denotes the emelementwise product.
"""
riemannian_gradient(M::ProbabilitySimplex, p, Y; kwargs...)

function riemannian_gradient!(M::ProbabilitySimplex, X, p, Y; kwargs...)
    X .= p .* Y - dot(p, Y) .* p
    return X
end

@doc raw"""
    riemann_tensor(::ProbabilitySimplex, p, X, Y, Z)

Compute the Riemann tensor ``R(X,Y)Z`` at point `p` on [`ProbabilitySimplex`](@ref) `M`.
It is computed using isometry with positive orthant of a sphere.
"""
riemann_tensor(::ProbabilitySimplex, p, X, Y, Z)

function riemann_tensor!(M::ProbabilitySimplex{N}, Xresult, p, X, Y, Z) where {N}
    pe = simplex_to_amplitude(M, p)
    Xrs = riemann_tensor(
        Sphere(N),
        pe,
        simplex_to_amplitude_diff(M, p, X),
        simplex_to_amplitude_diff(M, p, Y),
        simplex_to_amplitude_diff(M, p, Z),
    )
    amplitude_to_simplex_diff!(M, Xresult, pe, Xrs)
    return Xresult
end

function Base.show(io::IO, ::ProbabilitySimplex{n,boundary}) where {n,boundary}
    return print(io, "ProbabilitySimplex($(n); boundary=:$boundary)")
end

@doc raw"""
    volume_density(M::ProbabilitySimplex{N}, p, X) where {N}

Compute the volume density at point `p` on [`ProbabilitySimplex`](@ref) `M` for tangent
vector `X`. It is computed using isometry with positive orthant of a sphere.
"""
function volume_density(M::ProbabilitySimplex{N}, p, X) where {N}
    pe = simplex_to_amplitude(M, p)
    return volume_density(Sphere(N), pe, simplex_to_amplitude_diff(M, p, X))
end

@doc raw"""
    zero_vector(M::ProbabilitySimplex, p)

Return the zero tangent vector in the tangent space of the point `p`  from the
[`ProbabilitySimplex`](@ref) `M`, i.e. its representation by the zero vector in the embedding.
"""
zero_vector(::ProbabilitySimplex, ::Any)

zero_vector!(::ProbabilitySimplex, X, p) = fill!(X, 0)

@doc raw"""
    simplex_to_amplitude(M::ProbabilitySimplex, p)

Convert point `p` on [`ProbabilitySimplex`](@ref Manifolds.ProbabilitySimplex) to (real) probability amplitude. The
formula reads ``(\sqrt{p_1}, \sqrt{p_2}, …, \sqrt{p_{N+1}})``. This is an isometry from the
interior of the probability simplex to the interior of the positive orthant of a sphere.
"""
function simplex_to_amplitude(M::ProbabilitySimplex, p)
    return simplex_to_amplitude!(M, similar(p), p)
end

function simplex_to_amplitude!(::ProbabilitySimplex, q, p)
    q .= sqrt.(p)
    return q
end

@doc raw"""
    amplitude_to_simplex(M::ProbabilitySimplex{N}, p) where {N}

Convert point (real) probability amplitude `p` on to a point on [`ProbabilitySimplex`](@ref Manifolds.ProbabilitySimplex).
The formula reads ``(p_1^2, p_2^2, …, p_{N+1}^2)``. This is an isometry from the interior of
the positive orthant of a sphere to interior of the probability simplex.
"""
function amplitude_to_simplex(M::ProbabilitySimplex, p)
    return amplitude_to_simplex!(M, similar(p), p)
end

function amplitude_to_simplex!(::ProbabilitySimplex, q, p)
    q .= p .^ 2
    return q
end

@doc raw"""
    simplex_to_amplitude_diff(M::ProbabilitySimplex, p, X)

Compute differential of [`simplex_to_amplitude`](@ref Manifolds.simplex_to_amplitude) of a point on `p` one
[`ProbabilitySimplex`](@ref) at tangent vector `X` from the tangent space at `p` from
a sphere.
"""
function simplex_to_amplitude_diff(M::ProbabilitySimplex, p, X)
    return simplex_to_amplitude_diff!(M, similar(X), p, X)
end

function simplex_to_amplitude_diff!(::ProbabilitySimplex, Y, p, X)
    Y .= X ./ sqrt.(p)
    return Y
end

@doc raw"""
    amplitude_to_simplex_diff(M::ProbabilitySimplex, p, X)

Compute differential of [`amplitude_to_simplex`](@ref Manifolds.amplitude_to_simplex) of a point `p` on
[`ProbabilitySimplex`](@ref) at tangent vector `X` from the tangent space at `p` from
a sphere.
"""
function amplitude_to_simplex_diff(M::ProbabilitySimplex, p, X)
    return amplitude_to_simplex_diff!(M, similar(X), p, X)
end

function amplitude_to_simplex_diff!(::ProbabilitySimplex, Y, p, X)
    Y .= p .* X
    return Y
end
