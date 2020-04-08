@doc raw"""
    ProbabilitySimplex{n} <: AbstractEmbeddedManifold{ℝ,DefaultEmbeddingType}

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

We employ this set and its tangent spaces with the [`FisherRaoMetric`](@ref) exploiting
the isometry to the $n$-sphere of radius 2, i.e. a scaled version of the [`Sphere`](@ref).

This implementation follows the notation in [^ÅströmPetraSchmitzerSchnörr2017]

[^ÅströmPetraSchmitzerSchnörr2017]:
    > F. Åström, S. Petra, B. Schmitzer, C. Schnörr: “Image Labeling by Assignment”,
    > Journal of Mathematical Imaging and Vision, 58(2), pp. 221–238, 2017.
    > doi: [10.1007/s10851-016-0702-4](https://doi.org/10.1007/s10851-016-0702-4)
    > arxiv: [1603.05285](https://arxiv.org/abs/1603.05285).
"""
struct ProbabilitySimplex{n} <: AbstractEmbeddedManifold{ℝ,DefaultEmbeddingType} end

ProbabilitySimplex(n::Int) = ProbabilitySimplex{n}()

"""
    SoftmaxRetraction <: AbstractRetractionMethod

Describes a retraction that is based on the Softmax function.
"""
struct SoftmaxRetraction <: AbstractRetractionMethod end

"""
    SoftmaxInverseRetraction <: AbstractInverseRetractionMethod

Describes an inverse retraction that is based on the Softmax function.
"""
struct SoftmaxInverseRetraction <: AbstractInverseRetractionMethod end

"""
    FisherRaoMetric <: Metric

The Fisher-Rao metric or Fisher information metric is a particular Riemannian metric which
can be defined on a smooth statistical manifold, i.e., a smooth manifold whose points are
probability measures defined on a common probability space.

It is currently used in the discrete case for the [`ProbabilitySimplex`](@ref).
"""
struct FisherRaoMetric <: Metric end

"""
    check_manifold_point(M::ProbabilitySimplex, p; kwargs...)

Check whether `p` is a valid point on the [`ProbabilitySimplex`](@ref) `M`, i.e. is a point in
the embedding with positive entries that sum to one
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::ProbabilitySimplex, p; kwargs...)
    mpv = invoke(
        check_manifold_point,
        Tuple{(typeof(get_embedding(M))), typeof(p)},
        get_embedding(M),
        p;
        kwargs...
    )
    mpv === nothing || return mpv
    if any(p .<= 0)
        return DomainError(
            norm(p),
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
    check_tangent_vector(M::ProbabilitySimplex, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to `p` on the [`ProbabilitySimplex`](@ref) `M`, i.e.
after [`check_manifold_point`](@ref)`(M,p)`, `X` has to be of same dimension as `p`
and its elements have to sum to one.
The optional parameter `check_base_point` indicates, whether to call
[`check_manifold_point`](@ref)  for `p` or not.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(
    M::ProbabilitySimplex,
    p,
    X;
    check_base_point = true,
    kwargs...,
)
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{typeof(get_embedding(M)), typeof(p), typeof(X)},
        get_embedding(M),
        p,
        X;
        check_base_point = false, # already checked above
        kwargs...
    )
    mpv === nothing || return mpv
    if !isapprox(sum(X), 0.0; kwargs...)
        return DomainError(
            sum(X),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since its elements to not sum up to 0.",
        )
    end
    return nothing
end

decorated_manifold(M::ProbabilitySimplex) = Euclidean(representation_size(M)...; field = ℝ)

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
    return 2*acos(sumsqrt)
end

embed!(::ProbabilitySimplex, q, p) = copyto!(q, p)
embed!(::ProbabilitySimplex, Y, p, X) = copyto!(Y, X)

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
    q .= (cos(θ) .* s .+ usinc(θ) .* Xs).^2
    return q
end

@doc raw"""
    inner(M::ProbabilitySimplex,p,X,Y)

Compute the inner product of two tangent vectors `X`, `Y` from the tangent space $T_pΔ^n$ at
`p`. The formula reads
````math
g_p(X,Y) = \sum_{i=1}^{n+1}\frac{X_iY_i}{p}
````
"""
function inner(::ProbabilitySimplex, p, X, Y)
    d = zero(Base.promote_eltype(p,X,Y))
    @inbounds for i in eachindex(p,X,Y)
        d += X[i]*Y[i]/p[i]
    end
    return d
end

@doc raw"""
    inverse_retract(M::ProbabilitySimplex, p, q, ::SoftmaxInverseRetraction)

Compute a first order approximation by projection. The formula reads
````math
\operatorname{retr}^{-1}_p q = \bigl( I - \frac{1}{n}\mathbb{1}\mathbb{1}^\mathrm{T}\bigr)(\log(q)-\log(p))
````
where $\mathbb{1}$ is the column vector containing ones and $\log$ is applied elementwise.
"""
inverse_retract(::ProbabilitySimplex, ::Any, ::Any, ::SoftmaxInverseRetraction)

function inverse_retract!(::ProbabilitySimplex{n}, X, p, q, ::SoftmaxInverseRetraction) where {n}
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

function log!(M::ProbabilitySimplex, X, p, q)
    z = sqrt.(p .* q)
    s = sum(z)
    X .= 2acos(s) / sqrt(1 - s^2) .* (z .- s .* p)
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
    project(M::ProbabilitySimplex, p, Y)

project `Y` from the embedding onto the tangent space at `p` on
the [`ProbabilitySimplex`](@ref) `M`. The formula reads

````math
\operatorname{proj}_{Δ^n}(p,Y) = p\bigl(Y - ⟨p,Y⟩\mathbb{1}),
````

where multiplication is meant elementwise and $\mathbb{1}$ is the vector of ones.
"""
project(::ProbabilitySimplex, ::Any, ::Any)

function project!(::ProbabilitySimplex, X, p, Y)
    s = zero(eltype(X))
    @inbounds for i in eachindex(X, p, Y)
        X[i] = p[i] * Y[i]
        s += X[i]
    end
    X .-= s .* p
    return X
end

@doc raw"""
    representation_size(::ProbabilitySimplex{n})

return the representation size of points in the $n$-dimensional probability simplex,
i.e. an array size of `(n+1,)`.
"""
representation_size(::ProbabilitySimplex{n}) where {n} = (n+1,)

@doc raw"""
    retract(M::ProbabilitySimplex, p, X, ::SoftmaxRetraction)

Compute a first order approximation by applying the softmax function. The formula reads

````math
\operatorname{retr}_p X = \frac{p\mathrm{e}^X}{⟨p,\mathrm{e}^X⟩},
````

where multiplication, exponentiation and division are meant elementwise.
"""
retract(::ProbabilitySimplex, ::Any, ::Any, ::SoftmaxRetraction)

function retract!(::ProbabilitySimplex, q, p, X, ::SoftmaxRetraction)
    s = zero(eltype(q))
    @inbounds for i in eachindex(q, p, X)
        q[i] = p[i] * exp(X[i])
        s += q[i]
    end
    q ./= s
    return q
end

function show(io::IO, ::ProbabilitySimplex{n}) where {n}
    print(io, "ProbabilitySimplex($(n))")
end

@doc raw"""
    zero_tangent_vector(M::ProbabilitySimplex,p)

returns the zero tangent vector in the tangent space of the point `p`  from the
[`ProbabilitySimplex`](@ref) `M`, i.e. its representation by the zero vector in the embedding.
"""
zero_tangent_vector(::ProbabilitySimplex, ::Any)

zero_tangent_vector!(M::ProbabilitySimplex, v, p) = fill!(v, 0)
