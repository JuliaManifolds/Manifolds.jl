@doc raw"""
    ProbabilitySimplex{n} <: AbstractEmbeddedManifold{ℝ,DefaultEmbeddingType}

The (relative interior of) the probability simplex is the set
````math
𝒮^{n} := \bigl\{ p \in ℝ^{n+1}\ \big|\ p_i > 0 \text{for all} i=1,…,n+1
\text{ and } ⟨𝟙,p⟩ = \sum_{i=1}^{n+1} p_i = 1\bigr\},
````
where $𝟙=(1,…,1)^\mathrm{T}\in ℝ^{n+1}$ denotes the vector containing only ones.

The tangent space is given by
````math
T_p𝒮 = \bigl\{ X \in ℝ^{n+1}\ \big|\ ⟨𝟙,X⟩ = \sum_{i=1}^{n+1} X_i = 0 \bigr\}
````

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

@doc raw"""
    distance(M,p,q)

Compute the sistance of two points on the [`ProbabilitySimplex`](@ref) `M`.
The formula reads
````math
d_{𝒮}(p,q) = 2\arccos\Bigl( \sum_{i=1}^{n+1} \sqrt{p_iq_i})
````
"""
distance(::ProbabilitySimplex,p,q) = 2*arccos(sum(sqrt.(p.*q)))

embed!(M::ProbabilitySimplex, q, p) = (q .= p)
embed!(M::ProbabilitySimplex, Y, p, X) = (Y .= X)

@doc raw"""
    exp(M::ProbabilitySimplex,p,X)

Compute the exponential map on the probability simplex.

````math
\exp_pX = \frac{1}{2}\bigl(p+\frac{1}{\lVert X_p \rVert^2}X_p^2\bigr)
+ \frac{1}{2}\bigl(p - \frac{1}{\lVert X_p \rVert^2}X_p^2)\cos(\lVertX_p\rVert^2)
+ \frac{1}{\lVert Xp \rVert}\sqrt{p}\sin(\lVert X_p\rVert),
````
where $X_p = \frac{X}{\sqrt{p}}$, with its division meant elementwise, as well as for the
operations $X_p^2$ and $\sqrt{p}$.
"""
function exp!(::ProbabilitySimplex, q, p, X)
    Xp = X./sqrt.(p)
    Xp_n = Xp./norm(Xp)
    q .= 0.5*(p+Xp_n.^2) + 0.5*(p-Xp_n.^2).*cos(norm(Xp)) + Xp_n.*sqrt.(p).*sin(norm(Xp))
    return q
end

@doc raw"""
    inner(M::ProbabilitySimplex,p,X,Y)

Compute the inner product of two tangent vectors `X`, `Y` from the tangent space $T_p𝒮$ at
`p`. The formula reads
````math
g_p(X,Y) = \sum_{i=1}^{n+1}\frac{X_iY_i}{p}
````
"""
inner(::ProbabilitySimplex, p, X, Y) =

@doc raw"""
    log(M::ProbabilitySimplex, p, q)

Compute the logarithmic map of `p` and `q` on the [`ProbabilitySimplex`](@ref) `M`.

````math
\log_pq = \frac{d_{𝒮}(p,q)}
````
"""
function log!(M::ProbabilitySimplex, X, p, q)
    s = dot(sqrt.(p),sqrt.(q))
    X .= distance(M,p,q)./(1-s^2)*(sqrt.(p.*q) - s*p )
    return X
end

manifold_dimension(::ProbabilitySimplex{n}) where {n} = n

project!(::ProbabilitySimplex, X, p, Y) = X .= (p.*( Y .- dot(p,Y)))

representation_size(::ProbabilitySimplex{n}) where {n} = (n+1,)

@doc raw"""
    retract(M::ProbabilitySimplex, p, X, ::ProjectionRetracion)

Compute a first order approximation by projection. The formula reads
````math
\operatorname{retr}_p X = \frac{p\mathrm{e}^X}{⟨p,\mathrm{e}^X⟩},
````
where multiplication, exponentiation and division are meant elementwise.
"""
retract(::ProbabilitySimplex, ::Any, ::Any, ::ProjectionRetraction) =

function retract!(::ProbabilitySimplex, q, p, X, ::ProjectionRetraction)
    q .= p.*exp.(X)./(dot(p,exp.(X)))
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
