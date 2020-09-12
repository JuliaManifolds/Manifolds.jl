function check_manifold_point(M::Hyperbolic, p; kwargs...)
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    if !isapprox(minkowski_metric(p, p), -1.0; kwargs...)
        return DomainError(
            minkowski_metric(p, p),
            "The point $(p) does not lie on $(M) since its Minkowski inner product is not -1.",
        )
    end
    return nothing
end
function check_manifold_point(M::Hyperbolic, p::HyperboloidPoint; kwargs...)
    return check_manifold_point(M, p.value; kwargs...)
end

function check_tangent_vector(M::Hyperbolic, p, X; check_base_point = true, kwargs...)
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{supertype(typeof(M)),typeof(p),typeof(X)},
        M,
        p,
        X;
        check_base_point = false, # already checked above
        kwargs...,
    )
    mpv === nothing || return mpv
    if !isapprox(minkowski_metric(p, X), 0.0; kwargs...)
        return DomainError(
            abs(minkowski_metric(p, X)),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not orthogonal (with respect to the Minkowski inner product) in the embedding.",
        )
    end
    return nothing
end
function check_tangent_vector(
    M::Hyperbolic,
    p::HyperboloidPoint,
    X::HyperboloidTVector;
    kwargs...,
)
    return check_tangent_vector(M, p.value, X.value; kwargs...)
end

function convert(::Type{HyperboloidTVector}, x::T) where {T<:AbstractVector}
    return HyperboloidTVector(x)
end
convert(::Type{<:AbstractVector}, x::HyperboloidTVector) = x.value
function convert(::Type{HyperboloidPoint}, x::T) where {T<:AbstractVector}
    return HyperboloidPoint(x)
end
convert(::Type{<:AbstractVector}, x::HyperboloidPoint) = x.value

@doc raw"""
    convert(::Type{HyperboloidPoint}, p::PoincareBallPoint)
    convert(::Type{<:AbstractVector}, p::PoincareBallPoint)

convert a point [`PoincareBallPoint`](@ref) `x` (from $ℝ^n$) from the
Poincaré ball model of the [`Hyperbolic`](@ref) manifold $ℍ^n$ to a [`HyperboloidPoint`](@ref) $π(p) ∈ ℝ^{n+1}$.
The isometry is defined by

````math
π(p) = \frac{1}{1+\lVert p \rVert^2}
\begin{pmatrix}2p_1\\\vdots\\2p_n\\1+\lVert p \rVert^2\end{pmatrix}
````

Note that this is also used, when the type to convert to is a vector.
"""
function convert(::Type{HyperboloidPoint}, p::PoincareBallPoint)
    return HyperboloidPoint(convert(AbstractVector, p))
end
function convert(::Type{<:AbstractVector}, p::PoincareBallPoint)
    return 1 / (1 - norm(p.value)^2) .* [(2 .* p.value)..., 1 + norm(p.value)^2]
end

@doc raw"""
    convert(::Type{HyperboloidPoint, p::PoincareHalfSpacePoint)
    convert(::Type{<:AbstractVector}, x::PoincareHalfSpacePoint)

convert a point [`PoincareHalfSpacePoint`](@ref) `p` (from $ℝ^n$) from the
Poincaré half plane model of the [`Hyperbolic`](@ref) manifold $ℍ^n$ to a [`HyperboloidPoint`](@ref) $π(p) ∈ ℝ^{n+1}$.

This is done in two steps, namely transforming it to a Poincare ball point and from there further on to a Hyperboloid point.
"""

function convert(t::Type{HyperboloidPoint}, p::PoincareHalfSpacePoint)
    return convert(t, convert(PoincareBallPoint, p))
end
function convert(t::Type{<:AbstractVector}, p::PoincareHalfSpacePoint)
    return convert(t, convert(PoincareBallPoint, p))
end

@doc raw"""
    distance(M::Hyperbolic, p, q)
    distance(M::Hyperbolic, p::HyperboloidPoint, q::HyperboloidPoint)

Compute the distance on the [`Hyperbolic`](@ref) `M`, which reads

````math
d_{ℍ^n}(p,q) = \operatorname{acosh}( - ⟨p, q⟩_{\mathrm{M}}),
````

where $⟨\cdot,\cdot⟩_{\mathrm{M}}$ denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold.
"""
distance(::Hyperbolic, p, q) = acosh(max(-minkowski_metric(p, q), 1.0))
function distance(M::Hyperbolic, p::HyperboloidPoint, q::HyperboloidPoint)
    return distance(M, p.value, q.value)
end

function exp!(M::Hyperbolic, q, p, X)
    vn = sqrt(max(inner(M, p, X, X), 0.0))
    vn < eps(eltype(p)) && return copyto!(q, p)
    return copyto!(q, cosh(vn) * p + sinh(vn) / vn * X)
end

@doc raw"""
    inner(M::Hyperbolic{n}, p, X, Y)
    inner(M::Hyperbolic{n}, p::HyperboloidPoint, X::HyperboloidTVector, Y::HyperboloidTVector)

Cmpute the inner product in the Hyperboloid model, i.e. the [`minkowski_metric`](@ref) in
the embedding. The formula reads

````math
g_p(X,Y) = ⟨X,Y⟩_{\mathrm{M}} = -X_{n}Y_{n} + \displaystyle\sum_{k=1}^{n-1} X_kY_k.
````
This employs the metric of the embedding, see [`Lorentz`](@ref) space.
"""
function inner(
    M::Hyperbolic,
    p::HyperboloidPoint,
    X::HyperboloidTVector,
    Y::HyperboloidTVector,
)
    return inner(M, p.value, X.value, Y.value)
end

function log!(M::Hyperbolic, X, p, q)
    scp = minkowski_metric(p, q)
    w = q + scp * p
    wn = sqrt(max(scp .^ 2 - 1, 0.0))
    wn < eps(eltype(p)) && return zero_tangent_vector!(M, X, p)
    X .= acosh(max(1.0, -scp)) / wn .* w
    return X
end

function minkowski_metric(a::HyperboloidPoint, b::HyperboloidPoint)
    return minkowski_metric(convert(Vector, a), convert(Vector, b))
end

project!(::Hyperbolic, Y, p, X) = (Y .= X .+ minkowski_metric(p, X) .* p)
function project!(
    ::Hyperbolic,
    Y::HyperboloidTVector,
    p::HyperboloidPoint,
    X::HyperboloidTVector,
)
    return (Y.value .= X.value .+ minkowski_metric(p.value, X.value) .* p.value)
end

function vector_transport_to!(M::Hyperbolic, Y, p, X, q, ::ParallelTransport)
    w = log(M, p, q)
    wn = norm(M, p, w)
    wn < eps(eltype(p + q)) && return copyto!(Y, X)
    return copyto!(Y, X - (inner(M, p, w, X) * (w + log(M, q, p)) / wn^2))
end
