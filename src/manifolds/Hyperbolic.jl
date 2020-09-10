@doc raw"""
    Hyperbolic{N} <: AbstractEmbeddedManifold{ℝ,DefaultIsometricEmbeddingType}

The hyperbolic space $ℍ^n$ represented by $n+1$-Tuples, i.e. embedded in the
[`Lorentz`](@ref)ian manifold equipped with the [`MinkowskiMetric`](@ref)
$⟨\cdot,\cdot⟩_{\mathrm{M}}$. The space is defined as

```math
ℍ^n = \Bigl\{p ∈ ℝ^{n+1}\ \Big|\ ⟨p,p⟩_{\mathrm{M}}= -p_{n+1}^2
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
struct Hyperbolic{N} <: AbstractEmbeddedManifold{ℝ,DefaultIsometricEmbeddingType} end

Hyperbolic(n::Int) = Hyperbolic{n}()

#
# Representations
#

@doc raw"""
    HyperboloidPoint <: MPoint

In the Hyperboloid model of the [`Hyperbolic`](@ref) $ℍ^n$ points are represented
as vectors in $ℝ^{n+1}$ with [`MinkowskiMetric`](@ref) equal to $-1$.

This representation is the default, i.e. vectors are assumed to have this repesentation.
"""
struct HyperboloidPoint{TValue<:AbstractVector} <: MPoint
    value::TValue
end

@doc raw"""
    HyperboloidTVector <: TVector

In the Hyperboloid model of the [`Hyperbolic`](@ref) $ℍ^n$ tangent vctors are represented
as vectors in $ℝ^{n+1}$ with [`MinkowskiMetric`](@ref) $⟨p,X⟩_{\mathrm{M}}=0$ to their base
point $p$.

This representation is the default, i.e. vectors are assumed to have this repesentation.
"""
struct HyperboloidTVector{TValue<:AbstractVector} <: MPoint
    value::TValue
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
    PoincareBallPoint <: MPoint

A point on the [`Hyperbolic`](@ref) manifold $ℍ^n$ can be represented as a vector of norm
less than one in $\mathbb R^n$.
"""
struct PoincareBallPoint{TValue<:AbstractVector} <: MPoint
    value::TValue
end

@doc raw"""
    PoincareBallTVector <: TVector

In the Poincaré ball model of the [`Hyperbolic`](@ref) $ℍ^n$ tangent vctors are represented
as vectors in $ℝ^{n}$.
"""
struct PoincareBallTVector{TValue<:AbstractVector} <: MPoint
    value::TValue
end

@doc raw"""
    PoincareHalfSpacePoint <: MPoint

A point on the [`Hyperbolic`](@ref) manifold $ℍ^n$ can be represented as a vector in the
half plane, i.e. $x ∈ ℝ^n$ with $x_d > 0$.
"""
struct PoincareHalfSpacePoint{TValue<:AbstractVector} <: MPoint
    value::TValue
end

@doc raw"""
    PoincareHalfPlaneTVector <: TVector

In the Poincaré half plane model of the [`Hyperbolic`](@ref) $ℍ^n$ tangent vctors are
represented as vectors in $ℝ^{n}$.
"""
struct PoincareHalfSpaceTVector{TValue<:AbstractVector} <: MPoint
    value::TValue
end

@doc raw"""
    convert(::Type{PoincareBallPoint}, x::HyperboloidPoint)

convert a [`HyperboloidPoint`](@ref) $x∈ℝ^{n+1}$ from the hyperboloid model of the [`Hyperbolic`](@ref)
manifold $ℍ^n$ to a [`PoincareBallPoint`](@ref) $π(x)∈ℝ^{n}$ in the Poincaré ball model.
The isometry is defined by

````math
π(x) = \frac{1}{1+x_{n+1}} \begin{pmatrix}x_1\\\vdots\\x_n\end{pmatrix}
````

Note that this is also used, when `x` is a vector.
"""
function convert(t::Type{PoincareBallPoint}, x::HyperboloidPoint)
    return convert(t, x.value)
end
function convert(::Type{PoincareBallPoint}, x::T) where {T<:AbstractVector}
    return PoincareBallPoint(1 / (1 + last(x)) .* x[1:(end - 1)])
end

@doc raw"""
    convert(::Type{HyperboloidPoint}, x::PoincareBallPoint)

convert a point [`PoincareBallPoint`](@ref) `x` (from $ℝ^n$) from the
Poincaré ball model of the [`Hyperbolic`](@ref) manifold $ℍ^n$ to a [`HyperboloidPoint`](@ref) $π(x) ∈ ℝ^{n+1}$.
The isometry is defined by

````math
π(x) = \frac{1}{1+\lVert x \rVert^2}
\begin{pmatrix}2x_1\\\vdots\\2x_n\\1+\lVert x \rVert^2\end{pmatrix}
````

Note that this is also used, when the type to convert to is a vector.
"""
function convert(::Type{HyperboloidPoint}, x::PoincareBallPoint)
    return HyperboloidPoint(convert(AbstractVector, x))
end
function convert(::Type{<:AbstractVector}, x::PoincareBallPoint)
    return 1 / (1 - norm(x.value)^2) .* [(2 .* x.value)..., 1 + norm(x.value)^2]
end

@doc raw"""
    convert(::Type{PoincareHalfSpacePoint}, x::PoincareBallPoint)

convert a point [`PoincareBallPoint`](@ref) `x` (from $ℝ^n$) from the
Poincaré ball model of the [`Hyperbolic`](@ref) manifold $ℍ^n$ to a [`PoincareHalfSpacePoint`](@ref) $π(x) ∈ ℝ^n$.
Denote by $\tilde x = (x_1,\ldots,x_{d-1})$. Then the isometry is defined by

````math
π(x) = \frac{1}{\lVert \tilde x \rVert^2 - (x_n-1)^2}
\begin{pmatrix}2x_1\\\vdots\\2x_{n-1}\\1-\lVert\tilde x\rVert^2 - x_n^2-1\end{pmatrix}.
````
"""
function convert(::Type{PoincareHalfSpacePoint}, x::PoincareBallPoint)
    return PoincareHalfSpacePoint(
        1 / (norm(x.value[1:(end - 1)])^2 + (last(x.value) - 1)^2) .*
        [x.value[1:(end - 1)]..., 1 - norm(x.value[1:(end - 1)])^2 - last(x.value)^2],
    )
end

@doc raw"""
    convert(::Type{PoincareBallPoint}, x::PoincareHalfSpacePoint)

convert a point [`PoincareHalfSpacePoint`](@ref) `x` (from $ℝ^n$) from the
Poincaré half plane model of the [`Hyperbolic`](@ref) manifold $ℍ^n$ to a [`PoincareBallPoint`](@ref) $π(x) ∈ ℝ^n$.
Denote by $\tilde x = (x_1,\ldots,x_{d-1})$. Then the isometry is defined by

````math
π(x) = \frac{1}{\lVert \tilde x \rVert^2 + (x_n+1)^2}
\begin{pmatrix}2x_1\\\vdots\\2x_{n-1}\\\lVert\tilde x\rVert^2 + x_n^2-1\end{pmatrix}.
````
"""
function convert(::Type{PoincareBallPoint}, x::PoincareHalfSpacePoint)
    return PoincareBallPoint(
        1 / (norm(x.value[1:(end - 1)])^2 + (last(x.value) + 1)^2) .*
        [x.value[1:(end - 1)]..., norm(x.value[1:(end - 1)])^2 + last(x.value)^2 - 1],
    )
end

@doc raw"""
    convert(::Type{HyperboloidPoint, x::PoincareHalfSpacePoint)

convert a point [`PoincareHalfSpacePoint`](@ref) `x` (from $ℝ^n$) from the
Poincaré half plane model of the [`Hyperbolic`](@ref) manifold $ℍ^n$ to a [`HyperboloidPoint`](@ref) $π(x) ∈ ℝ^{n+1}$.

This is done in two steps, namely transforming it to a Poincare ball point and from there further on to a Hyperboloid point.
"""
function convert(t::Type{HyperboloidPoint}, x::PoincareHalfSpacePoint)
    return convert(t, convert(PoincareBallPoint, x))
end
@doc raw"""
    convert(::Type{<:AbstractVector}, x::PoincareHalfSpacePoint)

convert a point [`PoincareHalfSpacePoint`](@ref) `x` (from $ℝ^n$) from the
Poincaré half plane model of the [`Hyperbolic`](@ref) manifold $ℍ^n$ to a [`HyperboloidPoint`](@ref) $π(x) ∈ ℝ^{n+1}$.

This is done in two steps, namely transforming it to a Poincare ball point and from there further to an Poincaré half plane point.
"""
function convert(t::Type{<:AbstractVector}, x::PoincareHalfSpacePoint)
    return convert(t, convert(PoincareBallPoint, x))
end

@doc raw"""
    convert(::Type{PoincareHalfSpacePoint}, x::Hyperboloid)

convert a point [`HyperboloidPoint`](@ref) `x` (from $ℝ^{n+1}$) from the
Hyperboloid model of the [`Hyperbolic`](@ref) manifold $ℍ^n$ to a [`PoincareHalfSpacePoint`](@ref) $π(x) ∈ ℝ^{n}$.

This is done in two steps, namely transforming it to a Poincare ball point and from there further on to a PoincareHalfSpacePoint point.
"""
function convert(t::Type{PoincareHalfSpacePoint}, x::HyperboloidPoint)
    return convert(t, convert(PoincareBallPoint, x))
end
@doc raw"""
    convert(::Type{PoincareHalfSpacePoint}, x)

convert a point `x` (from $ℝ^{n+1}$) from the Hyperboloid model of the [`Hyperbolic`](@ref)
manifold $ℍ^n$ to a [`PoincareHalfSpacePoint`](@ref) $π(x) ∈ ℝ^{n}$.

This is done in two steps, namely transforming it to a Poincare ball point and from there further to a vector.
"""
function convert(t::Type{PoincareHalfSpacePoint}, x::T) where {T<:AbstractVector}
    return convert(t, convert(PoincareBallPoint, x))
end

@doc raw"""
    check_manifold_point(M::Hyperbolic, p; kwargs...)

Check whether `p` is a valid point on the [`Hyperbolic`](@ref) `M`.

For the [`HyperboloidPoint`](@ref) or plain vectors this means that, `p` is a vector of
length $n+1$ with inner product in the embedding of -1, see [`MinkowskiMetric`](@ref).
The tolerance for the last test can be set using the `kwargs...`.

For the [`PoincareBallPoint`](@ref) a valid point is a vector $p ∈ ℝ^n$ with a norm stricly
less than 1.

For the [`PoincareHalfSpacePoint`](@ref) a valid point is a vector from $p ∈ ℝ^n$ with a positive
last entry, i.e. $p_n>0$
"""
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
function check_manifold_point(M::Hyperbolic{N}, p::PoincareBallPoint; kwargs...) where {N}
    mpv = check_manifold_point(Euclidean(N), p.value; kwargs...)
    mpv === nothing || return mpv
    if !(norm(p.value) < 1)
        return DomainError(
            norm(p.value),
            "The point $(p) does not lie on $(M) since its norm is not less than 1.",
        )
    end
end
function check_manifold_point(
    M::Hyperbolic{N},
    p::PoincareHalfSpacePoint;
    kwargs...,
) where {N}
    mpv = check_manifold_point(Euclidean(N), p.value; kwargs...)
    mpv === nothing || return mpv
    if !(last(p.value) > 0)
        return DomainError(
            norm(p.value),
            "The point $(p) does not lie on $(M) since its last entry is nonpositive.",
        )
    end
end
@doc raw"""
    check_tangent_vector(M::Hyperbolic{n}, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to `p` on the [`Hyperbolic`](@ref) `M`, i.e.
after [`check_manifold_point`](@ref)`(M,p)`, `X` has to be of the same dimension as `p`.
The optional parameter `check_base_point` indicates whether to
call [`check_manifold_point`](@ref)  for `p`. The tolerance for the last test can be set
using the `kwargs...`.

For a the hyperboloid model or vectors, `X` has to be  orthogonal to `p` with respect
to the inner product from the embedding, see [`MinkowskiMetric`](@ref).

For a the Poincaré ball as well as the Poincaré half plane model, `X` has to be a vector from $ℝ^{n}$.
"""
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
function check_tangent_vector(
    M::Hyperbolic{N},
    p,
    X::Union{PoincareBallTVector,PoincareHalfSpaceTVector};
    check_base_point = true,
    kwargs...,
) where {N}
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    return check_manifold_point(Euclidean(N), X.value; kwargs...)
end


decorated_manifold(::Hyperbolic{N}) where {N} = Lorentz(N + 1, MinkowskiMetric())

default_metric_dispatch(::Hyperbolic, ::MinkowskiMetric) = Val(true)

@doc raw"""
    distance(M::Hyperbolic, p, q)

Compute the distance on the [`Hyperbolic`](@ref) `M`, which reads

````math
d_{ℍ^n}(p,q) = \operatorname{acosh}( - ⟨p, q⟩_{\mathrm{M}}),
````

where $⟨\cdot,\cdot⟩_{\mathrm{M}}$ denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold.
"""
distance(::Hyperbolic, p, q) = acosh(max(-minkowski_metric(p, q), 1.0))
function distance(M::Hyperbolic, p::PoincareBallPoint{T}, q::PoincareBallPoint{T}) where {T}
    return distance(M, convert(AbstractVector, p), convert(AbstractVector, q))
end
function distance(
    M::Hyperbolic,
    p::PoincareHalfSpacePoint{T},
    q::PoincareHalfSpacePoint{T},
) where {T}
    return distance(M, convert(AbstractVector, p), convert(AbstractVector, q))
end

embed!(::Hyperbolic, q, p::T) where {T<:AbstractVector} = (q .= p)
embed!(::Hyperbolic, Y, p, X::T) where {T<:AbstractVector} = (Y .= X)

embed!(::Hyperbolic, q, p::HyperboloidPoint) = (q .= p.value)
embed!(::Hyperbolic, Y, ::HyperboloidPoint, X::HyperboloidTVector) = (Y .= X.value)

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

function exp!(
    M::Hyperbolic,
    q::PoincareBallPoint,
    p::PoincareBallPoint,
    X::PoincareBallTVector,
)
    q.value .=
        convert(
            PoincareBallPoint,
            exp(M, convert(AbstractVector, p), convert(AbstractVector, X)),
        ).value
    return q
end

@doc raw"""
    injectivity_radius(M::Hyperbolic)
    injectivity_radius(M::Hyperbolic, p)

Return the injectivity radius on the [`Hyperbolic`](@ref), which is $∞$.
"""
injectivity_radius(M::Hyperbolic) = Inf
injectivity_radius(M::Hyperbolic, ::ExponentialRetraction) = Inf
injectivity_radius(M::Hyperbolic, ::Any) = Inf
injectivity_radius(M::Hyperbolic, ::Any, ::ExponentialRetraction) = Inf
eval(
    quote
        @invoke_maker 1 Manifold injectivity_radius(
            M::Hyperbolic,
            rm::AbstractRetractionMethod,
        )
    end,
)

function inner(
    ::Hyperbolic,
    p::PoincareHalfSpacePoint,
    X::PoincareHalfSpaceTVector,
    Y::PoincareHalfSpaceTVector,
)
    return dot(X.value, Y.value) / last(p.value)^2
end

@doc raw"""
    log(M::Hyperbolic, p, q)

Compute the logarithmic map on the [`Hyperbolic`](@ref) space $ℍ^n$, the tangent
vector representing the [`geodesic`](@ref) starting from `p`
reaches `q` after time 1. The formula reads for $p ≠ q$

```math
\log_p q = d_{ℍ^n}(p,q)
\frac{q-⟨p,q⟩_{\mathrm{M}} p}{\lVert q-⟨p,q⟩_{\mathrm{M}} p \rVert_2},
```

where $⟨\cdot,\cdot⟩_{\mathrm{M}}$ denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold. For $p=q$ the logarihmic map is equal to the zero vector.
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

function log!(
    M::Hyperbolic,
    X::PoincareBallTVector,
    p::PoincareBallPoint,
    q::PoincareBallPoint,
)
    X.value .=
        convert(
            PoincareBallTVector,
            exp(M, convert(AbstractVector, p), convert(AbstractVector, q)),
        ).value
    return X
end


@doc raw"""
    manifold_dimension(M::Hyperbolic)

Return the dimension of the hyperbolic space manifold $ℍ^n$, i.e. $\dim(ℍ^n) = n$.
"""
manifold_dimension(::Hyperbolic{N}) where {N} = N

"""
    mean(
        M::Hyperbolic,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = CyclicProximalPointEstimation();
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` on the
[`Hyperbolic`](@ref) space using [`CyclicProximalPointEstimation`](@ref).
"""
mean(::Hyperbolic, ::Any...)

function Statistics.mean!(M::Hyperbolic, p, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(M, p, x, w, CyclicProximalPointEstimation(); kwargs...)
end

@doc raw"""
    project(M::Hyperbolic, p, X)

Perform an orthogonal projection with respect to the Minkowski inner product of `X` onto
the tangent space at `p` of the [`Hyperbolic`](@ref) space `M`.

The formula reads
````math
Y = X + ⟨p,X⟩_{\mathrm{M}} p,
````
where $⟨\cdot, \cdot⟩_{\mathrm{M}}$ denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold.

!!! note

    Projection is only available for the (default) [`HyperboloidTVector`](@ref) representation,
    the others don't have such an embedding
"""
project(::Hyperbolic, ::Any, ::Any)

project!(::Hyperbolic, Y, p, X) = (Y .= X .+ minkowski_metric(p, X) .* p)
function project!(
    ::Hyperbolic,
    Y::HyperboloidTVector,
    p::HyperboloidPoint,
    X::HyperboloidTVector,
)
    return (Y.value .= X.value .+ minkowski_metric(p.value, X.value) .* p.value)
end

Base.show(io::IO, ::Hyperbolic{N}) where {N} = print(io, "Hyperbolic($(N))")
Base.show(io::IO, p::HyperboloidPoint) = print(io, "HyperboloidPoint($(p.value))")
Base.show(io::IO, v::HyperboloidTVector) = print(io, "HyperboloidTVector($(v.value))")
Base.show(io::IO, p::PoincareBallPoint) = print(io, "PoincareBallPoint($(p.value))")
Base.show(io::IO, v::PoincareBallTVector) = print(io, "PoincareBallTVector($(v.value))")
function Base.show(io::IO, p::PoincareHalfSpacePoint)
    return print(io, "PoincareHalfSpacePoint($(p.value))")
end
function Base.show(io::IO, v::PoincareHalfSpaceTVector)
    return print(io, "PoincareHalfPlaneTVector($(v.value))")
end

@doc raw"""
    vector_transport_to(M::Hyperbolic, p, X, q, ::ParallelTransport)

Compute the paralllel transport of the `X` from the tangent space at `p` on the
[`Hyperbolic`](@ref) space $ℍ^n$ to the tangent at `q` along the [`geodesic`](@ref)
connecting `p` and `q`. The formula reads

````math
\mathcal P_{q←p}X = X - \frac{⟨\log_p q,X⟩_p}{d^2_{ℍ^n}(p,q)}
\bigl(\log_p q + \log_qp \bigr),
````
where $⟨\cdot,\cdot⟩_p$ denotes the inner product in the tangent space at `p`.
"""
vector_transport_to(::Hyperbolic, ::Any, ::Any, ::Any, ::ParallelTransport)

function vector_transport_to!(M::Hyperbolic, Y, p, X, q, ::ParallelTransport)
    w = log(M, p, q)
    wn = norm(M, p, w)
    wn < eps(eltype(p + q)) && return copyto!(Y, X)
    return copyto!(Y, X - (inner(M, p, w, X) * (w + log(M, q, p)) / wn^2))
end

function zero_tangent_vector(::Hyperbolic, p::PoincareBallPoint)
    return PoincareBallTVector(zeros(p.value))
end
function zero_tangent_vector(::Hyperbolic, p::PoincareHalfSpacePoint)
    return PoincareBallTVector(zeros(p.value))
end

function zero_tangent_vector!(::Hyperbolic, X::PoincareBallTVector, ::PoincareBallPoint)
    return fill!(X.value, 0)
end
function zero_tangent_vector!(
    ::Hyperbolic,
    X::PoincareHalfSpacePoint,
    ::PoincareHalfSpacePoint,
)
    return fill!(X.value, 0)
end
