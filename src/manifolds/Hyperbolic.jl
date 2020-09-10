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

In the Hyperboloid model of the [`Hyperbolic`](@ref) $H^n$ points are represented
as vectors in $ℝ^{n+1}$ with [`MinkowskiMetric`](@ref) equal to $-1$.

This representation is the default, i.e. vectors are assumed to have this repesentation.
"""
struct HyperboloidPoint{T} <: MPoint
    value::Vector{T}
end
function convert(::Type{HyperboloidPoint{T1}}, x::Array{T2,1}) where {T1,T2}
    return HyperboloidPoint{T1}(x)
end
convert(::Array{T1,1}, x::HyperboloidPoint{T2}) where {T1,T2} = x.value

@doc raw"""
    HyperboloidTVector <: TVector

In the Hyperboloid model of the [`Hyperbolic`](@ref) $ℍ^n$ tangent vctors are represented
as vectors in $ℝ^{n+1}$ with [`MinkowskiMetric`](@ref) $⟨p,X⟩_{\mathrm{M}}=0$ to their base
point $p$.

This representation is the default, i.e. vectors are assumed to have this repesentation.
"""
struct HyperboloidTVector{T} <: MPoint
    value::Vector{T}
end
function convert(::Type{HyperboloidTVector{T1}}, x::Array{T2,1}) where {T1,T2}
    return HyperboloidTVector{T1}(x)
end
convert(::Array{T1,1}, x::HyperboloidTVector{T2}) where {T1,T2} = x.value

@doc raw"""
    PoincareBallPoint <: MPoint

A point on the [`Hyperbolic`](@ref) manifold $ℍ^n$ can be represented as a vector of norm
less than one in $\mathbb R^n$.
"""
struct PoincareBallPoint{T} <: MPoint
    value::Vector{T}
end

struct PoincareBallTVector{T} <: MPoint
    value::Vector{T}
end

@doc raw"""
    convert(::Type{PoincareBallPoint}, x::HyperboloidPoint)

convert a [`HyperboloidPoint`](@ref) $x∈ℝ^{n+1}$ from the hyperboloid model of the [`Hyperbolic`](@ref)
manifold $ℍ^n$ to a [`PoincareBallPoint`](@ref) $π(x)∈ℝ^{n}$ in the Poincaré ball model.
The isometry is defined by

````math
π(x) = \frac{1}{1+x_{d+1}} \begin{pmatrix}x_1\\\vdots\\x_d\end{pmatrix}
````

Note that this is also used, when `x` is a vector.
"""
function convert(t::Type{PoincareBallPoint{T1}}, x::HyperboloidPoint{T2}) where {T1,T2}
    return convert(t, x.value)
end
function convert(::Type{PoincareBallPoint{T1}}, x::Array{T2,1}) where {T1,T2}
    return PoincareBallPoint{T1}(1 / (1 + last(x)) .* x[1:(end - 1)])
end

@doc raw"""
    convert(::Type{HyperboloidPoint}, x::PoincareBallPoint)

convert a point [`PoincareBallPoint`](@ref) `x` (from $ℝ^n$) from the
Poincaré ball model of the [`Hyperbolic`](@ref) manifold $ℍ^n$ to a [`HyperboloidPoint`](@ref) $π(x) ∈ ℝ^{n+1}$
Poincaré ball model. The isometry is defined by

````math
π(x) = \frac{1}{1+\lVert x \rVert^2}
\begin{pmatrix}2x_1\\\vdots\\2x_d\\1+\lVert x \rVert^2\end{pmatrix}
````

Note that this is also used, when the type to convert to is a vector.
"""
function convert(::Type{HyperboloidPoint{T1}}, x::PoincareBallPoint{T2}) where {T1,T2}
    return HyperboloidPoint(convert(Array{T1,1}, x))
end
function convert(::Type{Array{T1,1}}, x::PoincareBallPoint{T2}) where {T1,T2}
    return 1 / (1 - norm(x.value)^2) .* [(2 .* x.value)..., 1 + norm(x.value)^2]
end

struct PoincareHalfPlanePoint{T} <: MPoint
    value::Vector{T}
end

struct PoincareHalfPlaneTVector{T} <: MPoint
    value::Vector{T}
end

@doc raw"""
    check_manifold_point(M::Hyperbolic, p; kwargs...)

Check whether `p` is a valid point on the [`Hyperbolic`](@ref) `M`.

For the [`HyperboloidPoint`](@ref) or plain arrays this means that, `p` is a vector of
length $n+1$ with inner product in the embedding of -1, see [`MinkowskiMetric`](@ref).
The tolerance for the last test can be set using the `kwargs...`.

For the [`PoincareBallPoint`](@ref) a valid point is a vector from $ℝ^n$ with a norm stricly
less than 1.
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
@doc raw"""
    check_tangent_vector(M::Hyperbolic{n}, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to `p` on the [`Hyperbolic`](@ref) `M`, i.e.
after [`check_manifold_point`](@ref)`(M,p)`, `X` has to be of the same dimension as `p`.
The optional parameter `check_base_point` indicates whether to
call [`check_manifold_point`](@ref)  for `p`. The tolerance for the last test can be set
using the `kwargs...`.

For a the hyperboloid model or vectors, `X` has to be  orthogonal to `p` with respect
to the inner product from the embedding, see [`MinkowskiMetric`](@ref).

For a the Poincaré ball model, `X` has to be a vector from $ℝ^{n}$.
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
    ::Hyperbolic{N},
    p,
    X::PoincareBallTVector;
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
    return distance(M, convert(Array{T,1}, p), convert(Array{T,1}, q))
end
function distance(
    M::Hyperbolic,
    p::PoincareHalfPlanePoint{T},
    q::PoincareHalfPlanePoint{T},
) where {T}
    return distance(M, convert(Array{T,1}, p), convert(Array{T,1}, q))
end

embed!(::Hyperbolic, q, p::Array) = (q .= p)
embed!(::Hyperbolic, Y, ::Array, X::Array) = (Y .= X)

embed!(::Hyperbolic, q, p::HyperboloidPoint) = (q .= p.value)
embed!(::Hyperbolic, Y::Array, p::HyperboloidPoint, X::HyperboloidTVector) = (Y .= X.value)

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
injectivity_radius(H::Hyperbolic) = Inf
injectivity_radius(H::Hyperbolic, ::ExponentialRetraction) = Inf
injectivity_radius(H::Hyperbolic, ::Any) = Inf
injectivity_radius(H::Hyperbolic, ::Any, ::ExponentialRetraction) = Inf
eval(
    quote
        @invoke_maker 1 Manifold injectivity_radius(
            M::Hyperbolic,
            rm::AbstractRetractionMethod,
        )
    end,
)

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
"""
project(::Hyperbolic, ::Any, ::Any)

project!(M::Hyperbolic, Y, p, X) = (Y .= X .+ minkowski_metric(p, X) .* p)

Base.show(io::IO, ::Hyperbolic{N}) where {N} = print(io, "Hyperbolic($(N))")
Base.show(io::IO, p::HyperboloidPoint) = print(io, "HyperboloidPoint($(p.value))")
Base.show(io::IO, v::HyperboloidTVector) = print(io, "HyperboloidTVector($(v.value))")
Base.show(io::IO, p::PoincareBallPoint) = print(io, "PoincareBallPoint($(p.value))")
Base.show(io::IO, v::PoincareBallTVector) = print(io, "PoincareBallTVector($(v.value))")
function Base.show(io::IO, p::PoincareHalfPlanePoint)
    return print(io, "PoincareHalfPlanePoint($(p.value))")
end
function Base.show(io::IO, v::PoincareHalfPlaneTVector)
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
function zero_tangent_vector(::Hyperbolic, p::PoincareBallPoint)
    return PoincareBallTVector(zeros(p.value))
end

function zero_tangent_vector!(::Hyperbolic, X::PoincareBallTVector, ::PoincareBallPoint)
    return fill!(X.value, 0)
end
function zero_tangent_vector!(::Hyperbolic, X::PoincareHalfPlanePoint, ::PoincareBallPoint)
    return fill!(X.value, 0)
end
