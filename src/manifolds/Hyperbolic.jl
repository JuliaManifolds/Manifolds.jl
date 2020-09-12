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

_HyperbolicPointTypes = [HyperboloidPoint, PoincareBallPoint, PoincareHalfSpacePoint]
_HyperbolicTangentTypes =
    [HyperboloidTVector, PoincareBallTVector, PoincareHalfSpaceTVector]
_HyperbolicTypes = [_HyperbolicPointTypes..., _HyperbolicTangentTypes...]

for T in _HyperbolicTangentTypes
    @eval begin
        Base.:*(v::$T, s::Number) = $T(v.value * s)
        Base.:*(s::Number, v::$T) = $T(s * v.value)
        Base.:/(v::$T, s::Number) = $T(v.value / s)
        Base.:\(s::Number, v::$T) = $T(s \ v.value)
        Base.:+(v::$T, w::$T) = $T(v.value + w.value)
        Base.:-(v::$T, w::$T) = $T(v.value - w.value)
        Base.:-(v::$T) = $T(-v.value)
        Base.:+(v::$T) = $T(v.value)
        Base.:(==)(v::$T, w::$T) = (v.value == w.value)
    end
end

for T in _HyperbolicTypes
    @eval begin
        allocate(p::$T) = $T(allocate(p.value))
        allocate(p::$T, ::Type{P}) where {P} = $T(allocate(p.value, P))
    end
end

for (P, T) in zip(_HyperbolicPointTypes, _HyperbolicTangentTypes)
    @eval allocate(p::$P, ::Type{$T}) = $T(allocate(p.value))
    @eval allocate_result_type(::Hyperbolic, ::typeof(log), ::Tuple{$P,$P}) = $T
    @eval allocate_result_type(::Hyperbolic, ::typeof(inverse_retract), ::Tuple{$P,$P}) = $T
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
    convert(::Type{HyperboloidPoint}, p::PoincareBallPoint)

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
    convert(
        ::Type{PoincareBallTVector},
        (p,X)::Tuple{HyperboloidPoint,HyperboloidTVector}
    )

convert a [`HyperboloidTVector`](@ref) `X` at `p` to a [`PoincareBallTVector`](@ref)
on the [`Hyperbolic`](@ref) manifold $ℍ^n$ by computing the push forward $π_*(p)[X]$ of
the isometry $π$ that maps from the Hyperboloid to the Poincaré ball,
cf. [`convert(::Type{PoincareBallPoint}, ::HyperboloidPoint)`](@ref).

The formula reads

````math
π_*(p)[X] = \frac{1}{p_{n+1}+1}\Bigl(\tilde X - \frac{X_{n+1}}{p_{n+1}+1}\tilde p \Bigl),
````

where $\tilde X = \begin{pmatrix}X_1\\\vdots\\X_n\end{pmatrix}$
and $\tilde p = \begin{pmatrix}p_1\\\vdots\\p_n\end{pmatrix}$.
"""
function convert(
    ::Type{PoincareBallTVector},
    (p, X)::Tuple{HyperboloidPoint,HyperboloidTVector},
)
    return PoincareBallTVector(
        1 / (p[end] + 1) .* (X[1:(end - 1)] .- (X[end] / (p[end] + 1) .* p[1:(end - 1)])),
    )
end

@doc raw"""
    convert(
        ::Type{Tuple{PoincareBallPoint,PoincareBallTVector}},
        (p,X)::Tuple{HyperboloidPoint,HyperboloidTVector}
    )

Convert a [`HyperboloidPoint`](@ref) `p` and a [`HyperboloidTVector`](@ref) `X`
to a [`PoincareBallPoint`](@ref) and a [`PoincareBallTVector`](@ref) simultaneously,
see [`convert(::Type{PoincareBallPoint}, ::HyperboloidPoint)`](@ref) and
[`convert(::Type{PoincareBallTVector}, ::Tuple{HyperboloidPoint,HyperboloidTVector})`](@ref)
for the formulae.
"""
function convert(
    ::Type{Tuple{PoincareBallPoint,PoincareBallTVector}},
    (p, X)::Tuple{HyperboloidPoint,HyperboloidTVector},
)
    return (convert(PoincareBallPoint, p), convert(PoincareBallTVector, (p, X)))
end

@doc raw"""
    convert(::Type{PoincareHalfSpacePoint}, x::PoincareBallPoint)

convert a point [`PoincareBallPoint`](@ref) `x` (from $ℝ^n$) from the
Poincaré ball model of the [`Hyperbolic`](@ref) manifold $ℍ^n$ to a [`PoincareHalfSpacePoint`](@ref) $π(x) ∈ ℝ^n$.
Denote by $\tilde x = (x_1,\ldots,x_{d-1})$. Then the isometry is defined by

````math
π(x) = \frac{1}{\lVert \tilde x \rVert^2 - (x_n-1)^2}
\begin{pmatrix}2x_1\\\vdots\\2x_{n-1}\\1-\lVert\tilde x\rVert^2 - x_n^2\end{pmatrix}.
````
"""
function convert(::Type{PoincareHalfSpacePoint}, x::PoincareBallPoint)
    return PoincareHalfSpacePoint(
        1 / (norm(x.value[1:(end - 1)])^2 + (last(x.value) - 1)^2) .*
        [x.value[1:(end - 1)]..., 1 - norm(x.value[1:(end - 1)])^2 - last(x.value)^2],
    )
end

@doc raw"""
    convert(
        ::Type{PoincareHalfSpaceTVector},
        (p,X)::Tuple{PoincareBallPoint,PoincareBallTVector}
    )

convert a [`PoincareBallTVector`](@ref) `X` at `p` to a [`PoincareHalfSpacePoint`](@ref)
on the [`Hyperbolic`](@ref) manifold $ℍ^n$ by computing the push forward $π_*(p)[X]$ of
the isometry $π$ that maps from the Poincaré ball to the Poincaré half space,
cf. [`convert(::Type{PoincareHalfSpacePoint}, ::PoincareBallPoint)`](@ref).

The formula reads

````math
π_*(p)[X] =
\frac{1}{\lVert \tilde p\rVert^2 + (1+p_n)^2}
\begin{pmatrix}
2X_1\\
⋮\\
2X_{n-1}\\
-2⟨X,p⟩
\end{pmatrix}
-
\frac{2}{(\lVert \tilde p\rVert^2 + (1+p_n)^2)^2}
\begin{pmatrix}
2p_1(⟨X,p⟩-X_1)\\
⋮\\
2p_{n-1}(⟨X,p⟩-X_{n-1})\\
(\lVert p \rVert^2-1)(⟨X,p⟩-X_n)
\end{pmatrix}
````
where $\tilde p = \begin{pmatrix}p_1\\\vdots\\p_{n-1}\end{pmatrix}$.
"""
function convert(
    ::Type{PoincareHalfSpaceTVector},
    (p, X)::Tuple{PoincareBallPoint,PoincareBallTVector},
)
    den = 1 + norm(p.value[1:end-1])^2 + (last(p.value)+1)^2
    scp = dot(p.value,X.value)
    c1 = (2/den .* X.value[1:end-1])
        .- 4 .* p.value[1:end-1] .* (scp .- X.value[1:end-1]) ./ (den^2)
    c2 = -2*scp/den + 2*(norm(p.value)^2-1)*(scp-last(X.value)) / (den^2)
    return PoincareHalfSpaceTVector([c1...,c2])
end

@doc raw"""
    convert(::Type{PoincareBallPoint}, p::PoincareHalfSpacePoint)

convert a point [`PoincareHalfSpacePoint`](@ref) `p` (from $ℝ^n$) from the
Poincaré half plane model of the [`Hyperbolic`](@ref) manifold $ℍ^n$ to a [`PoincareBallPoint`](@ref) $π(p) ∈ ℝ^n$.
Denote by $\tilde p = (p_1,\ldots,p_{d-1})$. Then the isometry is defined by

````math
π(p) = \frac{1}{\lVert \tilde p \rVert^2 + (p_n+1)^2}
\begin{pmatrix}2p_1\\\vdots\\2p_{n-1}\\\lVert p\rVert^2 - 1\end{pmatrix}.
````
"""
function convert(::Type{PoincareBallPoint}, p::PoincareHalfSpacePoint)
    return PoincareBallPoint(
        1 / (norm(p.value[1:(end - 1)])^2 + (last(p.value) + 1)^2) .*
        [p.value[1:(end - 1)]..., norm(p.value)^2 - 1],
    )
end

@doc raw"""
    convert(
        ::Type{PoincareBallTVector},
        (p,X)::Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTVector}
    )

convert a [`PoincareHalfSpaceTVector`](@ref) `X` at `p` to a [`PoincareBallTVector`](@ref)
on the [`Hyperbolic`](@ref) manifold $ℍ^n$ by computing the push forward $π_*(p)[X]$ of
the isometry $π$ that maps from the Poincaré half space to the Poincaré ball,
cf. [`convert(::Type{PoincareBallPoint}, ::PoincareHalfSpacePoint)`](@ref).

The formula reads

````math
π_*(p)[X] =
\frac{1}{\lVert \tilde p\rVert^2 + (1+p_n)^2}
\begin{pmatrix}
2X_1\\
⋮\\
2X_{n-1}\\
2⟨X,p⟩
\end{pmatrix}
-
\frac{2}{(\lVert \tilde p\rVert^2 + (1+p_n)^2)^2}
\begin{pmatrix}
2p_1(⟨X,p⟩+X_1)\\
⋮\\
2p_{n-1}(⟨X,p⟩+X_{n-1})\\
(1-\lVert p \rVert^2)(⟨X,p⟩+X_n)
\end{pmatrix}
````
where $\tilde p = \begin{pmatrix}p_1\\\vdots\\p_{n-1}\end{pmatrix}$.
"""
function convert(
    ::Type{PoincareBallTVector},
    (p, X)::Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTVector},
)
    den = 1 + norm(p.value[1:end-1])^2 + (last(p.value)+1)^2
    scp = dot(p.value,X.value)
    c1 = (2/den .* X.value[1:end-1])
        .- 4 .* p.value[1:end-1] .* (scp .+ X.value[1:end-1]) ./ (den^2)
    c2 = 2*scp/den + 2*(1-norm(p.value)^2)*(scp + last(X.value))
    return PoincareBallTVector([c1...,c2])
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

for T in _HyperbolicTypes
    @eval Base.copyto!(p::$T, q::$T) = copyto!(p.value, q.value)
end

decorated_manifold(::Hyperbolic{N}) where {N} = Lorentz(N + 1, MinkowskiMetric())

default_metric_dispatch(::Hyperbolic, ::MinkowskiMetric) = Val(true)

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
@doc raw"""
    distance(::Hyperbolic, p::PoincareHalfSpacePoint, q::PoincareHalfSpacePoint)

Compute the distance on the [`Hyperbolic`](@ref) manifold $ℍ^n$ represented in the
Poincaré half space model. The formula reads

````math
d_{ℍ^n}(p,q) = \operatorname{acosh}\Bigl( 1 + \frac{\lVert p - q \rVert^2}{2 p_n q_n} \Bigr)
````
"""
function distance(::Hyperbolic, p::PoincareHalfSpacePoint, q::PoincareHalfSpacePoint)
    return acosh(1 + norm(p.value .- q.value)^2 / (2 * p.value[end] * q.value[end]))
end

@doc raw"""
    distance(::Hyperbolic, p::PoincareBallPoint, q::PoincareBallPoint)

Compute the distance on the [`Hyperbolic`](@ref) manifold $ℍ^n$ represented in the
Poincaré ball model. The formula reads

````math
d_{ℍ^n}(p,q) =
\operatorname{acosh}\Bigl(
  1 + \frac{2\lVert p - q \rVert^2}{(1-\lVert p\rVert^2)(1-\lVert q\rVert^2)}
\Bigr)
````
"""
function distance(::Hyperbolic, p::PoincareBallPoint, q::PoincareBallPoint)
    return acosh(
        1 +
        2 * norm(p.value .- q.value)^2 / ((1 - norm(p.value)^2) * (1 - norm(q.value)^2)),
    )
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
for (P, T) in zip(_HyperbolicPointTypes, _HyperbolicTangentTypes)
    @eval function exp!(M::Hyperbolic, q::$P, p::$P, X::$T)
        q.value .=
            convert(
                $P,
                exp(M, convert(AbstractVector, p), convert(AbstractVector, X)),
            ).value
        return q
    end
end

@doc raw"""
    injectivity_radius(M::Hyperbolic)
    injectivity_radius(M::Hyperbolic, p)

Return the injectivity radius on the [`Hyperbolic`](@ref), which is $∞$.
"""
injectivity_radius(::Hyperbolic) = Inf
injectivity_radius(::Hyperbolic, ::ExponentialRetraction) = Inf
injectivity_radius(::Hyperbolic, ::Any) = Inf
injectivity_radius(::Hyperbolic, ::Any, ::ExponentialRetraction) = Inf
eval(
    quote
        @invoke_maker 1 Manifold injectivity_radius(
            M::Hyperbolic,
            rm::AbstractRetractionMethod,
        )
    end,
)

@doc raw"""
    inner(
        ::Hyperbolic{n},
        p::PoincareHalfSpacePoint,
        X::PoincareHalfSpaceTVector,
        Y::PoincareHalfSpaceTVector
    )

Compute the inner product in the Poincaré half space model. The formula reads
````math
g_p(X,Y) = \frac{⟨X,Y⟩}{p_n^2}.
````
"""
function inner(
    ::Hyperbolic,
    p::PoincareHalfSpacePoint,
    X::PoincareHalfSpaceTVector,
    Y::PoincareHalfSpaceTVector,
)
    return dot(X.value, Y.value) / last(p.value)^2
end
@doc raw"""
    inner(M::Hyperbolic{n}, p::HyperboloidPoint, X::HyperboloidTVector, Y::HyperboloidTVector)

Cmpute the inner product in the Hyperboloid model, i.e. the [`minkowski_metric`](@ref) in
the embedding. The formula reads

````math
g_p(X,Y) = ⟨X,Y⟩_{\mathrm{M}} = -X_{n}Y_{n} + \displaystyle\sum_{k=1}^{n-1} X_kY_k.
````
"""
function inner(
    M::Hyperbolic,
    p::HyperboloidPoint,
    X::HyperboloidTVector,
    Y::HyperboloidTVector,
)
    return inner(M, p.value, X.value, Y.value)
end

@doc raw"""
    inner(::Hyperbolic, p::PoincareBallPoint, X::PoincareBallTVector, Y::PoincareBallTVector)

Compute the inner producz in the Poincaré ball model. The formula reads
````math
g_p(X,Y) = \frac{4}{(1-\lVert p \rVert^2)^2}  ⟨X, Y⟩ .
````
"""
function inner(
    ::Hyperbolic,
    p::PoincareBallPoint,
    X::PoincareBallTVector,
    Y::PoincareBallTVector,
)
    return 4 / (1 - norm(p)^2)^2 * dot(X.value, Y.value)
end

for T in _HyperbolicPointTypes
    @eval function isapprox(::Hyperbolic, p::$T, q::$T; kwargs...)
        return isapprox(p.value, q.value; kwargs...)
    end
end
for (P, T) in zip(_HyperbolicPointTypes, _HyperbolicTangentTypes)
    @eval function isapprox(::Hyperbolic, ::$P, X::$T, Y::$T; kwargs...)
        return isapprox(X.value, Y.value; kwargs...)
    end
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

for (P, T) in zip(_HyperbolicPointTypes, _HyperbolicTangentTypes)
    @eval function log!(M::Hyperbolic, X::$T, p::$P, q::$P)
        X.value .=
            convert(
                $T,
                log(M, convert(AbstractVector, p), convert(AbstractVector, q)),
            ).value
        return X
    end
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

for T in _HyperbolicTypes
    @eval number_eltype(p::$T) = typeof(one(eltype(p.value)))
end

function minkowski_metric(a::HyperboloidPoint, b::HyperboloidPoint)
    return minkowski_metric(convert(Vector, a), convert(Vector, b))
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
for T in _HyperbolicTypes
    @eval Base.show(io::IO, p::$T) = print(io, "$($T)($(p.value))")
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

for (P, T) in zip(_HyperbolicPointTypes, _HyperbolicTangentTypes)
    @eval function vector_transport_to!(
        M::Hyperbolic,
        Y::$T,
        p::$P,
        X::$T,
        q::$P,
        m::ParallelTransport,
    )
        Y.value .=
            convert(
                $T,
                vector_transport_to(
                    M,
                    convert(Vector, p),
                    convert(Vector, X),
                    convert(Vector, q),
                    m,
                ),
            ).value
        return Y
    end
    @eval zero_tangent_vector(::Hyperbolic, p::$P) = $T(zero(p.value))
    @eval zero_tangent_vector!(::Hyperbolic, X::$T, ::$P) = fill!(X.value, 0)
end
