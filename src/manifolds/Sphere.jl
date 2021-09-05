"""
    AbstractSphere{𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType}

An abstract type to represent a unit sphere that is represented isometrically in the embedding.
"""
abstract type AbstractSphere{𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} end

@doc raw"""
    Sphere{n,𝔽} <: AbstractSphere{𝔽}

The (unit) sphere manifold $𝕊^{n}$ is the set of all unit norm vectors in $𝔽^{n+1}$.
The sphere is represented in the embedding, i.e.

````math
𝕊^{n} := \bigl\{ p \in 𝔽^{n+1}\ \big|\ \lVert p \rVert = 1 \bigr\}
````

where $𝔽\in\{ℝ,ℂ,ℍ\}$. Note that compared to the [`ArraySphere`](@ref), here the
argument `n` of the manifold is the dimension of the manifold, i.e. $𝕊^{n} ⊂ 𝔽^{n+1}$, $n\in ℕ$.

The tangent space at point $p$ is given by

````math
T_p𝕊^{n} := \bigl\{ X ∈ 𝔽^{n+1}\ |\ \Re(⟨p,X⟩) = 0 \bigr \},
````

where $𝔽\in\{ℝ,ℂ,ℍ\}$ and $⟨\cdot,\cdot⟩$ denotes the inner product in the
embedding $𝔽^{n+1}$.

For $𝔽=ℂ$, the manifold is the complex sphere, written $ℂ𝕊^n$, embedded in $ℂ^{n+1}$.
$ℂ𝕊^n$ is the complexification of the real sphere $𝕊^{2n+1}$.
Likewise, the quaternionic sphere $ℍ𝕊^n$ is the quaternionification of the real sphere
$𝕊^{4n+3}$.
Consequently, $ℂ𝕊^0$ is equivalent to $𝕊^1$ and [`Circle`](@ref), while $ℂ𝕊^1$ and $ℍ𝕊^0$
are equivalent to $𝕊^3$, though with different default representations.

This manifold is modeled as a special case of the more general case, i.e. as an embedded
manifold to the [`Euclidean`](@ref), and several functions like the [`inner`](@ref inner(::Euclidean, ::Any...)) product
and the [`zero_vector`](@ref zero_vector(::Euclidean, ::Any...)) are inherited from the embedding.

# Constructor

    Sphere(n[, field=ℝ])

Generate the (real-valued) sphere $𝕊^{n} ⊂ ℝ^{n+1}$, where `field` can also be used to
generate the complex- and quaternionic-valued sphere.
"""
struct Sphere{N,𝔽} <: AbstractSphere{𝔽} end
Sphere(n::Int, field::AbstractNumbers=ℝ) = Sphere{n,field}()

@doc raw"""
    ArraySphere{T<:Tuple,𝔽} <: AbstractSphere{𝔽}

The (unit) sphere manifold $𝕊^{n₁,n₂,...,nᵢ}$ is the set of all unit (Frobenius) norm elements of
$𝔽^{n₁,n₂,...,nᵢ}$, where $𝔽\in\{ℝ,ℂ,ℍ\}. The generalized sphere is
represented in the embedding, and supports arbitrary sized arrays or in other words arbitrary
tensors of unit norm. The set formally reads

````math
𝕊^{n_1, n_2, …, n_i} := \bigl\{ p \in 𝔽^{n_1, n_2, …, n_i}\ \big|\ \lVert p \rVert = 1 \bigr\}
````

where $𝔽\in\{ℝ,ℂ,ℍ\}$. Setting $i=1$ and $𝔽=ℝ$  this  simplifies to unit vectors in $ℝ^n$, see
[`Sphere`](@ref) for this special case. Note that compared to this classical case,
the argument for the generalized case here is given by the dimension of the embedding.
This means that `Sphere(2)` and `ArraySphere(3)` are the same manifold.

The tangent space at point $p$ is given by

````math
T_p 𝕊^{n_1, n_2, …, n_i} := \bigl\{ X ∈ 𝔽^{n_1, n_2, …, n_i}\ |\ \Re(⟨p,X⟩) = 0 \bigr \},
````

where $𝔽\in\{ℝ,ℂ,ℍ\}$ and $⟨\cdot,\cdot⟩$ denotes the (Frobenius) inner product in the
embedding $𝔽^{n_1, n_2, …, n_i}$.

This manifold is modeled as an embedded manifold to the [`Euclidean`](@ref), i.e.
several functions like the [`inner`](@ref inner(::Euclidean, ::Any...)) product and the
[`zero_vector`](@ref zero_vector(::Euclidean, ::Any...)) are inherited from the embedding.

# Constructor

    ArraySphere(n₁,n₂,...,nᵢ; field=ℝ)

Generate sphere in $𝔽^{n_1, n_2, …, n_i}$, where $𝔽$ defaults to the real-valued case $ℝ$.
"""
struct ArraySphere{N,𝔽} <: AbstractSphere{𝔽} where {N<:Tuple} end
function ArraySphere(n::Vararg{Int,I}; field::AbstractNumbers=ℝ) where {I}
    return ArraySphere{Tuple{n...},field}()
end

"""
    check_point(M::AbstractSphere, p; kwargs...)

Check whether `p` is a valid point on the [`AbstractSphere`](@ref) `M`, i.e. is a point in
the embedding of unit length.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_point(M::AbstractSphere, p; kwargs...)
    mpv = invoke(
        check_point,
        Tuple{(typeof(get_embedding(M))),typeof(p)},
        get_embedding(M),
        p;
        kwargs...,
    )
    mpv === nothing || return mpv
    if !isapprox(norm(p), 1.0; kwargs...)
        return DomainError(
            norm(p),
            "The point $(p) does not lie on the $(M) since its norm is not 1.",
        )
    end
    return nothing
end

"""
    check_vector(M::AbstractSphere, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`AbstractSphere`](@ref) `M`, i.e.
after [`check_point`](@ref)`(M,p)`, `X` has to be of same dimension as `p`
and orthogonal to `p`.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_vector(M::AbstractSphere, p, X; kwargs...)
    mpv = invoke(
        check_vector,
        Tuple{typeof(get_embedding(M)),typeof(p),typeof(X)},
        get_embedding(M),
        p,
        X;
        kwargs...,
    )
    mpv === nothing || return mpv
    if !isapprox(abs(real(dot(p, X))), 0.0; kwargs...)
        return DomainError(
            abs(dot(p, X)),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not orthogonal in the embedding.",
        )
    end
    return nothing
end

function decorated_manifold(M::AbstractSphere{𝔽}) where {𝔽}
    return Euclidean(representation_size(M)...; field=𝔽)
end
get_embedding(M::AbstractSphere{𝔽}) where {𝔽} = decorated_manifold(M)

@doc raw"""
    distance(M::AbstractSphere, p, q)

Compute the geodesic distance betweeen `p` and `q` on the [`AbstractSphere`](@ref) `M`.
The formula is given by the (shorter) great arc length on the (or a) great circle
both `p` and `q` lie on.

````math
d_{𝕊}(p,q) = \arccos(\Re(⟨p,q⟩)).
````
"""
distance(::AbstractSphere, p, q) = acos(clamp(real(dot(p, q)), -1, 1))

@doc raw"""
    exp(M::AbstractSphere, p, X)

Compute the exponential map from `p` in the tangent direction `X` on the [`AbstractSphere`](@ref)
`M` by following the great arc eminating from `p` in direction `X`.

````math
\exp_p X = \cos(\lVert X \rVert_p)p + \sin(\lVert X \rVert_p)\frac{X}{\lVert X \rVert_p},
````
where $\lVert X \rVert_p$ is the [`norm`](@ref norm(::AbstractSphere,p,X)) on the
tangent space at `p` of the [`AbstractSphere`](@ref) `M`.
"""
exp(::AbstractSphere, ::Any...)

function exp!(M::AbstractSphere, q, p, X)
    θ = norm(M, p, X)
    q .= cos(θ) .* p .+ usinc(θ) .* X
    return q
end

function get_basis(M::Sphere{n,ℝ}, p, B::DiagonalizingOrthonormalBasis{ℝ}) where {n}
    A = zeros(n + 1, n + 1)
    A[1, :] = transpose(p)
    A[2, :] = transpose(B.frame_direction)
    V = nullspace(A)
    κ = ones(n)
    if !iszero(B.frame_direction)
        # if we have a nonzero direction for the geodesic, add it and it gets curvature zero from the tensor
        V = hcat(B.frame_direction / norm(M, p, B.frame_direction), V)
        κ[1] = 0 # no curvature along the geodesic direction, if x!=y
    end
    T = typeof(similar(B.frame_direction))
    Ξ = [convert(T, V[:, i]) for i in 1:n]
    return CachedBasis(B, κ, Ξ)
end

@doc raw"""
    get_coordinates(M::AbstractSphere{ℝ}, p, X, B::DefaultOrthonormalBasis)

Represent the tangent vector `X` at point `p` from the [`AbstractSphere`](@ref) `M` in
an orthonormal basis by rotating the hyperplane containing `X` to a hyperplane whose
normal is the $x$-axis.

Given $q = p λ + x$, where $λ = \operatorname{sgn}(⟨x, p⟩)$, and $⟨⋅, ⋅⟩_{\mathrm{F}}$
denotes the Frobenius inner product, the formula for $Y$ is
````math
\begin{pmatrix}0 \\ Y\end{pmatrix} = X - q\frac{2 ⟨q, X⟩_{\mathrm{F}}}{⟨q, q⟩_{\mathrm{F}}}.
````
"""
get_coordinates(::AbstractSphere{ℝ}, p, X, ::DefaultOrthonormalBasis)

function get_coordinates!(
    M::AbstractSphere{ℝ},
    Y,
    p,
    X,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
)
    n = manifold_dimension(M)
    p1 = p[1]
    cosθ = abs(p1)
    λ = nzsign(p1, cosθ)
    pend, Xend = view(p, 2:(n + 1)), view(X, 2:(n + 1))
    factor = λ * X[1] / (1 + cosθ)
    Y .= Xend .- pend .* factor
    return Y
end

@doc raw"""
    get_vector(M::AbstractSphere{ℝ}, p, X, B::DefaultOrthonormalBasis)

Convert a one-dimensional vector of coefficients `X` in the basis `B` of the tangent space
at `p` on the [`AbstractSphere`](@ref) `M` to a tangent vector `Y` at `p` by rotating the
hyperplane containing `X`, whose normal is the $x$-axis, to the hyperplane whose normal is
`p`.

Given $q = p λ + x$, where $λ = \operatorname{sgn}(⟨x, p⟩)$, and $⟨⋅, ⋅⟩_{\mathrm{F}}$
denotes the Frobenius inner product, the formula for $Y$ is
````math
Y = X - q\frac{2 \left\langle q, \begin{pmatrix}0 \\ X\end{pmatrix}\right\rangle_{\mathrm{F}}}{⟨q, q⟩_{\mathrm{F}}}.
````
"""
get_vector(::AbstractSphere{ℝ}, p, X, ::DefaultOrthonormalBasis)

function get_vector!(
    M::AbstractSphere{ℝ},
    Y,
    p,
    X,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
)
    n = manifold_dimension(M)
    p1 = p[1]
    cosθ = abs(p1)
    λ = nzsign(p1, cosθ)
    pend = view(p, 2:(n + 1))
    pX = dot(pend, X)
    factor = pX / (1 + cosθ)
    Y[1] = -λ * pX
    Y[2:(n + 1)] .= X .- pend .* factor
    return Y
end

@doc raw"""
    injectivity_radius(M::AbstractSphere[, p])

Return the injectivity radius for the [`AbstractSphere`](@ref) `M`, which is globally $π$.

    injectivity_radius(M::Sphere, x, ::ProjectionRetraction)

Return the injectivity radius for the [`ProjectionRetraction`](@ref) on the
[`AbstractSphere`](@ref), which is globally $\frac{π}{2}$.
"""
injectivity_radius(::AbstractSphere) = π
injectivity_radius(::AbstractSphere, ::ExponentialRetraction) = π
injectivity_radius(::AbstractSphere, ::ProjectionRetraction) = π / 2
injectivity_radius(::AbstractSphere, ::Any) = π
injectivity_radius(::AbstractSphere, ::Any, ::ExponentialRetraction) = π
injectivity_radius(::AbstractSphere, ::Any, ::ProjectionRetraction) = π / 2
eval(
    quote
        @invoke_maker 1 AbstractManifold injectivity_radius(
            M::AbstractSphere,
            rm::AbstractRetractionMethod,
        )
    end,
)

@doc raw"""
    inverse_retract(M::AbstractSphere, p, q, ::ProjectionInverseRetraction)

Compute the inverse of the projection based retraction on the [`AbstractSphere`](@ref) `M`,
i.e. rearranging $p+X = q\lVert p+X\rVert_2$ yields
since $\Re(⟨p,X⟩) = 0$ and when $d_{𝕊^2}(p,q) ≤ \frac{π}{2}$ that

````math
\operatorname{retr}_p^{-1}(q) = \frac{q}{\Re(⟨p, q⟩)} - p.
````
"""
inverse_retract(::AbstractSphere, ::Any, ::Any, ::ProjectionInverseRetraction)

function inverse_retract!(::AbstractSphere, X, p, q, ::ProjectionInverseRetraction)
    return (X .= q ./ real(dot(p, q)) .- p)
end

@doc raw"""
    log(M::AbstractSphere, p, q)

Compute the logarithmic map on the [`AbstractSphere`](@ref) `M`, i.e. the tangent vector,
whose geodesic starting from `p` reaches `q` after time 1.
The formula reads for $x ≠ -y$

````math
\log_p q = d_{𝕊}(p,q) \frac{q-\Re(⟨p,q⟩) p}{\lVert q-\Re(⟨p,q⟩) p \rVert_2},
````

and a deterministic choice from the set of tangent vectors is returned if $x=-y$, i.e. for
opposite points.
"""
log(::AbstractSphere, ::Any...)

function log!(M::AbstractSphere, X, p, q)
    cosθ = clamp(real(dot(p, q)), -1, 1)
    if cosθ ≈ -1 # appr. opposing points, return deterministic choice from set-valued log
        fill!(X, zero(eltype(X)))
        if p[1] ≈ 1
            X[2] = 1
        else
            X[1] = 1
        end
        copyto!(X, X .- real(dot(p, X)) .* p)
        X .*= π / norm(X)
    else
        θ = acos(cosθ)
        X .= (q .- cosθ .* p) ./ usinc(θ)
    end
    return project!(M, X, p, X)
end

@doc raw"""
    manifold_dimension(M::AbstractSphere)

Return the dimension of the [`AbstractSphere`](@ref) `M`, respectively i.e. the
dimension of the embedding -1.
"""
manifold_dimension(M::AbstractSphere) = manifold_dimension(get_embedding(M)) - 1

"""
    mean(
        S::AbstractSphere,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/2);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::AbstractSphere, ::Any...)

function Statistics.mean!(
    S::AbstractSphere,
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)
    return mean!(S, p, x, w, GeodesicInterpolationWithinRadius(π / 2); kwargs...)
end

function mid_point!(S::Sphere, q, p1, p2)
    q .= p1 .+ p2
    project!(S, q, q)
    return q
end

@doc raw"""
    project(M::AbstractSphere, p)

Project the point `p` from the embedding onto the [`Sphere`](@ref) `M`.

````math
\operatorname{proj}(p) = \frac{p}{\lVert p \rVert},
````
where $\lVert\cdot\rVert$ denotes the usual 2-norm for vectors if $m=1$ and the Frobenius
norm for the case $m>1$.
"""
project(::AbstractSphere, ::Any)

project!(::AbstractSphere, q, p) = (q .= p ./ norm(p))

@doc raw"""
    project(M::AbstractSphere, p, X)

Project the point `X` onto the tangent space at `p` on the [`Sphere`](@ref) `M`.

````math
\operatorname{proj}_{p}(X) = X - \Re(⟨p, X⟩)p
````
"""
project(::AbstractSphere, ::Any, ::Any)

project!(::AbstractSphere, Y, p, X) = (Y .= X .- real(dot(p, X)) .* p)

@doc raw"""
    representation_size(M::AbstractSphere)

Return the size points on the [`AbstractSphere`](@ref) `M` are represented as, i.e., the
representation size of the embedding.
"""
@generated representation_size(M::ArraySphere{N}) where {N} = size_to_tuple(N)
@generated representation_size(M::Sphere{N}) where {N} = (N + 1,)

@doc raw"""
    retract(M::AbstractSphere, p, X, ::ProjectionRetraction)

Compute the retraction that is based on projection, i.e.

````math
\operatorname{retr}_p(X) = \frac{p+X}{\lVert p+X \rVert_2}
````
"""
retract(::AbstractSphere, ::Any, ::Any, ::ProjectionRetraction)

function retract!(M::AbstractSphere, q, p, X, ::ProjectionRetraction)
    q .= p .+ X
    return project!(M, q, q)
end

Base.show(io::IO, ::Sphere{n,𝔽}) where {n,𝔽} = print(io, "Sphere($(n), $(𝔽))")
function Base.show(io::IO, ::ArraySphere{N,𝔽}) where {N,𝔽}
    return print(io, "ArraySphere($(join(N.parameters, ", ")); field = $(𝔽))")
end

"""
    uniform_distribution(M::Sphere{n,ℝ}, p) where {n}

Uniform distribution on given [`Sphere`](@ref) `M`. Generated points will be of
similar type as `p`.
"""
function uniform_distribution(M::Sphere{n,ℝ}, p) where {n}
    d = Distributions.MvNormal(zero(p), 1.0)
    return ProjectedPointDistribution(M, d, project!, p)
end

@doc raw"""
    vector_transport_to(M::AbstractSphere, p, X, q, ::ParallelTransport)

Compute the parallel transport on the [`Sphere`](@ref) of the tangent vector `X` at `p`
to `q`, provided, the [`geodesic`](@ref) between `p` and `q` is unique. The formula reads

````math
P_{p←q}(X) = X - \frac{\Re(⟨\log_p q,X⟩_p)}{d^2_𝕊(p,q)}
\bigl(\log_p q + \log_q p \bigr).
````
"""
vector_transport_to(::AbstractSphere, ::Any, ::Any, ::Any, ::Any, ::ParallelTransport)

function vector_transport_to!(::AbstractSphere, Y, p, X, q, ::ParallelTransport)
    m = p .+ q
    mnorm2 = real(dot(m, m))
    factor = 2 * real(dot(X, q)) / mnorm2
    Y .= X .- m .* factor
    return Y
end

"""
    StereographicAtlas()

The stereographic atlas of ``S^n`` with two charts: one with the singular
point (-1, 0, ..., 0) (called `:north`) and one with the singular
point (1, 0, ..., 0) (called `:south`).
"""
struct StereographicAtlas <: AbstractAtlas{ℝ} end

function get_chart_index(::Sphere{n,ℝ}, ::StereographicAtlas, p) where {n}
    if p[1] < 0
        return :south
    else
        return :north
    end
end

function get_parameters!(::Sphere{n,ℝ}, x, ::StereographicAtlas, i::Symbol, p) where {n}
    if i === :north
        return x .= p[2:end] ./ (1 + p[1])
    else
        return x .= p[2:end] ./ (1 - p[1])
    end
end

function get_point!(::Sphere{n,ℝ}, p, ::StereographicAtlas, i::Symbol, x) where {n}
    xnorm2 = dot(x, x)
    if i === :north
        p[1] = (1 - xnorm2) / (xnorm2 + 1)
    else
        p[1] = (xnorm2 - 1) / (xnorm2 + 1)
    end
    p[2:end] .= 2 * x / (xnorm2 + 1)
    return p
end

function get_coordinates!(
    ::Sphere{n,ℝ},
    Y,
    p,
    X,
    B::InducedBasis{ℝ,TangentSpaceType,<:StereographicAtlas},
) where {n}
    if B.i === :north
        for i in 1:n
            Y[i] = X[i + 1] / (1 + p[1]) - X[1] * p[i + 1] / (1 + p[1])^2
        end
    else
        for i in 1:n
            Y[i] = X[i + 1] / (-1 + p[1]) - X[1] * p[i + 1] / (-1 + p[1])^2
        end
    end
    return Y
end

function get_vector!(
    M::Sphere{n,ℝ},
    Y,
    p,
    X,
    B::InducedBasis{ℝ,TangentSpaceType,<:StereographicAtlas},
) where {n}
    a = get_parameters(M, B.A, B.i, p)
    mult = inv(1 + dot(a, a))^2

    Y[1] = 0
    for j in 1:n
        Y[1] -= 4 * a[j] * mult * X[j]
    end
    for i in 2:(n + 1)
        Y[i] = 0
        for j in 1:n
            if i == j + 1
                Y[i] += 2 * (1 + dot(a, a) - 2 * a[i - 1]^2) * mult * X[j]
            else
                Y[i] -= 4 * a[i - 1] * a[j] * mult * X[j]
            end
        end
        if B.i === :south
            Y[i] *= -1
        end
    end
    return Y
end

function local_metric(
    M::Sphere{n,ℝ},
    p,
    B::InducedBasis{ℝ,TangentSpaceType,StereographicAtlas,Symbol},
) where {n}
    a = get_parameters(M, B.A, B.i, p)
    return (4 / (1 + dot(a, a))^2) * I
end
