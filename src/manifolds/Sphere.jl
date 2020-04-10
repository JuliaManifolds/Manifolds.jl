"""
    AbstractSphere{𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType}

An abstract type to represent a unit sphere that is represented isometrically in the embedding.
"""
abstract type AbstractSphere{𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} end

@doc raw"""
    Sphere{n,𝔽} <: AbstractSphere{𝔽}

The (unit) sphere manifold $𝕊^{n}$ is the set of all unit norm vectors in $𝔽^{N}$, $𝔽 elements.
The sphere is represented in the embedding, i.e.

````math
𝕊^{n} := \bigl\{ p \in 𝔽^{n+1}\ \big|\ \lVert p \rVert = 1 \bigr\}
````

where $𝔽\in\{ℝ,ℂ\}$. Note that compared to the [`ArraySphere`](@ref), here the
argument `n` of the manifold is the dimension of the manifold, i.e. $𝕊^{n} ⊂ 𝔽^{n+1}$, $n\in ℕ$.

The tangent space at point $p$ is given by

````math
T_p𝕊^{n} := \bigl\{ X ∈ 𝔽^{n+1}\ |\ ⟨p,X⟩ = 0 \bigr \},
````

where $𝔽\in\{ℝ,ℂ\} and $⟨\cdot,\cdot⟩$ denotes the inner product in the
embedding $\mathbb 𝔽^{n+1}$.

This manifold is modeled as a special case of the more general case, i.e. as an embedded
manifold to the [`Euclidean`](@ref), and several functions like the [`inner`](@ref inner(::Euclidean, ::Any...)) product
and the [`zero_tangent_vector`](@ref zero_tangent_vector(::Euclidean, ::Any...)) are inherited from the embedding.

# Constructor

    Sphere(n[, field=ℝ])

Generate the (real-valued) sphere $𝕊^{n} ⊂ ℝ^{n+1}$, where `field` can also be used to
generate the complex-valued sphere.
"""
struct Sphere{N,𝔽} <: AbstractSphere{𝔽} end
Sphere(n::Int, field::AbstractNumbers = ℝ) = Sphere{n,field}()


@doc raw"""
    ArraySphere{T<:Tuple,𝔽} <: AbstractSphere{𝔽}

The (unit) sphere manifold $𝕊^{n₁,n₂,...,nᵢ}$ is the set of all unit (Frobenius) norm elements of
$𝔽^{n₁,n₂,...,nᵢ}$, where $𝔽\in\{ℝ,ℂ\}. The generalized sphere is
represented in the embedding, and supports arbitrary sized arrays or in other words arbitrary
tensors of unit norm. The set formally reads

````math
𝕊^{n_1, n_2, …, n_i} := \bigl\{ p \in 𝔽^{n_1, n_2, …, n_i}\ \big|\ \lVert p \rVert = 1 \bigr\}
````

where $𝔽\in\{ℝ,ℂ\}$. Setting $i=1$ and $𝔽=ℝ$  this  simplifies to unit vectors in $ℝ^n$, see
[`Sphere`](@ref) for this special case. Note that compared to this classical case,
the argument for the generalized case here is given by the dimension of the embedding.
This means that `Sphere(2)` and `ArraySphere(3)` are the same manifold.

The tangent space at point p is given by

````math
T_p𝕊^{n_1, n_2, …, n_i} := \bigl\{ X ∈ 𝔽^{n_1, n_2, …, n_i}\ |\ ⟨p,X⟩ = 0 \bigr \},
````

where $𝔽\in\{ℝ,ℂ\}$ and $⟨\cdot,\cdot⟩$ denotes the inner product in the
embedding $\mathbb 𝔽^{n_1, n_2, …, n_i}$.

This manifold is modeled as an embedded manifold to the [`Euclidean`](@ref), i.e.
several functions like the [`inner`](@ref inner(::Euclidean, ::Any...)) product and the
[`zero_tangent_vector`](@ref zero_tangent_vector(::Euclidean, ::Any...)) are inherited from the embedding.

# Constructor

    ArraySphere(n₁,n₂,...,nᵢ; field=ℝ))

Generate sphere in $𝔽^{n_1, n_2, …, n_i}$, where 𝔽 defaults to the real-valued case ℝ.
"""
struct ArraySphere{N,𝔽} <: AbstractSphere{𝔽} where {N<:Tuple} end
function ArraySphere(n::Vararg{Int,I}; field::AbstractNumbers = ℝ) where {I}
    return ArraySphere{Tuple{n...},field}()
end

"""
    check_manifold_point(M::AbstractSphere, p; kwargs...)

Check whether `p` is a valid point on the [`AbstractSphere`](@ref) `M`, i.e. is a point in
the embedding of unit length.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::AbstractSphere, p; kwargs...)
    mpv = invoke(
        check_manifold_point,
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
    check_tangent_vector(M::AbstractSphere, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to `p` on the [`AbstractSphere`](@ref) `M`, i.e.
after [`check_manifold_point`](@ref)`(M,p)`, `X` has to be of same dimension as `p`
and orthogonal to `p`.
The optional parameter `check_base_point` indicates, whether to call
[`check_manifold_point`](@ref)  for `p` or not.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(M::AbstractSphere, p, X; check_base_point = true, kwargs...)
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{typeof(get_embedding(M)),typeof(p),typeof(X)},
        get_embedding(M),
        p,
        X;
        check_base_point = false, # already checked above
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
    return Euclidean(representation_size(M)...; field = 𝔽)
end
get_embedding(M::AbstractSphere{𝔽}) where {𝔽} = decorated_manifold(M)


@doc raw"""
    distance(M::AbstractSphere, p, q)

Compute the geodesic distance betweeen `p` and `q` on the [`AbstractSphere`](@ref) `M`.
The formula is given by the (shorter) great arc length on the (or a) great circle
both `p` and `q` lie on.

````math
d_{𝕊}(p,q) = \arccos(⟨p,q⟩).
````
"""
distance(::AbstractSphere, p, q) = acos(clamp(real(dot(p, q)), -1, 1))

embed!(::AbstractSphere, q, p) = (q .= p)

embed!(::AbstractSphere, Y, p, X) = (Y .= X)

@doc raw"""
    exp(M::AbstractSphere, p, X)

Compute the exponential map from `p` in the tangent direction `X` on the [`AbstractSphere`](@ref)
`M` by following the great arc eminating from `p` in direction `X`.

````math
\exp_p X = \cos(\lVert X \rVert_p)p + \sin(\lVert X \rVert_p)\frac{X}{\lVert X \rVert_p}X,
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

flat!(M::AbstractSphere, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

function get_basis(M::Sphere{n,ℝ}, p, B::DiagonalizingOrthonormalBasis{ℝ}) where {n}
    A = zeros(n + 1, n + 1)
    A[1, :] = transpose(p)
    A[2, :] = transpose(B.frame_direction)
    V = nullspace(A)
    κ = ones(n)
    if !iszero(B.frame_direction)
        # if we have a nonzero direction for the geodesic, add it and it gets curvature zero from the tensor
        V = cat(B.frame_direction / norm(M, p, B.frame_direction), V; dims = 2)
        κ[1] = 0 # no curvature along the geodesic direction, if x!=y
    end
    Ξ = [V[:, i] for i in 1:n]
    return CachedBasis(B, κ, Ξ)
end

@doc raw"""
    get_coordinates(M::Sphere, p, X, B::DefaultOrthonormalBasis)

Represent the tangent vector `X` at point `p` from the [`Sphere`](@ref) `M` in
an orthonormal basis by rotating the vector `X` using the rotation matrix
$2\frac{q q^\mathrm{T}}{q^\mathrm{T} q} - I$ where $q = p + (1, 0, …, 0)$.
"""
function get_coordinates(M::Sphere{n,ℝ}, p, X, B::DefaultOrthonormalBasis) where {n}
    isapprox(p[1], 1) && return X[2:end]
    xp1 = p .+ ntuple(i -> ifelse(i == 1, 1, 0), n + 1)
    return (2 * xp1 * dot(xp1, X) / dot(xp1, xp1) - X)[2:end]
end

function get_coordinates!(M::Sphere, Y, p, X, B::DefaultOrthonormalBasis)
    return copyto!(Y, get_coordinates(M, p, X, B))
end

function get_vector(M::Sphere{n,ℝ}, p, X, B::DefaultOrthonormalBasis) where {n}
    p[1] ≈ 1 && return vcat(0, X)
    xp1 = p .+ ntuple(i -> ifelse(i == 1, 1, 0), n + 1)
    X0 = vcat(0, X)
    return 2 * xp1 * real(dot(xp1, X0)) / real(dot(xp1, xp1)) - X0
end

function get_vector!(M::Sphere, Y::AbstractVector, p, X, B::DefaultOrthonormalBasis)
    return copyto!(Y, get_vector(M, p, X, B))
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
        @invoke_maker 1 Manifold injectivity_radius(
            M::AbstractSphere,
            rm::AbstractRetractionMethod,
        )
    end,
)

@doc raw"""
    inverse_retract(M::AbstractSphere, p, q, ::ProjectionInverseRetraction)

Compute the inverse of the projection based retraction on the [`AbstractSphere`](@ref) `M`,
i.e. rearranging $p+X = q\lVert p+X\rVert_2$ yields
since $⟨p,X⟩ = 0$ and when $d_{𝕊^2}(p,q) ≤ \frac{π}{2}$ that

````math
\operatorname{retr}_p^{-1}(q) = \frac{q}{⟨p, q⟩} - p.
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
\log_p q = d_{𝕊}(p,q) \frac{q-⟨p,q⟩ p}{\lVert q-⟨p,q⟩ p \rVert_2},
````

and a deterministic choice from the set of tangent vectors is returned if $x=-y$, i.e. for
opposite points.
"""
log(::AbstractSphere, ::Any...)

function log!(M::AbstractSphere, X, p, q)
    cosθ = real(dot(p, q))
    if cosθ ≈ -1 # appr. opposing points, return deterministic choice from set-valued log
        fill!(X, 0)
        if p[1] ≈ 1
            X[2] = 1
        else
            X[1] = 1
        end
        copyto!(X, X .- real(dot(p, X)) .* p)
        X .*= π / norm(X)
    else
        cosθ = cosθ > 1 ? one(cosθ) : cosθ
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

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` using
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

"""
    normal_tvector_distribution(S::Sphere{n,ℝ}, p, σ)

Generate a distribution in the tangent space at `p` by generating a
normal distribution in ambient space with standard deviation `σ`
projected to the tangent space at `p`.
"""
function normal_tvector_distribution(S::Sphere{n,ℝ}, p, σ) where {n}
    d = Distributions.MvNormal(zero(p), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(S), p, d, project!, p)
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

project!(S::AbstractSphere, q, p) = copyto!(q, p ./ norm(p))

@doc raw"""
    project(M::AbstractSphere, p, X)

Project the point `X` onto the tangent space at `p` on the [`Sphere`](@ref) `M`.

````math
\operatorname{proj}_{p}(X) = X - ⟨p, X⟩p
````
"""
project(::AbstractSphere, ::Any, ::Any)

project!(S::AbstractSphere, Y, p, X) = (Y .= X .- real(dot(p, X)) .* p)

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

Base.show(io::IO, ::Sphere{n,𝔽}) where {n,𝔽} = print(io, "Sphere($(n); field = $(𝔽))")
function Base.show(io::IO, ::ArraySphere{N,𝔽}) where {N,𝔽}
    return print(io, "ArraySphere($(join(N.parameters, ", ")); field = $(𝔽))")
end

"""
    uniform_distribution(M::AbstractSphere, p)

Uniform distribution on given [`AbstractSphere`](@ref) `M`. Generated points will be of
similar type as `p`.
"""
function uniform_distribution(M::AbstractSphere, p)
    d = Distributions.MvNormal(zero(p), 1.0)
    return ProjectedPointDistribution(M, d, project!, p)
end

@doc raw"""
    vector_transport_to(M::AbstractSphere, p, X, q, ::ParallelTransport)

Compute the parallel transport on the [`Sphere`](@ref) of the tangent vector `X` at `p`
to `q`, provided, the [`geodesic`](@ref) between `p` and `q` is unique. The formula reads

````math
P_{p←q}(X) = X - \frac{\langle \log_p q,X\rangle_p}{d^2_𝕊(p,q)}
\bigl(\log_xy + \log_yx \bigr).
````
"""
vector_transport_to(::AbstractSphere, ::Any, ::Any, ::Any, ::Any, ::ParallelTransport)

function vector_transport_to!(M::AbstractSphere, Y, p, X, q, ::ParallelTransport)
    X_pq = log(M, p, q)
    Xl = norm(M, p, X_pq)
    copyto!(Y, X)
    if Xl > 0
        factor = 2 * real(dot(X, q)) / (norm(p + q)^2)
        Y .-= factor .* (p .+ q)
    end
    return Y
end
