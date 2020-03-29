@doc raw"""
    Sphere{N,M,𝔽} <: AbstractEmbeddedManifold{DefaultEmbeddingType}

The (unit) sphere manifold $𝕊^{m,n}$ is the set of all unit norm elements. The sphere is
represented in the embedding, and currently supports both vectors and matrices, i.e.

````math
𝕊^{n,m} := \bigl\{ p \in 𝔽^{m,n}\ \big|\ \lVert p \rVert = 1 \bigr\}
````

where $𝔽\in\{ℝ,ℂ\}. For $m=1$ and $𝔽=ℝ$ this  simplifies to unit vectors in $ℝ^n$ and the
more common notation is $𝕊^{n,m} ≜ 𝕊^{n-1}$.

The tangent space at point p is given by

````math
T_p𝕊^{n,m} := \bigl\{ X ∈ 𝔽^{n,m} : ⟨p,X⟩ = 0 \bigr \},
````

where $𝔽\in\{ℝ,ℂ\} and $⟨\cdot,\cdot⟩$ denotes the inner product in the
embedding $\mathbb 𝔽^{n+1}$.

This manifold is modeled as an embedded manifold to the [`Euclidean`](@ref), i.e.
several functions like the [`inner`](@ref) product and the [`zero_tangent_vector`](@ref)
are inherited from the embedding.

# Constructor

    Sphere(n[, field=ℝ])

Generate $𝕊^{n+1,1} ⊂ ℝ^{n+1}$.

    Sphere(n,m[, field=ℝ])

Generate $𝕊^{n,m} ⊂ ℝ^{n,m}$.
"""
struct Sphere{n,m,𝔽} <: AbstractEmbeddedManifold{DefaultIsometricEmbeddingType} end

Sphere(n::Int, field::AbstractNumbers=ℝ) = Sphere{n+1,1,field}()
Sphere(n::Int, m::Int, field::AbstractNumbers=ℝ) = Sphere{n,m,field}()

"""
    check_manifold_point(M, p; kwargs...)

Check whether `p` is a valid point on the [`Sphere`](@ref) `M`, i.e. is a vector
of length [`manifold_dimension`](@ref)`(M)+1` (approximately) of unit length.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::Sphere, p; kwargs...)
    mpv = invoke(
        check_manifold_point,
        Tuple{(typeof(get_embedding(M))), typeof(p)},
        get_embedding(M),
        p;
        kwargs...
    )
    mpv === nothing || return mpv
    if !isapprox(norm(p), 1.0; kwargs...)
        return DomainError(
            norm(p),
            "The point $(p) does not lie on the sphere $(M) since its norm is not 1.",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to `p` on the [`Sphere`](@ref) `M`, i.e.
after [`check_manifold_point`](@ref)`(M,p)`, `X` has to be of same dimension as `p`
and orthogonal to `p`.
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `p`.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(
    M::Sphere,
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
    if !isapprox(abs(real(dot(p, X))), 0.0; kwargs...)
        return DomainError(
            abs(dot(p, X)),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not orthogonal in the embedding.",
        )
    end
    return nothing
end

decorated_manifold(M::Sphere{n,m,𝔽}) where {n,m,𝔽}= Euclidean(n,m; field=𝔽)
decorated_manifold(M::Sphere{n,1,𝔽}) where {n,𝔽}= Euclidean(n; field=𝔽)

@doc raw"""
    distance(M::Sphere, p, q)

Compute the geodesic distance betweeen `p` and `q` on the [`Sphere`](@ref) `M`.
The formula is given by the (shorter) great arc length on the (or a) great circle
both `p` and `q` lie on.

````math
d_{𝕊^n}(p,q) = \arccos(⟨p,q⟩).
````
"""
distance(::Sphere, p, q) = acos(clamp(real(dot(p, q)), -1, 1))

embed!(::Sphere, q, p) = (q .= p)

embed!(::Sphere, Y, p, X) = (Y .= X)

@doc raw"""
    exp(M::Sphere, p, X)

Compute the exponential map from `p` in the tangent direction `X` on the [`Sphere`](@ref)
`M` by following the great arc eminating from `p` in direction `X`.

````math
\exp_p X = \cos(\lVert X \rVert_p)p + \sin(\lVert X \rVert_p)\frac{X}{\lVert X \rVert_p}X,
````
where $\lVert X \rVert_p$ is the [`norm`](@ref norm(::Sphere,p,X)) on the
[`Sphere`](@ref) `M`.
"""
exp(::Sphere, ::Any...)

function exp!(M::Sphere, q, p, X)
    θ = norm(M, p, X)
    q .= cos(θ) .* p .+ usinc(θ) .* X
    return q
end

flat!(M::Sphere, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

function get_basis(M::Sphere{n,1,ℝ}, p, B::DiagonalizingOrthonormalBasis{T,ℝ}) where {T,n}
    A = zeros(n, n)
    A[1, :] = transpose(p)
    A[2, :] = transpose(B.frame_direction)
    V = nullspace(A)
    κ = ones(n)
    if !iszero(B.frame_direction)
        # if we have a nonzero direction for the geodesic, add it and it gets curvature zero from the tensor
        V = cat(B.frame_direction / norm(M, p, B.frame_direction), V; dims = 2)
        κ[1] = 0 # no curvature along the geodesic direction, if x!=y
    end
    Ξ = [V[:, i] for i = 1:manifold_dimension(M)]
    return CachedBasis(B, κ, Ξ)
end

@doc raw"""
    get_coordinates(M::Sphere, p, X, B::DefaultOrthonormalBasis)

Represent the tangent vector `X` at point `p` from the [`Sphere`](@ref) `M` in
an orthonormal basis by rotating the vector `X` using the rotation matrix
$2\frac{q q^\mathrm{T}}{q^\mathrm{T} q} - I$ where $q = p + (1, 0, …, 0)$.
"""
function get_coordinates(M::Sphere{n,1,ℝ}, p, X, B::DefaultOrthonormalBasis) where {n}
    if isapprox(abs(p[1]), 1)
        return X[2:end]
    else
        xp1 = p .+ ntuple(i -> ifelse(i == 1, 1, 0), n + 1)
        return (2*xp1*real(dot(xp1, X))/real(dot(xp1, xp1))-X)[2:end]
    end
end

function get_coordinates!(M::Sphere, Y, p, X, B::DefaultOrthonormalBasis)
    return copyto!(Y, get_coordinates(M, p, X, B))
end

function get_vector(M::Sphere{n,1,ℝ}, p, X, B::DefaultOrthonormalBasis) where {n}
    p[1] ≈ 1 && return vcat(0, X)
    xp1 = p .+ ntuple(i -> ifelse(i == 1, 1, 0), n + 1)
    X0 = vcat(0, X)
    return 2 * xp1 * real(dot(xp1, X0)) / real(dot(xp1, xp1)) - X0
end

function get_vector!(M::Sphere, Y::AbstractVector, p, X, B::DefaultOrthonormalBasis)
    return copyto!(Y, get_vector(M, p, X, B))
end

@doc raw"""
    injectivity_radius(M::Sphere[, p])

Return the injectivity radius for the [`Sphere`](@ref) `M`, which is globally $π$.

    injectivity_radius(M::Sphere, x, ::ProjectionRetraction)

Return the injectivity radius for the [`ProjectionRetraction`](@ref) on the
[`Sphere`](@ref), which is globally $\frac{π}{2}$.
"""
injectivity_radius(::Sphere) = π
injectivity_radius(::Sphere, ::ExponentialRetraction) = π
injectivity_radius(::Sphere, ::ProjectionRetraction) = π / 2
injectivity_radius(::Sphere, ::Any) = π
injectivity_radius(::Sphere, ::Any, ::ExponentialRetraction) = π
injectivity_radius(::Sphere, ::Any, ::ProjectionRetraction) = π / 2
eval(quote
    @invoke_maker 1 Manifold injectivity_radius(M::Sphere, rm::AbstractRetractionMethod)
end)

@doc raw"""
    inverse_retract(M::Sphere, p, q, ::ProjectionInverseRetraction)

Compute the inverse of the projection based retraction on the [`Sphere`](@ref),
i.e. rearranging $p+X = q\lVert p+X\rVert_2$ yields
since $⟨p,X⟩ = 0$ and when $d_{𝕊^2}(p,q) ≤ \frac{π}{2}$ that

````math
\operatorname{retr}_p^{-1}(q) = \frac{q}{⟨p, q⟩} - p.
````
"""
inverse_retract(::Sphere, ::Any, ::Any, ::ProjectionInverseRetraction)

function inverse_retract!(::Sphere, X, p, q, ::ProjectionInverseRetraction)
    return (X .= q ./ real(dot(p, q)) .- p)
end

@doc raw"""
    log(M::Sphere, p, q)

Compute the logarithmic map on the [`Sphere`](@ref) `M`, i.e. the tangent vector,
whose geodesic starting from `p` reaches `q` after time 1.
The formula reads for $x ≠ -y$

````math
\log_p q = d_{𝕊^n}(p,q) \frac{q-⟨p,q⟩ p}{\lVert q-⟨p,q⟩ p \rVert_2},
````

and a deterministic choice from the set of tangent vectors is returned if $x=-y$, i.e. for
opposite points.
"""
log(::Sphere, ::Any...)

function log!(S::Sphere, X, p, q)
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
    return project!(S, X, p, X)
end

@doc raw"""
    manifold_dimension(M::Sphere)

Return the dimension of the [`Sphere`](@ref)`(n) `M`, i.e. $𝕊^n$, which is $\dim(𝕊^n) = n$.
"""
manifold_dimension(S::Sphere{n,m,ℝ}) where {n,m} = n*m-1
manifold_dimension(S::Sphere{n,m,ℂ}) where {n,m} = 2*n*m-1

"""
    mean(
        S::Sphere,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/2);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::Sphere, ::Any...)

function mean!(S::Sphere, p, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(S, p, x, w, GeodesicInterpolationWithinRadius(π / 2); kwargs...)
end

"""
    normal_tvector_distribution(S::Sphere, p, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `p`.
"""
function normal_tvector_distribution(S::Sphere, p, σ)
    d = Distributions.MvNormal(zero(p), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(S), p, d, project_vector!, p)
end

@doc raw"""
    project(M::Sphere, p)

Project the point `p` from the embedding onto the [`Sphere`](@ref) `M`.

````math
    \operatorname{proj}_{𝕊^{n,m}}(p) = \frac{p}{\lVert p \rVert},
````
where $\lVert\cdot\rVert$ denotes the usual 2-norm for vectors if $m=1$ and the Frobenius
norm for the case $m>1$.
"""
project(::Sphere, ::Any)

project!(S::Sphere, q, p) = copyto!(q, p./ norm(p))

@doc raw"""
    project(M::Sphere, p, X)

Project the point `X` onto the tangent space at `p` on the [`Sphere`](@ref) `M`.

````math
\operatorname{proj}_{p}(X) = X - ⟨p, X⟩p
````
"""
project(::Sphere, ::Any, ::Any)

project!(S::Sphere, Y, p, X) = (Y .= X .- real(dot(p, X)) .* p)

@doc raw"""
    representation_size(M::Sphere)

Return the size points on the [`Sphere`](@ref) `M` are represented as, i.e.
for the `n`-dimensional [`Sphere`](@ref) it is vectors of size `(n+1,)` and
for (Forbenius-)unit-norm matrixes (n,m).
"""
@generated representation_size(::Sphere{n,1}) where {n} = (n,)
@generated representation_size(::Sphere{n,m}) where {n,m} = (n,m,)

@doc raw"""
    retract(M::Sphere, p, X, ::ProjectionRetraction)

Compute the retraction that is based on projection, i.e.

````math
\operatorname{retr}_p(X) = \frac{p+X}{\lVert p+X \rVert_2}
````
"""
retract(::Sphere, ::Any, ::Any, ::ProjectionRetraction)

function retract!(M::Sphere, q, p, X, ::ProjectionRetraction)
    q .= p .+ X
    return project!(M, q, q)
end

show(io::IO, ::Sphere{n,1,ℝ}) where {n} = print(io, "Sphere($(n-1))")
show(io::IO, ::Sphere{n,m,ℝ}) where {n,m} = print(io, "Sphere($(n),$(m))")
show(io::IO, ::Sphere{n,1,ℂ}) where {n} = print(io, "Sphere($(n-1); field = ℂ)")
show(io::IO, ::Sphere{n,m,ℂ}) where {n,m} = print(io, "Sphere($(n),$(m); field = ℂ)")

"""
    uniform_distribution(M::Sphere, p)

Uniform distribution on given [`Sphere`](@ref) `M`. Generated points will be of
similar type as `p`.
"""
function uniform_distribution(M::Sphere, p)
    d = Distributions.MvNormal(zero(p), 1.0)
    return ProjectedPointDistribution(M, d, project!, p)
end

@doc doc"""
    vector_transport_to(M::Sphere, p, X, q, ::ParallelTransport)

Compute the paralllel transport on the [`Sphere`](@ref) of the tangent vector `X` at `p`
to `q`, provided, the [`geodesic`](@ref) between `p` and `q` is unique. The formula reads

````math
P_{p←q}(X) = X - \frac{\langle \log_p q,X\rangle_p}{d^2_{𝕊^{n,m}}(p,q)}
\bigl(\log_xy + \log_yx \bigr).
````
"""
vector_transport_to(::Sphere, ::Any, ::Any, ::Any, ::Any, ::ParallelTransport)

function vector_transport_to!(M::Sphere, Y, p, X, q, ::ParallelTransport)
    X_pq = log(M, p, q)
    Xl = norm(M, p, X_pq)
    copyto!(Y, X)
    if Xl > 0
        factor = 2 * real(dot(X, q)) / (norm(p + q)^2)
        Y .-= factor .* (p .+ q)
    end
    return Y
end
