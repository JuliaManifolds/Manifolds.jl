@doc raw"""
    Sphere{N} <: Manifold

The unit sphere manifold $𝕊^n$ represented by $n+1$-Tuples, i.e. in by
vectors in $ℝ^{n+1}$ of unit length

# Constructor

    Sphere(n)

Generate the $𝕊^{n}\subset ℝ^{n+1}$
"""
struct Sphere{N} <: Manifold end

Sphere(n::Int) = Sphere{n}()

function get_basis(M::Sphere{N}, x, B::DiagonalizingOrthonormalBasis) where {N}
    A = zeros(N + 1, N + 1)
    A[1, :] = transpose(x)
    A[2, :] = transpose(B.frame_direction)
    V = nullspace(A)
    κ = ones(N)
    if !iszero(B.frame_direction)
        # if we have a nonzero direction for the geodesic, add it and it gets curvature zero from the tensor
        V = cat(B.frame_direction / norm(M, x, B.frame_direction), V; dims = 2)
        κ[1] = 0 # no curvature along the geodesic direction, if x!=y
    end
    vecs = [V[:, i] for i = 1:N]
    return PrecomputedDiagonalizingOrthonormalBasis(vecs, κ)
end

"""
    check_manifold_point(M, p; kwargs...)

Check whether `p` is a valid point on the [`Sphere`](@ref) `M`, i.e. is a vector
of length [`manifold_dimension`](@ref)`(M)+1` (approximately) of unit length.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::Sphere{N}, p; kwargs...) where {N}
    if size(p) != representation_size(M)
        return DomainError(
            size(p),
            "The point $(p) does not lie on $M, since its size is not $(N+1).",
        )
    end
    if !isapprox(norm(p), 1.0; kwargs...)
        return DomainError(
            norm(p),
            "The point $(p) does not lie on the sphere $(M) since its norm is not 1.",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`Sphere`](@ref) `M`, i.e.
after [`check_manifold_point`](@ref)`(M,p)`, `X` has to be of same dimension as `p`
and orthogonal to `p`.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(M::Sphere{N}, p, X; kwargs...) where {N}
    perr = check_manifold_point(M, p)
    perr === nothing || return perr
    if size(X) != representation_size(M)
        return DomainError(
            size(X),
            "The vector $(X) is not a tangent to a point on $M since its size does not match $(N+1).",
        )
    end
    if !isapprox(abs(dot(p, X)), 0.0; kwargs...)
        return DomainError(
            abs(dot(p, X)),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not orthogonal in the embedding.",
        )
    end
    return nothing
end

@doc raw"""
    distance(M::Sphere, p, q)

Compute the geodesic distance betweeen `p` and `q` on the [`Sphere`](@ref) `M`.
The formula is given by the (shorter) great arc length on the (or a) great circle
both `p` and `q` lie on.

````math
d_{𝕊^n}(p,q) = \operatorname{acos}(⟨p,q⟩).
````
"""
distance(::Sphere, x, y) = acos(clamp(dot(x, y), -1, 1))

@doc raw"""
    exp(M::Sphere, p, X)

Compute the exponential map from `p` into the tangent direction `X` on the [`Sphere`](@ref)
`M` by following the great arc emanating from `p` in direction `X`.

````math
\exp_p X = \cos(\lVert X \rVert_p)p + \sin(\lVert X \rVert_p)\frac{X}{\lVert X \rVert_p},
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

@doc raw"""
    injectivity_radius(M::Sphere[, p])

Return the injectivity radius for the [`Sphere`](@ref) `M`, which is globally $π$.

    injectivity_radius(M::Sphere, x, ::ProjectionRetraction)

Return the injectivity radius for the [`ProjectionRetraction`](@ref) on the
[`Sphere`](@ref), which is globally $\frac{π}{2}$.
"""
injectivity_radius(::Sphere, ::Any...) = π
injectivity_radius(::Sphere, ::Any, ::ProjectionRetraction) = π / 2

@doc raw"""
    inner(M::Sphere, p, X, Y)

Compute the inner product of the two tangent vectors `X`, `Y` from the tangent
space at `p` on the [`Sphere`](@ref) `M` using the restriction of the
metric from the embedding, i.e. $ g_p(X,Y) = X^\mathrm{T}Y$.
"""
@inline inner(S::Sphere, p, X, Y) = dot(X, Y)

function get_vector(M::Sphere{N}, p, X, B::ArbitraryOrthonormalBasis) where {N}
    p[1] ≈ 1 && return vcat(0, X)
    xp1 = p .+ ntuple(i -> ifelse(i == 1, 1, 0), N + 1)
    v0 = vcat(0, X)
    return 2 * xp1 * dot(xp1, v0) / dot(xp1, xp1) - v0
end

@doc raw"""
    inverse_retract(M::Sphere, p, q, ::ProjectionInverseRetraction)

Compute the inverse of the projection based retraction on the [`Sphere`](@ref),
i.e. rearranging $p+X = q\lVert p+X\rVert_2$ yields
since $⟨p,X⟩ = 0$ and when $d_{𝕊^2}(p,q) \leq \frac{π}{2}$ that

````math
\operatorname{retr}_p^{-1}(q) = \frac{q}{⟨p, q⟩} - p.
````
"""
inverse_retract(::Sphere, ::Any, ::Any, ::ProjectionInverseRetraction)

function inverse_retract!(::Sphere, X, p, q, ::ProjectionInverseRetraction)
    return (X .= q ./ dot(p, q) .- p)
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
    cosθ = dot(p, q)
    if cosθ ≈ -1 # appr. opposing points, return deterministic choice from set-valued log
        fill!(X, 0)
        if p[1] ≈ 1
            X[2] = 1
        else
            X[1] = 1
        end
        copyto!(X, X .- dot(p, X) .* p)
        X .*= π / norm(X)
    else
        cosθ = cosθ > 1 ? one(cosθ) : cosθ
        θ = acos(cosθ)
        X .= (q .- cosθ .* p) ./ usinc(θ)
    end
    return project_tangent!(S, X, p, X)
end

@doc raw"""
    manifold_dimension(M::Sphere)

Return the dimension of the [`Sphere`](@ref)`(n) `M`, i.e. $𝕊^n$, which is $\dim(𝕊^n) = n$.
"""
manifold_dimension(S::Sphere{N}) where {N} = N

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

@doc raw"""
    norm(M::Sphere, p, X)

Compute the length of the tangent vector `v` from the tangent space at `p` on the
[`Sphere`](@ref) `M`, which is the norm in the embedding, i.e.
````math
\lVert X \rVert_p = \lVert X \rVert_2.
````
"""
norm(M::Sphere, p, X) = norm(X)

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
    project_point(M::Sphere, p)

Project the point `p` from the embedding onto the [`Sphere`](@ref) `M`.

````math
\operatorname{proj}_{𝕊^n}(p) = \frac{p}{\lVert p \rVert_2}.
````
"""
project_point(::Sphere, ::Any...)

project_point!(S::Sphere, p) = (p ./= norm(p))

@doc raw"""
    project_tangent(M::Sphere, p, X)

Project the point `X` onto the tangent space at `p` on the [`Sphere`](@ref) `M`.

````math
\operatorname{proj}_{p}(X) = X - ⟨p, X⟩p
````
"""
project_tangent(::Sphere, ::Any...)

project_tangent!(S::Sphere, Y, p, X) = (Y .= X .- dot(p, X) .* p)

@doc raw"""
    get_coordinates(M::Sphere, p, X, B::ArbitraryOrthonormalBasis)

Represent the tangent vector `X` at point `p` from the [`Sphere`](@ref) `M` in
an orthonormal basis by rotating the vector `X` using the rotation matrix
$2\frac{q q^\mathrm{T}}{q^\mathrm{T} q} - I$ where $q = p + (1, 0, …, 0)$.
"""
function get_coordinates(M::Sphere{N}, p, X, B::ArbitraryOrthonormalBasis) where {N}
    if isapprox(p[1], 1)
        return X[2:end]
    else
        xp1 = p .+ ntuple(i -> ifelse(i == 1, 1, 0), N + 1)
        return (2*xp1*dot(xp1, X)/dot(xp1, xp1)-X)[2:end]
    end
end

@doc raw"""
    representation_size(M::Sphere)

Return the size points on the [`Sphere`](@ref) `M` are represented as, i.e.
for the `n`-dimensional [`Sphere`](@ref) it is vectors of size `(n+1,)`.
"""
@generated representation_size(::Sphere{N}) where {N} = (N + 1,)

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
    return project_point!(M, q)
end

sharp!(M::Sphere, X::TFVector, p, ξ::CoTFVector) = copyto!(X, ξ)

show(io::IO, ::Sphere{N}) where {N} = print(io, "Sphere($(N))")

"""
    uniform_distribution(M::Sphere, p)

Uniform distribution on given [`Sphere`](@ref) `M`. Generated points will be of
similar type as `p`.
"""
function uniform_distribution(M::Sphere, p)
    d = Distributions.MvNormal(zero(p), 1.0)
    return ProjectedPointDistribution(M, d, project_point!, p)
end

@doc raw"""
    vector_transport_to(M::Sphere, p, X, q, ::ParallelTransport)

Compute the [`ParallelTransport`](@ref) on the [`Sphere`](@ref) `M`, which is given by

````math
\mathcal P_{q←p}(X) = X - \frac{⟨\log_pq,X⟩_p}{d^2_{𝕊^n}(p,q)}
\bigl(\log_pq + \log_qp \bigr).
````
"""
vector_transport_to(::Sphere, ::Any, ::Any, ::Any, ::ParallelTransport)

function vector_transport_to!(M::Sphere, Y, p, X, q, ::ParallelTransport)
    v_xy = log(M, p, q)
    vl = norm(M, p, v_xy)
    copyto!(Y, X)
    if vl > 0
        factor = 2 * dot(X, q) / (norm(p + q)^2)
        Y .-= factor .* (p .+ q)
    end
    return Y
end

@doc raw"""
    zero_tangent_vector(M::Sphere, p)

Return the zero tangent vector from the tangent space at `p` on the [`Sphere`](@ref) `M`,
which is the zero vector in the embedding.
"""
zero_tangent_vector(::Sphere, ::Any...)

zero_tangent_vector!(::Sphere, X, p) = fill!(X, 0)
