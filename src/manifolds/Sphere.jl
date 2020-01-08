@doc doc"""
    Sphere{N} <: Manifold

The unit sphere manifold $\mathbb S^n$ represented by $n+1$-Tuples, i.e. in by
vectors in $\mathbb R^{n+1}$ of unit length

# Constructor

    Sphere(n)

Generate the $\mathbb S^{n}\subset \mathbb R^{n+1}$
"""
struct Sphere{N} <: Manifold end
Sphere(n::Int) = Sphere{n}()

"""
    check_manifold_point(S, x; kwargs...)

Check whether `x` is a valid point on the [`Sphere`](@ref) `S`, i.e. is a vector
of length [`manifold_dimension`](@ref)`(S)+1` (approximately) of unit length.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(S::Sphere{N},x; kwargs...) where {N}
    if size(x) != representation_size(S)
        return DomainError(size(x),"The point $(x) does not lie on $S, since its size is not $(N+1).")
    end
    if !isapprox(norm(x), 1.; kwargs...)
        return DomainError(norm(x), "The point $(x) does not lie on the sphere $(S) since its norm is not 1.")
    end
    return nothing
end

"""
    check_tangent_vector(S, x, v; kwargs... )

Check whether `v` is a tangent vector to `x` on the [`Sphere`](@ref) `S`, i.e.
after [`check_manifold_point`](@ref)`(S,x)`, `v` has to be of same dimension as `x`
and orthogonal to `x`.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(S::Sphere{N},x,v; kwargs...) where N
    perr = check_manifold_point(S,x)
    perr === nothing || return perr
    if size(v) != representation_size(S)
        return DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $S since its size does not match $(N+1).")
    end
    if !isapprox( abs(dot(x,v)), 0.; kwargs...)
        return DomainError(abs(dot(x,v)),
            "The vector $(v) is not a tangent vector to $(x) on $(S), since it is not orthogonal in the embedding."
        )
    end
    return nothing
end

@doc doc"""
    distance(M::Sphere, x, y)

Compute the geodesic distance betweeen `x` and `y` on the [`Sphere`](@ref) `M`.
The formula is given by the (shorter) great arc length on the (or a) great circle
both `x` and `y` lie on.

````math
d_{\mathbb S^n}(x,y) = \operatorname{acos}(\langle x, y\rangle).
````
"""
distance(S::Sphere, x, y) = acos(clamp(dot(x, y), -1, 1))

@doc doc"""
    exp(M::Sphere, x, v)

Compute the exponential map from `x` into the tangent direction `v` on the [`Sphere`](@ref)
`M` by following the great arc eminating from `x` in direction `v` along with length of `v`.

````math
\exp_x v = \cos(\lVert v \rVert_x)x + \sin(\lVert v \rVert_x)\frac{v}{\lVert v \rVert_x},
````
where $\lVert v \rVert_x$ is the [`norm`](@ref norm(::Sphere,x,v)) on the
[`Sphere`](@ref) `M`.
"""
exp(::Sphere, ::Any...)
function exp!(M::Sphere, y, x, v)
    θ = norm(M, x, v)
    y .= cos(θ) .* x .+ usinc(θ) .* v
    return y
end

function flat!(M::Sphere, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    copyto!(v.data, w.data)
    return v
end

@doc doc"""
    injectivity_radius(M::Sphere[, x])

Return the injectivity radius for the [`Sphere`](@ref) `M`, which is globally $\pi$.
"""
injectivity_radius(::Sphere, ::Any...) = π

@doc doc"""
    injectivity_radius(M::Sphere, x, ::ProjectionRetraction)

Return the injectivity radius for the [`ProjectionRetraction`](@ref) on the
[`Sphere`](@ref), which is globally $\frac{\pi}{2}$.
"""
injectivity_radius(::Sphere, ::Any, ::ProjectionRetraction) = π/2

@doc doc"""
    inner(S::Sphere, x, w, v)

Compute the inner product of the two tangent vectors `w,v` from the tangent
plane at `x` on the sphere `S=`$\mathbb S^n$ using the restriction of the
metric from the embedding, i.e. $ (v,w)_x = v^\mathrm{T}w $.
"""
@inline inner(S::Sphere, x, w, v) = dot(w, v)

function inverse_represent_in_basis(M::Sphere{N}, x, v, B::ArbitraryOrthonormalBasis) where N
    if isapprox(x[1], 1)
        return vcat(0, v)
    else
        xp1 = x .+ ntuple(i -> ifelse(i == 1, 1, 0), N+1)
        v0 = vcat(0, v)
        return 2*xp1*dot(xp1, v0)/dot(xp1, xp1) - v0
    end
end

@doc doc"""
    inverse_retract(M::Sphere, x, y, ::ProjectionInverseRetraction)

Compute the inverse of the projection based retraction on the [`Sphere`](@ref),
i.e. rearranging $x+v = y\lVert x+d \rVert_2$ yields
since $\langle x,v\rangle = 0$ and when $d_{\mathbb S^2}(x,y) \leq \frac{\pi}{2}$ that

````math
\operatorname{retr}_x^{-1}(y) = \frac{y}{\langle x, y \rangle} - x.
````
"""
inverse_retract(::Sphere, ::Any, ::Any, ::ProjectionInverseRetraction)
inverse_retract!(::Sphere, v, x, y, ::ProjectionInverseRetraction) = (v .= y./dot(x,y) .- x)

@doc doc"""
    log(M::Sphere, x, y)

Compute the logarithmic map on the [`Sphere`](@ref) `M`, i.e. the tangent vector, whose
geodesic starting from `x` reaches `y` after time 1.
The formula reads for $x\neq -y$

````math
\log_x y = d_{\mathbb S^n}(x,y)
\frac{y-\langle x,y\rangle x}{\lVert y-\langle x,y\rangle x \rVert_2},
````
and a deterministic choice from the set of tangent vectors is returned if $x=-y$, i.e. for
opposite points.
"""
log(::Sphere, ::Any...)

function log!(S::Sphere, v, x, y)
    cosθ = dot(x, y)
    if cosθ ≈ -1 # appr. opposing points, return deterministic choice from set-valued log
        fill!(v, 0)
        if x[1] ≈ 1
            v[2] = 1
        else
            v[1] = 1
        end
        copyto!(v, v .- dot(x, v) .* x)
        v .*= π / norm(v)
    else
        cosθ = cosθ > 1 ? one(cosθ) : cosθ
        θ = acos(cosθ)
        v .= (y .- cosθ .* x) ./ usinc(θ)
    end
    project_tangent!(S, v, x, v)
    return v
end

@doc doc"""
    manifold_dimension(S::Sphere)

Return the dimension of the manifold $\mathbb S^n$, i.e. $n$.
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
mean!(S::Sphere, y, x::AbstractVector, w::AbstractVector; kwargs...) =
    mean!(S, y, x, w, GeodesicInterpolationWithinRadius(π/2); kwargs...)

@doc doc"""
    norm(M::Sphere, x, v)

Compute the length of the tangent vector `v` from the tangent space at `x` on the
[`Sphere`](@ref) `M`, which is the norm in the embedding, i.e.
````math
\lVert v \rVert_x = \lVert v \rVert_2.
````
"""
norm(M::Sphere, x, v) = norm(v)

"""
    normal_tvector_distribution(S::Sphere, x, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `x`.
"""
function normal_tvector_distribution(S::Sphere, x, σ)
    d = Distributions.MvNormal(zero(x), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(S), x, d, project_vector!, x)
end

@doc doc"""
    project_point(M::Sphere, x)

Project the point `x` from the embedding onto the [`Sphere`](@ref) `M`.

````math
\operatorname{proj}_{\mathbb S^n}(x) = \frac{x}{\lVert x \rVert_2}.
````
"""
project_point(::Sphere, ::Any...)
project_point!(S::Sphere, x) = (x ./= norm(x))

@doc doc"""
    project_tangent(M::Sphere, x, v)

Project the point `v` onto the tangent space at `x` on the [`Sphere`](@ref) `M`.

````math
\operatorname{proj}_{x}(v) = v - \langle x, v \rangle_x
````
"""
project_tangent(::Sphere, ::Any...)
project_tangent!(S::Sphere, w, x, v) = (w .= v .- dot(x, v) .* x)

@doc doc"""
    represent_in_basis(M::Sphere, x, v, B::ArbitraryOrthonormalBasis)

Represent the tangent vector `v` at point `x` from a sphere `M` in
an orthonormal basis by rotating the vector `v` using rotation matrix
$2\frac{x_p x_p^T}{x_p^T x_p} - I$ where $x_p = x + (1, 0, \dots, 0)$.
"""
function represent_in_basis(M::Sphere{N}, x, v, B::ArbitraryOrthonormalBasis) where N
    if isapprox(x[1], 1)
        return v[2:end]
    else
        xp1 = x .+ ntuple(i -> ifelse(i == 1, 1, 0), N+1)
        return (2*xp1*dot(xp1, v)/dot(xp1, xp1) - v)[2:end]
    end
end

@doc doc"""
    representation_size(M::Sphere)

Return the size points on the [`Sphere`](@ref) `M` are represented as, i.e.
for the `n`-dimensional [`Sphere`](@ref) it is vectors of size `(n+1,)`.
"""
representation_size(::Sphere{N}) where N = (N+1,)

@doc doc"""
    retract(M::Sphere, x, y, ::ProjectionRetraction)

Compute the retraction that is based on projection, i.e.

````math
\operatorname{retr}_x(v) = \frac{x+v}{\lVert x+v \rVert_2}
````
"""
retract(::Sphere, ::Any, ::Any, ::ProjectionRetraction)
function retract!(M::Sphere, y, x, v, ::ProjectionRetraction)
    y .= x .+ v
    project_point!(M, y)
    return y
end

function sharp!(M::Sphere, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
    copyto!(v.data, w.data)
    return v
end

"""
    uniform_distribution(S::Sphere, x)

Uniform distribution on given sphere. Generated points will be of similar
type to `x`.
"""
function uniform_distribution(S::Sphere, x)
    d = Distributions.MvNormal(zero(x), 1.0)
    return ProjectedPointDistribution(S, d, project_point!, x)
end

@doc doc"""
    vector_transport_to(M::Sphere, x, v, y, ::ParallelTransport)

Compute the [`ParallelTransport`](@ref) on the [`Sphere`](@ref) `M`, which is given by

````math
P_{y\gets x}(v) = v - \frac{\langle \log_xy,v\rangle_x}{d^2_{\mathbb S^n}(x,y)}
\bigl(\log_xy + \log_yx \bigr).
````
"""
vector_transport_to(::Sphere, ::Any, ::Any, ::Any, ::ParallelTransport)
function vector_transport_to!(M::Sphere, vto, x, v, y, ::ParallelTransport)
    v_xy = log(M, x, y)
    vl = norm(M, x, v_xy)
    vto .= v
    if vl > 0
        factor = 2*dot(v, y)/(norm(x + y)^2)
        vto .-= factor.*(x .+ y)
    end
    return vto
end

@doc doc"""
    zero_tangent_vector(M::Sphere, x)

Return the zero tangent vector from the tangent space at `x` on the [`Sphere`](@ref) `M`,
which is the zero vector in the embedding.
"""
zero_tangent_vector(::Sphere, ::Any...)
function zero_tangent_vector!(S::Sphere, v, x)
    fill!(v, 0)
    return v
end
