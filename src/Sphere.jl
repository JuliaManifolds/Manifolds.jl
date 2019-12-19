@doc doc"""
    Sphere{N} <: Manifold

The unit sphere manifold $\mathbb S^n$ represented by $n+1$-Tuples, i.e. in by
vectors in $\mathbb R^{n+1}$ of unit length

# Constructor

    Sphere(n)

generates the $\mathbb S^{n}\subset \mathbb R^{n+1}$
"""
struct Sphere{N} <: Manifold end
Sphere(n::Int) = Sphere{n}()

function representation_size(::Sphere{N}) where N
    return (N+1,)
end

@doc doc"""
    manifold_dimension(S::Sphere)

Return the dimension of the manifold $\mathbb S^n$, i.e. $n$.
"""
manifold_dimension(S::Sphere{N}) where {N} = N

project_point!(S::Sphere, x) = (x ./= norm(x))

project_tangent!(S::Sphere, w, x, v) = (w .= v .- dot(x, v) .* x)

@doc doc"""
    inner(S::Sphere, x, w, v)

compute the inner product of the two tangent vectors `w,v` from the tangent
plane at `x` on the sphere `S=`$\mathbb S^n$ using the restriction of the
metric from the embedding, i.e. $ (v,w)_x = v^\mathrm{T}w $.
"""
@inline inner(S::Sphere, x, w, v) = dot(w, v)

norm(S::Sphere, x, v) = norm(v)

distance(S::Sphere, x, y) = acos(clamp(dot(x, y), -1, 1))

function exp!(S::Sphere, y, x, v)
    θ = norm(S, x, v)
    y .= cos(θ) .* x .+ usinc(θ) .* v
    return y
end

function log!(S::Sphere, v, x, y)
    cosθ = dot(x, y)
    if cosθ ≈ -1
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

injectivity_radius(S::Sphere, args...) = π

function zero_tangent_vector!(S::Sphere, v, x)
    fill!(v, 0)
    return v
end

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

function flat!(M::Sphere, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    copyto!(v.data, w.data)
    return v
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

"""
    check_manifold_point(S,x; kwargs...)

checks, whether `x` is a valid point on the [`Sphere`](@ref) `S`, i.e. is a vector
of length [`manifold_dimension`](@ref)`(S)+1` (approximately) of unit length.
The tolerance for the last test can be set using the ´kwargs...`.
"""
function check_manifold_point(S::Sphere{N},x; kwargs...) where {N}
    if length(x) != N+1
        return DomainError(size(x),"The point $(x) does not lie on $S, since its size is not $(N+1).")
    end
    if !isapprox(norm(x), 1.; kwargs...)
        return DomainError(norm(x), "The point $(x) does not lie on the sphere $(S) since its norm is not 1.")
    end
    return nothing
end

"""
    check_tangent_vector(S,x,v; kwargs... )

checks whether `v` is a tangent vector to `x` on the [`Sphere`](@ref) `S`, i.e.
after [`check_manifold_point`](@ref)`(S,x)`, `v` has to be of same dimension as `x`
and orthogonal to `x`.
The tolerance for the last test can be set using the ´kwargs...`.
"""
function check_tangent_vector(S::Sphere{N},x,v; kwargs...) where N
    perr = check_manifold_point(S,x)
    perr === nothing || return perr
    if length(v) != N+1
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

"""
    normal_tvector_distribution(S::Sphere, x, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `x`.
"""
function normal_tvector_distribution(S::Sphere, x, σ)
    d = Distributions.MvNormal(zero(x), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(S), x, d, project_vector!, x)
end

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
mean(::Sphere, args...)

mean!(S::Sphere, y, x::AbstractVector, w::AbstractVector; kwargs...) =
    mean!(S, y, x, w, GeodesicInterpolationWithinRadius(π/2); kwargs...)
