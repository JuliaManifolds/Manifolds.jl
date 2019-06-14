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

@doc doc"""
    dimension(S::Sphere)

returns the dimension of the manifold $\mathbb S^n$, i.e. $n$.
"""
dimension(S::Sphere{N}) where {N} = N

@doc doc"""
    dot(S,x,w,v)

compute the inner product of the two tangent vectors `w,v` from the tangent
plane at `x` on the sphere `S=`$\mathbb S^n$ using the restriction of the metric from the
embedding, i.e. $ (v,w)_x = v^\mathrm{T}w $.
"""
dot(S::Sphere, x, w, v) = dot(w, v)

proj!(S::Sphere, x) = (x ./= norm(x))

project_tangent!(S::Sphere, w, x, v) = (w .= v .- dot(x, v).*x)
distance(S::Sphere, x, y) = acos(dot(x, y))

function exp!(S::Sphere, y, x, v)
    nv = norm(S, x, v)
    if nv ≈ 0.0
        y .= x
    else
        y .= cos(nv).*x .+ (sin(nv)/nv).*v
    end
    return y
end

function log!(S::Sphere, v, x, y)
    dot_xy = dot(x, y)
    θ = acos(dot_xy)
    if θ ≈ 0.0
        zero_tangent_vector!(S, v, x)
    else
        v .= (θ/sin(θ)) .* (y .- dot_xy.*x)
    end
    return v
end

zero_tangent_vector(S::Sphere, x) = zero(x)
zero_tangent_vector!(S::Sphere, v, x) = (v .= zero(x))

"""
	uniform_sphere_distribution(S::Sphere, x)

Uniform distribution on given sphere. Generated points will be of similar
type to `x`.
"""
function uniform_distribution(S::Sphere, x)
	d = Distributions.MvNormal(zero(x), 1.0)
	return ProjectedPointDistribution(S, d, proj!, x)
end

"""
    is_manifold_point(S,x; kwargs...)

checks, whether `x` is a valid point on the [`Sphere`](@ref) `S`, i.e. is a vector
of length [`manifold_dimension`](@ref)`(S)+1` (approximately) of unit length.
The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_manifold_point(S::Sphere{N},x; kwargs...) where {N}
    if length(x) != N+1
        throw(DomainError(size(x),"The point $(x) does not lie on $S, since its size is not $(N+1)."))
    end
    if !isapprox(norm(x), 1.; kwargs...)
        throw(DomainError(norm(x), "The point $(x) does not lie on the sphere $(S) since its norm is not 1."))
    end
    return true
end

"""
    is_tangent_vector(S,x,v; kwargs... )

checks whether `v` is a tangent vector to `x` on the [`Sphere`](@ref) `S`, i.e.
atfer [`is_manifold_point`](@ref)`(S,x)`, `v` has to be of same dimension as `x`
and orthogonal to `x`.
The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_tangent_vector(S::Sphere{N},x,v; kwargs...) where N
    is_manifold_point(S,x)
    if length(v) != N+1
        throw(DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $S since its size does not match $(N+1)."))
    end
    if !isapprox( abs(dot(x,v)), 0.; kwargs...)
        throw(DomainError(abs(dot(x,v)),
            "The vector $(v) is not a tangent vector to $(x) on $(S), since it is not orthogonal in the embedding."
        ))
    end
end

"""
	gaussian_sphere_tvector_distribution(S::Sphere, x, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `x`.
"""
function normal_tvector_distribution(S::Sphere, x, σ)
	d = Distributions.MvNormal(zero(x), σ)
	return ProjectedTVectorDistribution(S, x, d, project_tangent!, x)
end
