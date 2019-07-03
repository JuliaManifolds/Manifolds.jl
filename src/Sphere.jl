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
show(io::IO, S::Sphere{N}) where {N} = print(io, "Sphere($(N))");

@traitimpl HasMetric{Sphere,EuclideanMetric}

const CompactSymplectic1 = GroupManifold{Sphere{3},QuaternionMultiplicationOperation}

function CompactSymplectic1()
    return CompactSymplectic1(Sphere(3), QuaternionMultiplicationOperation())
end

@doc doc"""
    CompactSymplectic1 <: GroupManifold{Sphere{3},QuaternionMultiplicationOperation}

The compact symplectic group $Sp(1)$, which is also the quaternionic unitary
group, is the group of 1-dimensional matrices whose elements are quaternions.
Topologically, the base manifold is the 3-Sphere. Elements are represented
as quaternions with the elements $q=[s, v_1, v_2, v_3]$.

# Constructor

    CompactSymplectic1()
    CompactSymplectic(1)
"""
function CompactSymplectic(n::Int)
    n == 1 || error("CompactSymplectic($(n)) is not implemented.")
    return CompactSymplectic1()
end

show(io::IO, ::CompactSymplectic1) = print(io, "CompactSymplectic(1)");

@doc doc"""
    manifold_dimension(S::Sphere)

Return the dimension of the manifold $\mathbb S^n$, i.e. $n$.
"""
manifold_dimension(S::Sphere{N}) where {N} = N

proj!(S::Sphere, x) = (x ./= norm(x))

project_tangent!(S::Sphere, w, x, v) = (w .= v .- dot(x, v) .* x)

@doc doc"""
    inner(S::Sphere, x, w, v)

compute the inner product of the two tangent vectors `w,v` from the tangent
plane at `x` on the sphere `S=`$\mathbb S^n$ using the restriction of the
metric from the embedding, i.e. $ (v,w)_x = v^\mathrm{T}w $.
"""
inner(S::Sphere, x, w, v) = dot(w, v)

norm(S::Sphere, x, v) = norm(v)

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

injectivity_radius(S::Sphere, args...) = π

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
    return true
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
