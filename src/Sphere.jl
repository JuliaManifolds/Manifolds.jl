@doc doc"""
    Sphere{T<:Tuple} Manifold

The unit sphere manifold $\mathbb S^n$ represented by $n+1$-Tuples, i.e. in by
vectors in $\mathbb R^{n+1}$ of unit length

# Constructor

    Sphere(n)

generates the $\mathbb S^{n}\subset \mathbb R^{n+1}$
"""
struct Sphere{T<:Tuple} <: Manifold where {T} end
Sphere(n::Int) = Sphere{Tuple{n+1}}()

@doc doc"""
    manifold_dimension(S::Sphere)

returns the dimension of the manifold $\mathbb S^n$, i.e. $n$.
"""
@generated manifold_dimension(::Sphere{T}) where {T} = sum(T.parameters)-1

@doc doc"""
    dot(S,x,w,v)

compute the inner product of the two tangent vectors `w,v` from the tangent
plane at `x` on the sphere `S=`$\mathbb S^n$ using the restriction of the metric from the
embedding, i.e. $ (v,w)_x = v^\mathrm{T}w $.
"""
dot(S::Sphere, x, w, v) = dot(w, v)

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

dimension(S::Sphere) = sum(S.shape)
random_point(S::Sphere) = (x = randn(S.shape); x / norm(x))

function random_tangent_vector(S::Sphere, x)
    v = randn(S.shape)
    return project_tangent!(S, v, x, v)
end

zero_tangent_vector(S::Sphere, x) = zero(x)
zero_tangent_vector!(S::Sphere, v, x) = (v .= zero(x))