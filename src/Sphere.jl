struct Sphere{T} <: Manifold where {T}
    shape::T
end

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
    θ = acos(dot(x, y))
    if θ ≈ 0.0
        zero_tangent_vector!(S, v, x)
    else
        v .= (θ/sin(θ)) .* (y .- cos(θ).*x)
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

function is_manifold_point(S::Sphere,x)
    if size(x) != S.shape
        throw(DomainError(size(x),"The point $(x) does not lie on $S, since its size is not $(S.shape)."))
    end
    if abs(norm(x)-1) > 10^(-15)
        throw(DomainError(norm(x), "The point $(x) does not lie on the sphere $(S) since its norm is not 1."))
    end
    return true
end

function is_tangent_vector(S::Sphere,x,v)
    is_manifold_point(S,x)
    if size(v) != S.shape
        throw(DomainError(size(v),
            "The vector $(v) can not be tangent to a point on $S since its size does not match $(S.shape)."))
    end
    if dot(x,v) > 10^(-15)
        throw(DomainError(dot(x,v),
            "The vector $(v) can not be a tangent vector to $(x) on $(S), since it is not orthogonal to $x in the embedding"
        ))
    end
end
