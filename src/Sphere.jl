struct Sphere{T} <: Manifold where {T}
    shape::T
end

project_tangent!(S::Sphere, w, x, v) = (w .= v .- dot(x, v).*x)

distance(g::EuclideanMetric{<:Sphere}, x, y) = acos(dot(x, y))

function exp!(g::EuclideanMetric{<:Sphere}, y, x, v)
    nv = norm(g, x, v)
    if nv ≈ 0.0
        y .= x
    else
        y .= cos(nv).*x .+ (sin(nv)/nv).*v
    end
    return y
end

function log!(g::EuclideanMetric{<:Sphere}, v, x, y)
    θ = acos(dot(x, y))
    if θ ≈ 0.0
        zero_tangent_vector!(manifold(g), v, x)
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
