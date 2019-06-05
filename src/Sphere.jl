struct Sphere{T} <: Manifold where {T}
    shape::T
end

project_tangent!(S::Sphere,w,x,v) = (w .= w - dot(x,v)*x)
distance(S::Sphere,x,y) = norm(x-y) # FIXME
exp!(S::Sphere,y,x,v) = retract!(S,y,x,v)
dimension(S::Sphere) = sum(S.shape)
random_point(S::Sphere) = (x = randn(S.shape); x / norm(x))
random_tangent_vector(S::Sphere, x) = (v = randn(S.shape); project_tangent!(v,x,v))
