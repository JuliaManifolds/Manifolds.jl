"""
    ArrayManifold{M <: Manifold} <: Manifold

A manifold to encapsulate manifolds working on array representations of
`MPoints` and `TVectors` in a transparent way, such that for these manifolds its
not necessary to introduce explicit types for the points and tangent vectors,
but they are encapusalted/stripped automatically when needed.
"""
struct ArrayManifold{M <: Manifold} <: Manifold
    manifold::M
end
convert(::Type{M},m::ArrayManifold{M}) where M <: Manifold = m.manifold
convert(::Type{ArrayManifold{M}},m::M) where M <: Manifold = ArrayManifold(M)

struct ArrayMPoint{V <: AbstractArray{<:Number}} <: MPoint
    value::V
end
convert(::Type{V},x::ArrayMPoint{V}) where V <: AbstractArray{<:Number} = x.value
convert(::Type{ArrayMPoint{V}},x::V) where V <: AbstractArray{<:Number} = ArrayPoint{V}(x)

struct ArrayTVector{V <: AbstractArray{<:Number}} <: TVector
    value::V
end
convert(::Type{V},v::ArrayTVector{V}) where V <: AbstractArray{<:Number} = v.value
convert(::Type{ArrayTVector{V}},v::V) where V <: AbstractArray{<:Number} = ArrayTVector{V}(v)

isapprox(M::ArrayManifold,x,y;kwargs...) = isapprox(M,ArrayMPoint(x),y; kwargs...)
isapprox(M::ArrayManifold,x::ArrayMPoint,y;kwargs...) = isapprox(M,x,ArrayMPoint(y); kwargs...)
function isapprox(M::ArrayManifold, x::ArrayMPoint, y::ArrayMPoint; kwargs...)
    is_manifold_point(M.manifold,x.value; kwargs...)
    is_manifold_point(M.manifold,y.value; kwargs...)
    return isapprox(M.manifold, x.value, y.value; kwargs...)
end

project_tangent!(M::ArrayManifold,w,x,v; kwargs...) = project_tangent!(M,ArrayTVector(w), x, v; kwargs...)
project_tangent!(M::ArrayManifold,w::ArrayTVector,x,v; kwargs...) = project_tangent!(M, w, ArrayMPoint(x), v; kwargs...)
function project_tangent!(M::ArrayManifold, w::ArrayTVector, x::ArrayMPoint, v; kwargs...)
    is_manifold_point(M.manifold,x.value; kwargs...)
    project_tangent!(M.manifold,w.value,x.value,v.value)
    is_tangent_vector(M.manifold,x.value,w.value; kwargs...)
    return w
end

distance(M::ArrayManifold,x,y; kwargs...) = distance(M, ArrayMPoint(x), y; kwargs...)
distance(M::ArrayManifold,x::ArrayMPoint,y; kwargs...) = distance(M,x,ArrayMPoint(y); kwargs...)
function distance(M::ArrayManifold, x::ArrayMPoint, y::ArrayMPoint; kwargs...)
    is_manifold_point(M.manifold, x.value; kwargs...)
    is_manifold_point(M.manifold, y.value; kwargs...)
    return distance(M.manifold,x.value, y.value)
end

dot(M::ArrayManifold,x,v,w; kwargs...) = dot(M, ArrayMPoint(x), v,w; kwargs...)
dot(M::ArrayManifold,x::ArrayMPoint,v,w; kwargs...) = dot(M, x, ArrayTVector(v), w; kwargs...)
dot(M::ArrayManifold,x::ArrayMPoint,v::ArrayTVector,w; kwargs...) = dot(M, x, v, ArrayTVector(w); kwargs...)
function dot(M::ArrayManifold, x::ArrayMPoint, v::ArrayTVector, w::ArrayTVector; kwargs...)
    is_manifold_point(M.manifold,x.value; kwargs...)
    is_tangent_vector(M.manifold,x.value, v.value; kwargs...)
    is_tangent_vector(M.manifold,x.value, w.value; kwargs...)
    return dot(M.manifold, x.value, v.value, w.value)
end

exp!(M::ArrayManifold, y, x, v; kwargs...) = exp!(M, ArrayMPoint(y), x, v; kwargs...)
exp!(M::ArrayManifold, y::ArrayMPoint, x, v; kwargs...) = exp!(M, y, ArrayMPoint(x), v; kwargs...)
exp!(M::ArrayManifold, y::ArrayMPoint, x::ArrayMPoint, v; kwargs...) = exp!(M, y, x, ArrayTVector(v); kwargs...)
function exp!(M::ArrayManifold, y::ArrayMPoint, x::ArrayMPoint, v::ArrayTVector; kwargs...)
    is_manifold_point(M.manifold,x.value; kwargs...)
    is_tangent_vector(M.manifold,x.value, v.value; kwargs...)
    exp!(M.manifold,y.value, x.value, v.value;)
    is_manifold_point(M.manifold, y.value; kwargs...)
    return y
end

log(M::ArrayManifold,x,y; kwargs...) = log(M, ArrayMPoint(x),y; kwargs...)
log(M::ArrayManifold,x::ArrayMPoint,y; kwargs...) = log(M, x, ArrayMPoint(y); kwargs...)
function log(M::ArrayManifold, x::ArrayMPoint, y::ArrayMPoint; kwargs...)
    is_manifold_point(M.manifold,x.value; kwargs...)
    is_manifold_point(M.manifold,y.value; kwargs...)
    v = ArrayTVector(log(M.manifold, x.value, y.value))
    is_tangent_vector(M.manifold, x.value, v.value; kwargs...)
    return v
end

log!(M::ArrayManifold, v, x, y; kwargs...) = log!(M, ArrayTVector(v), x, y; kwargs...)
log!(M::ArrayManifold, v::ArrayTVector, x, y; kwargs...) = log!(M, v, ArrayMPoint(x), y; kwargs...)
log!(M::ArrayManifold, v::ArrayTVector, x::ArrayMPoint, y; kwargs...) = log!(M, v, x, ArrayMPoint(y); kwargs...)
function log!(M::ArrayManifold, v::ArrayTVector, x::ArrayMPoint, y::ArrayMPoint; kwargs...)
    is_manifold_point(M.manifold,x.value; kwargs...)
    is_manifold_point(M.manifold,y.value; kwargs...)
    log!(M.manifold, v.value, x.value, y.value)
    is_tangent_vector(M.manifold, x.value, v.value; kwargs...)
    return v
end

zero_tangent_vector!(M::ArrayManifold, v, x; kwargs...) = zero_tangent_vector!(M, ArrayTVector(v), x; kwargs...)
zero_tangent_vector!(M::ArrayManifold, v::ArrayTVector, x; kwargs...) = zero_tangent_vector!(M, v, ArrayMPoint(x); kwargs...)
function zero_tangent_vector!(M::ArrayManifold, v::ArrayTVector, x::ArrayMPoint; kwargs...)
    is_manifold_point(M.manifold, x.value; kwargs...)
    zero_tangent_vector!(M.manifold, v.value, x.value; kwargs...)
    is_tangent_vector(M.manifold, x.value, v.value; kwargs...)
    return v
end

zero_tangent_vector(M::ArrayManifold, x) = zero_tangent_vector!(M, ArrayMPoint(x); kwargs...)
function zero_tangent_vector(M::ArrayManifold, x::ArrayMPoint; kwargs...)
    is_manifold_point(M,x; kwargs...)
    w = zero_tangent_vector(M.manifold, x.value)
    is_tangent_vector(M,x,w; kwargs...)
    return w
end

export ArrayManifold,
    ArrayMPoint,
    ArrayTVector