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

function isapprox(M::ArrayManifold, x::ArrayMPoint, y::ArrayMPoint; kwargs...)
    is_manifold_point(M.manifold,x.value)
    is_manifold_point(M.manifold,y.value)
    return isapprox(M.manifold, x.value, y.value; kwargs...)
end

function project_tangent!(M::ArrayManifold, w::ArrayTVector, x::ArrayMPoint, v)
    is_manifold_point(M.manifold,x.value)
    project_tangent!(M.manifold,w.value,x.value,v.value)
    is_tangent_vector(M.manifold,x.value,w.value)
    return w
end

function distance(M::ArrayManifold, x::ArrayMPoint, y::ArrayMPoint)
    is_manifold_point(M.manifold, x.value)
    is_manifold_point(M.manifold, y.value)
    return distance(M.manifold,x.value, y.value)
end

function dot(M::ArrayManifold, x::ArrayMPoint, v::ArrayTVector, w::ArrayTVector)
    is_manifold_point(M.manifold,x.value)
    is_tangent_vector(M.manifold,x.value, v.value)
    is_tangent_vector(M.manifold,x.value, w.value)
    return dot(M.manifold, x.value, v.value, w.value)
end

function exp!(M::ArrayManifold, y::ArrayMPoint, x::ArrayMPoint, v::ArrayTVector)
    is_manifold_point(M.manifold,x.value)
    is_tangent_vector(M.manifold,x.value, v.value)
    exp!(M.manifold,y.value, x.value, v.value)
    is_manifold_point(M.manifold, y.value)
    return y
end

function log!(M::ArrayManifold, v::ArrayTVector, x::ArrayMPoint, y::ArrayMPoint)
    is_manifold_point(M.manifold,x.value)
    is_manifold_point(M.manifold,y.value)
    log!(M.manifold, v.value, x.value, y.value)
    is_tangent_vector(M.manifold, x.value, v.value)
    return v
end

function zero_tangent_vector!(M::ArrayManifold, v::ArrayTVector, x::ArrayMPoint)
    is_manifold_point(M.manifold, x.value)
    zero_tangent_vector!(M.manifold, v.value, x.value)
    is_tangent_vector(M.manifold, x.value, v.value)
    return v
end

function zero_tangent_vector(M::ArrayManifold, x::ArrayMPoint)
    is_manifold_point(M,x)
    return zero_tangent_vector(M.manifold, x.value)    
end

export ArrayManifold,
    ArrayMPoint,
    ArrayTVector