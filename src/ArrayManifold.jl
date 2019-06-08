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
convert(::Type{V},x::ArrayTVector{V}) where V <: AbstractArray{<:Number} = x.value
convert(::Type{ArrayTVector{V}},x::V) where V <: AbstractArray{<:Number} = ArrayTVector{V}(x)

function isapprox(M::ArrayManifold, x, y; kwargs...)
    is_manifold_point(M.manifold,x)
    is_manifold_point(M.manifold,y)
    return isapprox(M.manifold, x, y; kwargs...)
end

function project_tangent!(M::ArrayManifold, w, x, v)
    is_manifold_point(M.manifold,x)
    project_tangent!(M.manifold,w,x,v)
    is_tangent_vector(M.manifold,x,w)
    return w
end

function distance(M::ArrayManifold, x, y)
    is_manifold_point(M.manifold,x)
    is_manifold_point(M.manifold,y)
    return distance(M.manifold,x,y)
end

function dot(M::ArrayManifold, x, v, w)
    is_manifold_point(M.manifold,x)
    is_tangent_vector(M.manifold,x,v)
    is_tangent_vector(M.manifold,x,w)
    return dot(M.manifold,x,v,w)
end

function exp!(M::ArrayManifold, y, x, v)
    is_manifold_point(M.manifold,x)
    is_tangent_vector(M.manifold,x,v)
    exp!(M.manifold,y,x,v)
    is_manifold_point(M.manifold,y)
    return y
end

function log!(M::ArrayManifold, v, x, y)
    is_manifold_point(M.manifold,x)
    is_manifold_point(M.manifold,y)
    log!(M.manifold, v,y,x)
    is_tangent_vector(M.manifold,x,v)
    return v
end

zero_tangent_vector!(M::ArrayManifold, v, x) = zero_tangent_vector!(M.manifold, v, x)
zero_tangent_vector(M::ArrayManifold, x) = zero_tangent_vector(M.manifold, x)

export ArrayManifold,
    ArrayMPoint,
    ArrayTVector