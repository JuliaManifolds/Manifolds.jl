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
convert(::Type{ArrayMPoint{V}},x::V) where V <: AbstractArray{<:Number} = MatrixPoint{V}(x)

struct MatrixTVector{V <: AbstractArray{<:Number}} <: TVector
    value::V
end
convert(::Type{V},x::MatrixTVector{V}) where V <: AbstractArray{<:Number} = x.value
convert(::Type{MatrixTVector{V}},x::V) where V <: AbstractArray{<:Number} = MatrixPoint{V}(x)

export ArrayManifold,
    ArrayMPoint,
    MatrixTVector