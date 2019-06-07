"""
    MatrixManifold{M <: Manifold} <: Manifold

A manifold to encapsulate manifolds working on array representations of
`MPoints` and `TVectors` in a transparent way, such that for these manifolds its
not necessary to introduce explicit types for the points and tangent vectors,
but they are encapusalted/stripped automatically when needed.
"""
struct MatrixManifold{M <: Manifold} <: Manifold
    manifold::M
end
convert(::Type{M},m::MatrixManifold{M}) where M <: Manifold = m.manifold
convert(::Type{MatrixManifold{M}},m::M) where M <: Manifold = MatrixManifold(M)

struct MatrixMPoint{V <: AbstractArray{<:Number}} <: MPoint
    value::V
end
convert(::Type{V},x::MatrixMPoint{V}) where V <: AbstractArray{<:Number} = x.value
convert(::Type{MatrixMPoint{V}},x::V) where V <: AbstractArray{<:Number} = MatrixPoint{V}(x)

struct MatrixTVector{V <: AbstractArray{<:Number}} <: TVector
    value::V
end
convert(::Type{V},x::MatrixTVector{V}) where V <: AbstractArray{<:Number} = x.value
convert(::Type{MatrixTVector{V}},x::V) where V <: AbstractArray{<:Number} = MatrixPoint{V}(x)

export MatrixManifold,
    MatrixMPoint,
    MatrixTVector