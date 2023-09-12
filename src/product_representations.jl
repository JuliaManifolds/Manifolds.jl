
@doc raw"""
    submanifold_component(M::AbstractManifold, p, i::Integer)
    submanifold_component(M::AbstractManifold, p, ::Val{i}) where {i}
    submanifold_component(p, i::Integer)
    submanifold_component(p, ::Val{i}) where {i}

Project the product array `p` on `M` to its `i`th component. A new array is returned.
"""
submanifold_component(::Any...)
@inline function submanifold_component(M::AbstractManifold, p, i::Integer)
    return submanifold_component(M, p, Val(i))
end
@inline submanifold_component(M::AbstractManifold, p, i::Val) = submanifold_component(p, i)
@inline submanifold_component(p::ArrayPartition, ::Val{I}) where {I} = p.x[I]
@inline submanifold_component(p, i::Integer) = submanifold_component(p, Val(i))

@doc raw"""
    submanifold_components(M::AbstractManifold, p)
    submanifold_components(p)

Get the projected components of `p` on the submanifolds of `M`. The components are returned in a Tuple.
"""
submanifold_components(::Any...)
@inline submanifold_components(::AbstractManifold, p) = submanifold_components(p)
@inline submanifold_components(p::ArrayPartition) = p.x

## ArrayPartition

ManifoldsBase._get_vector_cache_broadcast(::ArrayPartition) = Val(false)

allocate(a::AbstractArray{<:ArrayPartition}) = map(allocate, a)
allocate(x::ArrayPartition) = ArrayPartition(map(allocate, x.x)...)
function allocate(x::ArrayPartition, T::Type)
    return ArrayPartition(map(t -> allocate(t, T), submanifold_components(x))...)
end
