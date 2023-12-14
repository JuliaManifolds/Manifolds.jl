function allocate_result(G::SemidirectProductGroup, ::typeof(identity_element))
    M = base_manifold(G)
    N, H = M.manifolds
    np = allocate_result(N, identity_element)
    hp = allocate_result(H, identity_element)
    return ArrayPartition(np, hp)
end

Base.@propagate_inbounds function Base.getindex(
    p::ArrayPartition,
    M::SemidirectProductGroup,
    i::Union{Integer,Colon,AbstractVector,Val},
)
    return getindex(p, base_manifold(M), i)
end

Base.@propagate_inbounds function Base.setindex!(
    q::ArrayPartition,
    p,
    M::SemidirectProductGroup,
    i::Union{Integer,Colon,AbstractVector,Val},
)
    return setindex!(q, p, base_manifold(M), i)
end
