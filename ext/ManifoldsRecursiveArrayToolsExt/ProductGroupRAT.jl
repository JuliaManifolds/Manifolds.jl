
function allocate_result(G::SemidirectProductGroup, ::typeof(identity_element))
    M = base_manifold(G)
    N, H = M.manifolds
    np = allocate_result(N, identity_element)
    hp = allocate_result(H, identity_element)
    return ArrayPartition(np, hp)
end

function _common_product_translate_diff(
    G::ProductGroup,
    p,
    q,
    X::ArrayPartition,
    conv::ActionDirectionAndSide,
)
    M = G.manifold
    return ArrayPartition(
        map(
            translate_diff,
            M.manifolds,
            submanifold_components(G, p),
            submanifold_components(G, q),
            submanifold_components(G, X),
            repeated(conv),
        )...,
    )
end

function _compose(M::ProductManifold, p::ArrayPartition, q::ArrayPartition)
    return ArrayPartition(
        map(
            compose,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
        )...,
    )
end

function Base.exp(M::ProductGroup, p::Identity{ProductOperation}, X::ArrayPartition)
    return ArrayPartition(
        map(
            exp,
            M.manifold.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, X),
        )...,
    )
end

function exp_lie(G::ProductGroup, X)
    M = G.manifold
    return ArrayPartition(map(exp_lie, M.manifolds, submanifold_components(G, X))...)
end

Base.@propagate_inbounds function Base.getindex(
    p::ArrayPartition,
    M::ProductGroup,
    i::Union{Integer,Colon,AbstractVector,Val},
)
    return getindex(p, base_manifold(M), i)
end
Base.@propagate_inbounds function Base.getindex(
    p::ArrayPartition,
    M::SemidirectProductGroup,
    i::Union{Integer,Colon,AbstractVector,Val},
)
    return getindex(p, base_manifold(M), i)
end

function identity_element(G::ProductGroup)
    M = G.manifold
    return ArrayPartition(map(identity_element, M.manifolds))
end

function inverse_translate(
    G::ProductGroup,
    p::ArrayPartition,
    q::ArrayPartition,
    conv::ActionDirectionAndSide,
)
    M = G.manifold
    return ArrayPartition(
        map(
            inverse_translate,
            M.manifolds,
            submanifold_components(G, p),
            submanifold_components(G, q),
            repeated(conv),
        )...,
    )
end

function inverse_translate_diff(
    G::ProductGroup,
    p::ArrayPartition,
    q::ArrayPartition,
    X::ArrayPartition,
    conv::ActionDirectionAndSide,
)
    M = G.manifold
    return ArrayPartition(
        map(
            inverse_translate_diff,
            M.manifolds,
            submanifold_components(G, p),
            submanifold_components(G, q),
            submanifold_components(G, X),
            repeated(conv),
        )...,
    )
end

function Base.log(M::ProductGroup, p::Identity{ProductOperation}, q::ArrayPartition)
    return ArrayPartition(
        map(
            log,
            M.manifold.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
        )...,
    )
end

Base.@propagate_inbounds function Base.setindex!(
    q::ArrayPartition,
    p,
    M::ProductGroup,
    i::Union{Integer,Colon,AbstractVector,Val},
)
    return setindex!(q, p, base_manifold(M), i)
end
Base.@propagate_inbounds function Base.setindex!(
    q::ArrayPartition,
    p,
    M::SemidirectProductGroup,
    i::Union{Integer,Colon,AbstractVector,Val},
)
    return setindex!(q, p, base_manifold(M), i)
end

function translate(
    M::ProductGroup,
    p::ArrayPartition,
    q::ArrayPartition,
    conv::ActionDirectionAndSide,
)
    return ArrayPartition(
        map(
            translate,
            M.manifold.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
            repeated(conv),
        )...,
    )
end

# these isapprox methods are here just to reduce time-to-first-isapprox
function isapprox(G::ProductGroup, p::ArrayPartition, q::ArrayPartition; kwargs...)
    return isapprox(G.manifold, p, q; kwargs...)
end
function isapprox(
    G::ProductGroup,
    p::ArrayPartition,
    X::ArrayPartition,
    Y::ArrayPartition;
    kwargs...,
)
    return isapprox(G.manifold, p, X, Y; kwargs...)
end
