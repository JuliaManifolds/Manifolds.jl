"""
    ProductOperation <: AbstractGroupOperation

Direct product group operation.
"""
struct ProductOperation <: AbstractGroupOperation end

const ProductGroup{𝔽,T} = GroupManifold{𝔽,ProductManifold{𝔽,T},ProductOperation}

"""
    ProductGroup{𝔽,T} <: GroupManifold{𝔽,ProductManifold{T},ProductOperation}

Decorate a product manifold with a [`ProductOperation`](@ref).

Each submanifold must also be an [`AbstractGroupManifold`](@ref) or a decorated instance of
one. This type is mostly useful for equipping the direct product of group manifolds with an
[`Identity`](@ref) element.

# Constructor
    ProductGroup(manifold::ProductManifold)
"""
function ProductGroup(manifold::ProductManifold{𝔽}) where {𝔽}
    if !all(is_group_decorator, manifold.manifolds)
        error("All submanifolds of product manifold must be or decorate groups.")
    end
    op = ProductOperation()
    return GroupManifold(manifold, op)
end

function decorator_transparent_dispatch(::typeof(exp_lie!), M::ProductGroup, q, X)
    return Val(:transparent)
end
function decorator_transparent_dispatch(::typeof(log_lie!), M::ProductGroup, X, q)
    return Val(:transparent)
end

function identity_element(G::ProductGroup)
    M = G.manifold
    return ProductRepr(map(identity_element, M.manifolds))
end
function identity_element!(G::ProductGroup, p)
    pes = submanifold_components(G, p)
    M = G.manifold
    map(identity_element!, M.manifolds, pes)
    return p
end

function is_identity(G::ProductGroup, p; kwargs...)
    pes = submanifold_components(G, p)
    M = G.manifold # Inner prodct manifold (of groups)
    return all(map((M, pe) -> is_identity(M, pe; kwargs...), M.manifolds, pes))
end

function Base.show(io::IO, ::MIME"text/plain", G::ProductGroup)
    print(
        io,
        "ProductGroup with $(length(base_manifold(G).manifolds)) subgroup$(length(base_manifold(G).manifolds) == 1 ? "" : "s"):",
    )
    return _show_product_manifold_no_header(io, base_manifold(G))
end

function Base.show(io::IO, G::ProductGroup)
    M = base_manifold(G)
    return print(io, "ProductGroup(", join(M.manifolds, ", "), ")")
end

submanifold(G::ProductGroup, i) = submanifold(base_manifold(G), i)

function submanifold_component(
    G::GroupManifold{𝔽,MT,O},
    ::Identity{O},
    ::Val{I},
) where {I,MT<:ProductManifold,𝔽,O}
    # the identity on a product manifold with is a group consists of a tuple of identities
    return Identity(G.manifolds[I])
end

function submanifold_components(
    G::GroupManifold{𝔽,MT,O},
    ::Identity{O},
) where {MT<:ProductManifold,𝔽,O<:AbstractGroupOperation}
    M = base_manifold(G)
    return map(N -> Identity(N), M.manifolds)
end

inv!(G::ProductGroup, q, ::Identity) = identity_element!(G, q)
function inv!(G::ProductGroup, q, p)
    M = G.manifold
    map(inv!, M.manifolds, submanifold_components(G, q), submanifold_components(G, p))
    return q
end

_compose(G::ProductGroup, p, q) = _compose(G.manifold, p, q)
function _compose(M::ProductManifold, p::ProductRepr, q::ProductRepr)
    return ProductRepr(
        map(
            compose,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
        )...,
    )
end
function _compose(M::ProductManifold, p, q)
    x = allocate_result(M, compose, p, q)
    return _compose!(M, x, p, q)
end

_compose!(G::ProductGroup, x, p, q) = _compose!(G.manifold, x, p, q)
function _compose!(M::ProductManifold, x, p, q)
    map(
        compose!,
        M.manifolds,
        submanifold_components(M, x),
        submanifold_components(M, p),
        submanifold_components(M, q),
    )
    return x
end

translate(G::ProductGroup, p, q, conv::ActionDirection) = translate(G.manifold, p, q, conv)
function translate(
    M::ProductManifold,
    p::ProductRepr,
    q::ProductRepr,
    conv::ActionDirection,
)
    return ProductRepr(
        map(
            translate,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
            repeated(conv),
        )...,
    )
end
function translate(M::ProductManifold, p, q, conv::ActionDirection)
    x = allocate_result(M, translate, p, q)
    return translate!(M, x, p, q, conv)
end

function translate!(G::ProductGroup, x, p, q, conv::ActionDirection)
    return translate!(G.manifold, x, p, q, conv)
end
function translate!(M::ProductManifold, x, p, q, conv::ActionDirection)
    map(
        translate!,
        M.manifolds,
        submanifold_components(M, x),
        submanifold_components(M, p),
        submanifold_components(M, q),
        repeated(conv),
    )
    return x
end

function inverse_translate(G::ProductGroup, p, q, conv::ActionDirection)
    M = G.manifold
    return ProductRepr(
        map(
            inverse_translate,
            M.manifolds,
            submanifold_components(G, p),
            submanifold_components(G, q),
            repeated(conv),
        )...,
    )
end

function inverse_translate!(G::ProductGroup, x, p, q, conv::ActionDirection)
    M = G.manifold
    map(
        inverse_translate!,
        M.manifolds,
        submanifold_components(G, x),
        submanifold_components(G, p),
        submanifold_components(G, q),
        repeated(conv),
    )
    return x
end

function translate_diff(G::ProductGroup, p, q, X, conv::ActionDirection)
    M = G.manifold
    return ProductRepr(
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

function translate_diff!(G::ProductGroup, Y, p, q, X, conv::ActionDirection)
    M = G.manifold
    map(
        translate_diff!,
        M.manifolds,
        submanifold_components(G, Y),
        submanifold_components(G, p),
        submanifold_components(G, q),
        submanifold_components(G, X),
        repeated(conv),
    )
    return Y
end

function inverse_translate_diff(G::ProductGroup, p, q, X, conv::ActionDirection)
    M = G.manifold
    return ProductRepr(
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

function inverse_translate_diff!(G::ProductGroup, Y, p, q, X, conv::ActionDirection)
    M = G.manifold
    map(
        inverse_translate_diff!,
        M.manifolds,
        submanifold_components(G, Y),
        submanifold_components(G, p),
        submanifold_components(G, q),
        submanifold_components(G, X),
        repeated(conv),
    )
    return Y
end

function exp_lie(G::ProductGroup, X::Union{ProductRepr,ProductArray})
    M = G.manifold
    return ProductRepr(map(exp_lie, M.manifolds, submanifold_components(G, X))...)
end
function exp_lie(G::ProductGroup, X)
    q = allocate_result(G, exp_lie, X)
    return exp_lie!(G, q, X)
end

function exp_lie!(G::ProductGroup, q, X)
    M = G.manifold
    map(exp_lie!, M.manifolds, submanifold_components(G, q), submanifold_components(G, X))
    return q
end

function _log_lie(G::ProductGroup, q)
    M = G.manifold
    return ProductRepr(map(_log_lie, M.manifolds, submanifold_components(G, q))...)
end

#overwrite identity case to avoid allocating identity too early.
function log_lie!(G::ProductGroup, X, q::Identity{<:ProductOperation})
    M = G.manifold
    map(log_lie!, M.manifolds, submanifold_components(G, X), submanifold_components(G, q))
    return X
end

function _log_lie!(G::ProductGroup, X, q)
    M = G.manifold
    map(_log_lie!, M.manifolds, submanifold_components(G, X), submanifold_components(G, q))
    return X
end
