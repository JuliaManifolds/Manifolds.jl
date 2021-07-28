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
    return ProductRepr(map(identity_element, G.manifolds))
end
function identity_element!(G::ProductGroup, p)
    pes = submanifold_components(G, p)
    map(identity_element!, G.manifolds, pes)
    return p
end

function is_identity(G::ProductGroup, p; kwargs...)
    pes = submanifold_components(G, p)
    return all(map((M, pe) -> is_identity(M, pe; kwargs...), G.manifolds, pes))
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
function Base.inv(M::ProductManifold, x::ProductRepr)
    return ProductRepr(map(inv, M.manifolds, submanifold_components(M, x))...)
end
function Base.inv(M::ProductManifold, p)
    q = allocate_result(M, inv, p)
    return inv!(M, q, p)
end

inv!(G::ProductGroup, q, p) = inv!(G.manifold, q, p)
function inv!(M::ProductManifold, q, p)
    map(inv!, M.manifolds, submanifold_components(M, q), submanifold_components(M, p))
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
    return compose!(M, x, p, q)
end

_compose!(G::ProductGroup, x, p, q) = compose!(G.manifold, x, p, q)
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
    return inverse_translate(G.manifold, p, q, conv)
end
function inverse_translate(
    M::ProductManifold,
    p::ProductRepr,
    q::ProductRepr,
    conv::ActionDirection,
)
    return ProductRepr(
        map(
            inverse_translate,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
            repeated(conv),
        )...,
    )
end
function inverse_translate(M::ProductManifold, p, q, conv::ActionDirection)
    x = allocate_result(M, inverse_translate, p, q)
    return inverse_translate!(M, x, p, q, conv)
end

function inverse_translate!(G::ProductGroup, x, p, q, conv::ActionDirection)
    return inverse_translate!(G.manifold, x, p, q, conv)
end
function inverse_translate!(M::ProductManifold, x, p, q, conv::ActionDirection)
    map(
        inverse_translate!,
        M.manifolds,
        submanifold_components(M, x),
        submanifold_components(M, p),
        submanifold_components(M, q),
        repeated(conv),
    )
    return x
end

function translate_diff(G::ProductGroup, p, q, X, conv::ActionDirection)
    return translate_diff(G.manifold, p, q, X, conv)
end
function translate_diff(
    M::ProductManifold,
    p::ProductRepr,
    q::ProductRepr,
    X::ProductRepr,
    conv::ActionDirection,
)
    return ProductRepr(
        map(
            translate_diff,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
            submanifold_components(M, X),
            repeated(conv),
        )...,
    )
end
function translate_diff(M::ProductManifold, p, q, X, conv::ActionDirection)
    Y = allocate_result(M, translate_diff, X, p, q)
    return translate_diff!(M, Y, p, q, X, conv)
end

function translate_diff!(G::ProductGroup, Y, p, q, X, conv::ActionDirection)
    return translate_diff!(G.manifold, Y, p, q, X, conv)
end
function translate_diff!(M::ProductManifold, Y, p, q, X, conv::ActionDirection)
    map(
        translate_diff!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, q),
        submanifold_components(M, X),
        repeated(conv),
    )
    return Y
end

function inverse_translate_diff(G::ProductGroup, p, q, X, conv::ActionDirection)
    return inverse_translate_diff(G.manifold, p, q, X, conv)
end
function inverse_translate_diff(
    M::ProductManifold,
    p::ProductRepr,
    q::ProductRepr,
    X::ProductRepr,
    conv::ActionDirection,
)
    return ProductRepr(
        map(
            inverse_translate_diff,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
            submanifold_components(M, X),
            repeated(conv),
        )...,
    )
end
function inverse_translate_diff(M::ProductManifold, p, q, X, conv::ActionDirection)
    Y = allocate_result(M, inverse_translate_diff, X, p, q)
    return inverse_translate_diff!(M, Y, p, q, X, conv)
end

function inverse_translate_diff!(G::ProductGroup, Y, p, q, X, conv::ActionDirection)
    return inverse_translate_diff!(G.manifold, Y, p, q, X, conv)
end
function inverse_translate_diff!(M::ProductManifold, Y, p, q, X, conv::ActionDirection)
    map(
        inverse_translate_diff!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, q),
        submanifold_components(M, X),
        repeated(conv),
    )
    return Y
end

function exp_lie(M::ProductManifold, X::ProductRepr)
    return ProductRepr(map(exp_lie, M.manifolds, submanifold_components(M, X))...)
end
function exp_lie(M::ProductManifold, X)
    q = allocate_result(M, exp_lie, X)
    return exp_lie!(M, q, X)
end

function exp_lie!(M::ProductManifold, q, X)
    map(exp_lie!, M.manifolds, submanifold_components(M, q), submanifold_components(M, X))
    return q
end

function log_lie(M::ProductManifold, q::ProductRepr)
    return ProductRepr(map(log_lie, M.manifolds, submanifold_components(M, q))...)
end
function log_lie(M::ProductManifold, q)
    X = allocate_result(M, log_lie, q)
    return log_lie!(M, X, q)
end

function log_lie!(M::ProductManifold, X, q)
    map(log_lie!, M.manifolds, submanifold_components(M, X), submanifold_components(M, q))
    return X
end
