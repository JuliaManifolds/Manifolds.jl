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

function decorator_transparent_dispatch(::typeof(group_exp!), M::ProductGroup, q, X)
    return Val(:transparent)
end
function decorator_transparent_dispatch(::typeof(group_log!), M::ProductGroup, X, q)
    return Val(:transparent)
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
    e::Identity{GT},
    ::Val{I},
) where {I,MT<:ProductManifold,𝔽,GT<:GroupManifold{𝔽,MT}}
    return Identity(submanifold(e.group, I), submanifold_component(e.p, I))
end

function submanifold_components(
    e::Identity{GT},
) where {MT<:ProductManifold,𝔽,GT<:GroupManifold{𝔽,MT}}
    M = base_manifold(e.group)
    return map(Identity, M.manifolds, submanifold_components(e.group, e.p))
end

Base.inv(G::ProductGroup, p) = inv(G.manifold, p)
Base.inv(::GT, e::Identity{GT}) where {GT<:ProductGroup} = e
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

Base.identity(G::ProductGroup, p) = identity(G.manifold, p)
Base.identity(::GT, e::Identity{GT}) where {GT<:ProductGroup} = e
function Base.identity(M::ProductManifold, p::ProductRepr)
    return ProductRepr(map(identity, M.manifolds, submanifold_components(M, p))...)
end
function Base.identity(M::ProductManifold, p)
    q = allocate_result(M, identity, p)
    return identity!(M, q, p)
end

identity!(G::ProductGroup, q, p) = identity!(G.manifold, q, p)
function identity!(M::ProductManifold, q, p)
    map(identity!, M.manifolds, submanifold_components(M, q), submanifold_components(M, p))
    return q
end

compose(G::ProductGroup, p, q) = compose(G.manifold, p, q)
compose(G::GT, ::Identity{GT}, p) where {GT<:ProductGroup} = p
compose(G::GT, p, ::Identity{GT}) where {GT<:ProductGroup} = p
compose(G::GT, e::E, ::E) where {GT<:ProductGroup,E<:Identity{GT}} = e
function compose(M::ProductManifold, p::ProductRepr, q::ProductRepr)
    return ProductRepr(map(
        compose,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, q),
    )...)
end
function compose(M::ProductManifold, p, q)
    x = allocate_result(M, compose, p, q)
    return compose!(M, x, p, q)
end

compose!(G::ProductGroup, x, p, q) = compose!(G.manifold, x, p, q)
function compose!(M::ProductManifold, x, p, q)
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
    return ProductRepr(map(
        translate,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, q),
        repeated(conv),
    )...)
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
    return ProductRepr(map(
        inverse_translate,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, q),
        repeated(conv),
    )...)
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
    return ProductRepr(map(
        translate_diff,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, q),
        submanifold_components(M, X),
        repeated(conv),
    )...)
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
    return ProductRepr(map(
        inverse_translate_diff,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, q),
        submanifold_components(M, X),
        repeated(conv),
    )...)
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

function group_exp(M::ProductManifold, X::ProductRepr)
    return ProductRepr(map(group_exp, M.manifolds, submanifold_components(M, X))...)
end
function group_exp(M::ProductManifold, X)
    q = allocate_result(M, group_exp, X)
    return group_exp!(M, q, X)
end

function group_exp!(M::ProductManifold, q, X)
    map(group_exp!, M.manifolds, submanifold_components(M, q), submanifold_components(M, X))
    return q
end

function group_log(M::ProductManifold, q::ProductRepr)
    return ProductRepr(map(group_log, M.manifolds, submanifold_components(M, q))...)
end
function group_log(M::ProductManifold, q)
    X = allocate_result(M, group_log, q)
    return group_log!(M, X, q)
end

function group_log!(M::ProductManifold, X, q)
    map(group_log!, M.manifolds, submanifold_components(M, X), submanifold_components(M, q))
    return X
end
