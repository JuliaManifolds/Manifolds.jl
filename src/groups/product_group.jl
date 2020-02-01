"""
    ProductOperation <: AbstractGroupOperation

Direct product group operation.
"""
struct ProductOperation <: AbstractGroupOperation end

const ProductGroup{T} = GroupManifold{ProductManifold{T},ProductOperation}

"""
    ProductGroup{T} <: GroupManifold{ProductManifold{T},ProductOperation}

Decorate a product manifold with a [`ProductOperation`](@ref).

Each submanifold must also be an [`AbstractGroupManifold`](@ref) or a decorated instance of
one. This type is mostly useful for equipping the direct product of group manifolds with an
[`Identity`](@ref) element.

# Constructor
    ProductGroup(manifold::ProductManifold)
"""
function ProductGroup(manifold::ProductManifold)
    if !all(M -> (is_decorator_group(M) === Val(true)), manifold.manifolds)
        error("All submanifolds of product manifold must be or decorate groups.")
    end
    op = ProductOperation()
    return GroupManifold(manifold, op)
end

function show(io::IO, ::MIME"text/plain", G::ProductGroup)
    M = base_manifold(G)
    n = length(M.manifolds)
    print(io, "ProductGroup with $(n) subgroup$(n == 1 ? "" : "s"):")
    _show_product_manifold_no_header(io, M)
end

function show(io::IO, G::ProductGroup)
    M = base_manifold(G)
    print(io, "ProductGroup(", join(M.manifolds, ", "), ")")
end

function submanifold_component(
    e::Identity{GT},
    ::Val{I},
) where {I,MT<:ProductManifold,GT<:GroupManifold{MT}}
    M = base_manifold(e.group)
    return Identity(submanifold(M, I))
end

function submanifold_components(
    e::Identity{GT},
) where {MT<:ProductManifold,GT<:GroupManifold{MT}}
    M = base_manifold(e.group)
    return Identity.(M.manifolds)
end

inv(G::ProductGroup, p) = inv(G.manifold, p)
inv(::GT, e::Identity{GT}) where {GT<:ProductGroup} = e
function inv(M::ProductManifold, x::ProductRepr)
    return ProductRepr(map(inv, M.manifolds, submanifold_components(M, x))...)
end
function inv(M::ProductManifold, p)
    q = allocate_result(M, inv, p)
    return inv!(M, q, p)
end

inv!(G::ProductGroup, q, p) = inv!(G.manifold, q, p)
function inv!(M::ProductManifold, q, p)
    map(inv!, M.manifolds, submanifold_components(M, q), submanifold_components(M, p))
    return q
end

identity(G::ProductGroup, p) = identity(G.manifold, p)
identity(::GT, e::Identity{GT}) where {GT<:ProductGroup} = e
function identity(M::ProductManifold, p::ProductRepr)
    return ProductRepr(map(identity, M.manifolds, submanifold_components(M, p))...)
end
function identity(M::ProductManifold, p)
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
    z = allocate_result(M, compose, p, q)
    return compose!(M, z, p, q)
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
    vout = allocate_result(M, translate_diff, X, p, q)
    return translate_diff!(M, vout, p, q, X, conv)
end

function translate_diff!(G::ProductGroup, Y, p, q, X, conv::ActionDirection)
    return translate_diff!(G.manifold, Y, p, q, X, conv)
end
function translate_diff!(M::ProductManifold, Y, p, q, X, conv::ActionDirection)
    map(
        translate_diff!,
        M.manifolds,
        submanifold_components(Y),
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
    vout = allocate_result(M, inverse_translate_diff, X, p, q)
    return inverse_translate_diff!(M, vout, p, q, X, conv)
end

function inverse_translate_diff!(G::ProductGroup, Y, p, q, X, conv::ActionDirection)
    return inverse_translate_diff!(G.manifold, Y, p, q, X, conv)
end
function inverse_translate_diff!(M::ProductManifold, Y, p, q, X, conv::ActionDirection)
    map(
        inverse_translate_diff!,
        M.manifolds,
        submanifold_components(Y),
        submanifold_components(M, p),
        submanifold_components(M, q),
        submanifold_components(M, X),
        repeated(conv),
    )
    return Y
end
