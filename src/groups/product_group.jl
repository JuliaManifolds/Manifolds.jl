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

inv(G::ProductGroup, x) = inv(G.manifold, x)
inv(::GT, e::Identity{GT}) where {GT<:ProductGroup} = e
function inv(M::ProductManifold, x::ProductRepr)
    return ProductRepr(map(inv, M.manifolds, submanifold_components(M, x))...)
end
function inv(M::ProductManifold, x)
    y = allocate_result(M, inv, x)
    return inv!(M, y, x)
end

inv!(G::ProductGroup, y, x) = inv!(G.manifold, y, x)
function inv!(M::ProductManifold, y, x)
    map(inv!, M.manifolds, submanifold_components(M, y), submanifold_components(M, x))
    return y
end

identity(G::ProductGroup, x) = identity(G.manifold, x)
identity(::GT, e::Identity{GT}) where {GT<:ProductGroup} = e
function identity(M::ProductManifold, x::ProductRepr)
    return ProductRepr(map(identity, M.manifolds, submanifold_components(M, x))...)
end
function identity(M::ProductManifold, x)
    y = allocate_result(M, identity, x)
    return identity!(M, y, x)
end

identity!(G::ProductGroup, y, x) = identity!(G.manifold, y, x)
function identity!(M::ProductManifold, y, x)
    map(identity!, M.manifolds, submanifold_components(M, y), submanifold_components(M, x))
    return y
end

compose(G::ProductGroup, x, y) = compose(G.manifold, x, y)
compose(G::GT, ::Identity{GT}, x) where {GT<:ProductGroup} = x
compose(G::GT, x, ::Identity{GT}) where {GT<:ProductGroup} = x
compose(G::GT, e::E, ::E) where {GT<:ProductGroup,E<:Identity{GT}} = e
function compose(M::ProductManifold, x::ProductRepr, y::ProductRepr)
    return ProductRepr(map(
        compose,
        M.manifolds,
        submanifold_components(M, x),
        submanifold_components(M, y),
    )...)
end
function compose(M::ProductManifold, x, y)
    z = allocate_result(M, compose, x, y)
    return compose!(M, z, x, y)
end

compose!(G::ProductGroup, z, x, y) = compose!(G.manifold, z, x, y)
function compose!(M::ProductManifold, z, x, y)
    map(
        compose!,
        M.manifolds,
        submanifold_components(M, z),
        submanifold_components(M, x),
        submanifold_components(M, y),
    )
    return z
end

translate(G::ProductGroup, x, y, conv::ActionDirection) = translate(G.manifold, x, y, conv)
function translate(
    M::ProductManifold,
    x::ProductRepr,
    y::ProductRepr,
    conv::ActionDirection,
)
    return ProductRepr(map(
        translate,
        M.manifolds,
        submanifold_components(M, x),
        submanifold_components(M, y),
        repeated(conv),
    )...)
end
function translate(M::ProductManifold, x, y, conv::ActionDirection)
    z = allocate_result(M, translate, x, y)
    return translate!(M, z, x, y, conv)
end

function translate!(G::ProductGroup, z, x, y, conv::ActionDirection)
    return translate!(G.manifold, z, x, y, conv)
end
function translate!(M::ProductManifold, z, x, y, conv::ActionDirection)
    map(
        translate!,
        M.manifolds,
        submanifold_components(M, z),
        submanifold_components(M, x),
        submanifold_components(M, y),
        repeated(conv),
    )
    return z
end

function inverse_translate(G::ProductGroup, x, y, conv::ActionDirection)
    return inverse_translate(G.manifold, x, y, conv)
end
function inverse_translate(
    M::ProductManifold,
    x::ProductRepr,
    y::ProductRepr,
    conv::ActionDirection,
)
    return ProductRepr(map(
        inverse_translate,
        M.manifolds,
        submanifold_components(M, x),
        submanifold_components(M, y),
        repeated(conv),
    )...)
end
function inverse_translate(M::ProductManifold, x, y, conv::ActionDirection)
    z = allocate_result(M, inverse_translate, x, y)
    return inverse_translate!(M, z, x, y, conv)
end

function inverse_translate!(G::ProductGroup, z, x, y, conv::ActionDirection)
    return inverse_translate!(G.manifold, z, x, y, conv)
end
function inverse_translate!(M::ProductManifold, z, x, y, conv::ActionDirection)
    map(
        inverse_translate!,
        M.manifolds,
        submanifold_components(M, z),
        submanifold_components(M, x),
        submanifold_components(M, y),
        repeated(conv),
    )
    return z
end

function translate_diff(G::ProductGroup, x, y, v, conv::ActionDirection)
    return translate_diff(G.manifold, x, y, v, conv)
end
function translate_diff(
    M::ProductManifold,
    x::ProductRepr,
    y::ProductRepr,
    v::ProductRepr,
    conv::ActionDirection,
)
    return ProductRepr(map(
        translate_diff,
        M.manifolds,
        submanifold_components(M, x),
        submanifold_components(M, y),
        submanifold_components(M, v),
        repeated(conv),
    )...)
end
function translate_diff(M::ProductManifold, x, y, v, conv::ActionDirection)
    vout = allocate_result(M, translate_diff, v, x, y)
    return translate_diff!(M, vout, x, y, v, conv)
end

function translate_diff!(G::ProductGroup, vout, x, y, v, conv::ActionDirection)
    return translate_diff!(G.manifold, vout, x, y, v, conv)
end
function translate_diff!(M::ProductManifold, vout, x, y, v, conv::ActionDirection)
    map(
        translate_diff!,
        M.manifolds,
        submanifold_components(vout),
        submanifold_components(M, x),
        submanifold_components(M, y),
        submanifold_components(M, v),
        repeated(conv),
    )
    return vout
end

function inverse_translate_diff(G::ProductGroup, x, y, v, conv::ActionDirection)
    return inverse_translate_diff(G.manifold, x, y, v, conv)
end
function inverse_translate_diff(
    M::ProductManifold,
    x::ProductRepr,
    y::ProductRepr,
    v::ProductRepr,
    conv::ActionDirection,
)
    return ProductRepr(map(
        inverse_translate_diff,
        M.manifolds,
        submanifold_components(M, x),
        submanifold_components(M, y),
        submanifold_components(M, v),
        repeated(conv),
    )...)
end
function inverse_translate_diff(M::ProductManifold, x, y, v, conv::ActionDirection)
    vout = allocate_result(M, inverse_translate_diff, v, x, y)
    return inverse_translate_diff!(M, vout, x, y, v, conv)
end

function inverse_translate_diff!(G::ProductGroup, vout, x, y, v, conv::ActionDirection)
    return inverse_translate_diff!(G.manifold, vout, x, y, v, conv)
end
function inverse_translate_diff!(M::ProductManifold, vout, x, y, v, conv::ActionDirection)
    map(
        inverse_translate_diff!,
        M.manifolds,
        submanifold_components(vout),
        submanifold_components(M, x),
        submanifold_components(M, y),
        submanifold_components(M, v),
        repeated(conv),
    )
    return vout
end
