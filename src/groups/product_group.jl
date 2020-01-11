"""
    ProductOperation <: AbstractGroupOperation

Direct product group operation.
"""
struct ProductOperation <: AbstractGroupOperation end

const ProductGroup{T} = GroupManifold{ProductManifold{T},ProductOperation}

"""
    ProductGroup{T} <: GroupManifold{ProductManifold{T},ProductOperation}

Decorate a product manifold with a [`ProductOperation`](@ref).

Each submanifold must also be an `[AbstractGroupManifold]` or a decorated instance of one.
This type is mostly useful for equipping the direct product of group manifolds with an
[`Identity`](@ref) element.

# Constructors
    ProductGroup(manifold::ProductManifold)
"""
function ProductGroup(manifold::ProductManifold)
    if !all(is_decorator_group, manifold.manifolds)
        error("All submanifolds of product manifold must be or decorate groups.")
    end
    op = ProductOperation()
    return GroupManifold(manifold, op)
end

show(io::IO, G::ProductGroup) = print(io, "ProductGroup($(G.manifold.manifolds))")

function submanifold_components(
    e::Identity{GT},
) where {MT<:ProductManifold,GT<:GroupManifold{MT}}
    M = base_manifold(e.group)
    return Identity.(M.manifolds)
end

function submanifold_component(
    e::Identity{GT},
    ::Val{I},
) where {I,MT<:ProductManifold,GT<:GroupManifold{MT}}
    M = base_manifold(e.group)
    return Identity(submanifold(M, I))
end

inv(G::ProductGroup, x) = inv(G.manifold, x)
inv(::GT, e::Identity{GT}) where {GT<:ProductGroup} = e
function inv(M::ProductManifold, x::ProductRepr)
    return ProductRepr(map(inv, M.manifolds, submanifold_components(M, x))...)
end
function inv(M::ProductManifold, x)
    y = similar_result(M, inv, x)
    inv!(M, y, x)
    return y
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
    y = similar_result(M, identity, x)
    identity!(M, y, x)
    return y
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
    z = similar_result(M, compose, x, y)
    compose!(M, z, x, y)
    return z
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
    z = similar_result(M, translate, x, y)
    translate!(M, z, x, y, conv)
    return z
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
    z = similar_result(M, inverse_translate, x, y)
    inverse_translate!(M, z, x, y, conv)
    return z
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
        conv,
    )...)
end
function translate_diff(M::ProductManifold, x, y, v, conv::ActionDirection)
    vout = similar_result(M, translate_diff, v, x, y)
    translate_diff!(M, vout, x, y, v, conv)
    return vout
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
    vout = similar_result(M, inverse_translate_diff, v, x, y)
    inverse_translate_diff!(M, vout, x, y, v, conv)
    return vout
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
