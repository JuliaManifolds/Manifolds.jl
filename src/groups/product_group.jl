"""
    ProductOperation <: AbstractGroupOperation

Direct product group operation.
"""
struct ProductOperation <: AbstractGroupOperation end

const ProductGroup{𝔽,T} = GroupManifold{𝔽,ProductManifold{𝔽,T},ProductOperation}

"""
    ProductGroup{𝔽,T} <: GroupManifold{𝔽,ProductManifold{T},ProductOperation}

Decorate a product manifold with a [`ProductOperation`](@ref).

Each submanifold must also have a [`IsGroupManifold`](@ref) or a decorated instance of
one. This type is mostly useful for equipping the direct product of group manifolds with an
[`Identity`](@ref) element.

# Constructor
    ProductGroup(manifold::ProductManifold)
"""
function ProductGroup(manifold::ProductManifold, vectors::AbstractGroupVectorRepresentation)
    if !all(is_group_manifold, manifold.manifolds)
        error("All submanifolds of product manifold must be or decorate groups.")
    end
    op = ProductOperation()
    return GroupManifold(manifold, op, vectors)
end

@inline function active_traits(f, M::ProductGroup, args...)
    if is_metric_function(f)
        #pass to manifold by default - but keep Group Decorator for the retraction
        return merge_traits(IsGroupManifold(M.op, M.vectors), IsExplicitDecorator())
    else
        return merge_traits(
            IsGroupManifold(M.op, M.vectors),
            active_traits(f, M.manifold, args...),
            IsExplicitDecorator(),
        )
    end
end

function _common_product_adjoint_action!(G, Y, p, X, conv)
    M = G.manifold
    map(
        adjoint_action!,
        M.manifolds,
        submanifold_components(G, Y),
        submanifold_components(G, p),
        submanifold_components(G, X),
        repeated(conv),
    )
    return Y
end

function adjoint_action!(G::ProductGroup, Y, p, X, conv::LeftAction)
    return _common_product_adjoint_action!(G, Y, p, X, conv)
end
function adjoint_action!(G::ProductGroup, Y, p, X, conv::RightAction)
    return _common_product_adjoint_action!(G, Y, p, X, conv)
end

function adjoint_inv_diff!(G::ProductGroup, Y, p, X)
    M = G.manifold
    map(
        adjoint_inv_diff!,
        M.manifolds,
        submanifold_components(G, Y),
        submanifold_components(G, p),
        submanifold_components(G, X),
    )
    return Y
end

function identity_element!(G::ProductGroup, p)
    pes = submanifold_components(G, p)
    M = G.manifold
    map(identity_element!, M.manifolds, pes)
    return p
end

function is_identity(G::ProductGroup, p::Identity{<:ProductOperation}; kwargs...)
    pes = submanifold_components(G, p)
    M = G.manifold # Inner product manifold (of groups)
    return all(map((M, pe) -> is_identity(M, pe; kwargs...), M.manifolds, pes))
end

function Base.show(io::IO, ::MIME"text/plain", G::ProductGroup)
    print(
        io,
        "ProductGroup with $(length(base_manifold(G).manifolds)) subgroup$(length(base_manifold(G).manifolds) == 1 ? "" : "s"):",
    )
    return ManifoldsBase._show_product_manifold_no_header(io, base_manifold(G))
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
    M = G.manifold
    # the identity on a product manifold with is a group consists of a tuple of identities
    return Identity(M.manifolds[I])
end

function submanifold_components(
    G::GroupManifold{𝔽,MT,O},
    ::Identity{O},
) where {MT<:ProductManifold,𝔽,O<:AbstractGroupOperation}
    M = base_manifold(G)
    return map(N -> Identity(N), M.manifolds)
end

function submanifold_components(M::ProductGroup, ::Identity{ProductOperation})
    return map(N -> Identity(N), M.manifold.manifolds)
end

inv!(G::ProductGroup, q, ::Identity{ProductOperation}) = identity_element!(G, q)
function inv!(G::ProductGroup, q, p)
    M = G.manifold
    map(inv!, M.manifolds, submanifold_components(G, q), submanifold_components(G, p))
    return q
end
inv!(::ProductGroup, q::Identity{ProductOperation}, ::Identity{ProductOperation}) = q

function inv_diff!(G::ProductGroup, Y, p, X)
    M = G.manifold
    map(
        inv_diff!,
        M.manifolds,
        submanifold_components(G, Y),
        submanifold_components(G, p),
        submanifold_components(G, X),
    )
    return Y
end

_compose(G::ProductGroup, p, q) = _compose(G.manifold, p, q)

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

function translate!(M::ProductGroup, x, p, q, conv::ActionDirectionAndSide)
    map(
        translate!,
        M.manifold.manifolds,
        submanifold_components(M, x),
        submanifold_components(M, p),
        submanifold_components(M, q),
        repeated(conv),
    )
    return x
end

function inverse_translate!(G::ProductGroup, x, p, q, conv::ActionDirectionAndSide)
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

function _common_product_translate_diff end

function translate_diff(G::ProductGroup, p, q, X, conv::LeftForwardAction)
    return _common_product_translate_diff(G, p, q, X, conv)
end
function translate_diff(G::ProductGroup, p, q, X, conv::RightForwardAction)
    return _common_product_translate_diff(G, p, q, X, conv)
end
function translate_diff(G::ProductGroup, p, q, X, conv::LeftBackwardAction)
    return _common_product_translate_diff(G, p, q, X, conv)
end
function translate_diff(G::ProductGroup, p, q, X, conv::RightBackwardAction)
    return _common_product_translate_diff(G, p, q, X, conv)
end

translate_diff(::ProductGroup, ::Identity, q, X, ::LeftForwardAction) = X
translate_diff(::ProductGroup, ::Identity, q, X, ::RightForwardAction) = X
translate_diff(::ProductGroup, ::Identity, q, X, ::LeftBackwardAction) = X
translate_diff(::ProductGroup, ::Identity, q, X, ::RightBackwardAction) = X

function _common_product_translate_diff!(
    G::ProductGroup,
    Y,
    p,
    q,
    X,
    conv::ActionDirectionAndSide,
)
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

function translate_diff!(G::ProductGroup, Y, p, q, X, conv::LeftForwardAction)
    return _common_product_translate_diff!(G, Y, p, q, X, conv)
end
function translate_diff!(G::ProductGroup, Y, p, q, X, conv::RightForwardAction)
    return _common_product_translate_diff!(G, Y, p, q, X, conv)
end
function translate_diff!(G::ProductGroup, Y, p, q, X, conv::LeftBackwardAction)
    return _common_product_translate_diff!(G, Y, p, q, X, conv)
end
function translate_diff!(G::ProductGroup, Y, p, q, X, conv::RightBackwardAction)
    return _common_product_translate_diff!(G, Y, p, q, X, conv)
end

function inverse_translate_diff!(G::ProductGroup, Y, p, q, X, conv::ActionDirectionAndSide)
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

function exp!(M::ProductGroup, q, p::Identity{ProductOperation}, X)
    map(
        exp!,
        M.manifold.manifolds,
        submanifold_components(M, q),
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    return q
end

function exp_lie!(G::ProductGroup, q, X)
    M = G.manifold
    map(exp_lie!, M.manifolds, submanifold_components(G, q), submanifold_components(G, X))
    return q
end

function log!(M::ProductGroup, X, p::Identity{ProductOperation}, q)
    map(
        log!,
        M.manifold.manifolds,
        submanifold_components(M, X),
        submanifold_components(M, p),
        submanifold_components(M, q),
    )
    return X
end

# on this meta level we first pass down before we resolve identity.
function log_lie!(G::ProductGroup, X, q)
    M = G.manifold
    map(log_lie!, M.manifolds, submanifold_components(G, X), submanifold_components(G, q))
    return X
end

#overwrite identity case to avoid allocating identity too early.
function log_lie!(G::ProductGroup, X, q::Identity{ProductOperation})
    M = G.manifold
    map(log_lie!, M.manifolds, submanifold_components(G, X), submanifold_components(G, q))
    return X
end

# this isapprox method is here just to reduce time-to-first-isapprox
function isapprox(G::ProductGroup, ::Identity{ProductOperation}, X, Y; kwargs...)
    return isapprox(G.manifold, identity_element(G), X, Y; kwargs...)
end
