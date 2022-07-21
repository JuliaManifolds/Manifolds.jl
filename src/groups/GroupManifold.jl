"""
    GroupManifold{𝔽,M<:AbstractManifold{𝔽},O<:AbstractGroupOperation} <: AbstractDecoratorManifold{𝔽}

Decorator for a smooth manifold that equips the manifold with a group operation, thus making
it a Lie group. See [`IsGroupManifold`](@ref) for more details.

Group manifolds by default forward metric-related operations to the wrapped manifold.


# Constructor

    GroupManifold(manifold, op)
"""
struct GroupManifold{𝔽,M<:AbstractManifold{𝔽},O<:AbstractGroupOperation} <:
       AbstractDecoratorManifold{𝔽}
    manifold::M
    op::O
end

@inline function active_traits(f, M::GroupManifold, args...)
    return merge_traits(
        IsGroupManifold(M.op),
        active_traits(f, M.manifold, args...),
        IsExplicitDecorator(),
    )
end

decorated_manifold(G::GroupManifold) = G.manifold

(op::AbstractGroupOperation)(M::AbstractManifold) = GroupManifold(M, op)
function (::Type{T})(M::AbstractManifold) where {T<:AbstractGroupOperation}
    return GroupManifold(M, T())
end

function inverse_retract(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    p,
    q,
    method::GroupLogarithmicInverseRetraction,
)
    conv = direction(method)
    pinvq = inverse_translate(G, p, q, conv)
    Xₑ = log_lie(G, pinvq)
    return translate_diff(G, p, Identity(G), Xₑ, conv)
end

function inverse_retract!(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    X,
    p,
    q,
    method::GroupLogarithmicInverseRetraction,
)
    conv = direction(method)
    pinvq = inverse_translate(G, p, q, conv)
    Xₑ = log_lie(G, pinvq)
    return translate_diff!(G, X, p, Identity(G), Xₑ, conv)
end

function is_point(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    e::Identity,
    te=false;
    kwargs...,
)
    ie = is_identity(G, e; kwargs...)
    (te && !ie) && throw(DomainError(e, "The provided identity is not a point on $G."))
    return ie
end

function is_vector(
    t::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    e::Identity,
    X,
    te=false,
    cbp=true;
    kwargs...,
)
    if cbp
        ie = is_identity(G, e; kwargs...)
        (te && !ie) && throw(DomainError(e, "The provided identity is not a point on $G."))
        (!te && !ie) && return false
    end
    return is_vector(G.manifold, identity_element(G), X, te, false; kwargs...)
end

function Random.rand!(G::GroupManifold, pX; kwargs...)
    rand!(G.manifold, pX; kwargs...)
    return pX
end
function Random.rand!(rng::AbstractRNG, G::GroupManifold, pX; kwargs...)
    rand!(rng, G.manifold, pX; kwargs...)
    return pX
end

Base.show(io::IO, G::GroupManifold) = print(io, "GroupManifold($(G.manifold), $(G.op))")
