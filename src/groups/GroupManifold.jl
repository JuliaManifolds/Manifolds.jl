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
@inline function active_traits(f, ::AbstractRNG, M::GroupManifold, args...)
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
    conv = direction_and_side(method)
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
    conv = direction_and_side(method)
    pinvq = inverse_translate(G, p, q, conv)
    Xₑ = log_lie(G, pinvq)
    return translate_diff!(G, X, p, Identity(G), Xₑ, conv)
end

function is_point(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    e::Identity;
    error::Symbol=:none,
    kwargs...,
)
    ie = is_identity(G, e; kwargs...)
    if !ie
        s = "The provided identity is not a point on $G."
        (error === :error) && throw(DomainError(e, s))
        (error === :info) && @info s
        (error === :warn) && @warn s
    end
    return ie
end

function is_vector(
    t::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    e::Identity,
    X,
    cbp::Bool;
    error::Symbol=:none,
    kwargs...,
)
    if cbp
        ie = is_identity(G, e; kwargs...)
        if !ie
            s = "The provided identity is not a point on $G."
            (error === :error) && throw(DomainError(e, s))
            (error === :info) && @info s
            (error === :warn) && @warn s
            return false
        end
    end
    return is_vector(G.manifold, identity_element(G), X, false, te; kwargs...)
end

Base.show(io::IO, G::GroupManifold) = print(io, "GroupManifold($(G.manifold), $(G.op))")
