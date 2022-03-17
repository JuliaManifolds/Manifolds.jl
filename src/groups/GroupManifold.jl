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

function is_group_manifold(
    ::TraitList{<:IsGroupManifold{<:O}},
    ::GroupManifold{𝔽,<:M,<:O},
) where {𝔽,O<:AbstractGroupOperation,M<:AbstractManifold}
    return true
end

function exp(::TraitList{<:IsGroupManifold}, G::GroupManifold, p, X)
    return exp(G.manifold, p, X)
end

function exp!(::TraitList{<:IsGroupManifold}, G::GroupManifold, q, p, X)
    return exp!(G.manifold, q, p, X)
end

function get_basis(::TraitList{<:IsGroupManifold}, G::GroupManifold, p, B::AbstractBasis)
    return get_basis(G.manifold, p, B)
end

function get_coordinates(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    p,
    X,
    B::AbstractBasis,
)
    return get_coordinates(G.manifold, p, X, B)
end

function get_coordinates!(
    ::TraitList{<:IsGroupManifold},
    G::GraphManifold,
    Y,
    p,
    X,
    B::AbstractBasis,
)
    return get_coordinates!(G.manifold, Y, p, X, B)
end

get_embedding(G::GroupManifold) = get_embedding(G.manifold)

function get_vector(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    p,
    c,
    B::AbstractBasis,
)
    return get_vector(G.manifold, p, c, B)
end

function get_vector!(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    Y,
    p,
    c,
    B::AbstractBasis,
)
    return get_vector!(G.manifold, Y, p, c, B)
end

function injectivity_radius(::TraitList{<:IsGroupManifold}, G::GroupManifold)
    return injectivity_radius(G.manifold)
end
function injectivity_radius(::TraitList{<:IsGroupManifold}, G::GroupManifold, p)
    return injectivity_radius(G.manifold, p)
end
function injectivity_radius(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    m::AbstractRetractionMethod,
)
    return injectivity_radius(G.manifold, m)
end
function injectivity_radius(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    p,
    m::AbstractRetractionMethod,
)
    return injectivity_radius(G.manifold, p, m)
end

function inner(::TraitList{<:IsGroupManifold}, G::GroupManifold, p, X, Y)
    return inner(G.manifold, p, X, Y)
end

function inverse_retract(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    p,
    q,
    m::AbstractInverseRetractionMethod,
)
    return inverse_retract(G.manifold, p, q, m)
end
function inverse_retract(::TraitList{<:IsGroupManifold}, G::GroupManifold, p, q)
    return inverse_retract(G.manifold, p, q)
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
    m::AbstractInverseRetractionMethod,
)
    return inverse_retract!(G.manifold, X, p, q, m)
end
function inverse_retract!(::TraitList{<:IsGroupManifold}, G::GroupManifold, X, p, q)
    return inverse_retract!(G.manifold, G, X, p, q)
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

function is_point(::TraitList{<:IsGroupManifold}, G::GroupManifold, p, te=false; kwargs...)
    return is_point(G.manifold, p, te; kwargs...)
end
function is_point(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    e::Identity,
    te=false;
    kwargs...,
)
    ie = is_identity(G, e; kwargs...)
    if te && !ie
        return DomainError(e, "The provided identity is not a point on $G.")
    end
    return ie
end

function is_vector(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    p,
    X,
    te=false,
    cbp=true;
    kwargs...,
)
    return is_vector(G.manifold, p, X, te, cbp; kwargs...)
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
        (!te) && return ie
    end
    return is_vector(G.manifold, identity_element(G), X, te, false; kwargs...)
end

function log(::TraitList{<:IsGroupManifold}, G::GroupManifold, p, q)
    return log(G.manifold, p, q)
end

function log!(::TraitList{<:IsGroupManifold}, G::GroupManifold, X, p, q)
    return log!(G.manifold, X, p, q)
end

manifold_dimension(G::GroupManifold) = manifold_dimension(G.manifold)

function norm(::TraitList{<:IsGroupManifold}, G::GroupManifold, p, X)
    return norm(G.manifold, p, X)
end

function parallel_transport_along(::TraitList{<:IsGroupManifold}, G::GroupManifold, p, X, c)
    return parallel_transport_along(G.manifold, p, X, c)
end

function parallel_transport_along!(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    Y,
    p,
    X,
    c,
)
    return parallel_transport_along!(G.manifold, Y, p, X, c)
end

function parallel_transport_direction(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    p,
    X,
    q,
)
    return parallel_transport_direction(G.manifold, p, X, q)
end

function parallel_transport_direction!(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    Y,
    p,
    X,
    q,
)
    return parallel_transport_direction!(G.manifold, Y, p, X, q)
end

function parallel_transport_to(::TraitList{<:IsGroupManifold}, G::GroupManifold, p, X, q)
    return parallel_transport_to(G.manifold, p, X, q)
end

function parallel_transport_to!(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    Y,
    p,
    X,
    q,
)
    return parallel_transport_to!(G.manifold, Y, p, X, q)
end

function project(::TraitList{<:IsGroupManifold}, G::GroupManifold, p)
    return project(G.manifold, p)
end
function project(::TraitList{<:IsGroupManifold}, G::GroupManifold, p, X)
    return project(G.manifold, p, X)
end

function project!(::TraitList{<:IsGroupManifold}, G::GroupManifold, q, p)
    return project!(G.manifold, q, p)
end
function project!(::TraitList{<:IsGroupManifold}, G::GroupManifold, Y, p, X)
    return project!(G.manifold, Y, p, X)
end

function representation_size(::TraitList{<:IsGroupManifold}, G::GroupManifold)
    return representation_size(G.manifold)
end

function retract(::TraitList{<:IsGroupManifold}, G::GroupManifold, p, X)
    return retract(G.manifold, p, X)
end
function retract(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    p,
    X,
    m::AbstractRetractionMethod,
)
    return retract(G.manifold, p, X, m)
end
#resolve ambiguity
function retract(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    p,
    X,
    method::GroupExponentialRetraction,
)
    conv = direction(method)
    Xₑ = inverse_translate_diff(G, p, p, X, conv)
    pinvq = exp_lie(G, Xₑ)
    q = translate(G, p, pinvq, conv)
    return q
end

function retract!(::TraitList{<:IsGroupManifold}, G::GroupManifold, q, p, X)
    return retract!(G.manifold, q, p, X)
end
function retract!(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    q,
    p,
    X,
    m::AbstractRetractionMethod,
)
    return retract!(G.manifold, q, p, X, m)
end
#resolve ambiguity
function retract!(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    q,
    p,
    X,
    method::GroupExponentialRetraction,
)
    conv = direction(method)
    Xₑ = inverse_translate_diff(G, p, p, X, conv)
    pinvq = exp_lie(G, Xₑ)
    return translate!(G, q, p, pinvq, conv)
end

function vector_transport_along(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    p,
    X,
    c,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_along(G.manifold, p, X, c, m)
end

function vector_transport_along!(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    Y,
    p,
    X,
    c::AbstractVector,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_along!(G.manifold, Y, p, X, c, m)
end

function vector_transport_direction(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    p,
    X,
    d,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_direction(G.manifold, p, X, d, m)
end

function vector_transport_direction!(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    Y,
    p,
    X,
    d,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_direction!(G.manifold, Y, p, X, d, m)
end

function vector_transport_to(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_to(G.manifold, p, X, q, m)
end

function vector_transport_to!(
    ::TraitList{<:IsGroupManifold},
    G::GroupManifold,
    Y,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_to!(G.manifold, Y, p, X, q, m)
end

function zero_vector(::TraitList{<:IsGroupManifold}, G::GroupManifold, p)
    return zero_vector(G.manifold, p)
end

function zero_vector!(::TraitList{<:IsGroupManifold}, G::GroupManifold, X, p)
    return zero_vector!(G.manifold, X, p)
end

Base.show(io::IO, G::GroupManifold) = print(io, "GroupManifold($(G.manifold), $(G.op))")
