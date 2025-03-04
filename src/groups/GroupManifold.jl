#
# a small helper to deprecate functions, default: warn that this function will move to LieGroups.jl
function _lie_groups_depwarn_move(f::Function, comment::String="")
    return Base.depwarn(
        "The function $f will move to LieGroups.jl.$(length(comment)>0 ? "\n" : "")$(comment)",
        Symbol(f),
    )
end
function _lie_groups_depwarn_move(f::Function, newname::Symbol, comment::String="")
    return Base.depwarn(
        "The function $f will move to LieGroups.jl and be renamed to $newname.$(length(comment)>0 ? "\n" : "")$(comment)",
        Symbol(f),
    )
end
function _lie_groups_depwarn_removed(f::Function, comment::String="")
    return Base.depwarn(
        "The function $f will removed from Manifolds.jl. Its functionality is modelled different in LieGroups.jl. Check their transition tutorial for the replacement.$(length(comment)>0 ? "\n" : "")$(comment)",
        Symbol(f),
    )
end
# for types force=true so they show up more often when people “start using” the old groups
function _lie_groups_depwarn_move(f::Type, comment::String="")
    return Base.depwarn(
        "$T will move to LieGroups.jl.$(length(comment)>0 ? "\n" : "")$(comment)",
        Symbol(f);
        force=true,
    )
end
function _lie_groups_depwarn_move(f::Type, newname::Symbol, comment::String="")
    return Base.depwarn(
        "$T will move to LieGroups.jl and be renamed to $newname.$(length(comment)>0 ? "\n" : "")$(comment)",
        Symbol(f);
        force=true,
    )
end

"""
    GroupManifold{𝔽,M<:AbstractManifold{𝔽},O<:AbstractGroupOperation} <: AbstractDecoratorManifold{𝔽}

Concrete decorator for a smooth manifold that equips the manifold with a group operation,
thus making it a Lie group. See [`IsGroupManifold`](@ref) for more details.

Group manifolds by default forward metric-related operations to the wrapped manifold.

# Constructor

    GroupManifold(
        manifold::AbstractManifold,
        op::AbstractGroupOperation,
        vectors::AbstractGroupVectorRepresentation=LeftInvariantRepresentation(),
    )

Define the group operation `op` acting on the manifold `manifold`, hence if `op` acts smoothly,
this forms a Lie group.
"""
struct GroupManifold{
    𝔽,
    M<:AbstractManifold{𝔽},
    O<:AbstractGroupOperation,
    VR<:AbstractGroupVectorRepresentation,
} <: AbstractDecoratorManifold{𝔽}
    manifold::M
    op::O
    vectors::VR
end

function GroupManifold(M::AbstractManifold{𝔽}, op::AbstractGroupOperation) where {𝔽}
    rep = LeftInvariantRepresentation()
    Base.depwarn(
        "GroupManifold is deprecated.\nIt is replaced by `LieGroup(M, op)` from LieGroups.jl.\nAll corresponding functions are deprecated as well and will move accordingly. \nTheir deprecation warnings are, however, not forced to display as this one is.",
        :GroupManifold;
        force=true,
    )
    return GroupManifold{𝔽,typeof(M),typeof(op),typeof(rep)}(M, op, rep)
end

"""
    vector_representation(M::GroupManifold)

Get the [`AbstractGroupVectorRepresentation`](@ref) of [`GroupManifold`](@ref) `M`.
"""
function vector_representation(M::GroupManifold)
    _lie_groups_depwarn_removed(vector_representation)
    return M.vectors
end
@inline function active_traits(f, M::GroupManifold, args...)
    return merge_traits(
        IsGroupManifold(M.op, M.vectors),
        active_traits(f, M.manifold, args...),
        IsExplicitDecorator(),
    )
end
@inline function active_traits(f, ::AbstractRNG, M::GroupManifold, args...)
    return merge_traits(
        IsGroupManifold(M.op, M.vectors),
        active_traits(f, M.manifold, args...),
        IsExplicitDecorator(),
    )
end
# This could maybe even moved to ManifoldsBase?
@inline function active_traits(f, ::AbstractRNG, M::AbstractDecoratorManifold, args...)
    return active_traits(f, M, args...)
end

decorated_manifold(G::GroupManifold) = G.manifold

function (op::AbstractGroupOperation)(
    M::AbstractManifold,
    vectors::AbstractGroupVectorRepresentation,
)
    return GroupManifold(M, op, vectors)
end
function (::Type{T})(
    M::AbstractManifold,
    vectors::AbstractGroupVectorRepresentation,
) where {T<:AbstractGroupOperation}
    return GroupManifold(M, T(), vectors)
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
    return is_vector(G.manifold, identity_element(G), X, false; error=error, kwargs...)
end

@doc raw"""
    rand(::GroupManifold; vector_at=nothing, σ=1.0)
    rand!(::GroupManifold, pX; vector_at=nothing, kwargs...)
    rand(::TraitList{<:IsGroupManifold}, M; vector_at=nothing, σ=1.0)
    rand!(TraitList{<:IsGroupManifold}, M, pX; vector_at=nothing, kwargs...)

Compute a random point or tangent vector on a Lie group.

For points this just means to generate a random point on the
underlying manifold itself.

For tangent vectors, an element in the Lie Algebra is generated.
"""
Random.rand(::GroupManifold; kwargs...)

function Random.rand!(
    T::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    pX;
    kwargs...,
)
    return rand!(T, Random.default_rng(), G, pX; kwargs...)
end

function Random.rand!(
    ::TraitList{<:IsGroupManifold},
    rng::AbstractRNG,
    G::AbstractDecoratorManifold,
    pX;
    vector_at=nothing,
    kwargs...,
)
    M = base_manifold(G)
    if vector_at === nothing
        # points we produce the same as on manifolds
        rand!(rng, M, pX, kwargs...)
    else
        # tangent vectors are represented in the Lie algebra
        # => materialize the identity and produce a tangent vector there
        rand!(rng, M, pX; vector_at=identity_element(G), kwargs...)
    end
end

Base.show(io::IO, G::GroupManifold) = print(io, "GroupManifold($(G.manifold), $(G.op))")

function Statistics.var(M::GroupManifold, x::AbstractVector; kwargs...)
    return var(M.manifold, x; kwargs...)
end
