@doc raw"""
    AbstractGroupOperation

Abstract type for smooth binary operations $∘$ on elements of a Lie group $\mathcal{G}$:
```math
∘ : \mathcal{G} × \mathcal{G} → \mathcal{G}
```
An operation can be either defined for a specific group manifold over
number system `𝔽` or in general, by defining for an operation `Op` the following methods:

    identity_element!(::AbstractDecoratorManifold, q, q)
    inv!(::AbstractDecoratorManifold, q, p)
    _compose!(::AbstractDecoratorManifold, x, p, q)

Note that a manifold is connected with an operation by wrapping it with a decorator,
[`AbstractDecoratorManifold`](@ref) using the [`IsGroupManifold`](@ref) to specify the operation.
For a concrete case the concrete wrapper [`GroupManifold`](@ref) can be used.
"""
abstract type AbstractGroupOperation end

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

"""
    IsGroupManifold{O<:AbstractGroupOperation} <: AbstractTrait

A trait to declare an [`AbstractManifold`](@ref) as a manifold with group structure
with operation of type `O`.

# Constructor

    IsGroupManifold(op)
"""
struct IsGroupManifold{O<:AbstractGroupOperation} <: AbstractTrait
    op::O
end

"""
    AbstractInvarianceTrait <: AbstractTrait

A common supertype for anz [`AbstractTrait`](@ref) related to metric invariance
"""
abstract type AbstractInvarianceTrait <: AbstractTrait end

"""
    HasLeftInvariantMetric <: AbstractInvarianceTrait

Specify that a certain the metric of a [`GroupManifold`](@ref) is a left-invariant metric
"""
struct HasLeftInvariantMetric <: AbstractInvarianceTrait end

"""
    HasRightInvariantMetric <: AbstractInvarianceTrait

Specify that a certain the metric of a [`GroupManifold`](@ref) is a right-invariant metric
"""
struct HasRightInvariantMetric <: AbstractInvarianceTrait end

"""
    HasBiinvariantMetric <: AbstractInvarianceTrait

Specify that a certain the metric of a [`GroupManifold`](@ref) is a bi-invariant metric
"""
struct HasBiinvariantMetric <: AbstractInvarianceTrait end
function parent_trait(::HasBiinvariantMetric)
    return ManifoldsBase.TraitList(HasLeftInvariantMetric(), HasRightInvariantMetric())
end

@inline function active_traits(f, M::GroupManifold, args...)
    return merge_traits(IsGroupManifold(M.op), active_traits(f, M.manifold, args...))
end

Base.show(io::IO, G::GroupManifold) = print(io, "GroupManifold($(G.manifold), $(G.op))")

decorated_manifold(G::GroupManifold) = G.manifold

(op::AbstractGroupOperation)(M::AbstractManifold) = GroupManifold(M, op)
function (::Type{T})(M::AbstractManifold) where {T<:AbstractGroupOperation}
    return GroupManifold(M, T())
end

manifold_dimension(G::GroupManifold) = manifold_dimension(G.manifold)

###################
# Action directions
###################

"""
    ActionDirection

Direction of action on a manifold, either [`LeftAction`](@ref) or [`RightAction`](@ref).
"""
abstract type ActionDirection end

"""
    LeftAction()

Left action of a group on a manifold.
"""
struct LeftAction <: ActionDirection end

"""
    RightAction()

Right action of a group on a manifold.
"""
struct RightAction <: ActionDirection end

"""
    switch_direction(::ActionDirection)

Returns a [`RightAction`](@ref) when given a [`LeftAction`](@ref) and vice versa.
"""
switch_direction(::ActionDirection)
switch_direction(::LeftAction) = RightAction()
switch_direction(::RightAction) = LeftAction()

##################################
# General Identity element methods
##################################

@doc raw"""
    Identity{O<:AbstractGroupOperation}

Represent the group identity element ``e ∈ \mathcal{G}`` on a Lie group ``\mathcal G``
with [`AbstractGroupOperation`](@ref) of type `O`.

Similar to the philosophy that points are agnostic of their group at hand, the identity
does not store the group `g` it belongs to. However it depends on the type of the [`AbstractGroupOperation`](@ref) used.

See also [`identity_element`](@ref) on how to obtain the corresponding [`AbstractManifoldPoint`](@ref) or array representation.

# Constructors

    Identity(G::AbstractDecoratorManifold{𝔽})
    Identity(o::O)
    Identity(::Type{O})

create the identity of the corresponding subtype `O<:`[`AbstractGroupOperation`](@ref)
"""
struct Identity{O<:AbstractGroupOperation} end

@trait_function Identity(M::AbstractDecoratorManifold)
function Identity(
    ::TraitList{<:IsGroupManifold{O}},
    ::AbstractDecoratorManifold,
) where {O<:AbstractGroupOperation}
    return Identity{O}()
end
Identity(::O) where {O<:AbstractGroupOperation} = Identity(O)
Identity(::Type{O}) where {O<:AbstractGroupOperation} = Identity{O}()

# To ensure allocate_result_type works in general if idenitty apears in the tuple
number_eltype(::Identity) = Bool

@doc raw"""
    identity_element(G)

Return a point representation of the [`Identity`](@ref) on the [`IsGroupManifold`](@ref) `G`.
By default this representation is the default array or number representation.
It should return the corresponding default representation of ``e`` as a point on `G` if
points are not represented by arrays.
"""
identity_element(G::AbstractDecoratorManifold)
@trait_function identity_element(G::AbstractDecoratorManifold)
function identity_element(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold)
    q = allocate_result(G, identity_element)
    return identity_element!(G, q)
end

@trait_function identity_element!(G::AbstractDecoratorManifold, p)

function allocate_result(G::AbstractDecoratorManifold, ::typeof(identity_element))
    return zeros(representation_size(G)...)
end

@doc raw"""
    identity_element(G::AbstractDecoratorManifold, p)

Return a point representation of the [`Identity`](@ref) on the [`IsGroupManifold`](@ref) `G`,
where `p` indicates the type to represent the identity.
"""
identity_element(G::AbstractDecoratorManifold, p)
@trait_function identity_element(G::AbstractDecoratorManifold, p)
function identity_element(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p)
    q = allocate_result(G, identity_element, p)
    return identity_element!(G, q)
end

@doc raw"""
    is_identity(G::AbstractDecoratorManifold, q; kwargs)

Check whether `q` is the identity on the [`IsGroupManifold`](@ref) `G`, i.e. it is either
the [`Identity`](@ref)`{O}` with the corresponding [`AbstractGroupOperation`](@ref) `O`, or
(approximately) the correct point representation.
"""
is_identity(G::AbstractDecoratorManifold, q)
@trait_function is_identity(G::AbstractDecoratorManifold, q; kwargs...)
function is_identity(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    q;
    kwargs...,
)
    return isapprox(G, identity_element(G), q; kwargs...)
end
function is_identity(
    ::TraitList{IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    ::Identity{O};
    kwargs...,
) where {𝔽,O<:AbstractGroupOperation}
    return true
end
function is_identity(
    ::TraitList{<:IsGroupManifold},
    ::AbstractDecoratorManifold,
    ::Identity;
    kwargs...,
)
    return false
end

@inline function isapprox(
    ::TraitList{IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    p::Identity{O},
    q;
    kwargs...,
) where {O<:AbstractGroupOperation}
    return is_identity(G, q; kwargs...)
end
@inline function isapprox(
    ::TraitList{IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    p,
    q::Identity{O};
    kwargs...,
) where {O<:AbstractGroupOperation}
    return is_identity(G, p; kwargs...)
end
function isapprox(
    ::TraitList{IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    p::Identity{O},
    q::Identity{O};
    kwargs...,
) where {O<:AbstractGroupOperation}
    return true
end
@inline function isapprox(
    ::TraitList{IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    p::Identity{O},
    X,
    Y;
    kwargs...,
) where {𝔽,O<:AbstractGroupOperation}
    return isapprox(G, identity_element(G), X, Y; kwargs...)
end
function Base.isapprox(
    ::TraitList{<:IsGroupManifold},
    ::AbstractDecoratorManifold,
    ::Identity,
    ::Identity;
    kwargs...,
)
    return false
end

function Base.show(io::IO, ::Identity{O}) where {O<:AbstractGroupOperation}
    return print(io, "Identity($O)")
end

function check_point(
    ::TraitList{IsGroupManifold{O1}},
    G::AbstractDecoratorManifold,
    e::Identity{O2};
    kwargs...,
) where {O1<:AbstractGroupOperation,O2<:AbstractGroupOperation}
    return DomainError(
        e,
        "The Identity $e does not lie on $G, since its the identity with respect to $O2 and not $O1.",
    )
end

##########################
# Metric function forwards
##########################

function exp(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p, X)
    return exp(base_manifold(G), p, X)
end

function exp!(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, q, p, X)
    return exp!(base_manifold(G), q, p, X)
end

get_embedding(G::GroupManifold) = get_embedding(G.manifold)

function get_basis(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    B::AbstractBasis,
)
    return get_basis(base_manifold(G), p, B)
end

function get_coordinates(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    X,
    B::AbstractBasis,
)
    return get_coordinates(base_manifold(G), p, X, B)
end

function get_coordinates!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    Y,
    p,
    X,
    B::AbstractBasis,
)
    return get_coordinates!(base_manifold(G), Y, p, X, B)
end

function get_vector(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    c,
    B::AbstractBasis,
)
    return get_vector(base_manifold(G), p, c, B)
end

function get_vector!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    Y,
    p,
    c,
    B::AbstractBasis,
)
    return get_vector!(base_manifold(G), Y, p, c, B)
end

function injectivity_radius(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold)
    return injectivity_radius(base_manifold(G))
end
function injectivity_radius(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p)
    return injectivity_radius(base_manifold(G), p)
end
function injectivity_radius(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    m::AbstractRetractionMethod,
)
    return injectivity_radius(base_manifold(G), m)
end
function injectivity_radius(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    m::AbstractRetractionMethod,
)
    return injectivity_radius(base_manifold(G), p, m)
end

function inner(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p, X, Y)
    return inner(base_manifold(G), p, X, Y)
end

function inverse_retract(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    q,
    m::AbstractInverseRetractionMethod,
)
    return inverse_retract(base_manifold(G), p, q, m)
end
function inverse_retract(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p, q)
    return inverse_retract(base_manifold(G), p, q)
end

function inverse_retract!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    X,
    p,
    q,
    m::AbstractInverseRetractionMethod,
)
    return inverse_retract!(base_manifold(G), X, p, q, m)
end
function inverse_retract!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    X,
    p,
    q,
)
    return inverse_retract!(base_manifold(G), X, p, q)
end

function is_point(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    te=false;
    kwargs...,
)
    return is_point(base_manifold(G), p, te; kwargs...)
end
function is_point(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    e::Identity,
    te=false;
    kwargs...,
)
    ie = is_identity(G, e; kwargs...)
    (!te) && return ie
    return (!ie) && DomainError(e, "The provided identity is not a point on $G.")
end

function is_vector(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    X,
    te=false,
    cbp=true;
    kwargs...,
)
    return is_vector(base_manifold(G), p, X, te, cbp; kwargs...)
end

function log(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p, q)
    return log(base_manifold(G), p, q)
end

function log!(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, X, p, q)
    return log!(base_manifold(G), X, p, q)
end

function norm(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p, X)
    return norm(base_manifold(G), p, X)
end

function parallel_transport_along(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    X,
    c,
)
    return parallel_transport_along(base_manifold(G), p, X, c)
end

function parallel_transport_along!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    Y,
    p,
    X,
    c,
)
    return parallel_transport_along!(base_manifold(G), Y, p, X, c)
end

function parallel_transport_direction(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    X,
    q,
)
    return parallel_transport_direction(base_manifold(G), p, X, q)
end

function parallel_transport_direction!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    Y,
    p,
    X,
    q,
)
    return parallel_transport_direction!(base_manifold(G), Y, p, X, q)
end

function parallel_transport_to(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    X,
    q,
)
    return parallel_transport_to(base_manifold(G), p, X, q)
end

function parallel_transport_to!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    Y,
    p,
    X,
    q,
)
    return parallel_transport_to!(base_manifold(G), Y, p, X, q)
end

function project(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p)
    return project(base_manifold(G), p)
end
function project(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p, X)
    return project(base_manifold(G), p, X)
end

function project!(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, q, p)
    return project!(base_manifold(G), q, p)
end
function project!(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, Y, p, X)
    return project!(base_manifold(G), Y, p, X)
end

function retract(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p, X)
    return retract(base_manifold(G), p, X)
end
function retract(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    X,
    m::AbstractRetractionMethod,
)
    return retract(base_manifold(G), p, X, m)
end

function retract!(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, q, p, X)
    return retract!(base_manifold(G), q, p, X)
end
function retract!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    q,
    p,
    X,
    m::AbstractRetractionMethod,
)
    return retract!(base_manifold(G), q, p, X, m)
end

function vector_transport_along(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    X,
    c,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_along(base_manifold(G), p, X, c, m)
end

function vector_transport_along!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    Y,
    p,
    X,
    c::AbstractVector,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_along!(base_manifold(G), Y, p, X, c, m)
end

function vector_transport_direction(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    X,
    d,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_direction(base_manifold(G), p, X, d, m)
end

function vector_transport_direction!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    Y,
    p,
    X,
    d,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_direction!(base_manifold(G), Y, p, X, d, m)
end

function vector_transport_to(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_to(base_manifold(G), p, X, q, m)
end

function vector_transport_to!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    Y,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_to!(base_manifold(G), Y, p, X, q, m)
end

function zero_vector(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p)
    return zero_vector(base_manifold(G), p)
end

function zero_vector!(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, X, p)
    return zero_vector!(base_manifold(G), X, p)
end

##########################
# Group-specific functions
##########################

@doc raw"""
    adjoint_action(G::AbstractDecoratorManifold, p, X)

Adjoint action of the element `p` of the Lie group `G` on the element `X`
of the corresponding Lie algebra.

It is defined as the differential of the group authomorphism ``Ψ_p(q) = pqp⁻¹`` at
the identity of `G`.

The formula reads
````math
\operatorname{Ad}_p(X) = dΨ_p(e)[X]
````
where $e$ is the identity element of `G`.

Note that the adjoint representation of a Lie group isn't generally faithful.
Notably the adjoint representation of SO(2) is trivial.
"""
adjoint_action(G::AbstractDecoratorManifold, p, X)
@trait_function adjoint_action(G::AbstractDecoratorManifold, p, Xₑ)
function adjoint_action(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p, Xₑ)
    Xₚ = translate_diff(G, p, Identity(G), Xₑ, LeftAction())
    Y = inverse_translate_diff(G, p, p, Xₚ, RightAction())
    return Y
end

@trait_function adjoint_action!(G::AbstractDecoratorManifold, Y, p, Xₑ)
function adjoint_action!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    Y,
    p,
    Xₑ,
)
    Xₚ = translate_diff(G, p, Identity(G), Xₑ, LeftAction())
    inverse_translate_diff!(G, Y, p, p, Xₚ, RightAction())
    return Y
end

@doc raw"""
    inv(G::AbstractDecoratorManifold, p)

Inverse $p^{-1} ∈ \mathcal{G}$ of an element $p ∈ \mathcal{G}$, such that
$p \circ p^{-1} = p^{-1} \circ p = e ∈ \mathcal{G}$, where $e$ is the [`Identity`](@ref)
element of $\mathcal{G}$.
"""
inv(::AbstractDecoratorManifold, ::Any...)
@trait_function Base.inv(G::AbstractDecoratorManifold, p)
function Base.inv(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p)
    q = allocate_result(G, inv, p)
    return inv!(G, q, p)
end

function Base.inv(
    ::TraitList{IsGroupManifold{O}},
    ::AbstractDecoratorManifold,
    e::Identity{O},
) where {O<:AbstractGroupOperation}
    return e
end

@trait_function inv!(G::AbstractDecoratorManifold, q, p)
function inv!(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, q, p)
    return inv!(G.manifold, q, p)
end

function inv!(
    ::TraitList{IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    q,
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return identity_element!(G, q)
end

function Base.copyto!(
    ::TraitList{IsGroupManifold{O}},
    ::AbstractDecoratorManifold,
    e::Identity{O},
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return e
end
function Base.copyto!(
    ::TraitList{IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    p,
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return identity_element!(G, p)
end

@doc raw"""
    compose(G::AbstractDecoratorManifold, p, q)

Compose elements ``p,q ∈ \mathcal{G}`` using the group operation ``p \circ q``.

For implementing composition on a new group manifold, please overload `_compose`
instead so that methods with [`Identity`](@ref) arguments are not ambiguous.
"""
compose(::AbstractDecoratorManifold, ::Any...)

@trait_function compose(G::AbstractDecoratorManifold, p, q)
function compose(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, p, q)
    return _compose(G, p, q)
end
function compose(
    ::TraitList{<:IsGroupManifold{O}},
    ::AbstractDecoratorManifold,
    ::Identity{O},
    p,
) where {O<:AbstractGroupOperation}
    return p
end
function compose(
    ::TraitList{<:IsGroupManifold{O}},
    ::AbstractDecoratorManifold,
    p,
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return p
end
function compose(
    ::TraitList{<:IsGroupManifold{O}},
    ::AbstractDecoratorManifold,
    e::Identity{O},
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return e
end

function _compose(G::AbstractDecoratorManifold, p, q)
    x = allocate_result(G, compose, p, q)
    return _compose!(G, x, p, q)
end

@trait_function compose!(M::AbstractDecoratorManifold, x, p, q)

function compose!(::TraitList{<:IsGroupManifold}, ::AbstractDecoratorManifold, x, q, p)
    return _compose!(G, x, q, p)
end
function compose!(
    ::TraitList{<:IsGroupManifold{O}},
    ::AbstractDecoratorManifold,
    q,
    p,
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return copyto!(G, q, p)
end
function compose!(
    ::TraitList{<:IsGroupManifold{O}},
    ::AbstractDecoratorManifold,
    q,
    ::Identity{O},
    p,
) where {O<:AbstractGroupOperation}
    return copyto!(G, q, p)
end
function compose!(
    ::TraitList{<:IsGroupManifold{O}},
    ::AbstractDecoratorManifold,
    q,
    ::Identity{O},
    e::Identity{O},
) where {O<:AbstractGroupOperation}
    return identity_element!(G, q)
end
function compose!(
    ::TraitList{<:IsGroupManifold{O}},
    ::AbstractDecoratorManifold,
    e::Identity{O},
    ::Identity{O},
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return e
end

transpose(e::Identity) = e

@doc raw"""
    hat(M::AbstractDecoratorManifold{𝔽,O}, ::Identity{O}, Xⁱ) where {𝔽,O<:AbstractGroupOperation}

Given a basis $e_i$ on the tangent space at a the [`Identity`}(@ref) and tangent
component vector ``X^i``, compute the equivalent vector representation
``X=X^i e_i**, where Einstein summation notation is used:

````math
∧ : X^i ↦ X^i e_i
````

For array manifolds, this converts a vector representation of the tangent
vector to an array representation. The [`vee`](@ref) map is the `hat` map's
inverse.
"""
function hat(
    M::AbstractDecoratorManifold,
    ::Identity{O},
    X,
) where {O<:AbstractGroupOperation}
    return get_vector_lie(M, X, VeeOrthogonalBasis())
end
function hat!(
    M::AbstractDecoratorManifold,
    Y,
    ::Identity{O},
    X,
) where {O<:AbstractGroupOperation}
    return get_vector_lie!(M, Y, X, VeeOrthogonalBasis())
end
function hat(M::AbstractManifold, e::Identity, ::Any)
    return throw(ErrorException("On $M there exsists no identity $e"))
end
function hat!(M::AbstractManifold, ::Any, e::Identity, ::Any)
    return throw(ErrorException("On $M there exsists no identity $e"))
end

@doc raw"""
    vee(M::AbstractManifold, p, X)

Given a basis $e_i$ on the tangent space at a point `p` and tangent
vector `X`, compute the vector components $X^i$, such that $X = X^i e_i$, where
Einstein summation notation is used:

````math
\vee : X^i e_i ↦ X^i
````

For array manifolds, this converts an array representation of the tangent
vector to a vector representation. The [`hat`](@ref) map is the `vee` map's
inverse.
"""
function vee(
    M::AbstractDecoratorManifold,
    ::Identity{O},
    X,
) where {O<:AbstractGroupOperation}
    return get_coordinates_lie(M, X, VeeOrthogonalBasis())
end
function vee!(
    M::AbstractDecoratorManifold,
    Y,
    ::Identity{O},
    X,
) where {O<:AbstractGroupOperation}
    return get_coordinates_lie!(M, Y, X, VeeOrthogonalBasis())
end
function vee(M::AbstractManifold, e::Identity, ::Any)
    return throw(ErrorException("On $M there exsists no identity $e"))
end
function vee!(M::AbstractManifold, ::Any, e::Identity, ::Any)
    return throw(ErrorException("On $M there exsists no identity $e"))
end

"""
    lie_bracket(G::AbstractDecoratorManifold, X, Y)

Lie bracket between elements `X` and `Y` of the Lie algebra corresponding to
the Lie group `G`, cf. [`IsGroupManifold`](@ref).

This can be used to compute the adjoint representation of a Lie algebra.
Note that this representation isn't generally faithful. Notably the adjoint
representation of 𝔰𝔬(2) is trivial.
"""
lie_bracket(G::AbstractDecoratorManifold, X, Y)
@trait_function lie_bracket(M::AbstractDecoratorManifold, X, Y)

_action_order(p, q, ::LeftAction) = (p, q)
_action_order(p, q, ::RightAction) = (q, p)

@doc raw"""
    translate(G::AbstractDecoratorManifold, p, q, conv::ActionDirection=LeftAction()])

Translate group element $q$ by $p$ with the translation $τ_p$ with the specified
`conv`ention, either left ($L_p$) or right ($R_p$), defined as
```math
\begin{aligned}
L_p &: q ↦ p \circ q\\
R_p &: q ↦ q \circ p.
\end{aligned}
```
"""
translate(::AbstractDecoratorManifold, ::Any...)
@trait_function translate(
    G::AbstractDecoratorManifold,
    p,
    q,
    conv::ActionDirection=LeftAction(),
)
function translate(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    q,
    conv::ActionDirection,
)
    return compose(G, _action_order(p, q, conv)...)
end

@trait_function translate!(
    G::AbstractDecoratorManifold,
    X,
    p,
    q,
    conv::ActionDirection=LeftAction(),
)
function translate!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    X,
    p,
    q,
    conv::ActionDirection,
)
    return compose!(G, X, _action_order(p, q, conv)...)
end

@doc raw"""
    inverse_translate(G::AbstractDecoratorManifold, p, q, conv::ActionDirection=LeftAction())

Inverse translate group element $q$ by $p$ with the inverse translation $τ_p^{-1}$ with the
specified `conv`ention, either left ($L_p^{-1}$) or right ($R_p^{-1}$), defined as
```math
\begin{aligned}
L_p^{-1} &: q ↦ p^{-1} \circ q\\
R_p^{-1} &: q ↦ q \circ p^{-1}.
\end{aligned}
```
"""
inverse_translate(::AbstractDecoratorManifold, ::Any...)
@trait_function inverse_translate(
    G::AbstractDecoratorManifold,
    p,
    q,
    conv::ActionDirection=LeftAction(),
)
function inverse_translate(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    q,
    conv::ActionDirection,
)
    return translate(G, inv(G, p), q, conv)
end

@trait_function inverse_translate!(
    G::AbstractDecoratorManifold,
    X,
    p,
    q,
    conv::ActionDirection=LeftAction(),
)
function inverse_translate!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    X,
    p,
    q,
    conv::ActionDirection,
)
    return translate!(G, X, inv(G, p), q, conv)
end

@doc raw"""
    translate_diff(G::AbstractDecoratorManifold, p, q, X, conv::ActionDirection=LeftAction())

For group elements $p, q ∈ \mathcal{G}$ and tangent vector $X ∈ T_q \mathcal{G}$, compute
the action of the differential of the translation $τ_p$ by $p$ on $X$, with the specified
left or right `conv`ention. The differential transports vectors:
```math
(\mathrm{d}τ_p)_q : T_q \mathcal{G} → T_{τ_p q} \mathcal{G}\\
```
"""
translate_diff(::AbstractDecoratorManifold, ::Any...)
@trait_function translate_diff(
    G::AbstractDecoratorManifold,
    p,
    q,
    X,
    conv::ActionDirection=LeftAction(),
)
function translate_diff(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    q,
    X,
    conv::ActionDirection,
)
    Y = allocate_result(G, translate_diff, X, p, q)
    translate_diff!(G, Y, p, q, X, conv)
    return Y
end
@trait_function translate_diff!(
    G::AbstractDecoratorManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirection=LeftAction(),
)

@doc raw"""
    inverse_translate_diff(G::AbstractDecoratorManifold, p, q, X, conv::ActionDirection=LeftAction())

For group elements $p, q ∈ \mathcal{G}$ and tangent vector $X ∈ T_q \mathcal{G}$, compute
the action on $X$ of the differential of the inverse translation $τ_p$ by $p$, with the
specified left or right `conv`ention. The differential transports vectors:
```math
(\mathrm{d}τ_p^{-1})_q : T_q \mathcal{G} → T_{τ_p^{-1} q} \mathcal{G}\\
```
"""
inverse_translate_diff(::AbstractDecoratorManifold, ::Any...)
@trait_function inverse_translate_diff(
    G::AbstractDecoratorManifold,
    p,
    q,
    X,
    conv::ActionDirection=LeftAction(),
)
function inverse_translate_diff(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    q,
    X,
    conv::ActionDirection,
)
    return translate_diff(G, inv(G, p), q, X, conv)
end

@trait_function inverse_translate_diff!(
    G::AbstractDecoratorManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirection=LeftAction(),
)
function inverse_translate_diff!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirection,
)
    return translate_diff!(G, Y, inv(G, p), q, X, conv)
end

function representation_size(::TraitList{<:IsGroupManifold}, M::AbstractDecoratorManifold)
    return representation_size(base_manifold(M))
end

@doc raw"""
    exp_lie(G, X)
    exp_lie!(G, q, X)

Compute the group exponential of the Lie algebra element `X`. It is equivalent to the
exponential map defined by the [`CartanSchoutenMinus`](@ref) connection.

Given an element $X ∈ 𝔤 = T_e \mathcal{G}$, where $e$ is the [`Identity`](@ref) element of
the group $\mathcal{G}$, and $𝔤$ is its Lie algebra, the group exponential is the map

````math
\exp : 𝔤 → \mathcal{G},
````
such that for $t,s ∈ ℝ$, $γ(t) = \exp (t X)$ defines a one-parameter subgroup with the
following properties:

````math
\begin{aligned}
γ(t) &= γ(-t)^{-1}\\
γ(t + s) &= γ(t) \circ γ(s) = γ(s) \circ γ(t)\\
γ(0) &= e\\
\lim_{t → 0} \frac{d}{dt} γ(t) &= X.
\end{aligned}
````

!!! note
    In general, the group exponential map is distinct from the Riemannian exponential map
    [`exp`](@ref).

For example for the [`MultiplicationOperation`](@ref) and either `Number` or `AbstractMatrix`
the Lie exponential is the numeric/matrix exponential.

````math
\exp X = \operatorname{Exp} X = \sum_{n=0}^∞ \frac{1}{n!} X^n.
````

Since this function also depends on the group operation, make sure to implement
the corresponding trait version `exp_lie(::TraitList{<:IsGroupManifold}, G, X)`.
"""
exp_lie(G::AbstractManifold, X)
function exp_lie(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, X)
    q = allocate_result(G, exp_lie, X)
    return exp_lie!(G, q, X)
end

@trait_function exp_lie!(M::AbstractDecoratorManifold, q, X)

@doc raw"""
    log_lie(G, q)
    log_lie!(G, X, q)

Compute the Lie group logarithm of the Lie group element `q`. It is equivalent to the
logarithmic map defined by the [`CartanSchoutenMinus`](@ref) connection.

Given an element $q ∈ \mathcal{G}$, compute the right inverse of the group exponential map
[`exp_lie`](@ref), that is, the element $\log q = X ∈ 𝔤 = T_e \mathcal{G}$, such that
$q = \exp X$

!!! note
    In general, the group logarithm map is distinct from the Riemannian logarithm map
    [`log`](@ref).

    For matrix Lie groups this is equal to the (matrix) logarithm:

````math
\log q = \operatorname{Log} q = \sum_{n=1}^∞ \frac{(-1)^{n+1}}{n} (q - e)^n,
````

where $e$ here is the [`Identity`](@ref) element, that is, $1$ for numeric $q$ or the
identity matrix $I_m$ for matrix $q ∈ ℝ^{m × m}$.

Since this function also depends on the group operation, make sure to implement
the corresponding trait version `log_lie(::TraitList{<:IsGroupManifold}, G, q)`.
"""
log_lie(::AbstractDecoratorManifold, q)
@trait_function log_lie(G::AbstractDecoratorManifold, q)
function log_lie(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, q)
    X = allocate_result(G, log_lie, q)
    return log_lie!(G, X, q)
end
function log_lie(
    ::TraitList{<:IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return zero_vector(G, identity_element(G))
end

@trait_function log_lie!(G::AbstractDecoratorManifold, X, q)
function log_lie!(
    ::TraitList{<:IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    X,
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return zero_vector!(G, X, identity_element(G))
end

############################
# Group-specific Retractions
############################

"""
    GroupExponentialRetraction{D<:ActionDirection} <: AbstractRetractionMethod

Retraction using the group exponential [`exp_lie`](@ref) "translated" to any point on the
manifold.

For more details, see
[`retract`](@ref retract(::GroupManifold, p, X, ::GroupExponentialRetraction)).

# Constructor

    GroupExponentialRetraction(conv::ActionDirection = LeftAction())
"""
struct GroupExponentialRetraction{D<:ActionDirection} <: AbstractRetractionMethod end

function GroupExponentialRetraction(conv::ActionDirection=LeftAction())
    return GroupExponentialRetraction{typeof(conv)}()
end

"""
    GroupLogarithmicInverseRetraction{D<:ActionDirection} <: AbstractInverseRetractionMethod

Retraction using the group logarithm [`log_lie`](@ref) "translated" to any point on the
manifold.

For more details, see
[`inverse_retract`](@ref inverse_retract(::GroupManifold, p, q ::GroupLogarithmicInverseRetraction)).

# Constructor

    GroupLogarithmicInverseRetraction(conv::ActionDirection = LeftAction())
"""
struct GroupLogarithmicInverseRetraction{D<:ActionDirection} <:
       AbstractInverseRetractionMethod end

function GroupLogarithmicInverseRetraction(conv::ActionDirection=LeftAction())
    return GroupLogarithmicInverseRetraction{typeof(conv)}()
end

direction(::GroupExponentialRetraction{D}) where {D} = D()
direction(::GroupLogarithmicInverseRetraction{D}) where {D} = D()

@doc raw"""
    retract(
        G::AbstractDecoratorManifold,
        p,
        X,
        method::GroupExponentialRetraction{<:ActionDirection},
    )

Compute the retraction using the group exponential [`exp_lie`](@ref) "translated" to any
point on the manifold.
With a group translation ([`translate`](@ref)) $τ_p$ in a specified direction, the
retraction is

````math
\operatorname{retr}_p = τ_p \circ \exp \circ (\mathrm{d}τ_p^{-1})_p,
````

where $\exp$ is the group exponential ([`exp_lie`](@ref)), and $(\mathrm{d}τ_p^{-1})_p$ is
the action of the differential of inverse translation $τ_p^{-1}$ evaluated at $p$ (see
[`inverse_translate_diff`](@ref)).
"""
function retract(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
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

function retract!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
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

@doc raw"""
    inverse_retract(
        G::AbstractDecoratorManifold,
        p,
        X,
        method::GroupLogarithmicInverseRetraction{<:ActionDirection},
    )

Compute the inverse retraction using the group logarithm [`log_lie`](@ref) "translated"
to any point on the manifold.
With a group translation ([`translate`](@ref)) $τ_p$ in a specified direction, the
retraction is

````math
\operatorname{retr}_p^{-1} = (\mathrm{d}τ_p)_e \circ \log \circ τ_p^{-1},
````

where $\log$ is the group logarithm ([`log_lie`](@ref)), and $(\mathrm{d}τ_p)_e$ is the
action of the differential of translation $τ_p$ evaluated at the identity element $e$
(see [`translate_diff`](@ref)).
"""
function inverse_retract(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
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
    G::AbstractDecoratorManifold,
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

#################################
# Overloads for AdditionOperation
#################################

"""
    AdditionOperation <: AbstractGroupOperation

Group operation that consists of simple addition.
"""
struct AdditionOperation <: AbstractGroupOperation end

Base.:+(e::Identity{AdditionOperation}) = e
Base.:+(e::Identity{AdditionOperation}, ::Identity{AdditionOperation}) = e
Base.:+(::Identity{AdditionOperation}, p) = p
Base.:+(p, ::Identity{AdditionOperation}) = p

Base.:-(e::Identity{AdditionOperation}) = e
Base.:-(e::Identity{AdditionOperation}, ::Identity{AdditionOperation}) = e
Base.:-(::Identity{AdditionOperation}, p) = -p
Base.:-(p, ::Identity{AdditionOperation}) = p

Base.:*(e::Identity{AdditionOperation}, p) = e
Base.:*(p, e::Identity{AdditionOperation}) = e
Base.:*(e::Identity{AdditionOperation}, ::Identity{AdditionOperation}) = e

const AdditionGroupTrait = TraitList{<:IsGroupManifold{AdditionOperation}}

adjoint_action(::AdditionGroupTrait, G::AbstractDecoratorManifold, p, X) = X

function adjoint_action!(::AdditionGroupTrait, G::AbstractDecoratorManifold, Y, p, X)
    return copyto!(G, Y, p, X)
end

identity_element(::AdditionGroupTrait, G::AbstractDecoratorManifold, p::Number) = zero(p)

function identity_element!(::AdditionGroupTrait, G::AbstractDecoratorManifold, p) where {𝔽}
    return fill!(p, zero(eltype(p)))
end

Base.inv(::AdditionGroupTrait, G::AbstractDecoratorManifold, p) = -p
Base.inv(::AdditionGroupTrait, G::AbstractDecoratorManifold, e::Identity) = e

inv!(::AdditionGroupTrait, G::AbstractDecoratorManifold, q, p) = copyto!(G, q, -p)
function inv!(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    q,
    ::Identity{AdditionOperation},
)
    return identity_element!(G, q)
end
function inv!(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    q::Identity{AdditionOperation},
    e::Identity{AdditionOperation},
)
    return q
end

function is_identity(::AdditionGroupTrait, G::AbstractDecoratorManifold, p, q; kwargs...)
    return isapprox(G, q, zero(q); kwargs...)
end

_compose(::AdditionGroupTrait, G::AbstractDecoratorManifold, p, q) = p + q

function _compose!(::AdditionGroupTrait, G::AbstractDecoratorManifold, x, p, q)
    x .= p .+ q
    return x
end

function translate_diff(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    p,
    q,
    X,
    ::ActionDirection,
)
    return X
end

function translate_diff!(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    Y,
    p,
    q,
    X,
    ::ActionDirection,
)
    return copyto!(G, Y, p, X)
end

function inverse_translate_diff(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    p,
    q,
    X,
    ::ActionDirection,
)
    return X
end

function inverse_translate_diff!(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    Y,
    p,
    q,
    X,
    ::ActionDirection,
)
    return copyto!(G, Y, p, X)
end

exp_lie(::AdditionGroupTrait, G::AbstractDecoratorManifold, X) = X

exp_lie!(::AdditionGroupTrait, G::AbstractDecoratorManifold, q, X) = copyto!(G, q, X)

log_lie(::AdditionGroupTrait, G::AbstractDecoratorManifold, q) = q
function log_lie(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    ::Identity{AdditionOperation},
)
    return zero_vector(G, identity_element(G))
end

_log_lie!(::AdditionGroupTrait, G::AbstractDecoratorManifold, X, q) = copyto!(G, X, q)

lie_bracket(::AdditionGroupTrait, G::AbstractDecoratorManifold, X, Y) = zero(X)

lie_bracket!(::AdditionGroupTrait, G::AbstractDecoratorManifold, Z, X, Y) = fill!(Z, 0)

#######################################
# Overloads for MultiplicationOperation
#######################################

"""
    MultiplicationOperation <: AbstractGroupOperation

Group operation that consists of multiplication.
"""
struct MultiplicationOperation <: AbstractGroupOperation end

const MultiplicationGroupTrait = TraitList{<:IsGroupManifold{MultiplicationOperation}}

Base.:*(e::Identity{MultiplicationOperation}) = e
Base.:*(::Identity{MultiplicationOperation}, p) = p
Base.:*(p, ::Identity{MultiplicationOperation}) = p
Base.:*(e::Identity{MultiplicationOperation}, ::Identity{MultiplicationOperation}) = e
Base.:*(::Identity{MultiplicationOperation}, e::Identity{AdditionOperation}) = e
Base.:*(e::Identity{AdditionOperation}, ::Identity{MultiplicationOperation}) = e

Base.:/(p, ::Identity{MultiplicationOperation}) = p
Base.:/(::Identity{MultiplicationOperation}, p) = inv(p)
Base.:/(e::Identity{MultiplicationOperation}, ::Identity{MultiplicationOperation}) = e

Base.:\(p, ::Identity{MultiplicationOperation}) = inv(p)
Base.:\(::Identity{MultiplicationOperation}, p) = p
Base.:\(e::Identity{MultiplicationOperation}, ::Identity{MultiplicationOperation}) = e

LinearAlgebra.det(::Identity{MultiplicationOperation}) = true
LinearAlgebra.adjoint(e::Identity{MultiplicationOperation}) = e

function identity_element!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    p::AbstractMatrix,
)
    return copyto!(p, I)
end

function identity_element!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    p::AbstractArray,
)
    if length(p) == 1
        fill!(p, one(eltype(p)))
    else
        throw(DimensionMismatch("Array $p cannot be set to identity element of group $G"))
    end
    return p
end

function is_identity(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    q::Number;
    kwargs...,
)
    return isapprox(G, q, one(q); kwargs...)
end
function is_identity(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    q::AbstractVector;
    kwargs...,
)
    return length(q) == 1 && isapprox(G, q[], one(q[]); kwargs...)
end
function is_identity(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    q::AbstractMatrix;
    kwargs...,
)
    return isapprox(G, q, I; kwargs...)
end

LinearAlgebra.mul!(q, ::Identity{MultiplicationOperation}, p) = copyto!(q, p)
LinearAlgebra.mul!(q, p, ::Identity{MultiplicationOperation}) = copyto!(q, p)
function LinearAlgebra.mul!(
    q::AbstractMatrix,
    ::Identity{MultiplicationOperation},
    ::Identity{MultiplicationOperation},
)
    return copyto!(q, I)
end
function LinearAlgebra.mul!(
    q,
    ::Identity{MultiplicationOperation},
    ::Identity{MultiplicationOperation},
)
    return copyto!(q, one(q))
end
function LinearAlgebra.mul!(
    q::Identity{MultiplicationOperation},
    ::Identity{MultiplicationOperation},
    ::Identity{MultiplicationOperation},
)
    return q
end
Base.one(e::Identity{MultiplicationOperation}) = e

Base.inv(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, p) = inv(p)
function Base.inv(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    e::Identity{MultiplicationOperation},
)
    return e
end

inv!(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, q, p) = copyto!(q, inv(G, p))
function inv!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    q,
    ::Identity{MultiplicationOperation},
)
    return identity_element!(G, q)
end
function inv!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    q::Identity{MultiplicationOperation},
    e::Identity{MultiplicationOperation},
)
    return q
end

_compose(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, p, q) = p * q

function _compose!(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, x, p, q)
    return mul!_safe(x, p, q)
end

function inverse_translate(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    p,
    q,
    ::LeftAction,
)
    return p \ q
end
function inverse_translate(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    p,
    q,
    ::RightAction,
)
    return q / p
end

function inverse_translate!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    x,
    p,
    q,
    conv::ActionDirection,
)
    return copyto!(x, inverse_translate(G, p, q, conv))
end

function exp_lie!(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, q, X)
    X isa Union{Number,AbstractMatrix} && return copyto!(q, exp(X))
    return error(
        "exp_lie! not implemented on $(typeof(G)) for vector $(typeof(X)) and element $(typeof(q)).",
    )
end

function log_lie!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    X::AbstractMatrix,
    q::AbstractMatrix,
)
    return log_safe!(X, q)
end

function lie_bracket(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, X, Y)
    return mul!(X * Y, Y, X, -1, true)
end

function lie_bracket!(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, Z, X, Y)
    mul!(Z, X, Y)
    mul!(Z, Y, X, -1, true)
    return Z
end

@doc raw"""
    get_vector_lie(G::AbstractDecoratorManifold, a, B::AbstractBasis)

Reconstruct a tangent vector from the Lie algebra of `G` from cooordinates `a` of a basis `B`.
This is similar to calling [`get_vector`](@ref) at the `p=`[`Identity`](@ref)`(G)`.
"""
function get_vector_lie(G::AbstractManifold, X, B::AbstractBasis)
    return get_vector(G, identity_element(G), X, B)
end
function get_vector_lie!(G::AbstractManifold, Y, X, B::AbstractBasis)
    return get_vector!(G, Y, identity_element(G), X, B)
end

@doc raw"""
    get_coordinates_lie(G::AbstractManifold, X, B::AbstractBasis)

Get the coordinates of an element `X` from the Lie algebra og `G` with respect to a basis `B`.
This is similar to calling [`get_coordinates`](@ref) at the `p=`[`Identity`](@ref)`(G)`.
"""
function get_coordinates_lie(G::AbstractManifold, X, B::AbstractBasis)
    return get_coordinates(G, identity_element(G), X, B)
end
function get_coordinates_lie!(G::AbstractManifold, a, X, B::AbstractBasis)
    return get_coordinates!(G, a, identity_element(G), X, B)
end
