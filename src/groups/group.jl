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
[`AbstractDecoratorManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/decorator.html#ManifoldsBase.AbstractDecoratorManifold) using the [`IsGroupManifold`](@ref) to specify the operation.
For a concrete case the concrete wrapper [`GroupManifold`](@ref) can be used.
"""
abstract type AbstractGroupOperation end

"""
    IsGroupManifold{O<:AbstractGroupOperation} <: AbstractTrait

A trait to declare an [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  as a manifold with group structure
with operation of type `O`.

Using this trait you can turn a manifold that you implement _implictly_ into a Lie group.
If you wish to decorate an existing manifold with one (or different) [`AbstractGroupAction`](@ref)s,
see [`GroupManifold`](@ref).

# Constructor

    IsGroupManifold(op)
"""
struct IsGroupManifold{O<:AbstractGroupOperation} <: AbstractTrait
    op::O
end

"""
    AbstractInvarianceTrait <: AbstractTrait

A common supertype for anz [`AbstractTrait`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/decorator.html#ManifoldsBase.AbstractTrait) related to metric invariance
"""
abstract type AbstractInvarianceTrait <: AbstractTrait end

"""
    HasLeftInvariantMetric <: AbstractInvarianceTrait

Specify that a certain the metric of a [`GroupManifold`](@ref) is a left-invariant metric
"""
struct HasLeftInvariantMetric <: AbstractInvarianceTrait end

direction(::HasLeftInvariantMetric) = LeftForwardAction()
direction(::Type{HasLeftInvariantMetric}) = LeftForwardAction()

"""
    HasRightInvariantMetric <: AbstractInvarianceTrait

Specify that a certain the metric of a [`GroupManifold`](@ref) is a right-invariant metric
"""
struct HasRightInvariantMetric <: AbstractInvarianceTrait end

direction(::HasRightInvariantMetric) = RightBackwardAction()
direction(::Type{HasRightInvariantMetric}) = RightBackwardAction()

"""
    HasBiinvariantMetric <: AbstractInvarianceTrait

Specify that a certain the metric of a [`GroupManifold`](@ref) is a bi-invariant metric
"""
struct HasBiinvariantMetric <: AbstractInvarianceTrait end
function parent_trait(::HasBiinvariantMetric)
    return ManifoldsBase.TraitList(HasLeftInvariantMetric(), HasRightInvariantMetric())
end

"""
    is_group_manifold(G::GroupManifold)
    is_group_manifoldd(G::AbstractManifold, o::AbstractGroupOperation)

returns whether an [`AbstractDecoratorManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/decorator.html#ManifoldsBase.AbstractDecoratorManifold) is a group manifold with
[`AbstractGroupOperation`](@ref) `o`.
For a [`GroupManifold`](@ref) `G` this checks whether the right operations is stored within `G`.
"""
is_group_manifold(::AbstractManifold, ::AbstractGroupOperation) = false

@trait_function is_group_manifold(M::AbstractDecoratorManifold, op::AbstractGroupOperation)
function is_group_manifold(
    ::TraitList{<:IsGroupManifold{<:O}},
    ::AbstractDecoratorManifold,
    ::O,
) where {O<:AbstractGroupOperation}
    return true
end
@trait_function is_group_manifold(M::AbstractDecoratorManifold)
is_group_manifold(::AbstractManifold) = false
function is_group_manifold(
    t::TraitList{<:IsGroupManifold{<:AbstractGroupOperation}},
    M::AbstractDecoratorManifold,
)
    return is_group_manifold(M, t.head.op)
end

base_group(M::MetricManifold) = decorated_manifold(M)
base_group(M::ConnectionManifold) = decorated_manifold(M)
base_group(M::AbstractDecoratorManifold) = M

"""
    ActionDirection

Direction of action on a manifold, either [`LeftForwardAction`](@ref),
[`LeftBackwardAction`](@ref), [`RightForwardAction`](@ref) or [`RightBackwardAction`](@ref).
"""
abstract type ActionDirection end

@doc raw"""
    LeftForwardAction()

Left action of a group on a manifold. For an action ``α: G × X → X`` it is characterized by 
```math
α(g, α(h, x)) = α(gh, x)
```
for all ``g, h ∈ G`` and ``x ∈ X``.
"""
struct LeftForwardAction <: ActionDirection end

@doc raw"""
    LeftBackwardAction()

Left action of a group on a manifold. For an action ``α: X × G → X`` it is characterized by 
```math
α(α(x, h), g) = α(x, gh)
```
for all ``g, h ∈ G`` and ``x ∈ X``.

Note that a left action may still act from the right side in an expression.
"""
struct LeftBackwardAction <: ActionDirection end

const LeftAction = LeftForwardAction

"""
    RightForwardAction()

Right action of a group on a manifold. For an action ``α: G × X → X`` it is characterized by 
```math
α(g, α(h, x)) = α(hg, x)
```
for all ``g, h ∈ G`` and ``x ∈ X``.

Note that a right action may still act from the left side in an expression.
"""
struct RightForwardAction <: ActionDirection end

"""
    RightBackwardAction()

Right action of a group on a manifold. For an action ``α: X × G → X`` it is characterized by 
```math
α(α(x, h), g) = α(x, hg)
```
for all ``g, h ∈ G`` and ``x ∈ X``.

Note that a right action may still act from the left side in an expression.
"""
struct RightBackwardAction <: ActionDirection end

const RightAction = RightBackwardAction

abstract type AbstractDirectionSwitchType end

"""
    struct LeftRightSwitch <: AbstractDirectionSwitchType end

Switch between left and right action, maintaining forward/backward direction.
"""
struct LeftRightSwitch <: AbstractDirectionSwitchType end

"""
    struct ForwardBackwardSwitch <: AbstractDirectionSwitchType end

Switch between forward and backward action, maintaining left/right direction.
"""
struct ForwardBackwardSwitch <: AbstractDirectionSwitchType end

"""
    struct LeftRightSwitch <: AbstractDirectionSwitchType end

Simultaneously switch left/right and forward/backward directions.
"""
struct SimultaneousSwitch <: AbstractDirectionSwitchType end

"""
    switch_direction(::ActionDirection, type::AbstractDirectionSwitchType = SimultaneousSwitch())

Returns type of action between left and right, forward or backward, or both at the same type,
depending on `type`, which is either of `LeftRightSwitch`, `ForwardBackwardSwitch` or
`SimultaneousSwitch`.
"""
switch_direction(::ActionDirection, type::AbstractDirectionSwitchType)
switch_direction(AD::ActionDirection) = switch_direction(AD, SimultaneousSwitch())

switch_direction(::LeftForwardAction, ::LeftRightSwitch) = RightForwardAction()
switch_direction(::LeftBackwardAction, ::LeftRightSwitch) = RightBackwardAction()
switch_direction(::RightForwardAction, ::LeftRightSwitch) = LeftForwardAction()
switch_direction(::RightBackwardAction, ::LeftRightSwitch) = LeftBackwardAction()

switch_direction(::LeftForwardAction, ::ForwardBackwardSwitch) = LeftBackwardAction()
switch_direction(::LeftBackwardAction, ::ForwardBackwardSwitch) = LeftForwardAction()
switch_direction(::RightForwardAction, ::ForwardBackwardSwitch) = RightBackwardAction()
switch_direction(::RightBackwardAction, ::ForwardBackwardSwitch) = RightForwardAction()

switch_direction(::LeftForwardAction, ::SimultaneousSwitch) = RightBackwardAction()
switch_direction(::LeftBackwardAction, ::SimultaneousSwitch) = RightForwardAction()
switch_direction(::RightForwardAction, ::SimultaneousSwitch) = LeftBackwardAction()
switch_direction(::RightBackwardAction, ::SimultaneousSwitch) = LeftForwardAction()

@doc raw"""
    Identity{O<:AbstractGroupOperation}

Represent the group identity element ``e ∈ \mathcal{G}`` on a Lie group ``\mathcal G``
with [`AbstractGroupOperation`](@ref) of type `O`.

Similar to the philosophy that points are agnostic of their group at hand, the identity
does not store the group `g` it belongs to. However it depends on the type of the [`AbstractGroupOperation`](@ref) used.

See also [`identity_element`](@ref) on how to obtain the corresponding [`AbstractManifoldPoint`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifoldPoint) or array representation.

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
    BG = base_group(G)
    q = allocate_result(BG, identity_element)
    return identity_element!(BG, q)
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
    BG = base_group(G)
    q = allocate_result(BG, identity_element, p)
    return identity_element!(BG, q)
end

Base.adjoint(e::Identity) = e

function check_size(
    ::TraitList{<:IsGroupManifold{O}},
    M::AbstractDecoratorManifold,
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return nothing
end
function check_size(::EmptyTrait, M::AbstractDecoratorManifold, e::Identity)
    return DomainError(0, "$M seems to not be a group manifold with $e.")
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
    BG = base_group(G)
    return isapprox(BG, identity_element(BG), q; kwargs...)
end
function is_identity(
    ::TraitList{<:IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    ::Identity{O};
    kwargs...,
) where {O<:AbstractGroupOperation}
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
    ::TraitList{<:IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    p::Identity{O},
    q;
    kwargs...,
) where {O<:AbstractGroupOperation}
    return is_identity(G, q; kwargs...)
end
@inline function isapprox(
    ::TraitList{<:IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    p,
    q::Identity{O};
    kwargs...,
) where {O<:AbstractGroupOperation}
    BG = base_group(G)
    return is_identity(BG, p; kwargs...)
end
function isapprox(
    ::TraitList{<:IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    p::Identity{O},
    q::Identity{O};
    kwargs...,
) where {O<:AbstractGroupOperation}
    return true
end
function isapprox(
    ::TraitList{<:IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    p::Identity{O},
    q::Identity;
    kwargs...,
) where {O<:AbstractGroupOperation}
    return false
end
function isapprox(
    ::TraitList{<:IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    p::Identity,
    q::Identity{O};
    kwargs...,
) where {O<:AbstractGroupOperation}
    return false
end

@inline function isapprox(
    ::TraitList{IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    p::Identity{O},
    X,
    Y;
    kwargs...,
) where {O<:AbstractGroupOperation}
    BG = base_group(G)
    return isapprox(BG, identity_element(BG), X, Y; kwargs...)
end
function isapprox(
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

function is_point(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    e::Identity,
    te::Bool=false;
    kwargs...,
)
    ie = is_identity(G, e; kwargs...)
    (te && !ie) && throw(DomainError(e, "The provided identity is not a point on $G."))
    return ie
end

function is_vector(
    t::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    e::Identity,
    X,
    te::Bool=false,
    cbp=true;
    kwargs...,
)
    if cbp
        # pass te down so this throws an error if te=true
        # if !te and is_point was false -> return false, otherwise continue
        (!te && !is_point(G, e, te; kwargs...)) && return false
    end
    return is_vector(next_trait(t), G, identity_element(G), X, te, false; kwargs...)
end

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
    Xₚ = translate_diff(G, p, Identity(G), Xₑ, LeftForwardAction())
    Y = inverse_translate_diff(G, p, p, Xₚ, RightBackwardAction())
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
    Xₚ = translate_diff(G, p, Identity(G), Xₑ, LeftForwardAction())
    inverse_translate_diff!(G, Y, p, p, Xₚ, RightBackwardAction())
    return Y
end

function ManifoldDiff.differential_exp_argument_lie_approx!(
    M::AbstractManifold,
    Z,
    p,
    X,
    Y;
    n=20,
)
    tmp = copy(M, p, Y)
    a = -1.0
    zero_vector!(M, Z, p)
    for k in 0:n
        a *= -1 // (k + 1)
        Z .+= a .* tmp
        if k < n
            copyto!(tmp, lie_bracket(M, X, tmp))
        end
    end
    q = exp(M, p, X)
    translate_diff!(M, Z, q, Identity(M), Z)
    return Z
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
    BG = base_group(G)
    return inv!(BG, q, p)
end

function Base.inv(
    ::TraitList{IsGroupManifold{O}},
    ::AbstractDecoratorManifold,
    e::Identity{O},
) where {O<:AbstractGroupOperation}
    return e
end

@trait_function inv!(G::AbstractDecoratorManifold, q, p)

function inv!(
    ::TraitList{IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    q,
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    BG = base_group(G)
    return identity_element!(BG, q)
end
function inv!(
    ::TraitList{IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    ::Identity{O},
    e::Identity{O},
) where {O<:AbstractGroupOperation}
    return e
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
    BG = base_group(G)
    return identity_element!(BG, p)
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
    return _compose(base_group(G), p, q)
end
function compose(
    ::AbstractDecoratorManifold,
    ::Identity{O},
    p,
) where {O<:AbstractGroupOperation}
    return p
end
function compose(
    ::AbstractDecoratorManifold,
    p,
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return p
end
function compose(
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

function compose!(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, x, q, p)
    return _compose!(base_group(G), x, q, p)
end
function compose!(
    G::AbstractDecoratorManifold,
    q,
    p,
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return copyto!(G, q, p)
end
function compose!(
    G::AbstractDecoratorManifold,
    q,
    ::Identity{O},
    p,
) where {O<:AbstractGroupOperation}
    return copyto!(G, q, p)
end
function compose!(
    G::AbstractDecoratorManifold,
    q,
    ::Identity{O},
    e::Identity{O},
) where {O<:AbstractGroupOperation}
    return identity_element!(G, q)
end
function compose!(
    ::AbstractDecoratorManifold,
    e::Identity{O},
    ::Identity{O},
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return e
end

Base.transpose(e::Identity) = e

@trait_function hat(M::AbstractDecoratorManifold, e::Identity, X)
@trait_function hat!(M::AbstractDecoratorManifold, Y, e::Identity, X)

@doc raw"""
    hat(M::AbstractDecoratorManifold{𝔽,O}, ::Identity{O}, Xⁱ) where {𝔽,O<:AbstractGroupOperation}

Given a basis $e_i$ on the tangent space at a the [`Identity`](@ref) and tangent
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
    ::TraitList{IsGroupManifold{O}},
    M::AbstractDecoratorManifold,
    ::Identity{O},
    X,
) where {O<:AbstractGroupOperation}
    return get_vector_lie(M, X, VeeOrthogonalBasis())
end
function hat!(
    ::TraitList{IsGroupManifold{O}},
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
function hat!(M::AbstractManifold, c, e::Identity, X)
    return throw(ErrorException("On $M there exsists no identity $e"))
end

@trait_function vee(M::AbstractDecoratorManifold, e::Identity, X)
@trait_function vee!(M::AbstractDecoratorManifold, Y, e::Identity, X)

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
    ::TraitList{IsGroupManifold{O}},
    M::AbstractDecoratorManifold,
    ::Identity{O},
    X,
) where {O<:AbstractGroupOperation}
    return get_coordinates_lie(M, X, VeeOrthogonalBasis())
end
function vee!(
    ::TraitList{IsGroupManifold{O}},
    M::AbstractDecoratorManifold,
    Y,
    ::Identity{O},
    X,
) where {O<:AbstractGroupOperation}
    return get_coordinates_lie!(M, Y, X, VeeOrthogonalBasis())
end
function vee(M::AbstractManifold, e::Identity, X)
    return throw(ErrorException("On $M there exsists no identity $e"))
end
function vee!(M::AbstractManifold, c, e::Identity, X)
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

@trait_function lie_bracket!(M::AbstractDecoratorManifold, Z, X, Y)

_action_order(BG::AbstractDecoratorManifold, p, q, ::LeftForwardAction) = (p, q)
_action_order(BG::AbstractDecoratorManifold, p, q, ::LeftBackwardAction) = (q, inv(BG, p))
_action_order(BG::AbstractDecoratorManifold, p, q, ::RightForwardAction) = (inv(BG, p), q)
_action_order(BG::AbstractDecoratorManifold, p, q, ::RightBackwardAction) = (q, p)

@doc raw"""
    translate(G::AbstractDecoratorManifold, p, q, conv::ActionDirection=LeftForwardAction()])

Translate group element $q$ by $p$ with the translation $τ_p$ with the specified
`conv`ention, either left forward ($L_p$), left backward ($R'_p$), right backward ($R_p$)
or right forward ($L'_p$), defined as
```math
\begin{aligned}
L_p &: q ↦ p \circ q\\
L'_p &: q ↦ p^{-1} \circ q\\
R_p &: q ↦ q \circ p\\
R'_p &: q ↦ q \circ p^{-1}.
\end{aligned}
```
"""
translate(::AbstractDecoratorManifold, ::Any...)
@trait_function translate(
    G::AbstractDecoratorManifold,
    p,
    q,
    conv::ActionDirection=LeftForwardAction(),
)
function translate(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    q,
    conv::ActionDirection,
)
    BG = base_group(G)
    return compose(BG, _action_order(BG, p, q, conv)...)
end

@trait_function translate!(
    G::AbstractDecoratorManifold,
    X,
    p,
    q,
    conv::ActionDirection=LeftForwardAction(),
)
function translate!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    X,
    p,
    q,
    conv::ActionDirection,
)
    BG = base_group(G)
    return compose!(BG, X, _action_order(BG, p, q, conv)...)
end

@doc raw"""
    inverse_translate(G::AbstractDecoratorManifold, p, q, conv::ActionDirection=LeftForwardAction())

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
    conv::ActionDirection=LeftForwardAction(),
)
function inverse_translate(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    q,
    conv::ActionDirection,
)
    BG = base_group(G)
    return translate(BG, inv(BG, p), q, conv)
end

@trait_function inverse_translate!(
    G::AbstractDecoratorManifold,
    X,
    p,
    q,
    conv::ActionDirection=LeftForwardAction(),
)
function inverse_translate!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    X,
    p,
    q,
    conv::ActionDirection,
)
    BG = base_group(G)
    return translate!(BG, X, inv(BG, p), q, conv)
end

@doc raw"""
    translate_diff(G::AbstractDecoratorManifold, p, q, X, conv::ActionDirection=LeftForwardAction())

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
    conv::ActionDirection=LeftForwardAction(),
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
    BG = base_group(G)
    translate_diff!(BG, Y, p, q, X, conv)
    return Y
end
@trait_function translate_diff!(
    G::AbstractDecoratorManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirection=LeftForwardAction(),
)

@doc raw"""
    inverse_translate_diff(G::AbstractDecoratorManifold, p, q, X, conv::ActionDirection=LeftForwardAction())

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
    conv::ActionDirection=LeftForwardAction(),
)
function inverse_translate_diff(
    ::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    q,
    X,
    conv::ActionDirection,
)
    BG = base_group(G)
    return translate_diff(BG, inv(BG, p), q, X, conv)
end

@trait_function inverse_translate_diff!(
    G::AbstractDecoratorManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirection=LeftForwardAction(),
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
    BG = base_group(G)
    return translate_diff!(BG, Y, inv(BG, p), q, X, conv)
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
following properties. Note that one-parameter subgroups are commutative (see [^Suhubi2013],
section 3.5), even if the Lie group itself is not commutative.

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

[^Suhubi2013]:
    > E. Suhubi, Exterior Analysis: Using Applications of Differential Forms, 1st edition.
    > Amsterdam: Academic Press, 2013.
"""
exp_lie(G::AbstractManifold, X)
@trait_function exp_lie(M::AbstractDecoratorManifold, X)
function exp_lie(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, X)
    BG = base_group(G)
    q = allocate_result(BG, exp_lie, X)
    return exp_lie!(BG, q, X)
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
either
* `_log_lie(G, q)` and `_log_lie!(G, X, q)` for the points not being the [`Identity`](@ref)
* the trait version `log_lie(::TraitList{<:IsGroupManifold}, G, e)`, `log_lie(::TraitList{<:IsGroupManifold}, G, X, e)` for own implementations of the identity case.
"""
log_lie(::AbstractDecoratorManifold, q)
@trait_function log_lie(G::AbstractDecoratorManifold, q)
function log_lie(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, q)
    BG = base_group(G)
    return _log_lie(BG, q)
end
function log_lie(
    ::TraitList{<:IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    BG = base_group(G)
    return zero_vector(BG, identity_element(BG))
end
# though identity was taken care of – as usual restart decorator dispatch
function _log_lie(G::AbstractDecoratorManifold, q)
    BG = base_group(G)
    X = allocate_result(BG, log_lie, q)
    return log_lie!(BG, X, q)
end

@trait_function log_lie!(G::AbstractDecoratorManifold, X, q)
function log_lie!(::TraitList{<:IsGroupManifold}, G::AbstractDecoratorManifold, X, q)
    BG = base_group(G)
    return _log_lie!(BG, X, q)
end
function log_lie!(
    ::TraitList{<:IsGroupManifold{O}},
    G::AbstractDecoratorManifold,
    X,
    ::Identity{O},
) where {O<:AbstractGroupOperation}
    return zero_vector!(G, X, identity_element(G))
end

"""
    GroupExponentialRetraction{D<:ActionDirection} <: AbstractRetractionMethod

Retraction using the group exponential [`exp_lie`](@ref) "translated" to any point on the
manifold.

For more details, see
[`retract`](@ref retract(::GroupManifold, p, X, ::GroupExponentialRetraction)).

# Constructor

    GroupExponentialRetraction(conv::ActionDirection = LeftForwardAction())
"""
struct GroupExponentialRetraction{D<:ActionDirection} <: AbstractRetractionMethod end

function GroupExponentialRetraction(conv::ActionDirection=LeftForwardAction())
    return GroupExponentialRetraction{typeof(conv)}()
end

"""
    GroupLogarithmicInverseRetraction{D<:ActionDirection} <: AbstractInverseRetractionMethod

Retraction using the group logarithm [`log_lie`](@ref) "translated" to any point on the
manifold.

For more details, see
[`inverse_retract`](@ref inverse_retract(::GroupManifold, p, q ::GroupLogarithmicInverseRetraction)).

# Constructor

    GroupLogarithmicInverseRetraction(conv::ActionDirection = LeftForwardAction())
"""
struct GroupLogarithmicInverseRetraction{D<:ActionDirection} <:
       AbstractInverseRetractionMethod end

function GroupLogarithmicInverseRetraction(conv::ActionDirection=LeftForwardAction())
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
function retract(
    tl::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    p,
    X,
    t::Number,
    method::GroupExponentialRetraction,
)
    return retract(tl, G, p, t * X, method)
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
function retract!(
    tl::TraitList{<:IsGroupManifold},
    G::AbstractDecoratorManifold,
    q,
    p,
    X,
    t::Number,
    method::GroupExponentialRetraction,
)
    return retract!(tl, G, q, p, t * X, method)
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

@trait_function get_vector_lie(G::AbstractManifold, X, B::AbstractBasis)
@trait_function get_vector_lie!(G::AbstractManifold, Y, X, B::AbstractBasis)

@doc raw"""
    get_vector_lie(G::AbstractDecoratorManifold, a, B::AbstractBasis)

Reconstruct a tangent vector from the Lie algebra of `G` from cooordinates `a` of a basis `B`.
This is similar to calling [`get_vector`](@ref) at the `p=`[`Identity`](@ref)`(G)`.
"""
function get_vector_lie(
    ::TraitList{<:IsGroupManifold},
    G::AbstractManifold,
    X,
    B::AbstractBasis,
)
    return get_vector(base_manifold(G), identity_element(G), X, B)
end
function get_vector_lie!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractManifold,
    Y,
    X,
    B::AbstractBasis,
)
    return get_vector!(base_manifold(G), Y, identity_element(G), X, B)
end

@trait_function get_coordinates_lie(G::AbstractManifold, X, B::AbstractBasis)
@trait_function get_coordinates_lie!(G::AbstractManifold, a, X, B::AbstractBasis)

@doc raw"""
    get_coordinates_lie(G::AbstractManifold, X, B::AbstractBasis)

Get the coordinates of an element `X` from the Lie algebra og `G` with respect to a basis `B`.
This is similar to calling [`get_coordinates`](@ref) at the `p=`[`Identity`](@ref)`(G)`.
"""
function get_coordinates_lie(
    ::TraitList{<:IsGroupManifold},
    G::AbstractManifold,
    X,
    B::AbstractBasis,
)
    return get_coordinates(base_manifold(G), identity_element(G), X, B)
end
function get_coordinates_lie!(
    ::TraitList{<:IsGroupManifold},
    G::AbstractManifold,
    a,
    X,
    B::AbstractBasis,
)
    return get_coordinates!(base_manifold(G), a, identity_element(G), X, B)
end

function ManifoldsBase._pick_basic_allocation_argument(
    ::AbstractManifold,
    f,
    ::Identity,
    x...,
)
    return x[1]
end
