
@doc raw"""
    FiberBundleProductVectorTransport{
        TMP<:AbstractVectorTransportMethod,
        TMV<:AbstractVectorTransportMethod,
    } <: AbstractVectorTransportMethod

Vector transport type on [`FiberBundle`](@ref).

# Fields

* `method_horizonal` – vector transport method of the horizontal part (related to manifold M)
* `method_vertical` – vector transport method of the vertical part (related to fibers).

The vector transport is derived as a product manifold-style vector transport.
The considered product manifold is the product between the manifold ``\mathcal M``
and the topological vector space isometric to the fiber.

# Constructor

    FiberBundleProductVectorTransport(
        M::AbstractManifold=DefaultManifold();
        vector_transport_method_horizontal::AbstractVectorTransportMethod = default_vector_transport_method(M),
        vector_transport_method_vertical::AbstractVectorTransportMethod = default_vector_transport_method(M),
    )

Construct the `FiberBundleProductVectorTransport` using the [`default_vector_transport_method`](@ref),
which uses [`ParallelTransport`](@extref `ManifoldsBase.ParallelTransport`)
if no manifold is provided.
"""
struct FiberBundleProductVectorTransport{
    TMP<:AbstractVectorTransportMethod,
    TMV<:AbstractVectorTransportMethod,
} <: AbstractVectorTransportMethod
    method_horizontal::TMP
    method_vertical::TMV
end
function FiberBundleProductVectorTransport(
    M::AbstractManifold=ManifoldsBase.DefaultManifold(),
    fiber::FiberType=ManifoldsBase.TangentSpaceType();
    vector_transport_method_horizontal::AbstractVectorTransportMethod=default_vector_transport_method(
        M,
    ),
    vector_transport_method_vertical::AbstractVectorTransportMethod=fiber_bundle_transport(
        M,
        fiber,
    ),
)
    return FiberBundleProductVectorTransport(
        vector_transport_method_horizontal,
        vector_transport_method_vertical,
    )
end

"""
    FiberBundle{𝔽,TVS<:FiberType,TM<:AbstractManifold{𝔽},TVT<:FiberBundleProductVectorTransport} <: AbstractManifold{𝔽}

Fiber bundle on a [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)
`M` of type [`FiberType`](@extref `ManifoldsBase.FiberType`).
Examples include vector bundles, principal bundles or unit tangent bundles, see also [📖 Fiber Bundle](https://en.wikipedia.org/wiki/Fiber_bundle).

# Fields
* `manifold` – the [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)
               manifold the Fiber bundle is defined on,
* `type`     – representing the type of fiber we use.

# Constructor

    FiberBundle(M::AbstractManifold, type::FiberType)
"""
struct FiberBundle{
    𝔽,
    TF<:FiberType,
    TM<:AbstractManifold{𝔽},
    TVT<:FiberBundleProductVectorTransport,
} <: AbstractManifold{𝔽}
    type::TF
    manifold::TM
    vector_transport::TVT
end

function FiberBundle(fiber::FiberType, M::AbstractManifold)
    vtmm = vector_bundle_transport(fiber, M)
    vtbm = FiberBundleProductVectorTransport(vtmm, vtmm)
    return FiberBundle(fiber, M, vtbm)
end

@doc raw"""
    struct FiberBundleInverseProductRetraction <: AbstractInverseRetractionMethod end

Inverse retraction of the point `y` at point `p` from vector bundle `B` over manifold
`B.fiber` (denoted ``\mathcal M``). The inverse retraction is derived as a product manifold-style
approximation to the logarithmic map in the Sasaki metric. The considered product manifold
is the product between the manifold ``\mathcal M`` and the topological vector space isometric
to the fiber.

## Notation
The point ``p = (x_p, V_p)`` where ``x_p ∈ \mathcal M`` and ``V_p`` belongs to
the fiber ``F=π^{-1}(\{x_p\})`` of the vector bundle ``B`` where ``π`` is the canonical
projection of that vector bundle ``B``. Similarly, ``q = (x_q, V_q)``.

The inverse retraction is calculated as

```math
\operatorname{retr}^{-1}_p q = (\operatorname{retr}^{-1}_{x_p}(x_q), V_{\operatorname{retr}^{-1}} - V_p)
```

where ``V_{\operatorname{retr}^{-1}}`` is the result of vector transport of ``V_q`` to the point ``x_p``.
The difference ``V_{\operatorname{retr}^{-1}} - V_p`` corresponds to the logarithmic map in
the vector space ``F``.

See also [`FiberBundleProductRetraction`](@ref).
"""
struct FiberBundleInverseProductRetraction <: AbstractInverseRetractionMethod end

@doc raw"""
    struct FiberBundleProductRetraction <: AbstractRetractionMethod end

Product retraction map of tangent vector ``X`` at point ``p`` from vector bundle `B` over
manifold `B.fiber` (denoted ``\mathcal M``). The retraction is derived as a product manifold-style
approximation to the exponential map in the Sasaki metric. The considered product manifold
is the product between the manifold ``\mathcal M`` and the topological vector space isometric
to the fiber.

## Notation:
* The point ``p = (x_p, V_p)`` where ``x_p ∈ \mathcal M`` and ``V_p`` belongs to the
  fiber ``F=π^{-1}(\{x_p\})`` of the vector bundle ``B`` where ``π`` is the
  canonical projection of that vector bundle ``B``.
* The tangent vector ``X = (V_{X,M}, V_{X,F}) ∈ T_pB`` where
  ``V_{X,M}`` is a tangent vector from the tangent space ``T_{x_p}\mathcal M`` and
  ``V_{X,F}`` is a tangent vector from the tangent space ``T_{V_p}F`` (isomorphic to ``F``).

The retraction is calculated as

```math
\operatorname{retr}_p(X) = (\exp_{x_p}(V_{X,M}), V_{\exp})
````

where ``V_{\exp}`` is the result of vector transport of ``V_p + V_{X,F}``
to the point ``\exp_{x_p}(V_{X,M})``.
The sum ``V_p + V_{X,F}`` corresponds to the exponential map in the vector space ``F``.

See also [`FiberBundleInverseProductRetraction`](@ref).
"""
struct FiberBundleProductRetraction <: AbstractRetractionMethod end

vector_bundle_transport(::FiberType, M::AbstractManifold) = ParallelTransport()

struct FiberBundleBasisData{BBasis<:CachedBasis,TBasis<:CachedBasis}
    base_basis::BBasis
    fiber_basis::TBasis
end

"""
    base_manifold(B::FiberBundle)

Return the manifold the [`FiberBundle`](@ref)s is build on.
"""
base_manifold(B::FiberBundle) = base_manifold(B.manifold)

@doc raw"""
    bundle_transport_to(B::FiberBundle, p, X, q)

Given a fiber bundle ``B=F \mathcal M``, points ``p, q\in\mathcal M``, an element ``X`` of
the fiber over ``p``, transport ``X`` to fiber over ``q``.

Exact meaning of the operation depends on the fiber bundle, or may even be undefined.
Some fiber bundles may declare a default local section around each point crossing `X`,
represented by this function.
"""
function bundle_transport_to(B::FiberBundle, p, X, q)
    Y = allocate(X)
    return bundle_transport_to!(B, Y, p, X, q)
end

@doc raw"""
    bundle_transport_tangent_direction(B::FiberBundle, p, pf, X, d)

Compute parallel transport of vertical vector `X` according to Ehresmann connection on
[`FiberBundle`](@ref) `B`, in direction ``d\in T_p \mathcal M``. ``X`` is an element of the
vertical bundle ``VF\mathcal M`` at `pf` from tangent to fiber ``\pi^{-1}({p})``,
``p\in \mathcal M``.
"""
function bundle_transport_tangent_direction(
    B::FiberBundle,
    p,
    pf,
    X,
    d,
    m::AbstractVectorTransportMethod=default_vector_transport_method(B.manifold),
)
    Y = allocate(X)
    return bundle_transport_tangent_direction!(B, Y, p, pf, X, d, m)
end

@doc raw"""
    bundle_transport_tangent_to(B::FiberBundle, p, pf, X, q)

Compute parallel transport of vertical vector `X` according to Ehresmann connection on
[`FiberBundle`](@ref) `B`, to point ``q\in \mathcal M``. ``X`` is an element of the vertical
bundle ``VF\mathcal M`` at `pf` from tangent to fiber ``\pi^{-1}({p})``,
``p\in \mathcal M``.
"""
function bundle_transport_tangent_to(
    B::FiberBundle,
    p,
    pf,
    X,
    q,
    m::AbstractVectorTransportMethod=default_vector_transport_method(B.manifold),
)
    Y = allocate(X)
    return bundle_transport_tangent_to!(B, Y, p, pf, X, q, m)
end

"""
    bundle_projection(B::FiberBundle, p)

Projection of point `p` from the bundle `M` to the base manifold.
Returns the point on the base manifold `B.manifold` at which the vector part
of `p` is attached.
"""
bundle_projection(B::FiberBundle, p) = submanifold_component(B.manifold, p, Val(1))

function get_basis(M::FiberBundle, p, B::AbstractBasis)
    xp1, xp2 = submanifold_components(M, p)
    base_basis = get_basis(M.manifold, xp1, B)
    F = Fiber(M.manifold, xp1, M.type)
    fiber_basis = get_basis(F, xp2, B)
    return CachedBasis(B, FiberBundleBasisData(base_basis, fiber_basis))
end
function get_basis(M::FiberBundle, p, B::CachedBasis)
    return invoke(get_basis, Tuple{AbstractManifold,Any,CachedBasis}, M, p, B)
end

function get_coordinates(M::FiberBundle, p, X, B::AbstractBasis)
    px, Vx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    F = Fiber(M.manifold, px, M.type)
    return vcat(get_coordinates(M.manifold, px, VXM, B), get_coordinates(F, Vx, VXF, B))
end

function get_coordinates!(M::FiberBundle, Y, p, X, B::AbstractBasis)
    px, Vx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    n = manifold_dimension(M.manifold)
    get_coordinates!(M.manifold, view(Y, 1:n), px, VXM, B)
    F = Fiber(M.manifold, px, M.type)
    get_coordinates!(F, view(Y, (n + 1):length(Y)), Vx, VXF, B)
    return Y
end

function get_coordinates(
    M::FiberBundle,
    p,
    X,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:FiberBundleBasisData},
) where {𝔽}
    px, Vx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    F = Fiber(M.manifold, px, M.type)
    return vcat(
        get_coordinates(M.manifold, px, VXM, B.data.base_basis),
        get_coordinates(F, Vx, VXF, B.data.fiber_basis),
    )
end

function get_coordinates!(
    M::FiberBundle,
    Y,
    p,
    X,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:FiberBundleBasisData},
) where {𝔽}
    px, Vx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    n = manifold_dimension(M.manifold)
    F = Fiber(M.manifold, px, M.type)
    get_coordinates!(M.manifold, view(Y, 1:n), px, VXM, B.data.base_basis)
    get_coordinates!(F, view(Y, (n + 1):length(Y)), Vx, VXF, B.data.fiber_basis)
    return Y
end

function get_vector!(M::FiberBundle, Y, p, X, B::AbstractBasis)
    n = manifold_dimension(M.manifold)
    xp1, xp2 = submanifold_components(M, p)
    Yp1, Yp2 = submanifold_components(M, Y)
    F = Fiber(M.manifold, xp1, M.type)
    get_vector!(M.manifold, Yp1, xp1, X[1:n], B)
    get_vector!(F, Yp2, xp2, X[(n + 1):end], B)
    return Y
end

function get_vector!(
    M::FiberBundle,
    Y,
    p,
    X,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:FiberBundleBasisData},
) where {𝔽}
    n = manifold_dimension(M.manifold)
    xp1, xp2 = submanifold_components(M, p)
    Yp1, Yp2 = submanifold_components(M, Y)
    F = Fiber(M.manifold, xp1, M.type)
    get_vector!(M.manifold, Yp1, xp1, X[1:n], B.data.base_basis)
    get_vector!(F, Yp2, xp2, X[(n + 1):end], B.data.fiber_basis)
    return Y
end

function _isapprox(B::FiberBundle, p, q; kwargs...)
    xp, Vp = submanifold_components(B.manifold, p)
    xq, Vq = submanifold_components(B.manifold, q)
    return isapprox(B.manifold, xp, xq; kwargs...) &&
           isapprox(Fiber(B.manifold, xp, B.type), Vp, Vq; kwargs...)
end
function _isapprox(B::FiberBundle, p, X, Y; kwargs...)
    px, Vx = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    VYM, VYF = submanifold_components(B.manifold, Y)
    return isapprox(B.manifold, VXM, VYM; kwargs...) &&
           isapprox(Fiber(B.manifold, px, B.type), Vx, VXF, VYF; kwargs...)
end

function manifold_dimension(B::FiberBundle)
    return manifold_dimension(B.manifold) + fiber_dimension(B.manifold, B.type)
end

function Random.rand!(M::FiberBundle, pX; vector_at=nothing)
    return rand!(Random.default_rng(), M, pX; vector_at=vector_at)
end
function Random.rand!(rng::AbstractRNG, M::FiberBundle, pX; vector_at=nothing)
    pXM, pXF = submanifold_components(M.manifold, pX)
    if vector_at === nothing
        rand!(rng, M.manifold, pXM)
        rand!(rng, Fiber(M.manifold, pXM, M.type), pXF)
    else
        vector_atM, vector_atF = submanifold_components(M.manifold, vector_at)
        rand!(rng, M.manifold, pXM; vector_at=vector_atM)
        rand!(rng, Fiber(M.manifold, pXM, M.type), pXF; vector_at=vector_atF)
    end
    return pX
end

@doc raw"""
    zero_vector(B::FiberBundle, p)

Zero tangent vector at point `p` from the fiber bundle `B`
over manifold `B.fiber` (denoted ``\mathcal M``). The zero vector belongs to the space ``T_{p}B``

Notation:
  * The point ``p = (x_p, V_p)`` where ``x_p ∈ \mathcal M`` and ``V_p`` belongs to the
    fiber ``F=π^{-1}(\{x_p\})`` of the vector bundle ``B`` where ``π`` is the
    canonical projection of that vector bundle ``B``.

The zero vector is calculated as

``\mathbf{0}_{p} = (\mathbf{0}_{x_p}, \mathbf{0}_F)``

where ``\mathbf{0}_{x_p}`` is the zero tangent vector from ``T_{x_p}\mathcal M`` and
``\mathbf{0}_F`` is the zero element of the vector space ``F``.
"""
zero_vector(::FiberBundle, ::Any...)

function zero_vector!(B::FiberBundle, X, p)
    xp, Vp = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    F = Fiber(B.manifold, xp, B.type)
    zero_vector!(B.manifold, VXM, xp)
    zero_vector!(F, VXF, Vp)
    return X
end

@inline function allocate_result(M::FiberBundle, f::TF) where {TF}
    p = allocate_result(M.manifold, f)
    X = allocate_result(Fiber(M.manifold, p, M.type), f)
    return ArrayPartition(p, X)
end

function get_vector(M::FiberBundle, p, X, B::AbstractBasis)
    n = manifold_dimension(M.manifold)
    xp1, xp2 = submanifold_components(M, p)
    F = Fiber(M.manifold, xp1, M.type)
    return ArrayPartition(
        get_vector(M.manifold, xp1, X[1:n], B),
        get_vector(F, xp2, X[(n + 1):end], B),
    )
end
function get_vector(
    M::FiberBundle,
    p,
    X,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:FiberBundleBasisData},
) where {𝔽}
    n = manifold_dimension(M.manifold)
    xp1, xp2 = submanifold_components(M, p)
    F = Fiber(M.manifold, xp1, M.type)
    return ArrayPartition(
        get_vector(M.manifold, xp1, X[1:n], B.data.base_basis),
        get_vector(F, xp2, X[(n + 1):end], B.data.fiber_basis),
    )
end

function get_vectors(
    M::FiberBundle,
    p::ArrayPartition,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:FiberBundleBasisData},
) where {𝔽}
    xp1, xp2 = submanifold_components(M, p)
    zero_m = zero_vector(M.manifold, xp1)
    F = Fiber(M.manifold, xp1, M.type)
    zero_f = zero_vector(F, xp1)
    vs = typeof(ArrayPartition(zero_m, zero_f))[]
    for bv in get_vectors(M.manifold, xp1, B.data.base_basis)
        push!(vs, ArrayPartition(bv, zero_f))
    end
    for bv in get_vectors(F, xp2, B.data.fiber_basis)
        push!(vs, ArrayPartition(zero_m, bv))
    end
    return vs
end

"""
    getindex(p::ArrayPartition, M::FiberBundle, s::Symbol)
    p[M::FiberBundle, s]

Access the element(s) at index `s` of a point `p` on a [`FiberBundle`](@ref) `M` by
using the symbols `:point` and `:vector` or `:fiber` for the base and vector or fiber
component, respectively.
"""
@inline function Base.getindex(p::ArrayPartition, M::FiberBundle, s::Symbol)
    (s === :point) && return p.x[1]
    (s === :vector || s === :fiber) && return p.x[2]
    return throw(DomainError(s, "unknown component $s on $M."))
end

"""
    setindex!(p::ArrayPartition, val, M::FiberBundle, s::Symbol)
    p[M::VectorBundle, s] = val

Set the element(s) at index `s` of a point `p` on a [`FiberBundle`](@ref) `M` to `val` by
using the symbols `:point` and `:fiber` or `:vector` for the base and fiber or vector
component, respectively.

!!! note

    The *content* of element of `p` is replaced, not the element itself.
"""
@inline function Base.setindex!(x::ArrayPartition, val, M::FiberBundle, s::Symbol)
    if s === :point
        return copyto!(x.x[1], val)
    elseif s === :vector || s === :fiber
        return copyto!(x.x[2], val)
    else
        throw(DomainError(s, "unknown component $s on $M."))
    end
end

function Base.show(io::IO, B::FiberBundle)
    return print(io, "FiberBundle($(B.type), $(B.manifold), $(B.vector_transport))")
end

@inline function Base.view(x::ArrayPartition, M::FiberBundle, s::Symbol)
    (s === :point) && return x.x[1]
    (s === :vector || s === :fiber) && return x.x[2]
    throw(DomainError(s, "unknown component $s on $M."))
end
