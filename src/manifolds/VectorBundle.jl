
"""
    fiber_bundle_transport(M::AbstractManifold, fiber::FiberType)

Determine the vector transport used for [`exp`](@ref exp(::FiberBundle, ::Any...)) and
[`log`](@ref log(::FiberBundle, ::Any...)) maps on a vector bundle with fiber type
`fiber` and manifold `M`.
"""
fiber_bundle_transport(M::AbstractManifold, ::FiberType) =
    default_vector_transport_method(M)

"""
    VectorBundle{𝔽,TVS,TM,VTV} = FiberBundle{𝔽,TVS,TM,TVT} where {TVS<:VectorSpaceType}

Alias for [`FiberBundle`](@ref) when fiber type is a `TVS` of type
[`VectorSpaceType`](@extref `ManifoldsBase.VectorSpaceType`).

`VectorSpaceFiberType` is used to encode vector spaces as fiber types.
"""
const VectorBundle{𝔽,TVS,TM,TVT} = FiberBundle{
    𝔽,
    TVS,
    TM,
    TVT,
} where {
    TVS<:VectorSpaceType,
    TM<:AbstractManifold{𝔽},
    TVT<:FiberBundleProductVectorTransport,
}

"""
    TangentBundle{𝔽,M} = VectorBundle{𝔽,TangentSpaceType,M} where {𝔽,M<:AbstractManifold{𝔽}}

Tangent bundle for manifold of type `M`, as a manifold with the Sasaki metric [Sasaki:1958](@cite).

Exact retraction and inverse retraction can be approximated using [`FiberBundleProductRetraction`](@ref),
[`FiberBundleInverseProductRetraction`](@ref) and [`SasakiRetraction`](@extref `ManifoldsBase.SasakiRetraction`).
[`FiberBundleProductVectorTransport`](@ref) can be used as a vector transport.

# Constructors

    TangentBundle(M::AbstractManifold)
    TangentBundle(M::AbstractManifold, vtm::FiberBundleProductVectorTransport)
"""
const TangentBundle{𝔽,M} =
    VectorBundle{𝔽,TangentSpaceType,M} where {𝔽,M<:AbstractManifold{𝔽}}

TangentBundle(M::AbstractManifold) = FiberBundle(TangentSpaceType(), M)
function TangentBundle(M::AbstractManifold, vtm::FiberBundleProductVectorTransport)
    return FiberBundle(TangentSpaceType(), M, vtm)
end

const CotangentBundle{𝔽,M} =
    VectorBundle{𝔽,CotangentSpaceType,M} where {𝔽,M<:AbstractManifold{𝔽}}

CotangentBundle(M::AbstractManifold) = FiberBundle(CotangentSpaceType(), M)
function CotangentBundle(M::AbstractManifold, vtm::FiberBundleProductVectorTransport)
    return FiberBundle(CotangentSpaceType(), M, vtm)
end

function bundle_transport_to!(B::TangentBundle, Y, p, X, q)
    return vector_transport_to!(B.manifold, Y, p, X, q, B.vector_transport.method_vertical)
end

function bundle_transport_tangent_direction!(
    B::TangentBundle,
    Y,
    p,
    pf,
    X,
    d,
    m::AbstractVectorTransportMethod=default_vector_transport_method(B.manifold),
)
    return vector_transport_direction!(B.manifold, Y, p, X, d, m)
end

function bundle_transport_tangent_to!(
    B::TangentBundle,
    Y,
    p,
    pf,
    X,
    q,
    m::AbstractVectorTransportMethod=default_vector_transport_method(B.manifold),
)
    return vector_transport_to!(B.manifold, Y, p, X, q, m)
end

function default_inverse_retraction_method(::TangentBundle)
    return FiberBundleInverseProductRetraction()
end

function default_retraction_method(::TangentBundle)
    return FiberBundleProductRetraction()
end

function default_vector_transport_method(B::TangentBundle)
    default_vt_m = default_vector_transport_method(B.manifold)
    return FiberBundleProductVectorTransport(default_vt_m, default_vt_m)
end

"""
    injectivity_radius(M::TangentBundle)

Injectivity radius of [`TangentBundle`](@ref) manifold is infinite if the base manifold
is flat and 0 otherwise.
See [https://mathoverflow.net/questions/94322/injectivity-radius-of-the-sasaki-metric](https://mathoverflow.net/questions/94322/injectivity-radius-of-the-sasaki-metric).
"""
function injectivity_radius(M::TangentBundle)
    if is_flat(M.manifold)
        return Inf
    else
        return 0.0
    end
end

@doc raw"""
    inner(B::VectorBundle, p, X, Y)

Inner product of tangent vectors `X` and `Y` at point `p` from the
vector bundle `B` over manifold `B.fiber` (denoted ``\mathcal M``).

Notation:
  * The point ``p = (x_p, V_p)`` where ``x_p ∈ \mathcal M`` and ``V_p`` belongs to the
    fiber ``F=π^{-1}(\{x_p\})`` of the vector bundle ``B`` where ``π`` is the
    canonical projection of that vector bundle ``B``.
  * The tangent vector ``v = (V_{X,M}, V_{X,F}) ∈ T_{x}B`` where
    ``V_{X,M}`` is a tangent vector from the tangent space ``T_{x_p}\mathcal M`` and
    ``V_{X,F}`` is a tangent vector from the tangent space ``T_{V_p}F`` (isomorphic to ``F``).
    Similarly for the other tangent vector ``w = (V_{Y,M}, V_{Y,F}) ∈ T_{x}B``.

The inner product is calculated as

``⟨X, Y⟩_p = ⟨V_{X,M}, V_{Y,M}⟩_{x_p} + ⟨V_{X,F}, V_{Y,F}⟩_{V_p}.``
"""
function inner(B::FiberBundle, p, X, Y)
    px, Vx = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    VYM, VYF = submanifold_components(B.manifold, Y)
    F = Fiber(B.manifold, px, B.type)
    # for tangent bundle Vx is discarded by the method of inner for TangentSpace
    # and px is actually used as the base point
    return inner(B.manifold, px, VXM, VYM) + inner(F, Vx, VXF, VYF)
end

function _inverse_retract(M::FiberBundle, p, q, ::FiberBundleInverseProductRetraction)
    return inverse_retract_product(M, p, q)
end

function _inverse_retract!(M::FiberBundle, X, p, q, ::FiberBundleInverseProductRetraction)
    return inverse_retract_product!(M, X, p, q)
end

"""
    inverse_retract(M::VectorBundle, p, q, ::FiberBundleInverseProductRetraction)

Compute the allocating variant of the [`FiberBundleInverseProductRetraction`](@ref),
which by default allocates and calls `inverse_retract_product!`.
"""
inverse_retract(::VectorBundle, p, q, ::FiberBundleInverseProductRetraction)

function inverse_retract_product(M::VectorBundle, p, q)
    X = allocate_result(M, inverse_retract, p, q)
    return inverse_retract_product!(M, X, p, q)
end

function inverse_retract_product!(B::VectorBundle, X, p, q)
    px, Vx = submanifold_components(B.manifold, p)
    py, Vy = submanifold_components(B.manifold, q)
    VXM, VXF = submanifold_components(B.manifold, X)
    log!(B.manifold, VXM, px, py)
    bundle_transport_to!(B, VXF, py, Vy, px)
    copyto!(VXF, VXF - Vx)
    return X
end

"""
    is_flat(::VectorBundle)

Return true if the underlying manifold of [`VectorBundle`](@ref) `M` is flat.
"""
is_flat(M::VectorBundle) = is_flat(M.manifold)

@doc raw"""
    project(B::VectorBundle, p)

Project the point `p` from the ambient space of the vector bundle `B`
over manifold `B.fiber` (denoted ``\mathcal M``) to the vector bundle.

Notation:
  * The point ``p = (x_p, V_p)`` where ``x_p`` belongs to the ambient space of ``\mathcal M``
    and ``V_p`` belongs to the ambient space of the
    fiber ``F=π^{-1}(\{x_p\})`` of the vector bundle ``B`` where ``π`` is the
    canonical projection of that vector bundle ``B``.

The projection is calculated by projecting the point ``x_p`` to the manifold ``\mathcal M``
and then projecting the vector ``V_p`` to the tangent space ``T_{x_p}\mathcal M``.
"""
project(::VectorBundle, ::Any)

function project!(B::VectorBundle, q, p)
    px, pVx = submanifold_components(B.manifold, p)
    qx, qVx = submanifold_components(B.manifold, q)
    project!(B.manifold, qx, px)
    F = Fiber(B.manifold, qx, B.type)
    project!(F, qVx, pVx)
    return q
end

@doc raw"""
    project(B::VectorBundle, p, X)

Project the element `X` of the ambient space of the tangent space ``T_p B``
to the tangent space ``T_p B``.

Notation:
  * The point ``p = (x_p, V_p)`` where ``x_p ∈ \mathcal M`` and ``V_p`` belongs to the
    fiber ``F=π^{-1}(\{x_p\})`` of the vector bundle ``B`` where ``π`` is the
    canonical projection of that vector bundle ``B``.
  * The vector ``x = (V_{X,M}, V_{X,F})`` where ``x_p`` belongs to the ambient space of
    ``T_{x_p}\mathcal M`` and ``V_{X,F}`` belongs to the ambient space of the
    fiber ``F=π^{-1}(\{x_p\})`` of the vector bundle ``B`` where ``π`` is the
    canonical projection of that vector bundle ``B``.

The projection is calculated by projecting ``V_{X,M}`` to tangent space ``T_{x_p}\mathcal M``
and then projecting the vector ``V_{X,F}`` to the fiber ``F``.
"""
project(::VectorBundle, ::Any, ::Any)

function project!(B::VectorBundle, Y, p, X)
    px, Vx = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    VYM, VYF = submanifold_components(B.manifold, Y)
    F = Fiber(B.manifold, px, B.type)
    project!(B.manifold, VYM, px, VXM)
    project!(F, VYF, Vx, VXF)
    return Y
end

function _retract!(M::VectorBundle, q, p, X, t::Number, ::FiberBundleProductRetraction)
    return retract_product!(M, q, p, X, t)
end

"""
    retract(M::VectorBundle, p, q, t::Number, ::FiberBundleProductRetraction)

Compute the allocating variant of the [`FiberBundleProductRetraction`](@ref),
which by default allocates and calls `retract_product!`.
"""
retract(::VectorBundle, p, q, t::Number, ::FiberBundleProductRetraction)

function _retract(M::VectorBundle, p, X, t::Number, ::FiberBundleProductRetraction)
    return retract_product(M, p, X, t)
end

function retract_product(M::VectorBundle, p, X, t::Number)
    q = allocate_result(M, retract, p, X)
    return retract_product!(M, q, p, X, t)
end

function retract_product!(B::VectorBundle, q, p, X, t::Number)
    tX = t * X
    xp, Xp = submanifold_components(B.manifold, p)
    xq, Xq = submanifold_components(B.manifold, q)
    VXM, VXF = submanifold_components(B.manifold, tX)
    # this temporary avoids overwriting `p` when `q` and `p` occupy the same memory
    xqt = exp(B.manifold, xp, VXM)
    vector_transport_direction!(
        B.manifold,
        Xq,
        xp,
        Xp + VXF,
        VXM,
        B.vector_transport.method_horizontal,
    )
    copyto!(B.manifold, xq, xqt)
    return q
end

function retract_sasaki!(B::TangentBundle, q, p, X, t::Number, m::SasakiRetraction)
    tX = t * X
    xp, Xp = submanifold_components(B.manifold, p)
    xq, Xq = submanifold_components(B.manifold, q)
    VXM, VXF = submanifold_components(B.manifold, tX)
    p_k = allocate(B.manifold, xp)
    copyto!(B.manifold, p_k, xp)
    X_k = allocate(B.manifold, Xp)
    copyto!(B.manifold, X_k, Xp)
    Y_k = allocate(B.manifold, VXM)
    copyto!(B.manifold, Y_k, VXM)
    Z_k = allocate(B.manifold, VXF)
    copyto!(B.manifold, Z_k, VXF)
    ϵ = 1 / m.L
    for k in 1:(m.L)
        p_kp1 = exp(B.manifold, p_k, ϵ * Y_k)
        X_kp1 = parallel_transport_to(B.manifold, p_k, X_k .+ ϵ .* Z_k, p_kp1)
        Y_kp1 = parallel_transport_to(
            B.manifold,
            p_k,
            Y_k .+ ϵ .* riemann_tensor(B.manifold, p_k, X_k, Z_k, Y_k),
            p_kp1,
        )
        Z_kp1 = parallel_transport_to(B.manifold, p_k, Z_k, p_kp1)
        copyto!(B.manifold, p_k, p_kp1)
        copyto!(B.manifold, X_k, p_kp1, X_kp1)
        copyto!(B.manifold, Y_k, p_kp1, Y_kp1)
        copyto!(B.manifold, Z_k, p_kp1, Z_kp1)
    end
    copyto!(B.manifold, xq, p_k)
    copyto!(B.manifold, Xq, xq, X_k)
    return q
end

Base.show(io::IO, vb::TangentBundle) = print(io, "TangentBundle($(vb.manifold))")

function vector_transport_direction(M::VectorBundle, p, X, d)
    return vector_transport_direction(M, p, X, d, M.vector_transport)
end

function vector_transport_direction!(M::VectorBundle, Y, p, X, d)
    return vector_transport_direction!(M, Y, p, X, d, M.vector_transport)
end

function _vector_transport_direction!(
    M::VectorBundle,
    Y,
    p,
    X,
    d,
    m::FiberBundleProductVectorTransport,
)
    VYM, VYF = submanifold_components(M.manifold, Y)
    px, pVx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    dx, dVx = submanifold_components(M.manifold, d)
    vector_transport_direction!(M.manifold, VYM, px, VXM, dx, m.method_horizontal),
    vector_transport_direction!(M.manifold, VYF, px, VXF, dx, m.method_vertical),
    return Y
end

@doc raw"""
    vector_transport_to(M::VectorBundle, p, X, q, m::FiberBundleProductVectorTransport)

Compute the vector transport the tangent vector `X`at `p` to `q` on the
[`VectorBundle`](@ref) `M` using the [`FiberBundleProductVectorTransport`](@ref) `m`.
"""
vector_transport_to(
    ::VectorBundle,
    ::Any,
    ::Any,
    ::Any,
    ::FiberBundleProductVectorTransport,
)

function vector_transport_to(M::VectorBundle, p, X, q)
    return vector_transport_to(M, p, X, q, M.vector_transport)
end

function vector_transport_to!(M::VectorBundle, Y, p, X, q)
    return vector_transport_to!(M, Y, p, X, q, M.vector_transport)
end
function vector_transport_to!(
    M::TangentBundle,
    Y,
    p,
    X,
    q,
    m::FiberBundleProductVectorTransport,
)
    px, pVx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    VYM, VYF = submanifold_components(M.manifold, Y)
    qx, qVx = submanifold_components(M.manifold, q)
    vector_transport_to!(M.manifold, VYM, px, VXM, qx, m.method_horizontal)
    bundle_transport_tangent_to!(M, VYF, px, pVx, VXF, qx, m.method_vertical)
    return Y
end

Base.show(io::IO, vb::CotangentBundle) = print(io, "CotangentBundle($(vb.manifold))")
