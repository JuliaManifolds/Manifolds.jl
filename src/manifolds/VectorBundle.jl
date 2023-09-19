
@doc raw"""
    struct SasakiRetraction <: AbstractRetractionMethod end

Exponential map on [`TangentBundle`](@ref) computed via Euler integration as described
in [MuralidharanFlecther:2012](@cite). The system of equations for $\gamma : ℝ \to T\mathcal M$ such that
$\gamma(1) = \exp_{p,X}(X_M, X_F)$ and $\gamma(0)=(p, X)$ reads

```math
\dot{\gamma}(t) = (\dot{p}(t), \dot{X}(t)) = (R(X(t), \dot{X}(t))\dot{p}(t), 0)
```

where $R$ is the Riemann curvature tensor (see [`riemann_tensor`](@ref)).

# Constructor

    SasakiRetraction(L::Int)

In this constructor `L` is the number of integration steps.
"""
struct SasakiRetraction <: AbstractRetractionMethod
    L::Int
end

"""
    const VectorBundleVectorTransport = FiberBundleProductVectorTransport

Deprecated: an alias for `FiberBundleProductVectorTransport`.
"""
const VectorBundleVectorTransport = FiberBundleProductVectorTransport

const VectorBundle{𝔽,TVS,TM,TVT} = FiberBundle{
    𝔽,
    VectorSpaceFiberType{TVS},
    TM,
    TVT,
} where {
    TVS<:VectorSpaceType,
    TM<:AbstractManifold{𝔽},
    TVT<:FiberBundleProductVectorTransport,
}

function VectorBundle(
    vst::VectorSpaceType,
    M::AbstractManifold,
    vtm::FiberBundleProductVectorTransport,
)
    return FiberBundle(VectorSpaceFiberType(vst), M, vtm)
end
function VectorBundle(vst::VectorSpaceType, M::AbstractManifold)
    return FiberBundle(VectorSpaceFiberType(vst), M)
end

"""
    TangentBundle{𝔽,M} = VectorBundle{𝔽,TangentSpaceType,M} where {𝔽,M<:AbstractManifold{𝔽}}

Tangent bundle for manifold of type `M`, as a manifold with the Sasaki metric [Sasaki:1958](@cite).

Exact retraction and inverse retraction can be approximated using [`FiberBundleProductRetraction`](@ref),
[`FiberBundleInverseProductRetraction`](@ref) and [`SasakiRetraction`](@ref).
[`FiberBundleProductVectorTransport`](@ref) can be used as a vector transport.

# Constructors

    TangentBundle(M::AbstractManifold)
    TangentBundle(M::AbstractManifold, vtm::FiberBundleProductVectorTransport)
"""
const TangentBundle{𝔽,M} =
    VectorBundle{𝔽,TangentSpaceType,M} where {𝔽,M<:AbstractManifold{𝔽}}

TangentBundle(M::AbstractManifold) = VectorBundle(TangentSpace, M)
function TangentBundle(M::AbstractManifold, vtm::FiberBundleProductVectorTransport)
    return VectorBundle(TangentSpace, M, vtm)
end

const CotangentBundle{𝔽,M} =
    VectorBundle{𝔽,CotangentSpaceType,M} where {𝔽,M<:AbstractManifold{𝔽}}

CotangentBundle(M::AbstractManifold) = VectorBundle(CotangentSpace, M)
function CotangentBundle(M::AbstractManifold, vtm::FiberBundleProductVectorTransport)
    return VectorBundle(CotangentSpace, M, vtm)
end

base_manifold(B::VectorBundle) = base_manifold(B.manifold)

"""
    bundle_projection(B::VectorBundle, p::ArrayPartition)

Projection of point `p` from the bundle `M` to the base manifold.
Returns the point on the base manifold `B.manifold` at which the vector part
of `p` is attached.
"""
bundle_projection(B::VectorBundle, p) = submanifold_component(B.manifold, p, Val(1))

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
    distance(B::BundleFibers, p, X, Y)

Distance between vectors `X` and `Y` from the vector space at point `p`
from the manifold `B.manifold`, that is the base manifold of `M`.
"""
distance(B::VectorBundleFibers, p, X, Y) = norm(B, p, X - Y)

function get_basis(M::TangentBundleFibers, p, B::AbstractBasis{<:Any,TangentSpaceType})
    return get_basis(M.manifold, p, B)
end

function get_coordinates(M::TangentBundleFibers, p, X, B::AbstractBasis)
    return get_coordinates(M.manifold, p, X, B)
end

function get_coordinates!(M::TangentBundleFibers, Y, p, X, B::AbstractBasis)
    return get_coordinates!(M.manifold, Y, p, X, B)
end

function get_vector(M::TangentBundleFibers, p, X, B::AbstractBasis)
    return get_vector(M.manifold, p, X, B)
end

function get_vector!(M::TangentBundleFibers, Y, p, X, B::AbstractBasis)
    return get_vector!(M.manifold, Y, p, X, B)
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

"""
    inner(B::BundleFibers, p, X, Y)

Inner product of vectors `X` and `Y` from the vector space of type `B.fiber`
at point `p` from manifold `B.manifold`.
"""
inner(B::BundleFibers, p, X, Y)

inner(B::TangentBundleFibers, p, X, Y) = inner(B.manifold, p, X, Y)
function inner(B::CotangentBundleFibers, p, X, Y)
    return inner(B.manifold, p, sharp(B.manifold, p, X), sharp(B.manifold, p, Y))
end
@doc raw"""
    inner(B::VectorBundle, p, X, Y)

Inner product of tangent vectors `X` and `Y` at point `p` from the
vector bundle `B` over manifold `B.fiber` (denoted $\mathcal M$).

Notation:
  * The point $p = (x_p, V_p)$ where $x_p ∈ \mathcal M$ and $V_p$ belongs to the
    fiber $F=π^{-1}(\{x_p\})$ of the vector bundle $B$ where $π$ is the
    canonical projection of that vector bundle $B$.
  * The tangent vector $v = (V_{X,M}, V_{X,F}) ∈ T_{x}B$ where
    $V_{X,M}$ is a tangent vector from the tangent space $T_{x_p}\mathcal M$ and
    $V_{X,F}$ is a tangent vector from the tangent space $T_{V_p}F$ (isomorphic to $F$).
    Similarly for the other tangent vector $w = (V_{Y,M}, V_{Y,F}) ∈ T_{x}B$.

The inner product is calculated as

$⟨X, Y⟩_p = ⟨V_{X,M}, V_{Y,M}⟩_{x_p} + ⟨V_{X,F}, V_{Y,F}⟩_{V_p}.$
"""
function inner(B::FiberBundle, p, X, Y)
    px, Vx = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    VYM, VYF = submanifold_components(B.manifold, Y)
    # for tangent bundle Vx is discarded by the method of inner for TangentSpaceAtPoint
    # and px is actually used as the base point
    return inner(B.manifold, px, VXM, VYM) + inner(FiberAtPoint(B.fiber, px), Vx, VXF, VYF)
end

function _inverse_retract(M::FiberBundle, p, q, ::FiberBundleInverseProductRetraction)
    return inverse_retract_product(M, p, q)
end

function _inverse_retract!(M::FiberBundle, X, p, q, ::FiberBundleInverseProductRetraction)
    return inverse_retract_product!(M, X, p, q)
end

"""
    inverse_retract_product(M::VectorBundle, p, q)

Compute the allocating variant of the [`FiberBundleInverseProductRetraction`](@ref),
which by default allocates and calls `inverse_retract_product!`.
"""
function inverse_retract_product(M::VectorBundle, p, q)
    X = allocate_result(M, inverse_retract, p, q)
    return inverse_retract_product!(M, X, p, q)
end

function inverse_retract_product!(B::VectorBundle, X, p, q)
    px, Vx = submanifold_components(B.manifold, p)
    py, Vy = submanifold_components(B.manifold, q)
    VXM, VXF = submanifold_components(B.manifold, X)
    log!(B.manifold, VXM, px, py)
    vector_transport_to!(B.fiber, VXF, py, Vy, px, B.vector_transport.method_fiber)
    copyto!(VXF, VXF - Vx)
    return X
end

"""
    is_flat(::VectorBundle)

Return true if the underlying manifold of [`VectorBundle`](@ref) `M` is flat.
"""
is_flat(M::VectorBundle) = is_flat(M.manifold)

"""
    norm(B::BundleFibers, p, q)

Norm of the vector `X` from the vector space of type `B.fiber`
at point `p` from manifold `B.manifold`.
"""
LinearAlgebra.norm(B::VectorBundleFibers, p, X) = sqrt(inner(B, p, X, X))
LinearAlgebra.norm(B::TangentBundleFibers, p, X) = norm(B.manifold, p, X)

@doc raw"""
    project(B::VectorBundle, p)

Project the point `p` from the ambient space of the vector bundle `B`
over manifold `B.fiber` (denoted $\mathcal M$) to the vector bundle.

Notation:
  * The point $p = (x_p, V_p)$ where $x_p$ belongs to the ambient space of $\mathcal M$
    and $V_p$ belongs to the ambient space of the
    fiber $F=π^{-1}(\{x_p\})$ of the vector bundle $B$ where $π$ is the
    canonical projection of that vector bundle $B$.

The projection is calculated by projecting the point $x_p$ to the manifold $\mathcal M$
and then projecting the vector $V_p$ to the tangent space $T_{x_p}\mathcal M$.
"""
project(::VectorBundle, ::Any)

function project!(B::VectorBundle, q, p)
    px, pVx = submanifold_components(B.manifold, p)
    qx, qVx = submanifold_components(B.manifold, q)
    project!(B.manifold, qx, px)
    project!(B.manifold, qVx, qx, pVx)
    return q
end

@doc raw"""
    project(B::VectorBundle, p, X)

Project the element `X` of the ambient space of the tangent space $T_p B$
to the tangent space $T_p B$.

Notation:
  * The point $p = (x_p, V_p)$ where $x_p ∈ \mathcal M$ and $V_p$ belongs to the
    fiber $F=π^{-1}(\{x_p\})$ of the vector bundle $B$ where $π$ is the
    canonical projection of that vector bundle $B$.
  * The vector $x = (V_{X,M}, V_{X,F})$ where $x_p$ belongs to the ambient space of $T_{x_p}\mathcal M$
    and $V_{X,F}$ belongs to the ambient space of the
    fiber $F=π^{-1}(\{x_p\})$ of the vector bundle $B$ where $π$ is the
    canonical projection of that vector bundle $B$.

The projection is calculated by projecting $V_{X,M}$ to tangent space $T_{x_p}\mathcal M$
and then projecting the vector $V_{X,F}$ to the fiber $F$.
"""
project(::VectorBundle, ::Any, ::Any)

function project!(B::VectorBundle, Y, p, X)
    px, Vx = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    VYM, VYF = submanifold_components(B.manifold, Y)
    project!(B.manifold, VYM, px, VXM)
    project!(B.manifold, VYF, px, VXF)
    return Y
end

"""
    project(B::BundleFibers, p, X)

Project vector `X` from the vector space of type `B.fiber` at point `p`.
"""
function project(B::BundleFibers, p, X)
    Y = allocate_result(B, project, p, X)
    return project!(B, Y, p, X)
end

function project!(B::TangentBundleFibers, Y, p, X)
    return project!(B.manifold, Y, p, X)
end

function _retract(M::VectorBundle, p, X, t::Number, ::FiberBundleProductRetraction)
    return retract_product(M, p, X, t)
end

function _retract!(M::VectorBundle, q, p, X, t::Number, ::FiberBundleProductRetraction)
    return retract_product!(M, q, p, X, t)
end

"""
    retract_product(M::VectorBundle, p, q, t::Number)

Compute the allocating variant of the [`FiberBundleProductRetraction`](@ref),
which by default allocates and calls `retract_product!`.
"""
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
        B.vector_transport.method_point,
    )
    copyto!(B.manifold, xq, xqt)
    return q
end

function _retract(M::AbstractManifold, p, X, t::Number, m::SasakiRetraction)
    return retract_sasaki(M, p, X, t, m)
end

function _retract!(M::AbstractManifold, q, p, X, t::Number, m::SasakiRetraction)
    return retract_sasaki!(M, q, p, X, t, m)
end

"""
    retract_sasaki(M::AbstractManifold, p, X, t::Number, m::SasakiRetraction)

Compute the allocating variant of the [`SasakiRetraction`](@ref),
which by default allocates and calls `retract_sasaki!`.
"""
function retract_sasaki(M::AbstractManifold, p, X, t::Number, m::SasakiRetraction)
    q = allocate_result(M, retract, p, X)
    return retract_sasaki!(M, q, p, X, t, m)
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

function representation_size(B::CotangentBundleFibers)
    return representation_size(B.manifold)
end
function representation_size(B::TangentBundleFibers)
    return representation_size(B.manifold)
end

function Base.show(io::IO, tpt::TensorProductType)
    return print(io, "TensorProductType(", join(tpt.spaces, ", "), ")")
end
function Base.show(io::IO, vb::VectorBundle)
    return print(io, "VectorBundle($(vb.type.fiber), $(vb.manifold))")
end
function Base.show(io::IO, vbf::VectorBundleFibers)
    return print(io, "VectorBundleFibers($(vbf.fiber.fiber), $(vbf.manifold))")
end
Base.show(io::IO, vb::TangentBundle) = print(io, "TangentBundle($(vb.manifold))")
Base.show(io::IO, vb::CotangentBundle) = print(io, "CotangentBundle($(vb.manifold))")

@inline function allocate_result(B::TangentBundleFibers, f::typeof(zero_vector), x...)
    return allocate_result(B.manifold, f, x...)
end
@inline function allocate_result(M::VectorBundle, f::TF) where {TF}
    return ArrayPartition(allocate_result(M.manifold, f), allocate_result(M.fiber, f))
end

"""
    fiber_bundle_transport(fiber::FiberType, M::AbstractManifold)

Determine the vector tranport used for [`exp`](@ref exp(::FiberBundle, ::Any...)) and
[`log`](@ref log(::FiberBundle, ::Any...)) maps on a vector bundle with fiber type
`fiber` and manifold `M`.
"""
fiber_bundle_transport(::VectorSpaceType, ::AbstractManifold) = ParallelTransport()

function vector_space_dimension(M::AbstractManifold, V::TensorProductType)
    dim = 1
    for space in V.spaces
        dim *= fiber_dimension(M, space)
    end
    return dim
end

function vector_space_dimension(B::TangentBundleFibers)
    return manifold_dimension(B.manifold)
end
function vector_space_dimension(B::CotangentBundleFibers)
    return manifold_dimension(B.manifold)
end
function vector_space_dimension(B::VectorBundleFibers)
    return vector_space_dimension(B.manifold, B.fiber.fiber)
end

function vector_transport_direction(M::VectorBundle, p, X, d)
    return vector_transport_direction(M, p, X, d, M.vector_transport)
end
function vector_transport_direction(
    M::TangentBundleFibers,
    p,
    X,
    d,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_direction(M.manifold, p, X, d, m)
end

function _vector_transport_direction(
    M::VectorBundle,
    p,
    X,
    d,
    m::FiberBundleProductVectorTransport,
)
    px, pVx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    dx, dVx = submanifold_components(M.manifold, d)
    return ArrayPartition(
        vector_transport_direction(M.manifold, px, VXM, dx, m.method_point),
        vector_transport_direction(M.fiber, px, VXF, dx, m.method_fiber),
    )
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
    vector_transport_direction!(M.manifold, VYM, px, VXM, dx, m.method_point),
    vector_transport_direction!(M.manifold, VYF, px, VXF, dx, m.method_fiber),
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

function _vector_transport_to(
    M::VectorBundle,
    p,
    X,
    q,
    m::FiberBundleProductVectorTransport,
)
    px, pVx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    qx, qVx = submanifold_components(M.manifold, q)
    return ArrayPartition(
        vector_transport_to(M.manifold, px, VXM, qx, m.method_point),
        vector_transport_to(M.manifold, px, VXF, qx, m.method_fiber),
    )
end

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
    vector_transport_to!(M.manifold, VYM, px, VXM, qx, m.method_point)
    vector_transport_to!(M.manifold, VYF, px, VXF, qx, m.method_fiber)
    return Y
end
function vector_transport_to!(
    M::TangentBundleFibers,
    Y,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod,
)
    vector_transport_to!(M.manifold, Y, p, X, q, m)
    return Y
end

"""
    zero_vector!(B::BundleFibers, X, p)

Save the zero vector from the vector space of type `B.fiber` at point `p`
from manifold `B.manifold` to `X`.
"""
zero_vector!(B::BundleFibers, X, p)

function zero_vector!(B::TangentBundleFibers, X, p)
    return zero_vector!(B.manifold, X, p)
end

@doc raw"""
    Y = Weingarten(M::TangentSpaceAtPoint, p, X, V)
    Weingarten!(M::TangentSpaceAtPoint, Y, p, X, V)

Compute the Weingarten map ``\mathcal W_p`` at `p` on the [`TangentSpaceAtPoint`](@ref) `M` with respect to the
tangent vector ``X \in T_p\mathcal M`` and the normal vector ``V \in N_p\mathcal M``.

Since this a flat space by itself, the result is always the zero tangent vector.
"""
Weingarten(::TangentSpaceAtPoint, p, X, V)

Weingarten!(::TangentSpaceAtPoint, Y, p, X, V) = fill!(Y, 0)
