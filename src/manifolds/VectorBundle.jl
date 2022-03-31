
"""
    TensorProductType(spaces::VectorSpaceType...)

Vector space type corresponding to the tensor product of given vector space
types.
"""
struct TensorProductType{TS<:Tuple} <: VectorSpaceType
    spaces::TS
end

TensorProductType(spaces::VectorSpaceType...) = TensorProductType{typeof(spaces)}(spaces)

"""
    VectorBundleFibers(fiber::VectorSpaceType, M::AbstractManifold)

Type representing a family of vector spaces (fibers) of a vector bundle over `M`
with vector spaces of type `fiber`. In contrast with `VectorBundle`, operations
on `VectorBundleFibers` expect point-like and vector-like parts to be
passed separately instead of being bundled together. It can be thought of
as a representation of vector spaces from a vector bundle but without
storing the point at which a vector space is attached (which is specified
separately in various functions).
"""
struct VectorBundleFibers{TVS<:VectorSpaceType,TM<:AbstractManifold}
    fiber::TVS
    manifold::TM
end

const TangentBundleFibers{M} =
    VectorBundleFibers{TangentSpaceType,M} where {M<:AbstractManifold}

TangentBundleFibers(M::AbstractManifold) = VectorBundleFibers(TangentSpace, M)

const CotangentBundleFibers{M} =
    VectorBundleFibers{CotangentSpaceType,M} where {M<:AbstractManifold}

CotangentBundleFibers(M::AbstractManifold) = VectorBundleFibers(CotangentSpace, M)

"""
    VectorSpaceAtPoint{
        𝔽,
        TFiber<:VectorBundleFibers{<:VectorSpaceType,<:AbstractManifold{𝔽}},
        TX,
    } <: AbstractManifold{𝔽}

A vector space at a point `p` on the manifold.
This is modelled using [`VectorBundleFibers`](@ref) with only a vector-like part
and fixing the point-like part to be just `p`.

This vector space itself is also a `manifold`. Especially, it's flat and hence isometric
to the [`Euclidean`](@ref) manifold.

# Constructor
    VectorSpaceAtPoint(fiber::VectorBundleFibers, p)

A vector space (fiber type `fiber` of a vector bundle) at point `p` from
the manifold `fiber.manifold`.
"""
struct VectorSpaceAtPoint{
    𝔽,
    TFiber<:VectorBundleFibers{<:VectorSpaceType,<:AbstractManifold{𝔽}},
    TX,
} <: AbstractManifold{𝔽}
    fiber::TFiber
    point::TX
end

const TangentSpaceAtPoint{M} =
    VectorSpaceAtPoint{𝔽,TangentBundleFibers{M}} where {𝔽,M<:AbstractManifold{𝔽}}

"""
    TangentSpaceAtPoint(M::AbstractManifold, p)

Return an object of type [`VectorSpaceAtPoint`](@ref) representing tangent
space at `p` on the [`AbstractManifold`](@ref) `M`.
"""
TangentSpaceAtPoint(M::AbstractManifold, p) = VectorSpaceAtPoint(TangentBundleFibers(M), p)

"""
    TangentSpace(M::AbstractManifold, p)

Return a [`TangentSpaceAtPoint`](@ref) representing tangent space at `p` on the [`AbstractManifold`](@ref) `M`.
"""
TangentSpace(M::AbstractManifold, p) = VectorSpaceAtPoint(TangentBundleFibers(M), p)

const CotangentSpaceAtPoint{M} =
    VectorSpaceAtPoint{𝔽,CotangentBundleFibers{M}} where {𝔽,M<:AbstractManifold{𝔽}}

"""
    CotangentSpaceAtPoint(M::AbstractManifold, p)

Return an object of type [`VectorSpaceAtPoint`](@ref) representing cotangent
space at `p`.
"""
function CotangentSpaceAtPoint(M::AbstractManifold, p)
    return VectorSpaceAtPoint(CotangentBundleFibers(M), p)
end

"""
    VectorBundleVectorTransport(
        method_point::AbstractVectorTransportMethod,
        method_vector::AbstractVectorTransportMethod,
    )

Vector transport type on [`VectorBundle`](@ref). `method_point` is used for vector transport
of the point part and `method_vector` is used for transport of the vector part
"""
struct VectorBundleVectorTransport{
    TMP<:AbstractVectorTransportMethod,
    TMV<:AbstractVectorTransportMethod,
} <: AbstractVectorTransportMethod
    method_point::TMP
    method_vector::TMV
end

"""
    VectorBundle{𝔽,TVS<:VectorSpaceType,TM<:AbstractManifold{𝔽}} <: AbstractManifold{𝔽}

Vector bundle on a [`AbstractManifold`](@ref) `M` of type [`VectorSpaceType`](@ref).

# Constructor

    VectorBundle(M::AbstractManifold, type::VectorSpaceType)
"""
struct VectorBundle{
    𝔽,
    TVS<:VectorSpaceType,
    TM<:AbstractManifold{𝔽},
    TVT<:VectorBundleVectorTransport,
} <: AbstractManifold{𝔽}
    type::TVS
    manifold::TM
    fiber::VectorBundleFibers{TVS,TM}
    vector_transport::TVT
end

function VectorBundle(
    fiber::TVS,
    M::TM,
    vtm::VectorBundleVectorTransport,
) where {TVS<:VectorSpaceType,TM<:AbstractManifold{𝔽}} where {𝔽}
    return VectorBundle{𝔽,TVS,TM,typeof(vtm)}(fiber, M, VectorBundleFibers(fiber, M), vtm)
end
function VectorBundle(fiber::VectorSpaceType, M::AbstractManifold)
    vtmm = vector_bundle_transport(fiber, M)
    vtbm = VectorBundleVectorTransport(vtmm, vtmm)
    return VectorBundle(fiber, M, vtbm)
end

const TangentBundle{𝔽,M} =
    VectorBundle{𝔽,TangentSpaceType,M} where {𝔽,M<:AbstractManifold{𝔽}}

TangentBundle(M::AbstractManifold) = VectorBundle(TangentSpace, M)
function TangentBundle(M::AbstractManifold, vtm::VectorBundleVectorTransport)
    return VectorBundle(TangentSpace, M, vtm)
end

const CotangentBundle{𝔽,M} =
    VectorBundle{𝔽,CotangentSpaceType,M} where {𝔽,M<:AbstractManifold{𝔽}}

CotangentBundle(M::AbstractManifold) = VectorBundle(CotangentSpace, M)
function CotangentBundle(M::AbstractManifold, vtm::VectorBundleVectorTransport)
    return VectorBundle(CotangentSpace, M, vtm)
end

struct VectorBundleBasisData{BBasis<:CachedBasis,TBasis<:CachedBasis}
    base_basis::BBasis
    vec_basis::TBasis
end

base_manifold(B::VectorBundleFibers) = base_manifold(B.manifold)
base_manifold(B::VectorSpaceAtPoint) = base_manifold(B.fiber)
base_manifold(B::VectorBundle) = base_manifold(B.manifold)

"""
    bundle_projection(B::VectorBundle, x::ProductRepr)

Projection of point `p` from the bundle `M` to the base manifold.
Returns the point on the base manifold `B.manifold` at which the vector part
of `p` is attached.
"""
bundle_projection(B::VectorBundle, p) = submanifold_component(B.manifold, p, Val(1))

"""
    distance(B::VectorBundleFibers, p, X, Y)

Distance between vectors `X` and `Y` from the vector space at point `p`
from the manifold `B.manifold`, that is the base manifold of `M`.
"""
distance(B::VectorBundleFibers, p, X, Y) = norm(B, p, X - Y)
@doc raw"""
    distance(B::VectorBundle, p, q)

Distance between points $x$ and $y$ from the
vector bundle `B` over manifold `B.fiber` (denoted $\mathcal M$).

Notation:
  * The point $p = (x_p, V_p)$ where $x_p  ∈ \mathcal M$ and $V_p$ belongs to the
    fiber $F=π^{-1}(\{x_p\})$ of the vector bundle $B$ where $π$ is the
    canonical projection of that vector bundle $B$.
    Similarly, $q = (x_q, V_q)$.

The distance is calculated as

$d_B(x, y) = \sqrt{d_M(x_p, x_q)^2 + d_F(V_p, V_{q←p})^2}$

where $d_\mathcal M$ is the distance on manifold $\mathcal M$, $d_F$ is the distance
between two vectors from the fiber $F$ and $V_{q←p}$ is the result
of parallel transport of vector $V_q$ to point $x_p$. The default
behavior of [`vector_transport_to`](@ref) is used to compute the vector
transport.
"""
function distance(B::VectorBundle, p, q)
    xp, Vp = submanifold_components(B.manifold, p)
    xq, Vq = submanifold_components(B.manifold, q)
    dist_man = distance(B.manifold, xp, xq)
    vy_x = vector_transport_to(B.manifold, xq, Vq, xp, B.vector_transport.method_point)
    dist_vec = distance(B.fiber, xp, Vp, vy_x)
    return sqrt(dist_man^2 + dist_vec^2)
end
"""
    distance(M::TangentSpaceAtPoint, p, q)

Distance between vectors `p` and `q` from the vector space `M`. It is calculated as the norm
of their difference.
"""
function distance(M::TangentSpaceAtPoint, p, q)
    return norm(M.fiber.manifold, M.point, q - p)
end

function embed!(M::TangentSpaceAtPoint, q, p)
    return embed!(M.fiber.manifold, q, M.point, p)
end
function embed!(M::TangentSpaceAtPoint, Y, p, X)
    return embed!(M.fiber.manifold, Y, M.point, X)
end

@doc raw"""
    exp(B::VectorBundle, p, X)

Exponential map of tangent vector $X$ at point $p$ from
vector bundle `B` over manifold `B.fiber` (denoted $\mathcal M$).

Notation:
  * The point $p = (x_p, V_p)$ where $x_p ∈ \mathcal M$ and $V_p$ belongs to the
    fiber $F=π^{-1}(\{x_p\})$ of the vector bundle $B$ where $π$ is the
    canonical projection of that vector bundle $B$.
  * The tangent vector $X = (V_{X,M}, V_{X,F}) ∈ T_pB$ where
    $V_{X,M}$ is a tangent vector from the tangent space $T_{x_p}\mathcal M$ and
    $V_{X,F}$ is a tangent vector from the tangent space $T_{V_p}F$ (isomorphic to $F$).

The exponential map is calculated as

$\exp_p(X) = (\exp_{x_p}(V_{X,M}), V_{\exp})$

where $V_{\exp}$ is the result of vector transport of $V_p + V_{X,F}$
to the point $\exp_{x_p}(V_{X,M})$.
The sum $V_p + V_{X,F}$ corresponds to the exponential map in the vector space $F$.
"""
exp(::VectorBundle, ::Any, ::Any)

@doc raw"""
    exp(M::TangentSpaceAtPoint, p, X)

Exponential map of tangent vectors `X` and `p` from the tangent space `M`. It is
calculated as their sum.
"""
exp(::TangentSpaceAtPoint, ::Any, ::Any)

function exp!(B::VectorBundle, q, p, X)
    xp, Xp = submanifold_components(B.manifold, p)
    xq, Xq = submanifold_components(B.manifold, q)
    VXM, VXF = submanifold_components(B.manifold, X)
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
function exp!(M::TangentSpaceAtPoint, q, p, X)
    copyto!(q, p + X)
    return q
end

function get_basis(M::VectorBundle, p, B::AbstractBasis)
    xp1 = submanifold_component(p, Val(1))
    base_basis = get_basis(M.manifold, xp1, B)
    vec_basis = get_basis(M.fiber, xp1, B)
    return CachedBasis(B, VectorBundleBasisData(base_basis, vec_basis))
end
function get_basis(M::VectorBundle, p, B::CachedBasis)
    return invoke(get_basis, Tuple{AbstractManifold,Any,CachedBasis}, M, p, B)
end
function get_basis(M::TangentSpaceAtPoint, p, B::CachedBasis)
    return invoke(
        get_basis,
        Tuple{AbstractManifold,Any,CachedBasis},
        M.fiber.manifold,
        M.point,
        B,
    )
end

function get_basis(M::VectorBundle, p, B::DiagonalizingOrthonormalBasis)
    xp1 = submanifold_component(p, Val(1))
    bv1 = DiagonalizingOrthonormalBasis(submanifold_component(B.frame_direction, Val(1)))
    b1 = get_basis(M.manifold, xp1, bv1)
    bv2 = DiagonalizingOrthonormalBasis(submanifold_component(B.frame_direction, Val(2)))
    b2 = get_basis(M.fiber, xp1, bv2)
    return CachedBasis(B, VectorBundleBasisData(b1, b2))
end

for BT in [
    DefaultOrthonormalBasis,
    DefaultOrthonormalBasis{<:Any,TangentSpaceType},
    ProjectedOrthonormalBasis{:gram_schmidt,ℝ},
    ProjectedOrthonormalBasis{:svd,ℝ},
]
    eval(quote
        @invoke_maker 3 AbstractBasis get_basis(M::VectorBundle, p, B::$BT)
    end)
    eval(
        quote
            @invoke_maker 3 AbstractBasis{<:Any,TangentSpaceType} get_basis(
                M::TangentSpaceAtPoint,
                p,
                B::$BT,
            )
        end,
    )
end
function get_basis(M::TangentBundleFibers, p, B::AbstractBasis{<:Any,TangentSpaceType})
    return get_basis(M.manifold, p, B)
end
function get_basis(M::TangentSpaceAtPoint, p, B::AbstractBasis{<:Any,TangentSpaceType})
    return get_basis(M.fiber.manifold, M.point, B)
end

function get_coordinates!(M::VectorBundle, Y, p, X, B::AbstractBasis)
    px, Vx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    n = manifold_dimension(M.manifold)
    get_coordinates!(M.manifold, view(Y, 1:n), px, VXM, B)
    get_coordinates!(M.fiber, view(Y, (n + 1):length(Y)), px, VXF, B)
    return Y
end
function get_coordinates!(
    M::VectorBundle,
    Y,
    p,
    X,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:VectorBundleBasisData},
) where {𝔽}
    px, Vx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    n = manifold_dimension(M.manifold)
    get_coordinates!(M.manifold, view(Y, 1:n), px, VXM, B.data.base_basis)
    get_coordinates!(M.fiber, view(Y, (n + 1):length(Y)), px, VXF, B.data.vec_basis)
    return Y
end
for BT in [
    DefaultBasis,
    DefaultOrthogonalBasis,
    DefaultOrthonormalBasis,
    ProjectedOrthonormalBasis{:gram_schmidt,ℝ},
    ProjectedOrthonormalBasis{:svd,ℝ},
    VeeOrthogonalBasis,
]
    eval(
        quote
            @invoke_maker 5 AbstractBasis get_coordinates!(
                M::VectorBundle,
                Y,
                p,
                X,
                B::$BT,
            )
        end,
    )
    eval(
        quote
            @invoke_maker 5 AbstractBasis{<:Any,TangentSpaceType} get_coordinates!(
                M::TangentSpaceAtPoint,
                Y,
                p,
                X,
                B::$BT,
            )
        end,
    )
end
function get_coordinates!(M::VectorBundle, Y, p, X, B::CachedBasis)
    return error(
        "get_coordinates! called on $M with an incorrect CachedBasis. Expected a CachedBasis with VectorBundleBasisData, given $B",
    )
end
function get_coordinates!(M::TangentSpaceAtPoint, Y, p, X, B::CachedBasis)
    return get_coordinates!(M.fiber.manifold, Y, M.point, X, B)
end

function get_coordinates!(
    M::TangentBundleFibers,
    Y,
    p,
    X,
    B::ManifoldsBase.all_uncached_bases{TangentSpaceType},
)
    return get_coordinates!(M.manifold, Y, p, X, B)
end
function get_coordinates!(
    M::TangentSpaceAtPoint,
    Y,
    p,
    X,
    B::ManifoldsBase.all_uncached_bases{TangentSpaceType},
)
    return get_coordinates!(M.fiber.manifold, Y, M.point, X, B)
end

function get_vector!(M::VectorBundle, Y, p, X, B::DefaultOrthonormalBasis)
    n = manifold_dimension(M.manifold)
    xp1 = submanifold_component(p, Val(1))
    get_vector!(M.manifold, submanifold_component(Y, Val(1)), xp1, X[1:n], B)
    get_vector!(M.fiber, submanifold_component(Y, Val(2)), xp1, X[(n + 1):end], B)
    return Y
end
function get_vector!(
    M::VectorBundle,
    Y,
    p,
    X,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:VectorBundleBasisData},
) where {𝔽}
    n = manifold_dimension(M.manifold)
    xp1 = submanifold_component(p, Val(1))
    get_vector!(
        M.manifold,
        submanifold_component(Y, Val(1)),
        xp1,
        X[1:n],
        B.data.base_basis,
    )
    get_vector!(
        M.fiber,
        submanifold_component(Y, Val(2)),
        xp1,
        X[(n + 1):end],
        B.data.vec_basis,
    )
    return Y
end
function get_vector!(
    M::TangentBundleFibers,
    Y,
    p,
    X,
    B::ManifoldsBase.all_uncached_bases{TangentSpaceType},
)
    return get_vector!(M.manifold, Y, p, X, B)
end
function get_vector!(
    M::TangentSpaceAtPoint,
    Y,
    p,
    X,
    B::ManifoldsBase.all_uncached_bases{TangentSpaceType},
)
    return get_vector!(M.fiber.manifold, Y, M.point, X, B)
end
for BT in [
    DefaultBasis,
    DefaultOrthogonalBasis,
    DefaultOrthonormalBasis,
    ProjectedOrthonormalBasis{:gram_schmidt,ℝ},
    ProjectedOrthonormalBasis{:svd,ℝ},
    VeeOrthogonalBasis,
]
    eval(
        quote
            @invoke_maker 5 AbstractBasis{<:Any,TangentSpaceType} get_vector!(
                M::TangentSpaceAtPoint,
                Y,
                p,
                X,
                B::$BT,
            )
        end,
    )
end
function get_vector!(M::TangentSpaceAtPoint, Y, p, X, B::CachedBasis)
    return get_vector!(M.fiber.manifold, Y, M.point, X, B)
end

function get_vectors(
    M::VectorBundle,
    p,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:VectorBundleBasisData},
) where {𝔽}
    xp1 = submanifold_component(p, Val(1))
    zero_m = zero_vector(M.manifold, xp1)
    zero_f = zero_vector(M.fiber, xp1)
    vs = typeof(ProductRepr(zero_m, zero_f))[]
    for bv in get_vectors(M.manifold, xp1, B.data.base_basis)
        push!(vs, ProductRepr(bv, zero_f))
    end
    for bv in get_vectors(M.fiber, xp1, B.data.vec_basis)
        push!(vs, ProductRepr(zero_m, bv))
    end
    return vs
end

function get_vectors(M::VectorBundleFibers, p, B::CachedBasis)
    return get_vectors(M.manifold, p, B)
end
function get_vectors(M::TangentSpaceAtPoint, p, B::CachedBasis)
    return get_vectors(M.fiber.manifold, M.point, B)
end

"""
    getindex(p::ProductRepr, M::VectorBundle, s::Symbol)
    p[M::VectorBundle, s]

Access the element(s) at index `s` of a point `p` on a [`VectorBundle`](@ref) `M` by
using the symbols `:point` and `:vector` for the base and vector component, respectively.
"""
@inline function Base.getindex(p::ProductRepr, M::VectorBundle, s::Symbol)
    (s === :point) && return submanifold_component(M, p, Val(1))
    (s === :vector) && return submanifold_component(M, p, Val(2))
    return throw(DomainError(s, "unknown component $s on $M."))
end

@doc raw"""
    injectivity_radius(M::TangentSpaceAtPoint)

Return the injectivity radius on the [`TangentSpaceAtPoint`](@ref) `M`, which is $∞$.
"""
injectivity_radius(::TangentSpaceAtPoint) = Inf

"""
    inner(B::VectorBundleFibers, p, X, Y)

Inner product of vectors `X` and `Y` from the vector space of type `B.fiber`
at point `p` from manifold `B.manifold`.
"""
function inner(B::VectorBundleFibers, p, X, Y)
    return error(
        "inner not defined for vector space family of type $(typeof(B)), " *
        "point of type $(typeof(p)) and " *
        "vectors of types $(typeof(X)) and $(typeof(Y)).",
    )
end
inner(B::VectorBundleFibers{<:TangentSpaceType}, p, X, Y) = inner(B.manifold, p, X, Y)
function inner(B::VectorBundleFibers{<:CotangentSpaceType}, p, X, Y)
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
function inner(B::VectorBundle, p, X, Y)
    px, Vx = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    VYM, VYF = submanifold_components(B.manifold, Y)
    # for tangent bundle Vx is discarded by the method of inner for TangentSpaceAtPoint
    # and px is actually used as the base point
    return inner(B.manifold, px, VXM, VYM) +
           inner(VectorSpaceAtPoint(B.fiber, px), Vx, VXF, VYF)
end

"""
    inner(M::TangentSpaceAtPoint, p, X, Y)

Inner product of vectors `X` and `Y` from the tangent space at `M`.
"""
function inner(M::TangentSpaceAtPoint, p, X, Y)
    return inner(M.fiber.manifold, M.point, X, Y)
end

function Base.isapprox(B::VectorBundle, p, q; kwargs...)
    xp, Vp = submanifold_components(B.manifold, p)
    xq, Vq = submanifold_components(B.manifold, q)
    return isapprox(B.manifold, xp, xq; kwargs...) &&
           isapprox(VectorSpaceAtPoint(B.fiber, xp), Vp, Vq; kwargs...)
end
function Base.isapprox(B::VectorBundle, p, X, Y; kwargs...)
    px, Vx = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    VYM, VYF = submanifold_components(B.manifold, Y)
    return isapprox(B.manifold, VXM, VYM; kwargs...) &&
           isapprox(VectorSpaceAtPoint(B.fiber, px), VXF, VYF; kwargs...)
end
function Base.isapprox(M::TangentSpaceAtPoint, X, Y; kwargs...)
    return isapprox(M.fiber.manifold, M.point, X, Y; kwargs...)
end

@doc raw"""
    log(B::VectorBundle, p, q)

Logarithmic map of the point `y` at point `p` from
vector bundle `B` over manifold `B.fiber` (denoted $\mathcal M$).

Notation:
  * The point $p = (x_p, V_p)$ where $x_p ∈ \mathcal M$ and $V_p$ belongs to the
    fiber $F=π^{-1}(\{x_p\})$ of the vector bundle $B$ where $π$ is the
    canonical projection of that vector bundle $B$.
    Similarly, $q = (x_q, V_q)$.

The logarithmic map is calculated as

$\log_p q = (\log_{x_p}(x_q), V_{\log} - V_p)$

where $V_{\log}$ is the result of vector transport of $V_q$ to the point $x_p$.
The difference $V_{\log} - V_p$ corresponds to the logarithmic map in the vector space $F$.
"""
log(::VectorBundle, ::Any...)
"""
    log(M::TangentSpaceAtPoint, p, q)

Logarithmic map on the tangent space manifold `M`, calculated as the difference of tangent
vectors `q` and `p` from `M`.
"""
log(::TangentSpaceAtPoint, ::Any...)

function log!(B::VectorBundle, X, p, q)
    px, Vx = submanifold_components(B.manifold, p)
    py, Vy = submanifold_components(B.manifold, q)
    VXM, VXF = submanifold_components(B.manifold, X)
    log!(B.manifold, VXM, px, py)
    vector_transport_to!(B.manifold, VXF, py, Vy, px, B.vector_transport.method_vector)
    copyto!(VXF, VXF - Vx)
    return X
end
function log!(::TangentSpaceAtPoint, X, p, q)
    copyto!(X, q - p)
    return X
end

function manifold_dimension(B::VectorBundle)
    return manifold_dimension(B.manifold) + vector_space_dimension(B.fiber)
end
function manifold_dimension(M::VectorSpaceAtPoint)
    return vector_space_dimension(M.fiber)
end

"""
    norm(B::VectorBundleFibers, p, q)

Norm of the vector `X` from the vector space of type `B.fiber`
at point `p` from manifold `B.manifold`.
"""
LinearAlgebra.norm(B::VectorBundleFibers, p, X) = sqrt(inner(B, p, X, X))
LinearAlgebra.norm(B::VectorBundleFibers{<:TangentSpaceType}, p, X) = norm(B.manifold, p, X)
LinearAlgebra.norm(M::VectorSpaceAtPoint, p, X) = norm(M.fiber.manifold, M.point, X)

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

@doc raw"""
    project(M::TangentSpaceAtPoint, p)

Project the point `p` from the tangent space `M`, that is project the vector `p`
tangent at `M.point`.
"""
project(::TangentSpaceAtPoint, ::Any)

function project!(B::VectorBundle, q, p)
    px, pVx = submanifold_components(B.manifold, p)
    qx, qVx = submanifold_components(B.manifold, q)
    project!(B.manifold, qx, px)
    project!(B.manifold, qVx, qx, pVx)
    return q
end
function project!(M::TangentSpaceAtPoint, q, p)
    return project!(M.fiber.manifold, q, M.point, p)
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
@doc raw"""
    project(M::TangentSpaceAtPoint, p, X)

Project the vector `X` from the tangent space `M`, that is project the vector `X`
tangent at `M.point`.
"""
project(::TangentSpaceAtPoint, ::Any, ::Any)

function project!(B::VectorBundle, Y, p, X)
    px, Vx = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    VYM, VYF = submanifold_components(B.manifold, Y)
    project!(B.manifold, VYM, px, VXM)
    project!(B.manifold, VYF, px, VXF)
    return Y
end
function project!(M::TangentSpaceAtPoint, Y, p, X)
    return project!(M.fiber.manifold, Y, M.point, X)
end

"""
    project(B::VectorBundleFibers, p, X)

Project vector `X` from the vector space of type `B.fiber` at point `p`.
"""
function project(B::VectorBundleFibers, p, X)
    Y = allocate_result(B, project, p, X)
    return project!(B, Y, p, X)
end

function project!(B::VectorBundleFibers{<:TangentSpaceType}, Y, p, X)
    return project!(B.manifold, Y, p, X)
end
function project!(B::VectorBundleFibers, Y, p, X)
    return error(
        "project! not implemented for vector space family of type $(typeof(B)), output vector of type $(typeof(Y)) and input vector at point $(typeof(p)) with type of w $(typeof(X)).",
    )
end

function Random.rand!(M::VectorBundle, pX; vector_at=nothing)
    pXM, pXF = submanifold_components(M.manifold, pX)
    if vector_at === nothing
        rand!(M.manifold, pXM)
        rand!(M.manifold, pXF; vector_at=pXM)
    else
        vector_atM, vector_atF = submanifold_components(M.manifold, vector_at)
        rand!(M.manifold, pXM; vector_at=vector_atM)
        rand!(M.manifold, pXF; vector_at=vector_atM)
    end
    return pX
end
function Random.rand!(rng::AbstractRNG, M::VectorBundle, pX; vector_at=nothing)
    pXM, pXF = submanifold_components(M.manifold, pX)
    if vector_at === nothing
        rand!(rng, M.manifold, pXM)
        rand!(rng, M.manifold, pXF; vector_at=pXM)
    else
        vector_atM, vector_atF = submanifold_components(M.manifold, vector_at)
        rand!(rng, M.manifold, pXM; vector_at=vector_atM)
        rand!(rng, M.manifold, pXF; vector_at=vector_atM)
    end
    return pX
end
function Random.rand!(M::TangentSpaceAtPoint, X; vector_at=nothing)
    rand!(M.fiber.manifold, X; vector_at=M.point)
    return X
end
function Random.rand!(rng::AbstractRNG, M::TangentSpaceAtPoint, X; vector_at=nothing)
    rand!(rng, M.fiber.manifold, X; vector_at=M.point)
    return X
end

"""
    setindex!(p::ProductRepr, val, M::VectorBundle, s::Symbol)
    p[M::VectorBundle, s] = val

Set the element(s) at index `s` of a point `p` on a [`VectorBundle`](@ref) `M` to `val` by
using the symbols `:point` and `:vector` for the base and vector component, respectively.

!!! note

    The *content* of element of `p` is replaced, not the element itself.
"""
@inline function Base.setindex!(x::ProductRepr, val, M::VectorBundle, s::Symbol)
    if s === :point
        return copyto!(submanifold_component(M, x, Val(1)), val)
    elseif s === :vector
        return copyto!(submanifold_component(M, x, Val(2)), val)
    else
        throw(DomainError(s, "unknown component $s on $M."))
    end
end

function representation_size(B::VectorBundleFibers{<:TCoTSpaceType})
    return representation_size(B.manifold)
end
function representation_size(B::VectorBundle)
    len_manifold = prod(representation_size(B.manifold))
    len_vs = prod(representation_size(B.fiber))
    return (len_manifold + len_vs,)
end
function representation_size(B::TangentSpaceAtPoint)
    return representation_size(B.fiber.manifold)
end

function Base.show(io::IO, tpt::TensorProductType)
    return print(io, "TensorProductType(", join(tpt.spaces, ", "), ")")
end
function Base.show(io::IO, fiber::VectorBundleFibers)
    return print(io, "VectorBundleFibers($(fiber.fiber), $(fiber.manifold))")
end
function Base.show(io::IO, ::MIME"text/plain", vs::VectorSpaceAtPoint)
    summary(io, vs)
    println(io, "\nFiber:")
    pre = " "
    sf = sprint(show, "text/plain", vs.fiber; context=io, sizehint=0)
    sf = replace(sf, '\n' => "\n$(pre)")
    println(io, pre, sf)
    println(io, "Base point:")
    sp = sprint(show, "text/plain", vs.point; context=io, sizehint=0)
    sp = replace(sp, '\n' => "\n$(pre)")
    return print(io, pre, sp)
end
function Base.show(io::IO, ::MIME"text/plain", TpM::TangentSpaceAtPoint)
    println(io, "Tangent space to the manifold $(base_manifold(TpM)) at point:")
    pre = " "
    sp = sprint(show, "text/plain", TpM.point; context=io, sizehint=0)
    sp = replace(sp, '\n' => "\n$(pre)")
    return print(io, pre, sp)
end
Base.show(io::IO, vb::VectorBundle) = print(io, "VectorBundle($(vb.type), $(vb.manifold))")
Base.show(io::IO, vb::TangentBundle) = print(io, "TangentBundle($(vb.manifold))")
Base.show(io::IO, vb::CotangentBundle) = print(io, "CotangentBundle($(vb.manifold))")

"""
    allocate_result(B::VectorBundleFibers, f, x...)

Allocates an array for the result of function `f` that is
an element of the vector space of type `B.fiber` on manifold `B.manifold`
and arguments `x...` for implementing the non-modifying operation
using the modifying operation.
"""
@inline function allocate_result(B::VectorBundleFibers, f::TF, x...) where {TF}
    if length(x) == 0
        # TODO: this may be incorrect when point and tangent vector representation are
        #       different for the manifold but there is no easy and generic way around that
        return allocate_result(B.manifold, f)
    else
        T = allocate_result_type(B, f, x)
        return allocate(x[1], T)
    end
end
@inline function allocate_result(M::VectorBundle, f::TF) where {TF}
    return ProductRepr(allocate_result(M.manifold, f), allocate_result(M.fiber, f))
end

"""
    allocate_result_type(B::VectorBundleFibers, f, args::NTuple{N,Any}) where N

Return type of element of the array that will represent the result of
function `f` for representing an operation with result in the vector space `fiber`
for manifold `M` on given arguments (passed at a tuple).
"""
@inline function allocate_result_type(
    ::VectorBundleFibers,
    f::TF,
    args::NTuple{N,Any},
) where {TF,N}
    return typeof(mapreduce(eti -> one(number_eltype(eti)), +, args))
end

"""
    vector_bundle_transport(fiber::VectorSpaceType, M::AbstractManifold)

Determine the vector tranport used for [`exp`](@ref exp(::VectorBundle, ::Any...)) and
[`log`](@ref log(::VectorBundle, ::Any...)) maps on a vector bundle with vector space type
`fiber` and manifold `M`.
"""
vector_bundle_transport(::VectorSpaceType, ::AbstractManifold) = ParallelTransport()

function vector_space_dimension(B::VectorBundleFibers)
    return vector_space_dimension(B.manifold, B.fiber)
end

function vector_space_dimension(M::AbstractManifold, V::TensorProductType)
    dim = 1
    for space in V.spaces
        dim *= vector_space_dimension(M, space)
    end
    return dim
end

function vector_transport_direction(M::VectorBundle, p, X, d)
    return vector_transport_direction(M, p, X, d, M.vector_transport)
end

function vector_transport_direction!(M::VectorBundle, Y, p, X, d)
    return vector_transport_direction!(M, Y, p, X, d, M.vector_transport)
end

@doc raw"""
    vector_transport_to(M::VectorBundle, p, X, q, m::VectorBundleVectorTransport)

Compute the vector transport the tangent vector `X`at `p` to `q` on the
[`VectorBundle`](@ref) `M` using the [`VectorBundleVectorTransport`](@ref) `m`.
"""
vector_transport_to(::VectorBundle, ::Any, ::Any, ::Any, ::VectorBundleVectorTransport)
function vector_transport_to(M::VectorBundle, p, X, q)
    return vector_transport_to(M, p, X, q, M.vector_transport)
end

function vector_transport_to!(M::VectorBundle, Y, p, X, q)
    return vector_transport_to!(M, Y, p, X, q, M.vector_transport)
end
function vector_transport_to!(M::TangentBundle, Y, p, X, q, m::VectorBundleVectorTransport)
    px, pVx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    VYM, VYF = submanifold_components(M.manifold, Y)
    qx, qVx = submanifold_components(M.manifold, q)
    vector_transport_to!(M.manifold, VYM, px, VXM, qx, m.method_point)
    vector_transport_to!(M.manifold, VYF, px, VXF, qx, m.method_vector)
    return Y
end
function vector_transport_to!(
    M::TangentBundle,
    Y,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod,
)
    return vector_transport_to!(M, Y, p, X, q, VectorBundleVectorTransport(m, m))
end
for VT in ManifoldsBase.VECTOR_TRANSPORT_DISAMBIGUATION
    eval(
        quote
            @invoke_maker 6 AbstractVectorTransportMethod vector_transport_to!(
                M::TangentBundle,
                Y,
                p,
                X,
                q,
                B::$VT,
            )
        end,
    )
end
function vector_transport_to!(M::TangentSpaceAtPoint, Y, p, X, q)
    return copyto!(Y, X)
end
function vector_transport_to!(M::TangentSpaceAtPoint, Y, p, X, q, ::ParallelTransport)
    return copyto!(Y, X)
end

"""
    zero_vector(B::VectorBundleFibers, p)

Compute the zero vector from the vector space of type `B.fiber` at point `p`
from manifold `B.manifold`.
"""
function zero_vector(B::VectorBundleFibers, p)
    X = allocate_result(B, zero_vector, p)
    return zero_vector!(B, X, p)
end

"""
    zero_vector!(B::VectorBundleFibers, X, p)

Save the zero vector from the vector space of type `B.fiber` at point `p`
from manifold `B.manifold` to `X`.
"""
function zero_vector!(B::VectorBundleFibers, X, p)
    return error(
        "zero_vector! not implemented for vector space family of type $(typeof(B)).",
    )
end
function zero_vector!(B::VectorBundleFibers{<:TangentSpaceType}, X, p)
    return zero_vector!(B.manifold, X, p)
end

@doc raw"""
    zero_vector(B::VectorBundle, p)

Zero tangent vector at point `p` from the vector bundle `B`
over manifold `B.fiber` (denoted $\mathcal M$). The zero vector belongs to the space $T_{p}B$

Notation:
  * The point $p = (x_p, V_p)$ where $x_p ∈ \mathcal M$ and $V_p$ belongs to the
    fiber $F=π^{-1}(\{x_p\})$ of the vector bundle $B$ where $π$ is the
    canonical projection of that vector bundle $B$.

The zero vector is calculated as

$\mathbf{0}_{p} = (\mathbf{0}_{x_p}, \mathbf{0}_F)$

where $\mathbf{0}_{x_p}$ is the zero tangent vector from $T_{x_p}\mathcal M$ and
$\mathbf{0}_F$ is the zero element of the vector space $F$.
"""
zero_vector(::VectorBundle, ::Any...)

@doc raw"""
    zero_vector(M::TangentSpaceAtPoint, p)

Zero tangent vector at point `p` from the tangent space `M`, that is the zero tangent vector
at point `M.point`.
"""
zero_vector(::TangentSpaceAtPoint, ::Any...)

function zero_vector!(B::VectorBundle, X, p)
    xp, Vp = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    zero_vector!(B.manifold, VXM, xp)
    zero_vector!(B.fiber, VXF, Vp)
    return X
end
function zero_vector!(M::TangentSpaceAtPoint, X, p)
    return zero_vector!(M.fiber.manifold, X, M.point)
end
