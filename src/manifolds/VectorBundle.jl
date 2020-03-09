"""
    VectorSpaceType

Abstract type for tangent spaces, cotangent spaces, their tensor products,
exterior products, etc.

Every vector space `fiber` is supposed to provide:
* a method of constructing vectors,
* basic operations: addition, subtraction, multiplication by a scalar
  and negation (unary minus),
* [`zero_vector!(fiber, X, p)`](@ref) to construct zero vectors at point `p`,
* `allocate(X)` and `allocate(X, T)` for vector `X` and type `T`,
* `copyto!(X, Y)` for vectors `X` and `Y`,
* `number_eltype(v)` for vector `v`,
* [`vector_space_dimension(::VectorBundleFibers{<:typeof(fiber)}) where fiber`](@ref).

Optionally:
* inner product via `inner` (used to provide Riemannian metric on vector
  bundles),
* [`flat`](@ref) and [`sharp`](@ref),
* `norm` (by default uses `inner`),
* [`project_vector`](@ref) (for embedded vector spaces),
* [`representation_size`](@ref) (if support for [`ProductArray`](@ref) is desired),
* broadcasting for basic operations.
"""
abstract type VectorSpaceType end

struct TangentSpaceType <: VectorSpaceType end

struct CotangentSpaceType <: VectorSpaceType end

TCoTSpaceType = Union{TangentSpaceType,CotangentSpaceType}

const TangentSpace = TangentSpaceType()
const CotangentSpace = CotangentSpaceType()

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
    VectorBundleFibers(fiber::VectorSpaceType, M::Manifold)

Type representing a family of vector spaces (fibers) of a vector bundle over `M`
with vector spaces of type `fiber`. In contrast with `VectorBundle`, operations
on `VectorBundleFibers` expect point-like and vector-like parts to be
passed separately instead of being bundled together. It can be thought of
as a representation of vector spaces from a vector bundle but without
storing the point at which a vector space is attached (which is specified
separately in various functions).
"""
struct VectorBundleFibers{TVS<:VectorSpaceType,TM<:Manifold}
    fiber::TVS
    manifold::TM
end

const TangentBundleFibers{M} = VectorBundleFibers{TangentSpaceType,M} where {M<:Manifold}

TangentBundleFibers(M::Manifold) = VectorBundleFibers(TangentSpace, M)

const CotangentBundleFibers{M} =
    VectorBundleFibers{CotangentSpaceType,M} where {M<:Manifold}

CotangentBundleFibers(M::Manifold) = VectorBundleFibers(CotangentSpace, M)

"""
    VectorSpaceAtPoint(fiber::VectorBundleFibers, p)

A vector space (fiber type `fiber` of a vector bundle) at point `p` from
the manifold `fiber.manifold`.
"""
struct VectorSpaceAtPoint{TFiber<:VectorBundleFibers,TX}
    fiber::TFiber
    point::TX
end

"""
    TangentSpaceAtPoint(M::Manifold, p)

Return an object of type [`VectorSpaceAtPoint`](@ref) representing tangent
space at `p`.
"""
TangentSpaceAtPoint(M::Manifold, p) = VectorSpaceAtPoint(TangentBundleFibers(M), p)

"""
    CotangentSpaceAtPoint(M::Manifold, p)

Return an object of type [`VectorSpaceAtPoint`](@ref) representing cotangent
space at `p`.
"""
CotangentSpaceAtPoint(M::Manifold, p) = VectorSpaceAtPoint(CotangentBundleFibers(M), p)

"""
    VectorBundle(M::Manifold, type::VectorSpaceType)

Vector bundle on manifold `M` of type `type`.
"""
struct VectorBundle{TVS<:VectorSpaceType,TM<:Manifold} <: Manifold
    type::TVS
    manifold::TM
    fiber::VectorBundleFibers{TVS,TM}
end

function VectorBundle(fiber::TVS, M::TM) where {TVS<:VectorSpaceType,TM<:Manifold}
    return VectorBundle{TVS,TM}(fiber, M, VectorBundleFibers(fiber, M))
end

const TangentBundle{M} = VectorBundle{TangentSpaceType,M} where {M<:Manifold}

TangentBundle(M::Manifold) = VectorBundle(TangentSpace, M)

const CotangentBundle{M} = VectorBundle{CotangentSpaceType,M} where {M<:Manifold}

CotangentBundle(M::Manifold) = VectorBundle(CotangentSpace, M)

"""
    FVector(type::VectorSpaceType, data)

Decorator indicating that the vector `data` is from a fiber of a vector bundle
of type `type`.
"""
struct FVector{TType<:VectorSpaceType,TData}
    type::TType
    data::TData
end

const TFVector = FVector{TangentSpaceType}
const CoTFVector = FVector{CotangentSpaceType}

struct VectorBundleBasisData{BBasis<:CachedBasis, TBasis<:CachedBasis}
    base_basis::BBasis
    vec_basis::TBasis
end

(+)(X::FVector, Y::FVector) = FVector(X.type, X.data + Y.data)

(-)(X::FVector, Y::FVector) = FVector(X.type, X.data - Y.data)
(-)(X::FVector) = FVector(X.type, -X.data)

(*)(a::Number, X::FVector) = FVector(X.type, a * X.data)

function copyto!(X::FVector, Y::FVector)
    copyto!(X.data, Y.data)
    return X
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
    vy_x = vector_transport_to(B.manifold, xq, Vq, xp)
    dist_vec = distance(B.fiber, xp, Vp, vy_x)
    return sqrt(dist_man^2 + dist_vec^2)
end

function number_eltype(::Type{FVector{TType,TData}}) where {TType<:VectorSpaceType,TData}
    return number_eltype(TData)
end
number_eltype(v::FVector) = number_eltype(v.data)

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
exp(::VectorBundle, ::Any)

function exp!(B::VectorBundle, q, p, X)
    xp, Xp = submanifold_components(B.manifold, p)
    xq, Xq = submanifold_components(B.manifold, q)
    VXM, VXF = submanifold_components(B.manifold, X)
    exp!(B.manifold, xq, xp, VXM)
    vector_transport_to!(B.manifold, Xq, xp, Xp + VXF, xq)
    return q
end

@doc raw"""
    flat(M::Manifold, p, X::FVector)

Compute the flat isomorphism (one of the musical isomorphisms) of tangent vector `X`
from the vector space of type `M` at point `p` from the underlying [`Manifold`](@ref).

The function can be used for example to transform vectors
from the tangent bundle to vectors from the cotangent bundle
$♭ : T\mathcal M → T^{*}\mathcal M$
"""
function flat(M::Manifold, p, X::FVector)
    ξ = allocate_result(M, flat, X, p)
    return flat!(M, ξ, p, X)
end

function flat!(M::Manifold, ξ::FVector, p, X::FVector)
    error(
        "flat! not implemented for vector bundle fibers space " *
        "of type $(typeof(M)), vector of type $(typeof(ξ)), point of " *
        "type $(typeof(p)) and vector of type $(typeof(X)).",
    )
end

@decorator_transparent_signature flat!(
    M::AbstractDecoratorManifold,
    ξ::CoTFVector,
    p,
    X::TFVector,
)

function get_basis(M::VectorBundle, p, B::AbstractBasis)
    xp1 = submanifold_component(p, Val(1))
    base_basis = get_basis(M.manifold, xp1, B)
    vec_basis = get_basis(M.fiber, xp1, B)
    return CachedBasis(B, VectorBundleBasisData(base_basis, vec_basis))
end
function get_basis(M::VectorBundle, p, B::CachedBasis)
    return invoke(get_basis, Tuple{Manifold, Any, CachedBasis}, M, p, B)
end

function get_basis(M::VectorBundle, p, B::DiagonalizingOrthonormalBasis)
    xp1 = submanifold_component(p, Val(1))
    bv1 = DiagonalizingOrthonormalBasis(submanifold_component(B.frame_direction, Val(1)))
    b1 = get_basis(M.manifold, xp1, bv1)
    bv2 = DiagonalizingOrthonormalBasis(submanifold_component(B.frame_direction, Val(2)))
    b2 = get_basis(M.fiber, xp1, bv2)
    return CachedBasis(B, VectorBundleBasisData(b1, b2))
end
function get_basis(M::VectorBundle, p, B::DefaultOrthonormalBasis)
    return invoke(get_basis, Tuple{VectorBundle,Any,AbstractBasis}, M, p, B)
end
function get_basis(M::TangentBundleFibers, p, B::AbstractBasis)
    return get_basis(M.manifold, p, B)
end

function get_coordinates(M::VectorBundle, p, X, B::AbstractBasis) where {N}
    px, Vx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    coord1 = get_coordinates(M.manifold, px, VXM, B)
    coord2 = get_coordinates(M.fiber, px, VXF, B)
    return vcat(coord1, coord2)
end
function get_coordinates(
    M::VectorBundle,
    p,
    X,
    B::CachedBasis{<:AbstractBasis,<:VectorBundleBasisData},
) where {N}
     px, Vx = submanifold_components(M.manifold, p)
     VXM, VXF = submanifold_components(M.manifold, X)
     coord1 = get_coordinates(M.manifold, px, VXM, B.data.base_basis)
     coord2 = get_coordinates(M.fiber, px, VXF, B.data.vec_basis)
     return vcat(coord1, coord2)
 end
function get_coordinates(M::TangentBundleFibers, p, X, B::all_uncached_bases) where {N}
    return get_coordinates(M.manifold, p, X, B)
end

function get_coordinates!(M::VectorBundle, Y, p, X, B::all_uncached_bases) where {N}
    px, Vx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    n = manifold_dimension(M.manifold)
    get_coordinates!(M.manifold, view(Y, 1:n), px, VXM, B)
    get_coordinates!(M.fiber, view(Y, n+1:length(Y)), px, VXF, B)
    return Y
end
function get_coordinates!(
    M::VectorBundle,
    Y,
    p,
    X,
    B::CachedBasis{<:AbstractBasis,<:VectorBundleBasisData},
) where {N}
     px, Vx = submanifold_components(M.manifold, p)
     VXM, VXF = submanifold_components(M.manifold, X)
     n = manifold_dimension(M.manifold)
     get_coordinates!(M.manifold, view(Y, 1:n), px, VXM, B.data.base_basis)
     get_coordinates!(M.fiber, view(Y, n+1:length(Y)), px, VXF, B.data.vec_basis)
     return Y
 end
function get_coordinates!(M::TangentBundleFibers, Y, p, X, B::all_uncached_bases) where {N}
    return get_coordinates!(M.manifold, Y, p, X, B)
end
function get_coordinates!(M::VectorBundle, Y, p, X, B::DefaultBasis) where {N}
    return invoke(
        get_coordinates!,
        Tuple{VectorBundle,Any,Any,Any,AbstractBasis},
        M,
        Y,
        p,
        X,
        B,
    )
end

function get_vector(M::VectorBundle, p, X, B::DefaultOrthonormalBasis) where {N}
    n = manifold_dimension(M.manifold)
    xp1 = submanifold_component(p, Val(1))
    v1 = get_vector(M.manifold, xp1, X[1:n], B)
    v2 = get_vector(M.fiber, xp1, X[n+1:end], B)
    return ProductRepr(v1, v2)
end
function get_vector(
    M::VectorBundle,
    p,
    X,
    B::CachedBasis{VectorBundleBasisData},
) where {N}
    n = manifold_dimension(M.manifold)
    xp1 = submanifold_component(p, Val(1))
    v1 = get_vector(M.manifold, xp1, X[1:n], B.data.base_basis)
    v2 = get_vector(M.fiber, xp1, X[n+1:end], B.data.vec_basis)
    return ProductRepr(v1, v2)
end
function get_vector(M::TangentBundleFibers, p, X, B::all_uncached_bases) where {N}
    return get_vector(M.manifold, p, X, B)
end

function get_vector!(M::VectorBundle, Y, p, X, B::DefaultOrthonormalBasis) where {N}
    n = manifold_dimension(M.manifold)
    xp1 = submanifold_component(p, Val(1))
    get_vector!(M.manifold, submanifold_component(Y, Val(1)), xp1, X[1:n], B)
    get_vector!(M.fiber, submanifold_component(Y, Val(2)), xp1, X[n+1:end], B)
    return Y
end
function get_vector!(
    M::VectorBundle,
    Y,
    p,
    X,
    B::CachedBasis{VectorBundleBasisData},
) where {N}
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
        X[n+1:end],
        B.data.vec_basis,
    )
    return Y
end
function get_vector!(M::TangentBundleFibers, Y, p, X, B::all_uncached_bases) where {N}
    return get_vector!(M.manifold, Y, p, X, B)
end

function get_vectors(
    M::VectorBundle,
    p,
    B::CachedBasis{<:AbstractBasis,<:VectorBundleBasisData},
)
    xp1 = submanifold_component(p, Val(1))
    zero_m = zero_tangent_vector(M.manifold, xp1)
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
Base.@propagate_inbounds getindex(x::FVector, i) = getindex(x.data, i)

"""
    inner(B::VectorBundleFibers, p, X, Y)

Inner product of vectors `X` and `Y` from the vector space of type `B.fiber`
at point `p` from manifold `B.manifold`.
"""
function inner(B::VectorBundleFibers, p, X, Y)
    error(
        "inner not defined for vector space family of type $(typeof(B)), " *
        "point of type $(typeof(p)) and " *
        "vectors of types $(typeof(X)) and $(typeof(Y)).",
    )
end
inner(B::VectorBundleFibers{<:TangentSpaceType}, p, X, Y) = inner(B.manifold, p, X, Y)
function inner(B::VectorBundleFibers{<:CotangentSpaceType}, p, X, Y)
    return inner(
        B.manifold,
        p,
        sharp(B.manifold, p, FVector(CotangentSpace, X)).data,
        sharp(B.manifold, p, FVector(CotangentSpace, Y)).data,
    )
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
    return inner(B.manifold, px, VXM, VYM) + inner(B.fiber, Vx, VXF, VYF)
end

function isapprox(B::VectorBundle, p, q; kwargs...)
    xp, Vp = submanifold_components(B.manifold, p)
    xq, Vq = submanifold_components(B.manifold, q)
    return isapprox(B.manifold, xp, xq; kwargs...) && isapprox(Vp, Vq; kwargs...)
end
function isapprox(B::VectorBundle, p, X, Y; kwargs...)
    px, Vx = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    VYM, VYF = submanifold_components(B.manifold, Y)
    return isapprox(B.manifold, VXM, VYM; kwargs...) &&
           isapprox(B.manifold, px, VXF, VYF; kwargs...)
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

function log!(B::VectorBundle, X, p, q)
    px, Vx = submanifold_components(B.manifold, p)
    py, Vy = submanifold_components(B.manifold, q)
    VXM, VXF = submanifold_components(B.manifold, X)
    log!(B.manifold, VXM, px, py)
    vector_transport_to!(B.manifold, VXF, py, Vy, px)
    copyto!(VXF, VXF - Vx)
    return X
end

function manifold_dimension(B::VectorBundle)
    return manifold_dimension(B.manifold) + vector_space_dimension(B.fiber)
end

"""
    norm(B::VectorBundleFibers, p, q)

Norm of the vector `X` from the vector space of type `B.fiber`
at point `p` from manifold `B.manifold`.
"""
norm(B::VectorBundleFibers, p, X) = sqrt(inner(B, p, X, X))
norm(B::VectorBundleFibers{<:TangentSpaceType}, p, X) = norm(B.manifold, p, X)

@doc raw"""
    project_point(B::VectorBundle, p)

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
project_point(::VectorBundle, ::Any...)

function project_point!(B::VectorBundle, q, p)
    px, pVx = submanifold_components(B.manifold, p)
    qx, qVx = submanifold_components(B.manifold, q)
    project_point!(B.manifold, qx, px)
    project_tangent!(B.manifold, qVx, qx, pVx)
    return q
end

@doc raw"""
    project_tangent(B::VectorBundle, p, X)

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
project_tangent(::VectorBundle, ::Any...)

function project_tangent!(B::VectorBundle, Y, p, X)
    px, Vx = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    VYM, VYF = submanifold_components(B.manifold, Y)
    project_tangent!(B.manifold, VYM, px, VXM)
    project_tangent!(B.manifold, VYF, px, VXF)
    return Y
end

"""
    project_vector(B::VectorBundleFibers, p, X)

Project vector `X` from the vector space of type `B.fiber` at point `p`.
"""
function project_vector(B::VectorBundleFibers, p, X)
    Y = allocate_result(B, project_vector, p, X)
    return project_vector!(B, Y, p, X)
end

function project_vector!(B::VectorBundleFibers{<:TangentSpaceType}, Y, p, X)
    return project_tangent!(B.manifold, Y, p, X)
end
function project_vector!(B::VectorBundleFibers, Y, p, X)
    error("project_vector! not implemented for vector space family of type $(typeof(B)), output vector of type $(typeof(Y)) and input vector at point $(typeof(p)) with type of w $(typeof(X)).")
end

Base.@propagate_inbounds setindex!(x::FVector, val, i) = setindex!(x.data, val, i)

function representation_size(B::VectorBundleFibers{<:TCoTSpaceType})
    return representation_size(B.manifold)
end
function representation_size(B::VectorBundle)
    len_manifold = prod(representation_size(B.manifold))
    len_vs = prod(representation_size(B.fiber))
    return (len_manifold + len_vs,)
end

@doc raw"""
    sharp(M::Manifold, p, ξ::FVector)

Compute the sharp isomorphism (one of the musical isomorphisms) of vector `ξ`
from the vector space `M` at point `p` from the underlying [`Manifold`](@ref).

The function can be used for example to transform vectors
from the cotangent bundle to vectors from the tangent bundle
$♯ : T^{*}\mathcal M → T\mathcal M$
"""
function sharp(M::Manifold, p, ξ::FVector)
    X = allocate_result(M, sharp, ξ, p)
    return sharp!(M, X, p, ξ)
end

function sharp!(M::Manifold, X::FVector, p, ξ::FVector)
    error(
        "sharp! not implemented for vector bundle fibers space " *
        "of type $(typeof(M)), vector of type $(typeof(X)), point of " *
        "type $(typeof(p)) and vector of type $(typeof(ξ)).",
    )
end

@decorator_transparent_signature sharp!(
    M::AbstractDecoratorManifold,
    X::TFVector,
    p,
    ξ::CoTFVector,
)

show(io::IO, ::TangentSpaceType) = print(io, "TangentSpace")
show(io::IO, ::CotangentSpaceType) = print(io, "CotangentSpace")
function show(io::IO, tpt::TensorProductType)
    print(io, "TensorProductType(", join(tpt.spaces, ", "), ")")
end
function show(io::IO, fiber::VectorBundleFibers)
    print(io, "VectorBundleFibers($(fiber.fiber), $(fiber.manifold))")
end
function show(io::IO, mime::MIME"text/plain", vs::VectorSpaceAtPoint)
    summary(io, vs)
    println(io, "\nFiber:")
    pre = " "
    sf = sprint(show, "text/plain", vs.fiber; context = io, sizehint = 0)
    sf = replace(sf, '\n' => "\n$(pre)")
    println(io, pre, sf)
    println(io, "Base point:")
    sp = sprint(show, "text/plain", vs.point; context = io, sizehint = 0)
    sp = replace(sp, '\n' => "\n$(pre)")
    print(io, pre, sp)
end
show(io::IO, vb::VectorBundle) = print(io, "VectorBundle($(vb.type), $(vb.manifold))")
show(io::IO, vb::TangentBundle) = print(io, "TangentBundle($(vb.manifold))")
show(io::IO, vb::CotangentBundle) = print(io, "CotangentBundle($(vb.manifold))")

allocate(x::FVector) = FVector(x.type, allocate(x.data))
allocate(x::FVector, ::Type{T}) where {T} = FVector(x.type, allocate(x.data, T))

"""
    allocate_result(B::VectorBundleFibers, f, x...)

Allocates an array for the result of function `f` that is
an element of the vector space of type `B.fiber` on manifold `B.manifold`
and arguments `x...` for implementing the non-modifying operation
using the modifying operation.
"""
function allocate_result(B::VectorBundleFibers, f, x...)
    T = allocate_result_type(B, f, x)
    return allocate(x[1], T)
end
function allocate_result(M::Manifold, ::typeof(flat), w::TFVector, x)
    return FVector(CotangentSpace, allocate(w.data))
end
function allocate_result(M::Manifold, ::typeof(sharp), w::CoTFVector, x)
    return FVector(TangentSpace, allocate(w.data))
end

"""
    allocate_result_type(B::VectorBundleFibers, f, args::NTuple{N,Any}) where N

Returns type of element of the array that will represent the result of
function `f` for representing an operation with result in the vector space `fiber`
for manifold `M` on given arguments (passed at a tuple).
"""
function allocate_result_type(B::VectorBundleFibers, f, args::NTuple{N,Any}) where {N}
    T = typeof(reduce(+, one(number_eltype(eti)) for eti ∈ args))
    return T
end

size(x::FVector) = size(x.data)

function submanifold_component(M::Manifold, x::FVector, i::Val)
    return submanifold_component(M, x.data, i)
end
submanifold_component(x::FVector, i::Val) = submanifold_component(x.data, i)

submanifold_components(M::Manifold, x::FVector) = submanifold_components(M, x.data)
submanifold_components(x::FVector) = submanifold_components(x.data)

"""
    vector_space_dimension(B::VectorBundleFibers)

Dimension of the vector space of type `B`.
"""
function vector_space_dimension(B::VectorBundleFibers)
    error("vector_space_dimension not implemented for vector space family $(typeof(B)).")
end
function vector_space_dimension(B::VectorBundleFibers{<:TCoTSpaceType})
    return manifold_dimension(B.manifold)
end
function vector_space_dimension(B::VectorBundleFibers{<:TensorProductType})
    dim = 1
    for space in B.fiber.spaces
        dim *= vector_space_dimension(VectorBundleFibers(space, B.manifold))
    end
    return dim
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
    error("zero_vector! not implemented for vector space family of type $(typeof(B)).")
end
function zero_vector!(B::VectorBundleFibers{<:TangentSpaceType}, X, p)
    return zero_tangent_vector!(B.manifold, X, p)
end

@doc raw"""
    zero_tangent_vector(B::VectorBundle, p)

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
zero_tangent_vector(::VectorBundle, ::Any...)

function zero_tangent_vector!(B::VectorBundle, X, p)
    xp, Vp = submanifold_components(B.manifold, p)
    VXM, VXF = submanifold_components(B.manifold, X)
    zero_tangent_vector!(B.manifold, VXM, xp)
    zero_vector!(B.fiber, VXF, Vp)
    return X
end
