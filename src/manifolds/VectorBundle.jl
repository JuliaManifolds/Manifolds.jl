
"""
    VectorSpaceType

Abstract type for tangent spaces, cotangent spaces, their tensor products,
exterior products, etc.

Every vector space `VS` is supposed to provide:
* a method of constructing vectors,
* basic operations: addition, subtraction, multiplication by a scalar
  and negation (unary minus),
* [`zero_vector!(VS, v, x)`](@ref) to construct zero vectors at point `x`,
* `similar(v, T)` for vector `v`,
* `copyto!(v, w)` for vectors `v` and `w`,
* `eltype(v)` for vector `v`,
* [`vector_space_dimension(::VectorBundleFibers{<:typeof(VS)}) where VS`](@ref).

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

TCoTSpaceType = Union{TangentSpaceType, CotangentSpaceType}

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
    VectorBundleFibers(VS::VectorSpaceType, M::Manifold)

Type representing a family of vector spaces (fibers) of a vector bundle over `M`
with vector spaces of type `VS`. In contrast with `VectorBundle`, operations
on `VectorBundleFibers` expect point-like and vector-like parts to be
passed separately instead of being bundled together. It can be thought of
as a representation of vector spaces from a vector bundle but without
storing the point at which a vector space is attached (which is specified
separately in various functions).
"""
struct VectorBundleFibers{TVS<:VectorSpaceType, TM<:Manifold}
    VS::TVS
    M::TM
end
TangentBundleFibers{M} = VectorBundleFibers{TangentSpaceType,M}
TangentBundleFibers(M::Manifold) = VectorBundleFibers(TangentSpace, M)
CotangentBundleFibers{M} = VectorBundleFibers{CotangentSpaceType,M}
CotangentBundleFibers(M::Manifold) = VectorBundleFibers(CotangentSpace, M)

"""
    VectorSpaceAtPoint(fiber::VectorBundleFibers, x)

A vector space (fiber type `fiber` of a vector bundle) at point `x` from
the manifold `fiber.M`.
"""
struct VectorSpaceAtPoint{TFiber<:VectorBundleFibers, TX}
    fiber::TFiber
    x::TX
end

"""
    TangentSpaceAtPoint(M::Manifold, x)

Return an object of type [`VectorSpaceAtPoint`](@ref) representing tangent
space at `x`.
"""
function TangentSpaceAtPoint(M::Manifold, x)
    return VectorSpaceAtPoint(TangentBundleFibers(M), x)
end

"""
    CotangentSpaceAtPoint(M::Manifold, x)

Return an object of type [`VectorSpaceAtPoint`](@ref) representing cotangent
space at `x`.
"""
function CotangentSpaceAtPoint(M::Manifold, x)
    return VectorSpaceAtPoint(CotangentBundleFibers(M), x)
end

"""
    VectorBundle(M::Manifold, type::VectorSpaceType)

Vector bundle on manifold `M` of type `type`.
"""
struct VectorBundle{TVS<:VectorSpaceType, TM<:Manifold} <: Manifold
    type::TVS
    M::TM
    VS::VectorBundleFibers{TVS, TM}
end
function VectorBundle(VS::TVS, M::TM) where {TVS<:VectorSpaceType, TM<:Manifold}
    return VectorBundle{TVS, TM}(VS, M, VectorBundleFibers(VS, M))
end
TangentBundle{M} = VectorBundle{TangentSpaceType,M}
TangentBundle(M::Manifold) = VectorBundle(TangentSpace, M)
CotangentBundle{M} = VectorBundle{CotangentSpaceType,M}
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

(+)(v1::FVector, v2::FVector) = FVector(v1.type, v1.data + v2.data)
(-)(v1::FVector, v2::FVector) = FVector(v1.type, v1.data - v2.data)
(-)(v::FVector) = FVector(v.type, -v.data)
(*)(a::Number, v::FVector) = FVector(v.type, a*v.data)

base_manifold(B::VectorBundleFibers) = base_manifold(B.M)
base_manifold(B::VectorSpaceAtPoint) = base_manifold(B.fiber)
base_manifold(B::VectorBundle) = base_manifold(B.M)

"""
    bundle_projection(B::VectorBundle, x::ProductRepr)

Projection of point `x` from the bundle `M` to the base manifold.
Returns the point on the base manifold `B.M` at which the vector part
of `x` is attached.
"""
function bundle_projection(B::VectorBundle, x)
    return submanifold_component(x, 1)
end

"""
    distance(B::VectorBundleFibers, x, v, w)

Distance between vectors `v` and `w` from the vector space at point `x`
from the manifold `M.M`, that is the base manifold of `M`.
"""
distance(B::VectorBundleFibers, x, v, w) = norm(B, x, v-w)
@doc doc"""
    distance(B::VectorBundle, x, y)

Distance between points $x$ and $y$ from the
vector bundle `B` over manifold `B.VS` (denoted $M$).

Notation:
  * The point $x = (p_x, \xi_x)$ where $p_x \in M$ and $\xi_x$ belongs to the
    fiber $F=\pi^{-1}(\{p_x\})$ of the vector bundle $B$ where $\pi$ is the
    canonical projection of that vector bundle $B$.
    Similarly, $y = (p_y, \xi_y)$.

The distance is calculated as

$d_B(x, y) = \sqrt{d_M(p_x, p_y)^2 + d_F(\xi_x, \xi_{y\to x})^2}$

where $d_M$ is the distance on manifold $M$, $d_F$ is the distance
between two vectors from the fiber $F$ and $\xi_{y\to x}$ is the result
of parallel transport of vector $\xi_y$ to point $p_x$. The default
behavior of [`vector_transport_to`](@ref) is used to compute the vector
transport.
"""
function distance(B::VectorBundle, x, y)
    dist_man = distance(B.M, x.parts[1], y.parts[1])
    vy_x = vector_transport_to(B.M, y.parts[1], y.parts[2], x.parts[1])
    dist_vec = distance(B.VS, x.parts[1], x.parts[2], vy_x)

    return sqrt(dist_man^2 + dist_vec^2)
end

eltype(::Type{FVector{TType,TData}}) where {TType<:VectorSpaceType,TData} = eltype(TData)

@doc doc"""
    exp(B::VectorBundle, x, v)

Exponential map of tangent vector $v$ at point $x$ from
vector bundle `B` over manifold `B.VS` (denoted $M$).

Notation:
  * The point $x = (p_x, \xi_x)$ where $p_x \in M$ and $\xi_x$ belongs to the
    fiber $F=\pi^{-1}(\{p_x\})$ of the vector bundle $B$ where $\pi$ is the
    canonical projection of that vector bundle $B$.
  * The tangent vector $v = (\xi_{v,M}, \xi_{v,F}) \in T_{x}B$ where
    $\xi_{v,M}$ is a tangent vector from the tangent space $T_{p_x}M$ and
    $\xi_{v,F}$ is a tangent vector from the tangent space $T_{\xi_x}F$ (isomorphic to $F$).

The exponential map is calculated as

$\exp_{x}(v) = (\exp_{p_x}(\xi_{v,M}), \xi_{\exp})$

where $\xi_{\exp}$ is the result of vector transport of $\xi_x + \xi_{v,F}$
to the point $\exp_{p_x}(\xi_{v,M})$.
The sum $\xi_x + \xi_{v,F}$ corresponds to the exponential map in the vector space $F$.
"""
exp(::VectorBundle, ::Any)
function exp!(B::VectorBundle, y, x, v)
    exp!(B.M, y.parts[1], x.parts[1], v.parts[1])
    vector_transport_to!(B.M, y.parts[2], x.parts[1], x.parts[2] + v.parts[2], y.parts[1])
    return y
end

@doc doc"""
    flat!(M::Manifold, x, w::FVector)

Compute the flat isomorphism (one of the musical isomorphisms) of tangent vector `w`
from the vector space of type `M` at point `x` from the underlying [`Manifold`](@ref).

The function can be used for example to transform vectors
from the tangent bundle to vectors from the cotangent bundle
$\flat \colon T\mathcal M \to T^{*}\mathcal M$
"""
function flat(M::Manifold, x, w::FVector)
    v = similar_result(M, flat, w, x)
    flat!(M, v, x, w)
    return v
end
function flat!(M::Manifold, v::FVector, x, w::FVector)
    error("flat! not implemented for vector bundle fibers space " *
        "of type $(typeof(M)), vector of type $(typeof(v)), point of " *
        "type $(typeof(x)) and vector of type $(typeof(w)).")
end

Base.@propagate_inbounds getindex(x::FVector, i) = getindex(x.data, i)

"""
    inner(B::VectorBundleFibers, x, v, w)

Inner product of vectors `v` and `w` from the vector space of type `B.VS`
at point `x` from manifold `B.M`.
"""
function inner(B::VectorBundleFibers, x, v, w)
    error("inner not defined for vector space family of type $(typeof(B)), " *
        "point of type $(typeof(x)) and " *
        "vectors of types $(typeof(v)) and $(typeof(w)).")
end
function inner(B::VectorBundleFibers{<:TangentSpaceType}, x, v, w)
    return inner(B.M, x, v, w)
end
function inner(B::VectorBundleFibers{<:CotangentSpaceType}, x, v, w)
    return inner(B.M,
                 x,
                 sharp(B.M, x, FVector(CotangentSpace, v)).data,
                 sharp(B.M, x, FVector(CotangentSpace, w)).data)
end
@doc doc"""
    inner(B::VectorBundle, x, v, w)

Inner product of tangent vectors `v` and `w` at point `x` from the
vector bundle `B` over manifold `B.VS` (denoted $M$).

Notation:
  * The point $x = (p_x, \xi_x)$ where $p_x \in M$ and $\xi_x$ belongs to the
    fiber $F=\pi^{-1}(\{p_x\})$ of the vector bundle $B$ where $\pi$ is the
    canonical projection of that vector bundle $B$.
  * The tangent vector $v = (\xi_{v,M}, \xi_{v,F}) \in T_{x}B$ where
    $\xi_{v,M}$ is a tangent vector from the tangent space $T_{p_x}M$ and
    $\xi_{v,F}$ is a tangent vector from the tangent space $T_{\xi_x}F$ (isomorphic to $F$).
    Similarly for the other tangent vector $w = (\xi_{w,M}, \xi_{w,F}) \in T_{x}B$.

The inner product is calculated as

$\langle v, w \rangle_{B} = \langle \xi_{v,M}, \xi_{w,M} \rangle_{M} + \langle \xi_{v,F}, \xi_{w,F} \rangle_{F}.$
"""
function inner(B::VectorBundle, x, v, w)
    return inner(B.M, x.parts[1], v.parts[1], w.parts[1]) +
           inner(B.VS, x.parts[2], v.parts[2], w.parts[2])
end
function isapprox(B::VectorBundle, x, y; kwargs...)
    return isapprox(B.M, x.parts[1], y.parts[1]; kwargs...) &&
        isapprox(x.parts[2], y.parts[2]; kwargs...)
end
function isapprox(B::VectorBundle, x, v, w; kwargs...)
    return isapprox(B.M, v.parts[1], w.parts[1]; kwargs...) &&
        isapprox(B.M, x.parts[1], v.parts[2], w.parts[2]; kwargs...)
end

@doc doc"""
    log(B::VectorBundle, x, y)

Logarithmic map of the point $y$ at point $x$ from
vector bundle `B` over manifold `B.VS` (denoted $M$).

Notation:
  * The point $x = (p_x, \xi_x)$ where $p_x \in M$ and $\xi_x$ belongs to the
    fiber $F=\pi^{-1}(\{p_x\})$ of the vector bundle $B$ where $\pi$ is the
    canonical projection of that vector bundle $B$.
    Similarly, $y = (p_y, \xi_y)$.

The logarithmic map is calculated as

$\log_{x}(y) = (\log_{p_x}(p_y), \xi_{\log} - \xi_x)$

where $\xi_{\log}$ is the result of vector transport of $\xi_y$
to the point $p_x$.
The difference $\xi_{\log} - \xi_x$ corresponds to the logarithmic map in the vector space $F$.
"""
log(::VectorBundle, ::Any...)
function log!(B::VectorBundle, v, x, y)
    log!(B.M, v.parts[1], x.parts[1], y.parts[1])
    vector_transport_to!(B.M, v.parts[2], y.parts[1], y.parts[2], x.parts[1])
    copyto!(v.parts[2], v.parts[2] - x.parts[2])
    return v
end

manifold_dimension(B::VectorBundle) = manifold_dimension(B.M) + vector_space_dimension(B.VS)

"""
    norm(B::VectorBundleFibers, x, v)

Norm of the vector `v` from the vector space of type `B.VS`
at point `x` from manifold `B.M`.
"""
norm(B::VectorBundleFibers, x, v) = sqrt(inner(B, x, v, v))

norm(B::VectorBundleFibers{<:TangentSpaceType}, x, v) = norm(B.M, x, v)

@doc doc"""
    project_point(B::VectorBundle, x)

Project the point $x$ from the ambient space of the vector bundle `B`
over manifold `B.VS` (denoted $M$) to the vector bundle.

Notation:
  * The point $x = (p_x, \xi_x)$ where $p_x$ belongs to the ambient space of $M$
    and $\xi_x$ belongs to the ambient space of the
    fiber $F=\pi^{-1}(\{p_x\})$ of the vector bundle $B$ where $\pi$ is the
    canonical projection of that vector bundle $B$.

The projection is calculated by projecting the point $p_x$ to the manifold $M$
and then projecting the vector $\xi_x$ to the tangent space $T_{p_x}M$.
"""
project_point(::VectorBundle, ::Any...)
function project_point!(B::VectorBundle, x)
    project_point!(B.M, x.parts[1])
    project_tangent!(B.M, x.parts[2], x.parts[1], x.parts[2])
    return x
end

@doc doc"""
    project_tangent(B::VectorBundle, x, v)

Project the element $v$ of the ambient space of the tangent space $T_x B$
to the tangent space $T_x B$.

Notation:
  * The point $x = (p_x, \xi_x)$ where $p_x \in M$ and $\xi_x$ belongs to the
    fiber $F=\pi^{-1}(\{p_x\})$ of the vector bundle $B$ where $\pi$ is the
    canonical projection of that vector bundle $B$.
  * The vector $x = (\xi_{v,M}, \xi_{v,F})$ where $p_x$ belongs to the ambient space of $T_{p_x}M$
    and $\xi_{v,F}$ belongs to the ambient space of the
    fiber $F=\pi^{-1}(\{p_x\})$ of the vector bundle $B$ where $\pi$ is the
    canonical projection of that vector bundle $B$.

The projection is calculated by projecting $\xi_{v,M}$ to tangent space $T_{p_x}M$
and then projecting the vector $\xi_{v,F}$ to the fiber $F$.
"""
project_tangent(::VectorBundle, ::Any...)
function project_tangent!(B::VectorBundle, w, x, v)
    project_tangent!(B.M, w.parts[1], x.parts[1], v.parts[1])
    project_tangent!(B.M, w.parts[2], x.parts[1], v.parts[2])
    return w
end

"""
    project_vector(B::VectorBundleFibers, x, w)

Project vector `w` from the vector space of type `B.VS` at point `x`.
"""
function project_vector(B::VectorBundleFibers, x, w)
    v = similar_result(B, project_vector, x, w)
    project_vector!(B, v, x, w)
    return v
end
function project_vector!(B::VectorBundleFibers{<:TangentSpaceType}, v, x, w)
    project_tangent!(B.M, v, x, w)
    return v
end
function project_vector!(B::VectorBundleFibers, v, x, w)
    error("project_vector! not implemented for vector space family of type $(typeof(B)), output vector of type $(typeof(v)) and input vector at point $(typeof(x)) with type of w $(typeof(w)).")
end

Base.@propagate_inbounds setindex!(x::FVector, val, i) = setindex!(x.data, val, i)

function representation_size(B::VectorBundleFibers{<:TCoTSpaceType})
    representation_size(B.M)
end
function representation_size(B::VectorBundle)
    len_manifold = prod(representation_size(B.M))
    len_vs = prod(representation_size(B.VS))
    return (len_manifold + len_vs,)
end

@doc doc"""
    sharp(M::Manifold, x, w::FVector)

Compute the sharp isomorphism (one of the musical isomorphisms) of vector `w`
from the vector space `M` at point `x` from the underlying [`Manifold`](@ref).

The function can be used for example to transform vectors
from the cotangent bundle to vectors from the tangent bundle
$\sharp \colon T^{*}\mathcal M \to T\mathcal M$
"""
function sharp(M::Manifold, x, w::FVector)
    v = similar_result(M, sharp, w, x)
    sharp!(M, v, x, w)
    return v
end

function sharp!(M::Manifold, v::FVector, x, w::FVector)
    error("sharp! not implemented for vector bundle fibers space " *
        "of type $(typeof(M)), vector of type $(typeof(v)), point of " *
        "type $(typeof(x)) and vector of type $(typeof(w)).")
end

similar(x::FVector) = FVector(x.type, similar(x.data))
similar(x::FVector, ::Type{T}) where {T} = FVector(x.type, similar(x.data, T))

"""
    similar_result(B::VectorBundleFibers, f, x...)

Allocates an array for the result of function `f` that is
an element of the vector space of type `B.VS` on manifold `B.M`
and arguments `x...` for implementing the non-modifying operation
using the modifying operation.
"""
function similar_result(B::VectorBundleFibers, f, x...)
    T = similar_result_type(B, f, x)
    return similar(x[1], T)
end
"""
    similar_result_type(B::VectorBundleFibers, f, args::NTuple{N,Any}) where N

Returns type of element of the array that will represent the result of
function `f` for representing an operation with result in the vector space `VS`
for manifold `M` on given arguments (passed at a tuple).
"""
function similar_result_type(B::VectorBundleFibers, f, args::NTuple{N,Any}) where N
    T = typeof(reduce(+, one(eltype(eti)) for eti ∈ args))
    return T
end
function similar_result(M::Manifold, ::typeof(flat), w::FVector{TangentSpaceType}, x)
    return FVector(CotangentSpace, similar(w.data))
end

size(x::FVector) = size(x.data)

function similar_result(M::Manifold, ::typeof(sharp), w::FVector{CotangentSpaceType}, x)
    return FVector(TangentSpace, similar(w.data))
end

"""
    vector_space_dimension(B::VectorBundleFibers)

Dimension of the vector space of type `B`.
"""
function vector_space_dimension(B::VectorBundleFibers)
    error("vector_space_dimension not implemented for vector space family $(typeof(B)).")
end
vector_space_dimension(B::VectorBundleFibers{<:TCoTSpaceType}) = manifold_dimension(B.M)
function vector_space_dimension(B::VectorBundleFibers{<:TensorProductType})
    dim = 1
    for space in B.VS.spaces
        dim *= vector_space_dimension(VectorBundleFibers(space, B.M))
    end
    return dim
end

"""
    zero_vector!(B::VectorBundleFibers, v, x)

Save the zero vector from the vector space of type `B.VS` at point `x`
from manifold `B.M` to `v`.
"""
function zero_vector!(B::VectorBundleFibers, v, x)
    error("zero_vector! not implemented for vector space family of type $(typeof(B)).")
end

function zero_vector!(B::VectorBundleFibers{<:TangentSpaceType}, v, x)
    zero_tangent_vector!(B.M, v, x)
    return v
end

"""
    zero_vector(B::VectorBundleFibers, x)

Compute the zero vector from the vector space of type `B.VS` at point `x`
from manifold `B.M`.
"""
function zero_vector(B::VectorBundleFibers, x)
    v = similar_result(B, zero_vector, x)
    zero_vector!(B, v, x)
    return v
end
@doc doc"""
    zero_tangent_vector!(B::VectorBundle, v, x)

Zero tangent vector at point $x$ from the vector bundle `B`
over manifold `B.VS` (denoted $M$). The zero vector belongs to the space $T_{x}B$
The result overwrites the tangent vector `v`.

Notation:
  * The point $x = (p_x, \xi_x)$ where $p_x \in M$ and $\xi_x$ belongs to the
    fiber $F=\pi^{-1}(\{p_x\})$ of the vector bundle $B$ where $\pi$ is the
    canonical projection of that vector bundle $B$.

The zero vector is calculated as

$\mathbf{0}_{x} = (\mathbf{0}_{p_x}, \mathbf{0}_F)$

where $\mathbf{0}_{p_x}$ is the zero tangent vector from $T_{p_x}M$ and
$\mathbf{0}_F$ is the zero element of the vector space $F$.
"""
function zero_tangent_vector!(B::VectorBundle, v, x)
    zero_tangent_vector!(B.M, v.parts[1], x.parts[1])
    zero_vector!(B.VS, v.parts[2], x.parts[2])
    return v
end
