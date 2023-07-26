
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
    abstract type FiberType end

An abstract type for fiber types.
"""
abstract type FiberType end

struct VectorSpaceFiberType{TVS<:VectorSpaceType} <: FiberType
    fiber::TVS
end

function Base.show(io::IO, vsf::VectorSpaceFiberType)
    return print(io, "VectorSpaceFiberType($(vsf.fiber))")
end

const TangentFiberType = VectorSpaceFiberType{TangentSpaceType}

const CotangentFiberType = VectorSpaceFiberType{CotangentSpaceType}

const TangentFiber = VectorSpaceFiberType{TangentSpaceType}(TangentSpace)
const CotangentFiber = VectorSpaceFiberType{CotangentSpaceType}(CotangentSpace)

"""
    BundleFibers(fiber::FiberType, M::AbstractManifold)

Type representing a family of vector spaces (fibers) of a vector bundle over `M`
with vector spaces of type `fiber`. In contrast with `FiberBundle`, operations
on `BundleFibers` expect point-like and vector-like parts to be
passed separately instead of being bundled together. It can be thought of
as a representation of vector spaces from a vector bundle but without
storing the point at which a vector space is attached (which is specified
separately in various functions).
"""
struct BundleFibers{TF<:FiberType,TM<:AbstractManifold}
    fiber::TF
    manifold::TM
end

"""
    VectorBundleFibers{TVS,TM}

Alias for [`BundleFibers`](@ref) when the fiber is a vector space.
"""
const VectorBundleFibers{TVS,TM} = BundleFibers{
    VectorSpaceFiberType{TVS},
    TM,
} where {TVS<:VectorSpaceType,TM<:AbstractManifold}

function VectorBundleFibers(fiber::VectorSpaceType, M::AbstractManifold)
    return BundleFibers(VectorSpaceFiberType(fiber), M)
end
function VectorBundleFibers(fiber::VectorSpaceFiberType, M::AbstractManifold)
    return BundleFibers(fiber, M)
end

const TangentBundleFibers{M} = BundleFibers{TangentFiberType,M} where {M<:AbstractManifold}

TangentBundleFibers(M::AbstractManifold) = BundleFibers(TangentFiber, M)

const CotangentBundleFibers{M} =
    BundleFibers{CotangentFiberType,M} where {M<:AbstractManifold}

CotangentBundleFibers(M::AbstractManifold) = BundleFibers(CotangentFiber, M)

"""
    FiberAtPoint{
        𝔽,
        TFiber<:BundleFibers{<:FiberType,<:AbstractManifold{𝔽}},
        TX,
    } <: AbstractManifold{𝔽}

A fiber of a [`FiberBundle`](@ref) at a point `p` on the manifold.
This is modelled using [`BundleFibers`](@ref) with only a fiber part
and fixing the point-like part to be just `p`.

This fiber itself is also a `manifold`. For vector fibers it's by default flat and hence
isometric to the [`Euclidean`](@ref) manifold.

# Constructor

    FiberAtPoint(fiber::BundleFibers, p)

A fiber of type `fiber` at point `p` from the manifold `fiber.manifold`.
"""
struct FiberAtPoint{𝔽,TFiber<:BundleFibers{<:FiberType,<:AbstractManifold{𝔽}},TX} <:
       AbstractManifold{𝔽}
    fiber::TFiber
    point::TX
end

"""
    VectorSpaceAtPoint{𝔽,TFiber}

Alias for [`FiberAtPoint`](@ref) when the fiber is a vector space.
"""
const VectorSpaceAtPoint{𝔽,TFiber} = FiberAtPoint{
    𝔽,
    TFiber,
} where {TFiber<:VectorBundleFibers{<:FiberType,<:AbstractManifold{𝔽}}}

function VectorSpaceAtPoint(M::AbstractManifold, fiber::VectorSpaceFiberType, p)
    return FiberAtPoint(BundleFibers(fiber, M), p)
end
VectorSpaceAtPoint(fiber::BundleFibers{<:VectorSpaceFiberType}, p) = FiberAtPoint(fiber, p)

"""
    TangentSpaceAtPoint{𝔽,M}

Alias for [`VectorSpaceAtPoint`](@ref) for the tangent space at a point.
"""
const TangentSpaceAtPoint{𝔽,M} =
    FiberAtPoint{𝔽,TangentBundleFibers{M}} where {𝔽,M<:AbstractManifold{𝔽}}

"""
    TangentSpaceAtPoint(M::AbstractManifold, p)

Return an object of type [`VectorSpaceAtPoint`](@ref) representing tangent
space at `p` on the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  `M`.
"""
TangentSpaceAtPoint(M::AbstractManifold, p) = VectorSpaceAtPoint(M, TangentFiber, p)

"""
    TangentSpace(M::AbstractManifold, p)

Return a [`TangentSpaceAtPoint`](@ref Manifolds.TangentSpaceAtPoint) representing tangent
space at `p` on the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold) `M`.
"""
TangentSpace(M::AbstractManifold, p) = TangentSpaceAtPoint(M, p)

const CotangentSpaceAtPoint{𝔽,M} =
    VectorSpaceAtPoint{CotangentBundleFibers{M}} where {𝔽,M<:AbstractManifold{𝔽}}

"""
    CotangentSpaceAtPoint(M::AbstractManifold, p)

Return an object of type [`VectorSpaceAtPoint`](@ref) representing cotangent
space at `p`.
"""
function CotangentSpaceAtPoint(M::AbstractManifold, p)
    return VectorSpaceAtPoint(M, CotangentFiber, p)
end

"""
    allocate_result(B::BundleFibers, f, x...)

Allocates an array for the result of function `f` that is
an element of the vector space of type `B.fiber` on manifold `B.manifold`
and arguments `x...` for implementing the non-modifying operation
using the modifying operation.
"""
@inline function allocate_result(B::BundleFibers, f::TF, x...) where {TF}
    if length(x) == 0
        # TODO: this may be incorrect when point and tangent vector representation are
        #       different for the manifold but there is no easy and generic way around that
        return allocate_result(B.manifold, f)
    else
        T = allocate_result_type(B, f, x)
        return allocate(x[1], T)
    end
end

"""
    allocate_result_type(B::BundleFibers, f, args::NTuple{N,Any}) where N

Return type of element of the array that will represent the result of
function `f` for representing an operation with result in the vector space `fiber`
for manifold `M` on given arguments (passed at a tuple).
"""
@inline function allocate_result_type(
    ::BundleFibers,
    f::TF,
    args::NTuple{N,Any},
) where {TF,N}
    return typeof(mapreduce(eti -> one(number_eltype(eti)), +, args))
end

base_manifold(B::BundleFibers) = base_manifold(B.manifold)
base_manifold(B::FiberAtPoint) = base_manifold(B.fiber)

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
    exp(M::TangentSpaceAtPoint, p, X)

Exponential map of tangent vectors `X` and `p` from the tangent space `M`. It is
calculated as their sum.
"""
exp(::TangentSpaceAtPoint, ::Any, ::Any)

function exp!(M::TangentSpaceAtPoint, q, p, X)
    copyto!(M.fiber.manifold, q, p + X)
    return q
end

function fiber_dimension(B::BundleFibers)
    return fiber_dimension(B.manifold, B.fiber)
end
fiber_dimension(M::TangentBundleFibers) = manifold_dimension(M.manifold)
fiber_dimension(M::CotangentBundleFibers) = manifold_dimension(M.manifold)
fiber_dimension(M::AbstractManifold, ::CotangentSpaceType) = manifold_dimension(M)
fiber_dimension(M::AbstractManifold, ::TangentSpaceType) = manifold_dimension(M)

function get_basis(M::TangentSpaceAtPoint, p, B::CachedBasis)
    return invoke(
        get_basis,
        Tuple{AbstractManifold,Any,CachedBasis},
        M.fiber.manifold,
        M.point,
        B,
    )
end
function get_basis(M::TangentSpaceAtPoint, p, B::AbstractBasis{<:Any,TangentSpaceType})
    return get_basis(M.fiber.manifold, M.point, B)
end

function get_coordinates(M::TangentSpaceAtPoint, p, X, B::AbstractBasis)
    return get_coordinates(M.fiber.manifold, M.point, X, B)
end

function get_coordinates!(M::TangentSpaceAtPoint, Y, p, X, B::AbstractBasis)
    return get_coordinates!(M.fiber.manifold, Y, M.point, X, B)
end

function get_vector(M::TangentSpaceAtPoint, p, X, B::AbstractBasis)
    return get_vector(M.fiber.manifold, M.point, X, B)
end

function get_vector!(M::TangentSpaceAtPoint, Y, p, X, B::AbstractBasis)
    return get_vector!(M.fiber.manifold, Y, M.point, X, B)
end

function get_vectors(M::TangentSpaceAtPoint, p, B::CachedBasis)
    return get_vectors(M.fiber.manifold, M.point, B)
end

@doc raw"""
    injectivity_radius(M::TangentSpaceAtPoint)

Return the injectivity radius on the [`TangentSpaceAtPoint`](@ref Manifolds.TangentSpaceAtPoint) `M`, which is $∞$.
"""
injectivity_radius(::TangentSpaceAtPoint) = Inf

"""
    inner(M::TangentSpaceAtPoint, p, X, Y)

Inner product of vectors `X` and `Y` from the tangent space at `M`.
"""
function inner(M::TangentSpaceAtPoint, p, X, Y)
    return inner(M.fiber.manifold, M.point, X, Y)
end

"""
    is_flat(::TangentSpaceAtPoint)

Return true. [`TangentSpaceAtPoint`](@ref Manifolds.TangentSpaceAtPoint) is a flat manifold.
"""
is_flat(::TangentSpaceAtPoint) = true

function _isapprox(M::TangentSpaceAtPoint, X, Y; kwargs...)
    return isapprox(M.fiber.manifold, M.point, X, Y; kwargs...)
end

"""
    log(M::TangentSpaceAtPoint, p, q)

Logarithmic map on the tangent space manifold `M`, calculated as the difference of tangent
vectors `q` and `p` from `M`.
"""
log(::TangentSpaceAtPoint, ::Any...)
function log!(::TangentSpaceAtPoint, X, p, q)
    copyto!(X, q - p)
    return X
end

function manifold_dimension(M::FiberAtPoint)
    return fiber_dimension(M.fiber)
end

LinearAlgebra.norm(M::VectorSpaceAtPoint, p, X) = norm(M.fiber.manifold, M.point, X)

function parallel_transport_to!(M::TangentSpaceAtPoint, Y, p, X, q)
    return copyto!(M.fiber.manifold, Y, p, X)
end

@doc raw"""
    project(M::TangentSpaceAtPoint, p)

Project the point `p` from the tangent space `M`, that is project the vector `p`
tangent at `M.point`.
"""
project(::TangentSpaceAtPoint, ::Any)

function project!(M::TangentSpaceAtPoint, q, p)
    return project!(M.fiber.manifold, q, M.point, p)
end

@doc raw"""
    project(M::TangentSpaceAtPoint, p, X)

Project the vector `X` from the tangent space `M`, that is project the vector `X`
tangent at `M.point`.
"""
project(::TangentSpaceAtPoint, ::Any, ::Any)

function project!(M::TangentSpaceAtPoint, Y, p, X)
    return project!(M.fiber.manifold, Y, M.point, X)
end

function Random.rand!(rng::AbstractRNG, M::TangentSpaceAtPoint, X; vector_at=nothing)
    rand!(rng, M.fiber.manifold, X; vector_at=M.point)
    return X
end

function representation_size(B::TangentSpaceAtPoint)
    return representation_size(B.fiber.manifold)
end

function Base.show(io::IO, fiber::BundleFibers)
    return print(io, "BundleFibers($(fiber.fiber), $(fiber.manifold))")
end
function Base.show(io::IO, ::MIME"text/plain", vs::FiberAtPoint)
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

function vector_transport_to!(
    M::TangentSpaceAtPoint,
    Y,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod,
)
    return copyto!(M.fiber.manifold, Y, p, X)
end

"""
    zero_vector(B::BundleFibers, p)

Compute the zero vector from the vector space of type `B.fiber` at point `p`
from manifold `B.manifold`.
"""
function zero_vector(B::BundleFibers, p)
    X = allocate_result(B, zero_vector, p)
    return zero_vector!(B, X, p)
end

@doc raw"""
    zero_vector(M::TangentSpaceAtPoint, p)

Zero tangent vector at point `p` from the tangent space `M`, that is the zero tangent vector
at point `M.point`.
"""
zero_vector(::TangentSpaceAtPoint, ::Any...)

function zero_vector!(M::TangentSpaceAtPoint, X, p)
    return zero_vector!(M.fiber.manifold, X, M.point)
end
