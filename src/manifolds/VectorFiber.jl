
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
    VectorSpaceFiberType{TVS<:VectorSpaceType}

`FiberType` of a [`FiberBundle`](@ref) corresponding to [`VectorSpaceType`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/bases.html#ManifoldsBase.VectorSpaceType)
`TVS`.
"""
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
    VectorSpaceAtPoint{𝔽,M,TFiber}

Alias for [`FiberAtPoint`](@ref) when the fiber is a vector space.
"""
const VectorSpaceAtPoint{𝔽,M,TSpaceType} = FiberAtPoint{
    𝔽,
    VectorBundleFibers{TSpaceType,M},
} where {𝔽,M<:AbstractManifold{𝔽},TSpaceType<:VectorSpaceType}

function VectorSpaceAtPoint(M::AbstractManifold, fiber::VectorSpaceFiberType, p)
    return FiberAtPoint(BundleFibers(fiber, M), p)
end

"""
    TangentSpaceAtPoint{𝔽,M}

Alias for [`VectorSpaceAtPoint`](@ref) for the tangent space at a point.
"""
const TangentSpaceAtPoint{𝔽,M} =
    VectorSpaceAtPoint{𝔽,M,TangentSpaceType} where {𝔽,M<:AbstractManifold{𝔽}}

"""
    TangentSpaceAtPoint(M::AbstractManifold, p)

Return an object of type [`VectorSpaceAtPoint`](@ref) representing tangent
space at `p` on the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  `M`.
"""
TangentSpaceAtPoint(M::AbstractManifold, p) = VectorSpaceAtPoint(M, TangentFiber, p)

"""
    TangentSpace(M::AbstractManifold, p)

Return a [`TangentSpaceAtPoint`] representing tangent
space at `p` on the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold) `M`.
"""
TangentSpace(M::AbstractManifold, p) = TangentSpaceAtPoint(M, p)

const CotangentSpaceAtPoint{𝔽,M} =
    VectorSpaceAtPoint{𝔽,CotangentBundleFibers{M}} where {𝔽,M<:AbstractManifold{𝔽}}

"""
    CotangentSpaceAtPoint(M::AbstractManifold, p)

Return an object of type [`VectorSpaceAtPoint`](@ref) representing cotangent
space at `p`.
"""
function CotangentSpaceAtPoint(M::AbstractManifold, p)
    return VectorSpaceAtPoint(M, CotangentFiber, p)
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
    exp(M::TangentSpaceAtPoint, p, X)

Exponential map of tangent vectors `X` and `p` from the tangent space `M`. It is
calculated as their sum.
"""
exp(::TangentSpaceAtPoint, ::Any, ::Any)

function exp!(M::TangentSpaceAtPoint, q, p, X)
    copyto!(M.fiber.manifold, q, p + X)
    return q
end

fiber_dimension(M::TangentBundleFibers) = manifold_dimension(M.manifold)
fiber_dimension(M::CotangentBundleFibers) = manifold_dimension(M.manifold)
fiber_dimension(M::AbstractManifold, V::VectorSpaceFiberType) = fiber_dimension(M, V.fiber)
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

Return the injectivity radius on the [`TangentSpaceAtPoint`](@ref) `M`, which is $∞$.
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

Return true. [`TangentSpaceAtPoint`](@ref) is a flat manifold.
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

@doc raw"""
    zero_vector(M::TangentSpaceAtPoint, p)

Zero tangent vector at point `p` from the tangent space `M`, that is the zero tangent vector
at point `M.point`.
"""
zero_vector(::TangentSpaceAtPoint, ::Any...)

function zero_vector!(M::VectorSpaceAtPoint, X, p)
    return zero_vector!(M.fiber.manifold, X, M.point)
end
