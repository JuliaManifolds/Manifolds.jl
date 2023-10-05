
const CotangentFiberType = VectorSpaceFiberType{CotangentSpaceType}

const CotangentBundleFibers{M} =
    BundleFibers{CotangentFiberType,M} where {M<:AbstractManifold}

CotangentBundleFibers(M::AbstractManifold) = BundleFibers(CotangentFiber, M)

const CotangentFiber = VectorSpaceFiberType{CotangentSpaceType}(CotangentSpace)

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

fiber_dimension(M::CotangentBundleFibers) = manifold_dimension(M.manifold)

"""
    TensorProductType(spaces::VectorSpaceType...)

Vector space type corresponding to the tensor product of given vector space
types.
"""
struct TensorProductType{TS<:Tuple} <: VectorSpaceType
    spaces::TS
end

TensorProductType(spaces::VectorSpaceType...) = TensorProductType{typeof(spaces)}(spaces)

function Base.show(io::IO, tpt::TensorProductType)
    return print(io, "TensorProductType(", join(tpt.spaces, ", "), ")")
end

function vector_space_dimension(M::AbstractManifold, V::TensorProductType)
    dim = 1
    for space in V.spaces
        dim *= fiber_dimension(M, space)
    end
    return dim
end
