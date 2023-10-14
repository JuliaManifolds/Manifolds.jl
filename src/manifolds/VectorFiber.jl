
"""
    TensorProductType(spaces::VectorSpaceType...)

Vector space type corresponding to the tensor product of given vector space
types.
"""
struct TensorProductType{TS<:Tuple} <: VectorSpaceType
    spaces::TS
end

TensorProductType(spaces::VectorSpaceType...) = TensorProductType{typeof(spaces)}(spaces)

function inner(B::CotangentSpace, p, X, Y)
    return inner(
        B.manifold,
        B.point,
        sharp(B.manifold, B.point, X),
        sharp(B.manifold, B.point, Y),
    )
end

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
