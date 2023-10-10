
"""
    BundleFibers(fiber::FiberType, M::AbstractManifold)

Type representing a family of fibers of a fiber bundle over `M`
with fiber of type `fiber`. In contrast with `FiberBundle`, operations
on `BundleFibers` expect point-like and fiber-like parts to be
passed separately instead of being bundled together. It can be thought of
as a representation of fibers from a fiber bundle but without
storing the point at which a fiber is attached (which is specified
separately in various functions).
"""
struct BundleFibers{TF<:FiberType,TM<:AbstractManifold}
    fiber::TF
    manifold::TM
end

"""
    allocate_result(B::BundleFibers, f, x...)

Allocates an array for the result of function `f` that is
an element of the fiber space of type `B.fiber` on manifold `B.manifold`
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
function `f` for representing an operation with result in the fiber `fiber`
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

function fiber_dimension(B::BundleFibers)
    return fiber_dimension(B.manifold, B.fiber)
end

function Base.show(io::IO, fiber::BundleFibers)
    return print(io, "BundleFibers($(fiber.fiber), $(fiber.manifold))")
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
