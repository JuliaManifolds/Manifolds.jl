
"""
    abstract type FiberType end

An abstract type for fiber types.
"""
abstract type FiberType end

"""
    BundleFibers(fiber::FiberType, M::AbstractManifold)

Type representing a family of fibers of a fiber bundle over `M`
with vectorfiber of type `fiber`. In contrast with `FiberBundle`, operations
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
base_manifold(B::FiberAtPoint) = base_manifold(B.fiber)

function fiber_dimension(B::BundleFibers)
    return fiber_dimension(B.manifold, B.fiber)
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

"""
    zero_vector(B::BundleFibers, p)

Compute the zero vector from the vector space of type `B.fiber` at point `p`
from manifold `B.manifold`.
"""
function zero_vector(B::BundleFibers, p)
    X = allocate_result(B, zero_vector, p)
    return zero_vector!(B, X, p)
end
