module ManifoldDistributions

using Distributions
using LinearAlgebra
using ManifoldsBase
using ManifoldsBase: AbstractDecoratorManifold, Manifold, @decorator_transparent_signature
using ..Manifolds
using ..Manifolds:
    TangentBundleFibers,
    VectorBundleFibers,
    VectorSpaceType,
    ProductRepr,
    AbstractPowerManifold,
    Euclidean,
    MetricManifold,
    ProductManifold,
    Rotations,
    Sphere,
    allocate_result,
    get_iterator,
    project_vector!,
    project!,
    representation_size,
    submanifold_components,
    zero_vector,
    _read,
    _write
using Random

"""
    FVectorvariate

Structure that subtypes `VariateForm`, indicating that a single sample
is a vector from a fiber of a vector bundle.
"""
struct FVectorvariate <: VariateForm end

"""
    FVectorSupport(space::Manifold, VectorBundleFibers)

Value support for vector bundle fiber-valued distributions (values from a fiber of a vector
bundle at a `point` from the given manifold).
For example used for tangent vector-valued distributions.
"""
struct FVectorSupport{TSpace<:VectorBundleFibers,T} <: ValueSupport
    space::TSpace
    point::T
end

"""
    FVectorDistribution{TSpace<:VectorBundleFibers, T}

An abstract distribution for vector bundle fiber-valued distributions (values from a fiber
of a vector bundle at point `x` from the given manifold).
For example used for tangent vector-valued distributions.
"""
abstract type FVectorDistribution{TSpace<:VectorBundleFibers,T} <:
              Distribution{FVectorvariate,FVectorSupport{TSpace,T}} end

"""
    MPointvariate

Structure that subtypes `VariateForm`, indicating that a single sample
is a point on a manifold.
"""
struct MPointvariate <: VariateForm end

"""
    MPointSupport(M::Manifold)

Value support for manifold-valued distributions (values from given
[`Manifold`](@ref) `M`).
"""
struct MPointSupport{TM<:Manifold} <: ValueSupport
    manifold::TM
end

"""
    MPointDistribution{TM<:Manifold}

An abstract distribution for points on manifold of type `TM`.
"""
abstract type MPointDistribution{TM<:Manifold} <:
              Distribution{MPointvariate,MPointSupport{TM}} end

"""
    support(d::FVectorDistribution)

Get the object of type `FVectorSupport` for the distribution `d`.
"""
function Distributions.support(::T) where {T<:FVectorDistribution}
    error("support not implemented for type $T")
end

@decorator_transparent_signature normal_tvector_distribution(
    M::AbstractDecoratorManifold,
    p,
    σ,
)

@decorator_transparent_signature uniform_distribution(M::AbstractDecoratorManifold, p)

include("projected_distribution.jl")
include("manifold_distribution.jl")

export normal_tvector_distribution, projected_distribution, uniform_distribution
export FVectorDistribution,
    MPointDistribution, ProjectedFVectorDistribution, ProjectedPointDistribution

end # module
