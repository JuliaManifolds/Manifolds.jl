module Manifolds

import Base: +, -, *, angle, axes, convert, copy, copyto!, dataids, eltype, exp, getindex,
    isapprox, length, log, promote_rule, setindex!, similar, size
import Distributions: _rand!, support
import LinearAlgebra: cross, det, Diagonal, dot, norm, I, UniformScaling
import ManifoldsBase: base_manifold, check_manifold_point, check_tangent_vector
import ManifoldsBase: distance, exp, exp!, geodesic, injectivity_radius, inner
import ManifoldsBase: isapprox, is_manifold_point, is_tangent_vector, is_decorator_manifold
import ManifoldsBase: inverse_retract, inverse_retract!, log, log!, manifold_dimension, norm
import ManifoldsBase: project_point, project_point!, project_tangent, project_tangent!
import ManifoldsBase: representation_size, retract, retract!, similar_result, shortest_geodesic
import ManifoldsBase: vector_transport_along, vector_transport_along!, vector_transport_direction,
    vector_transport_direction!, vector_transport_to, vector_transport_to!
import ManifoldsBase: zero_tangent_vector, zero_tangent_vector!
import Random: rand
import Statistics: mean, mean!, median, median!, std, var
import StatsBase: kurtosis, mean_and_std, mean_and_var, moment, skewness

using Einsum: @einsum
using FiniteDifferences
using HybridArrays
using LinearAlgebra
using ManifoldsBase: CoTVector, Manifold, MPoint, TVector
using ManifoldsBase: ArrayCoTVector, ArrayManifold, ArrayMPoint, ArrayTVector, ArrayCoTVector
using ManifoldsBase: AbstractRetractionMethod, ExponentialRetraction
using ManifoldsBase: AbstractInverseRetractionMethod, LogarithmicInverseRetraction
using ManifoldsBase: AbstractVectorTransportMethod, ParallelTransport, ProjectionTransport
using Markdown: @doc_str
using Random: AbstractRNG
using Requires
using StaticArrays
using UnsafeArrays

"""
    AbstractField

An abstract type to represent the field matrix manifolds are build upon,
following the idea of [TensorKit](https://github.com/Jutho/TensorKit.jl)
in order to hace concrete field types to dispatch on. The two most common
field types are `RealNumbers` (`ℝ` for short) and `ComplexNumbers` (`ℂ`).
"""
abstract type AbstractField end

struct RealNumbers <: AbstractField end
struct ComplexNumbers <: AbstractField end

const ℝ = RealNumbers()
const ℂ = ComplexNumbers()

Base.show(io::IO, ::RealNumbers) = print(io, "ℝ")
Base.show(io::IO, ::ComplexNumbers) = print(io, "ℂ")

"""
    AbstractEstimationMethod

Abstract type for defining statistical estimation methods.
"""
abstract type AbstractEstimationMethod end

@doc doc"""
    hat(M::Manifold, x, vⁱ)

Given a basis $e_i$ on the tangent space at a point $x$ and tangent
component vector $v^i$, compute the equivalent vector representation
$v=v^i e_i$, where Einstein summation notation is used:

````math
\wedge: v^i \mapsto v^i e_i
````

For matrix manifolds, this converts a vector representation of the tangent
vector to a matrix representation. The [`vee`](@ref) map is the `hat` map's
inverse.
"""
function hat(M::Manifold, x, vⁱ)
    repr_size = representation_size(TangentBundleFibers(M))
    v = MArray{Tuple{repr_size...},eltype(vⁱ)}(undef)
    hat!(M, v, x, vⁱ)
    return v
end
hat!(M::Manifold, v, x, vⁱ) = error("hat! operator not defined for manifold $(typeof(M)), vector $(typeof(vⁱ)), and matrix $(typeof(v))")


@doc doc"""
    vee(M::Manifold, x, v)

Given a basis $e_i$ on the tangent space at a point $x$ and tangent
vector $v$, compute the vector components $v^i$, such that $v = v^i e_i$, where
Einstein summation notation is used:

````math
\vee: v^i e_i \mapsto v^i
````

For matrix manifolds, this converts a  matrix representation of the tangent
vector to a vector representation. The [`hat`](@ref) map is the `vee` map's
inverse.
"""
function vee(M::Manifold, x, v)
    vⁱ = MVector{manifold_dimension(M),eltype(v)}(undef)
    vee!(M, vⁱ, x, v)
    return vⁱ
end
function vee!(M::Manifold, vⁱ, x, v)
    error("vee! operator not defined for manifold $(typeof(M)), matrix $(typeof(v)), and vector $(typeof(vⁱ))")
end
"""
    PolarRetraction <: AbstractRetractionMethod

Retractions that are based on singular value decompositions of the matrix / matrices
for point and tangent vector on a [`Manifold`](@ref)
"""
struct PolarRetraction <: AbstractRetractionMethod end

"""
    QRRetraction <: AbstractRetractionMethod

Retractions that are based on a QR decomposition of the
matrix / matrices for point and tangent vector on a [`Manifold`](@ref)
"""
struct QRRetraction <: AbstractRetractionMethod end

"""
    PolarInverseRetraction <: AbstractInverseRetractionMethod

Inverse retractions that are based on a singular value decomposition of the
matrix / matrices for point and tangent vector on a [`Manifold`](@ref)
"""
struct PolarInverseRetraction <: AbstractInverseRetractionMethod end

"""
    QRInverseRetraction <: AbstractInverseRetractionMethod

Inverse retractions that are based on a QR decomposition of the
matrix / matrices for point and tangent vector on a [`Manifold`](@ref)
"""
struct QRInverseRetraction <: AbstractInverseRetractionMethod end


include("utils.jl")
include("autodiff.jl")
include("SizedAbstractArray.jl")

include("ProductRepresentations.jl")
include("VectorBundle.jl")
include("Metric.jl")

include("DistributionsBase.jl")
include("ProjectedDistribution.jl")

include("ProductManifold.jl")
include("PowerManifold.jl")

include("CholeskySpace.jl")
include("manifolds/Circle.jl")
include("Euclidean.jl")
include("FixedRankMatrices.jl")
include("Grassmann.jl")
include("Hyperbolic.jl")
include("Rotations.jl")
include("Stiefel.jl")
include("Sphere.jl")
include("Symmetric.jl")
include("SymmetricPositiveDefinite.jl")

include("statistics.jl")

function __init__()
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
        using .ForwardDiff
        include("forward_diff.jl")
    end

    @require OrdinaryDiffEq="1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
        using .OrdinaryDiffEq: ODEProblem,
            AutoVern9,
            Rodas5,
            solve
        include("ode.jl")
    end
end
#
export CoTVector, Manifold, MPoint, TVector, Manifold
export Euclidean, CholeskySpace, Circle, FixedRankMatrices, Grassmann,
    Hyperbolic, Rotations,Sphere, Stiefel, SymmetricMatrices, SymmetricPositiveDefinite
export SVDMPoint, UMVTVector, AbstractField, ℝ, ℂ
# decorator manifolds
export ArrayManifold, ArrayMPoint, ArrayTVector, ArrayCoTVector
export CotangentBundle, CotangentSpaceAtPoint, CotangentBundleFibers, CotangentSpace, FVector
export PowerManifold, ProductManifold
export ProjectedPointDistribution, ProductRepr, TangentBundle, TangentBundleFibers
export TangentSpace, TangentSpaceAtPoint, VectorSpaceAtPoint, VectorSpaceType, VectorBundle
export VectorBundleFibers
export AbstractVectorTransportMethod, ParallelTransport, ProjectedPointDistribution
export Metric, RiemannianMetric, LorentzMetric, MinkowskiMetric, EuclideanMetric, MetricManifold,
    LinearAffineMetric, LogEuclideanMetric, LogCholeskyMetric
export AbstractVectorTransportMethod, ParallelTransport, ProjectionTransport
export AbstractRetractionMethod, QRRetraction, PolarRetraction
export AbstractInverseRetractionMethod, QRInverseRetraction, PolarInverseRetraction
export AbstractEstimationMethod, GradientDescentEstimation, CyclicProximalPointEstimation,
    GeodesicInterpolation, GeodesicInterpolationWithinRadius
export base_manifold, bundle_projection, christoffel_symbols_first, christoffel_symbols_second,
    christoffel_symbols_second_jacobian, complex_dot, det_local_metric, distance,
    einstein_tensor, exp, exp!, flat, flat!, gaussian_curvature, geodesic, hat, hat!,
    injectivity_radius, inner, inverse_local_metric, inverse_retract, inverse_retract!,
    isapprox, is_decorator_manifold, is_default_metric, is_manifold_point,
    is_tangent_vector, isapprox, inner, kurtosis, local_metric, local_metric_jacobian,
    log, log!, log_local_metric_density, manifold_dimension, metric, mean, mean!,
    mean_and_var, mean_and_std, median, median!, moment, norm, normal_tvector_distribution,
    one, project_point, project_point!, project_tangent, project_tangent!,
    projected_distribution, ricci_curvature, ricci_tensor, representation_size, retract,
    retract!, riemann_tensor, sharp, sharp!, shortest_geodesic, similar_result, skewness,
    std, sym_rem, submanifold, submanifold_component, tangent_orthonormal_basis, var,
    vector_space_dimension, vector_transport_along, vector_transport_along!,
    vector_transport_direction, vector_transport_direction!, vector_transport_to,
    vector_transport_to!, vee, vee!, zero_vector, zero_vector!, zero_tangent_vector,
    zero_tangent_vector!
end # module
