module Manifolds

import Base: isapprox,
    exp,
    log,
    angle,
    eltype,
    similar,
    getindex,
    setindex!,
    size,
    length,
    copy,
    copyto!,
    convert,
    dataids,
    axes,
    promote_rule,
    inv,
    +,
    -,
    *,
    \,
    /
import Statistics: mean,
    mean!,
    median,
    median!,
    var,
    std
import StatsBase: mean_and_std,
    mean_and_var,
    moment,
    kurtosis,
    skewness
import LinearAlgebra: dot,
    norm,
    det,
    cross,
    I,
    UniformScaling,
    Diagonal

using ManifoldsBase
using ManifoldsBase: Manifold,
    MPoint,
    TVector,
    CoTVector,
    ArrayCoTVector,
    ArrayManifold,
    ArrayMPoint,
    ArrayTVector,
    ArrayCoTVector,
    AbstractRetractionMethod,
    AbstractInverseRetractionMethod,
    AbstractVectorTransportMethod,
    ExponentialRetraction,
    LogarithmicInverseRetraction
    ParallelTransport,
    ProjectionTransport
import ManifoldsBase: base_manifold,
    check_manifold_point,
    check_tangent_vector,
    distance,
    exp,
    exp!,
    geodesic,
    injectivity_radius,
    inner,
    isapprox,
    is_manifold_point,
    is_tangent_vector,
    is_decorator_manifold,
    inverse_retract,
    inverse_retract!,
    log,
    log!,
    manifold_dimension,
    norm,
    project_point,
    project_point!,
    project_tangent,
    project_tangent!,
    representation_size,
    retract,
    retract!,
    shortest_geodesic,
    vector_transport_along,
    vector_transport_along!,
    vector_transport_direction,
    vector_transport_direction!,
    vector_transport_to,
    vector_transport_to!,
    zero_tangent_vector,
    zero_tangent_vector!,
    similar_result

using Requires
using StaticArrays
using HybridArrays
using Markdown: @doc_str
import Distributions: _rand!, support
import Random: rand
using LinearAlgebra
using Random: AbstractRNG
using FiniteDifferences
using UnsafeArrays
using Einsum: @einsum

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

$\wedge: v^i \mapsto v^i e_i$

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

$\vee: v^i e_i \mapsto v^i$

For matrix manifolds, this converts a  matrix representation of the tangent
vector to a vector representation. The [`hat`](@ref) map is the `vee` map's
inverse.
"""
function vee(M::Manifold, x, v)
    vⁱ = MVector{manifold_dimension(M),eltype(v)}(undef)
    vee!(M, vⁱ, x, v)
    return vⁱ
end
vee!(M::Manifold, vⁱ, x, v) = error("vee! operator not defined for manifold $(typeof(M)), matrix $(typeof(v)), and vector $(typeof(vⁱ))")

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
include("Euclidean.jl")
include("FixedRankMatrices.jl")
include("Grassmann.jl")
include("Rotations.jl")
include("Stiefel.jl")
include("Sphere.jl")
include("Symmetric.jl")
include("SymmetricPositiveDefinite.jl")

include("Group.jl")
include("GroupAction.jl")

include("SpecialOrthogonal.jl")
include("RotationAction.jl")

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

# Base Types
export Manifold,
    Euclidean,
    Sphere,
    SymmetricMatrices,
    MPoint,
    TVector,
    CoTVector,
    SVDMPoint,
    UMVTVector
# decorator manifolds
export ArrayManifold,
    ArrayMPoint,
    ArrayTVector,
    ArrayCoTVector,
    CotangentBundle,
    CotangentSpaceAtPoint,
    CotangentBundleFibers,
    CotangentSpace,
    FVector,
    PowerManifold,
    ProductManifold,
    ProjectedPointDistribution,
    ProductRepr,
    TangentBundle,
    TangentBundleFibers,
    TangentSpace,
    TangentSpaceAtPoint,
    VectorSpaceAtPoint,
    VectorSpaceType,
    VectorBundle,
    VectorBundleFibers,
    FVector,
    TangentBundle,
    CotangentBundle,
    TangentBundleFibers,
    CotangentBundleFibers,
    AbstractVectorTransportMethod,
    ParallelTransport,
    ProjectedPointDistribution
# Manifolds
export CholeskySpace,
    Euclidean,
    FixedRankMatrices,
    Grassmann,
    Rotations,
    Sphere,
    Stiefel,
    SymmetricPositiveDefinite
# Types
export Metric,
    RiemannianMetric,
    LorentzMetric,
    EuclideanMetric,
    MetricManifold,
    LinearAffineMetric,
    LogEuclideanMetric,
    LogCholeskyMetric,
    AbstractVectorTransportMethod,
    ParallelTransport,
    ProjectionTransport,
    AbstractRetractionMethod,
    QRRetraction,
    PolarRetraction,
    AbstractInverseRetractionMethod,
    QRInverseRetraction,
    PolarInverseRetraction,
    AbstractEstimationMethod,
    GradientDescentEstimation,
    CyclicProximalPointEstimation,
    GeodesicInterpolation,
    GeodesicInterpolationWithinRadius
export base_manifold,
    bundle_projection,
    christoffel_symbols_first,
    christoffel_symbols_second,
    christoffel_symbols_second_jacobian,
    det_local_metric,
    distance,
    einstein_tensor,
    exp,
    exp!,
    flat,
    flat!,
    gaussian_curvature,
    geodesic,
    hat,
    hat!,
    injectivity_radius,
    inner,
    inverse_local_metric,
    inverse_retract,
    inverse_retract!,
    isapprox,
    is_decorator_manifold,
    is_default_metric,
    is_manifold_point,
    is_tangent_vector,
    isapprox,
    inner,
    kurtosis,
    local_metric,
    local_metric_jacobian,
    log,
    log!,
    log_local_metric_density,
    manifold_dimension,
    metric,
    mean,
    mean!,
    mean_and_var,
    mean_and_std,
    median,
    median!,
    moment,
    norm,
    normal_tvector_distribution,
    one,
    project_point,
    project_point!,
    project_tangent,
    project_tangent!,
    projected_distribution,
    ricci_curvature,
    ricci_tensor,
    representation_size,
    retract,
    retract!,
    riemann_tensor,
    sharp,
    sharp!,
    shortest_geodesic,
    skewness,
    std,
    submanifold,
    submanifold_component,
    tangent_orthonormal_basis,
    var,
    vector_space_dimension,
    vector_transport_along,
    vector_transport_along!,
    vector_transport_direction,
    vector_transport_direction!,
    vector_transport_to,
    vector_transport_to!,
    vee,
    vee!,
    zero_vector,
    zero_vector!,
    zero_tangent_vector,
    zero_tangent_vector!
# Lie groups and actions
export AbstractGroupOperation,
    AbstractGroupManifold,
    LeftInvariantCanonicalMetric,
    RightInvariantCanonicalMetric,
    ActionDirection,
    LeftAction,
    RightAction,
    Identity,
    group_id,
    group_id!,
    inv,
    inv!,
    compose_left,
    compose_left!,
    translate,
    inverse_translate,
    AbstractActionOnManifold,
    base_group,
    action_on,
    apply_action,
    apply_action!,
    optimal_alignment,
    optimal_alignment!,
    center_of_orbit,
    SpecialOrthogonal,
    RotationAction
end # module
