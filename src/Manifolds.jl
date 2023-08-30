"""
`Manifolds.jl` provides a library of manifolds aiming for an easy-to-use and fast implementation.
"""
module Manifolds

import Base:
    angle,
    copyto!,
    convert,
    foreach,
    identity,
    in,
    inv,
    isempty,
    length,
    ndims,
    show,
    showerror,
    size,
    transpose
import ManifoldsBase:
    @trait_function,
    _access_nested,
    _get_basis,
    _injectivity_radius,
    _inverse_retract,
    _inverse_retract!,
    _read,
    _retract,
    _retract!,
    _write,
    active_traits,
    allocate,
    allocate_coordinates,
    allocate_result,
    allocate_result_type,
    allocation_promotion_function,
    array_value,
    base_manifold,
    change_basis,
    change_basis!,
    change_metric,
    change_metric!,
    change_representer,
    change_representer!,
    check_point,
    check_size,
    check_vector,
    copy,
    copyto!,
    default_inverse_retraction_method,
    default_retraction_method,
    default_vector_transport_method,
    decorated_manifold,
    distance,
    dual_basis,
    embed,
    embed!,
    exp,
    exp!,
    get_basis,
    get_basis_default,
    get_basis_diagonalizing,
    get_basis_orthogonal,
    get_basis_orthonormal,
    get_basis_vee,
    get_component,
    get_coordinates,
    get_coordinates!,
    get_coordinates_diagonalizing,
    get_coordinates_diagonalizing!,
    get_coordinates_orthogonal,
    get_coordinates_orthonormal,
    get_coordinates_orthogonal!,
    get_coordinates_orthonormal!,
    get_coordinates_vee!,
    get_embedding,
    get_iterator,
    get_vector,
    get_vector!,
    get_vector_diagonalizing,
    get_vector_diagonalizing!,
    get_vector_orthogonal,
    get_vector_orthonormal,
    get_vector_orthogonal!,
    get_vector_orthonormal!,
    get_vectors,
    gram_schmidt,
    hat,
    hat!,
    injectivity_radius,
    _injectivity_radius,
    injectivity_radius_exp,
    inner,
    isapprox,
    _isapprox,
    is_flat,
    is_point,
    is_vector,
    inverse_retract,
    inverse_retract!,
    _inverse_retract,
    _inverse_retract!,
    inverse_retract_cayley!,
    inverse_retract_embedded!,
    inverse_retract_pade!,
    inverse_retract_polar!,
    inverse_retract_project!,
    inverse_retract_qr!,
    inverse_retract_shooting!,
    inverse_retract_softmax!,
    log,
    log!,
    manifold_dimension,
    mid_point,
    mid_point!,
    norm,
    number_eltype,
    number_of_coordinates,
    parallel_transport_along,
    parallel_transport_along!,
    parallel_transport_direction,
    parallel_transport_direction!,
    parallel_transport_to,
    parallel_transport_to!,
    parent_trait,
    power_dimensions,
    project,
    project!,
    rand,
    rand!,
    representation_size,
    retract,
    retract!,
    retract_cayley!,
    retract_exp_ode!,
    retract_pade!,
    retract_polar!,
    retract_project!,
    retract_qr!,
    retract_softmax!,
    riemann_tensor,
    riemann_tensor!,
    set_component!,
    vector_space_dimension,
    vector_transport_along, # just specified in Euclidean - the next 5 as well
    vector_transport_along_diff,
    vector_transport_along_project,
    vector_transport_along!,
    vector_transport_along_diff!,
    vector_transport_along_project!,
    vector_transport_direction,
    vector_transport_direction_diff,
    vector_transport_direction!,
    vector_transport_direction_diff!,
    vector_transport_to,
    vector_transport_to_diff,
    vector_transport_to_project,
    vector_transport_to!,
    vector_transport_to_diff!,
    vector_transport_to_project!, # some overwrite layer 2
    _vector_transport_direction,
    _vector_transport_direction!,
    _vector_transport_to,
    _vector_transport_to!,
    vee,
    vee!,
    Weingarten,
    Weingarten!,
    zero_vector,
    zero_vector!,
    CotangentSpace,
    TangentSpace
import ManifoldDiff:
    adjoint_Jacobi_field,
    adjoint_Jacobi_field!,
    diagonalizing_projectors,
    jacobi_field,
    jacobi_field!,
    riemannian_gradient,
    riemannian_gradient!,
    riemannian_Hessian,
    riemannian_Hessian!

using Base.Iterators: repeated
using Distributions
using Einsum: @einsum
using HybridArrays
using Kronecker
using Graphs
using LinearAlgebra
using ManifoldsBase:
    @next_trait_function,
    ℝ,
    ℂ,
    ℍ,
    AbstractBasis,
    AbstractDecoratorManifold,
    AbstractInverseRetractionMethod,
    AbstractManifold,
    AbstractManifoldPoint,
    AbstractMetric,
    AbstractNumbers,
    AbstractOrthogonalBasis,
    AbstractOrthonormalBasis,
    AbstractPowerManifold,
    AbstractPowerRepresentation,
    AbstractRetractionMethod,
    AbstractTrait,
    AbstractVectorTransportMethod,
    AbstractLinearVectorTransportMethod,
    ApproximateInverseRetraction,
    ApproximateRetraction,
    CachedBasis,
    CayleyRetraction,
    CayleyInverseRetraction,
    ComplexNumbers,
    ComponentManifoldError,
    CompositeManifoldError,
    CotangentSpaceType,
    CoTFVector,
    DefaultBasis,
    DefaultOrthogonalBasis,
    DefaultOrthonormalBasis,
    DefaultOrDiagonalizingBasis,
    DiagonalizingBasisData,
    DiagonalizingOrthonormalBasis,
    DifferentiatedRetractionVectorTransport,
    EmbeddedManifold,
    EmptyTrait,
    EuclideanMetric,
    ExponentialRetraction,
    FVector,
    IsIsometricEmbeddedManifold,
    IsEmbeddedManifold,
    IsEmbeddedSubmanifold,
    IsExplicitDecorator,
    LogarithmicInverseRetraction,
    ManifoldsBase,
    NestedPowerRepresentation,
    NestedReplacingPowerRepresentation,
    TraitList,
    NLSolveInverseRetraction,
    ODEExponentialRetraction,
    OutOfInjectivityRadiusError,
    PadeRetraction,
    PadeInverseRetraction,
    ParallelTransport,
    PolarInverseRetraction,
    PolarRetraction,
    PoleLadderTransport,
    PowerManifold,
    PowerManifoldNested,
    PowerManifoldNestedReplacing,
    ProjectedOrthonormalBasis,
    ProjectionInverseRetraction,
    ProjectionRetraction,
    ProjectionTransport,
    QuaternionNumbers,
    QRInverseRetraction,
    QRRetraction,
    RealNumbers,
    RiemannianMetric,
    ScaledVectorTransport,
    SchildsLadderTransport,
    ShootingInverseRetraction,
    SoftmaxRetraction,
    SoftmaxInverseRetraction,
    TangentSpaceType,
    TCoTSpaceType,
    TFVector,
    TVector,
    ValidationManifold,
    ValidationMPoint,
    ValidationTVector,
    VectorSpaceType,
    VeeOrthogonalBasis,
    @invoke_maker,
    _euclidean_basis_vector,
    combine_allocation_promotion_functions,
    geodesic,
    geodesic!,
    merge_traits,
    next_trait,
    number_system,
    real_dimension,
    rep_size_to_colons,
    shortest_geodesic,
    shortest_geodesic!,
    size_to_tuple,
    trait
using ManifoldDiff: ManifoldDiff
using ManifoldDiff:
    default_differential_backend,
    _derivative,
    _derivative!,
    differential,
    differential!,
    gradient,
    gradient!,
    _gradient,
    _gradient!,
    hessian,
    _hessian,
    jacobian,
    _jacobian,
    _jacobian!,
    set_default_differential_backend!
using ManifoldDiff:
    AbstractDiffBackend,
    AbstractRiemannianDiffBackend,
    CoprojectorOntoVector,
    ExplicitEmbeddedBackend,
    IdentityProjector,
    NoneDiffBackend,
    ProjectorOntoVector,
    RiemannianProjectionBackend,
    TangentDiffBackend
using Markdown: @doc_str
using MatrixEquations: lyapc, sylvc
using Quaternions: Quaternions
using Random
using RecursiveArrayTools: ArrayPartition
using Requires
using SimpleWeightedGraphs: AbstractSimpleWeightedGraph, get_weight
using SpecialFunctions
using StaticArrays
using Statistics
using StatsBase
using StatsBase: AbstractWeights

include("utils.jl")

include("product_representations.jl")

include("manifold_fallbacks.jl")

# Main Meta Manifolds
include("manifolds/ConnectionManifold.jl")
include("manifolds/MetricManifold.jl")
include("manifolds/QuotientManifold.jl")
include("manifolds/VectorBundle.jl")
include("groups/group.jl")

# Features I: Which are extended on Meta Manifolds
include("distributions.jl")
include("projected_distribution.jl")
include("statistics.jl")

# Meta Manifolds II: Products
include("manifolds/ProductManifold.jl")

METAMANIFOLDS = [
    AbstractManifold,
    AbstractDecoratorManifold,
    AbstractPowerManifold,
    PowerManifoldNested,
    PowerManifoldNestedReplacing,
    ProductManifold,
    TangentSpaceAtPoint,
    ValidationManifold,
    VectorBundle,
]

# Features II: That require metas
include("atlases.jl")
include("differentiation/ode_callback.jl")
include("cotangent_space.jl")

# Meta Manifolds II: Power Manifolds
include("manifolds/PowerManifold.jl")
include("manifolds/GraphManifold.jl")

#
# Manifolds
#
include("manifolds/Euclidean.jl")
include("manifolds/Lorentz.jl")

include("manifolds/CenteredMatrices.jl")
include("manifolds/CholeskySpace.jl")
include("manifolds/Circle.jl")
include("manifolds/Elliptope.jl")
include("manifolds/EmbeddedTorus.jl")
include("manifolds/FixedRankMatrices.jl")
include("manifolds/Flag.jl")
include("manifolds/FlagOrthogonal.jl")
include("manifolds/FlagStiefel.jl")
include("manifolds/GeneralizedGrassmann.jl")
include("manifolds/GeneralizedStiefel.jl")
include("manifolds/Hyperbolic.jl")
include("manifolds/MultinomialDoublyStochastic.jl")
include("manifolds/MultinomialSymmetric.jl")
include("manifolds/PositiveNumbers.jl")
include("manifolds/ProjectiveSpace.jl")
include("manifolds/SkewHermitian.jl")
include("manifolds/Spectrahedron.jl")
include("manifolds/Stiefel.jl")
include("manifolds/StiefelEuclideanMetric.jl")
include("manifolds/StiefelCanonicalMetric.jl")
include("manifolds/StiefelSubmersionMetric.jl")
include("manifolds/Sphere.jl")
include("manifolds/SphereSymmetricMatrices.jl")
include("manifolds/Symmetric.jl")
include("manifolds/SymmetricPositiveDefinite.jl")
include("manifolds/SPDFixedDeterminant.jl")
include("manifolds/SymmetricPositiveDefiniteBuresWasserstein.jl")
include("manifolds/SymmetricPositiveDefiniteGeneralizedBuresWasserstein.jl")
include("manifolds/SymmetricPositiveDefiniteAffineInvariant.jl")
include("manifolds/SymmetricPositiveDefiniteLogCholesky.jl")
include("manifolds/SymmetricPositiveDefiniteLogEuclidean.jl")
include("manifolds/SymmetricPositiveSemidefiniteFixedRank.jl")
include("manifolds/Symplectic.jl")
include("manifolds/SymplecticStiefel.jl")
include("manifolds/Tucker.jl")
#
include("manifolds/ProbabilitySimplex.jl")
include("manifolds/ProbabilitySimplexEuclideanMetric.jl")
include("manifolds/GeneralUnitaryMatrices.jl")
include("manifolds/Unitary.jl")
include("manifolds/Rotations.jl")
include("manifolds/Orthogonal.jl")

# shape spaces require Sphere
include("manifolds/KendallsPreShapeSpace.jl")
include("manifolds/KendallsShapeSpace.jl")

# Introduce the quotient, Grassmann, only after Stiefel
include("manifolds/Grassmann.jl")

# Product or power based manifolds
include("manifolds/Torus.jl")
include("manifolds/Multinomial.jl")
include("manifolds/Oblique.jl")
include("manifolds/EssentialManifold.jl")

#
# Group Manifolds
include("groups/GroupManifold.jl")

# a) generics
include("groups/addition_operation.jl")
include("groups/multiplication_operation.jl")
include("groups/connections.jl")
include("groups/metric.jl")
include("groups/group_action.jl")
include("groups/group_operation_action.jl")
include("groups/validation_group.jl")
include("groups/product_group.jl")
include("groups/semidirect_product_group.jl")
include("groups/power_group.jl")

# generic group - commopn (special) unitary/orthogonal functions
include("groups/general_unitary_groups.jl")
# Special Group Manifolds
include("groups/general_linear.jl")
include("groups/special_linear.jl")
include("groups/translation_group.jl")
include("groups/unitary.jl")
include("groups/orthogonal.jl")
include("groups/special_unitary.jl")
include("groups/special_orthogonal.jl")
include("groups/circle_group.jl")
include("groups/heisenberg.jl")

include("groups/translation_action.jl")
include("groups/rotation_action.jl")

include("groups/special_euclidean.jl")

# final utilities
include("trait_recursion_breaking.jl")

@doc raw"""
    Base.in(p, M::AbstractManifold; kwargs...)
    p ∈ M

Check, whether a point `p` is a valid point (i.e. in) a [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  `M`.
This method employs [`is_point`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.is_point) deactivating the error throwing option.
"""
Base.in(p, M::AbstractManifold; kwargs...) = is_point(M, p, false; kwargs...)

@doc raw"""
    Base.in(p, TpM::TangentSpaceAtPoint; kwargs...)
    X ∈ TangentSpaceAtPoint(M,p)

Check whether `X` is a tangent vector from (in) the tangent space $T_p\mathcal M$, i.e.
the [`TangentSpaceAtPoint`](@ref) at `p` on the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  `M`.
This method uses [`is_vector`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.is_vector) deactivating the error throw option.
"""
function Base.in(X, TpM::TangentSpaceAtPoint; kwargs...)
    return is_vector(base_manifold(TpM), TpM.point, X, false; kwargs...)
end

@doc raw"""
    manifold_volume(M::AbstractManifold)

Volume of manifold `M` defined through integration of Riemannian volume element in a chart.
Note that for many manifolds there is no universal agreement over the exact ranges over
which the integration should happen. For details see [BoyaSudarshanTilma:2003](@cite).
"""
manifold_volume(::AbstractManifold)

@doc raw"""
    volume_density(M::AbstractManifold, p, X)

Volume density function of manifold `M`, i.e. determinant of the differential of exponential map
`exp(M, p, X)`. Determinant can be understood as computed in a basis, from the matrix
of the linear operator said differential corresponds to. Details are available in Section 4.1
of [ChevallierLiLuDunson:2022](@cite).

Note that volume density is well-defined only for `X` for which `exp(M, p, X)` is injective.
"""
volume_density(::AbstractManifold, p, X)

# functions populated with methods by extensions

function solve_chart_log_bvp end
function estimate_distance_from_bvp end

function solve_chart_exp_ode end
function solve_chart_parallel_transport_ode end

function find_eps end
function test_parallel_transport end
function test_manifold end
function test_group end
function test_action end

# end of functions populated with methods by extensions

function __init__()
    @static if isdefined(Base.Experimental, :register_error_hint)
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
            if exc.f === solve_exp_ode
                print(io, "\nDid you forget to load OrdinaryDiffEq? For example: ")
                printstyled(io, "`using OrdinaryDiffEq`", color=:cyan)
            end
        end
    end

    @static if !isdefined(Base, :get_extension)
        @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
            @require DiffEqCallbacks = "459566f4-90b8-5000-8ac3-15dfb0a30def" begin
                include("../ext/ManifoldsOrdinaryDiffEqDiffEqCallbacksExt.jl")
            end
        end

        @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
            include("../ext/ManifoldsOrdinaryDiffEqExt.jl")
        end

        @require BoundaryValueDiffEq = "764a87c0-6b3e-53db-9096-fe964310641d" begin
            include("../ext/ManifoldsBoundaryValueDiffEqExt.jl")
        end

        @require NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56" begin
            include("../ext/ManifoldsNLsolveExt.jl")
        end

        @require RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01" begin
            @require Colors = "5ae59095-9a9b-59fe-a467-6f913c188581" begin
                include("../ext/ManifoldsRecipesBaseExt.jl")
            end
        end

        @require Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40" begin
            include("../ext/ManifoldsTestExt/ManifoldsTestExt.jl")
        end
    end

    return nothing
end

include("deprecated.jl")

export test_manifold
export test_group, test_action

#
export CoTVector, AbstractManifold, AbstractManifoldPoint, TVector
export AbstractSphere, AbstractProjectiveSpace
export Euclidean,
    ArrayProjectiveSpace,
    ArraySphere,
    CenteredMatrices,
    CholeskySpace,
    Circle,
    Elliptope,
    EssentialManifold,
    FixedRankMatrices,
    Flag,
    GeneralizedGrassmann,
    GeneralizedStiefel,
    Grassmann,
    HeisenbergGroup,
    Hyperbolic,
    KendallsPreShapeSpace,
    KendallsShapeSpace,
    Lorentz,
    MultinomialDoubleStochastic,
    MultinomialMatrices,
    MultinomialSymmetric,
    Oblique,
    OrthogonalMatrices,
    PositiveArrays,
    PositiveMatrices,
    PositiveNumbers,
    PositiveVectors,
    ProbabilitySimplex,
    ProjectiveSpace,
    Rotations,
    SkewHermitianMatrices,
    SkewSymmetricMatrices,
    Spectrahedron,
    Sphere,
    SphereSymmetricMatrices,
    Stiefel,
    SymmetricMatrices,
    SymmetricPositiveDefinite,
    SPDFixedDeterminant,
    SymmetricPositiveSemidefiniteFixedRank,
    Symplectic,
    SymplecticStiefel,
    SymplecticMatrix,
    Torus,
    Tucker,
    UnitaryMatrices
# Point representation types
export HyperboloidPoint,
    PoincareBallPoint,
    PoincareHalfSpacePoint,
    SVDMPoint,
    TuckerPoint,
    StiefelPoint,
    ProjectorPoint,
    SPDPoint
# Tangent vector representation types
export HyperboloidTVector,
    PoincareBallTVector,
    PoincareHalfSpaceTVector,
    TuckerTVector,
    UMVTVector,
    ProjectorTVector,
    StiefelTVector
export AbstractNumbers, ℝ, ℂ, ℍ

# decorator manifolds
export AbstractDecoratorManifold
export IsIsometricEmbeddedManifold, IsEmbeddedManifold, IsEmbeddedSubmanifold
export IsDefaultMetric, IsDefaultConnection, IsMetricManifold, IsConnectionManifold
export ValidationManifold, ValidationMPoint, ValidationTVector, ValidationCoTVector
export CotangentBundle,
    CotangentSpaceAtPoint, CotangentBundleFibers, CotangentSpace, FVector
export AbstractPowerManifold,
    AbstractPowerRepresentation,
    ArrayPowerRepresentation,
    NestedPowerRepresentation,
    NestedReplacingPowerRepresentation,
    PowerManifold,
    QuotientManifold
export ProductManifold, EmbeddedManifold
export GraphManifold, GraphManifoldType, VertexManifold, EdgeManifold
export ProductRepr, ArrayPartition
export ProjectedPointDistribution, TangentBundle, TangentBundleFibers
export TangentSpace, TangentSpaceAtPoint, VectorSpaceAtPoint, VectorSpaceType, VectorBundle
export VectorBundleFibers
export AbstractVectorTransportMethod,
    DifferentiatedRetractionVectorTransport, ParallelTransport, ProjectedPointDistribution
export PoleLadderTransport, SchildsLadderTransport
export ProductVectorTransport
export AbstractAffineConnection,
    AbstractConnectionManifold, ConnectionManifold, LeviCivitaConnection
export AbstractCartanSchoutenConnection,
    CartanSchoutenMinus, CartanSchoutenPlus, CartanSchoutenZero
export MetricManifold
# Metric types
export AbstractMetric,
    AffineInvariantMetric,
    BuresWassersteinMetric,
    CanonicalMetric,
    EuclideanMetric,
    ExtendedSymplecticMetric,
    FisherRaoMetric,
    GeneralizedBuresWassersteinMetric,
    LogCholeskyMetric,
    LogEuclideanMetric,
    LorentzMetric,
    MinkowskiMetric,
    PowerMetric,
    ProductMetric,
    RealSymplecticMetric,
    RiemannianMetric,
    StiefelSubmersionMetric
export AbstractAtlas, RetractionAtlas
# Vector transport types
export AbstractVectorTransportMethod, ParallelTransport, ProjectionTransport
# Retraction types
export AbstractRetractionMethod,
    CayleyRetraction,
    ExponentialRetraction,
    QRRetraction,
    PolarRetraction,
    ProjectionRetraction,
    SoftmaxRetraction,
    ODEExponentialRetraction,
    PadeRetraction,
    ProductRetraction,
    PowerRetraction,
    SasakiRetraction
# Inverse Retraction types
export AbstractInverseRetractionMethod,
    ApproximateInverseRetraction,
    ApproximateLogarithmicMap,
    CayleyInverseRetraction,
    LogarithmicInverseRetraction,
    QRInverseRetraction,
    PolarInverseRetraction,
    ProjectionInverseRetraction,
    ShootingInverseRetraction,
    SoftmaxInverseRetraction
# Estimation methods for median and mean
export AbstractEstimationMethod,
    GradientDescentEstimation,
    CyclicProximalPointEstimation,
    GeodesicInterpolation,
    GeodesicInterpolationWithinRadius,
    ExtrinsicEstimation
# Tangent space bases
export CachedBasis,
    DefaultBasis,
    DefaultOrthogonalBasis,
    DefaultOrthonormalBasis,
    DiagonalizingOrthonormalBasis,
    InducedBasis,
    ProjectedOrthonormalBasis
# Errors on Manifolds
export ComponentManifoldError, CompositeManifoldError
# Functions on Manifolds
export ×,
    allocate,
    allocate_result,
    base_manifold,
    bundle_projection,
    canonical_project,
    canonical_project!,
    change_basis,
    change_basis!,
    change_metric,
    change_metric!,
    change_representer,
    change_representer!,
    check_chart_switch,
    check_point,
    check_vector,
    christoffel_symbols_first,
    christoffel_symbols_second,
    christoffel_symbols_second_jacobian,
    convert,
    complex_dot,
    decorated_manifold,
    default_vector_transport_method,
    default_inverse_retraction_method,
    default_retraction_method,
    det_local_metric,
    differential_canonical_project,
    differential_canonical_project!,
    distance,
    dual_basis,
    einstein_tensor,
    embed,
    embed!,
    equiv,
    exp,
    exp!,
    flat,
    flat!,
    gaussian_curvature,
    geodesic,
    geodesic!,
    get_default_atlas,
    get_component,
    get_embedding,
    get_orbit_action,
    get_total_space,
    grad_euclidean_to_manifold,
    grad_euclidean_to_manifold!,
    hat,
    hat!,
    horizontal_component,
    horizontal_component!,
    horizontal_lift,
    horizontal_lift!,
    identity_element,
    identity_element!,
    induced_basis,
    incident_log,
    injectivity_radius,
    inner,
    inverse_local_metric,
    inverse_retract,
    inverse_retract!,
    isapprox,
    is_default_connection,
    is_default_metric,
    is_flat,
    is_group_manifold,
    is_identity,
    is_point,
    is_vector,
    kurtosis,
    local_metric,
    local_metric_jacobian,
    log,
    log!,
    log_local_metric_density,
    manifold_dimension,
    manifold_volume,
    metric,
    mean,
    mean!,
    mean_and_var,
    mean_and_std,
    median,
    median!,
    mid_point,
    mid_point!,
    minkowski_metric,
    moment,
    norm,
    normal_tvector_distribution,
    number_eltype,
    one,
    power_dimensions,
    parallel_transport_along,
    parallel_transport_along!,
    parallel_transport_direction,
    parallel_transport_direction!,
    parallel_transport_to,
    parallel_transport_to!,
    project,
    project!,
    projected_distribution,
    rand,
    rand!,
    real_dimension,
    ricci_curvature,
    ricci_tensor,
    representation_size,
    retract,
    retract!,
    riemannian_gradient,
    riemannian_gradient!,
    riemannian_Hessian,
    riemannian_Hessian!,
    riemann_tensor,
    riemann_tensor!,
    set_component!,
    sharp,
    sharp!,
    shortest_geodesic,
    shortest_geodesic!,
    skewness,
    std,
    sym_rem,
    symplectic_inverse_times,
    symplectic_inverse_times!,
    submanifold,
    submanifold_component,
    submanifold_components,
    uniform_distribution,
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
    vertical_component,
    vertical_component!,
    volume_density,
    Weingarten,
    Weingarten!,
    zero_vector,
    zero_vector!
# Lie group types & functions
export AbstractGroupAction,
    AbstractGroupOperation,
    ActionDirection,
    AdditionOperation,
    CircleGroup,
    GeneralLinear,
    GroupManifold,
    GroupOperationAction,
    Identity,
    InvariantMetric,
    LeftAction,
    LeftBackwardAction,
    LeftForwardAction,
    LeftInvariantMetric,
    MultiplicationOperation,
    Orthogonal,
    PowerGroup,
    ProductGroup,
    ProductOperation,
    RealCircleGroup,
    RightAction,
    RightBackwardAction,
    RightForwardAction,
    RightInvariantMetric,
    RotationAction,
    SemidirectProductGroup,
    SpecialEuclidean,
    SpecialLinear,
    SpecialOrthogonal,
    SpecialUnitary,
    TranslationGroup,
    TranslationAction,
    Unitary
export AbstractInvarianceTrait
export IsMetricManifold, IsConnectionManifold
export IsGroupManifold,
    HasLeftInvariantMetric, HasRightInvariantMetric, HasBiinvariantMetric
export adjoint_action,
    adjoint_action!,
    adjoint_apply_diff_group,
    adjoint_apply_diff_group!,
    affine_matrix,
    apply,
    apply!,
    apply_diff,
    apply_diff!,
    apply_diff_group,
    apply_diff_group!,
    base_group,
    center_of_orbit,
    has_approx_invariant_metric,
    compose,
    compose!,
    direction,
    exp_lie,
    exp_lie!,
    group_manifold,
    geodesic,
    get_coordinates_lie,
    get_coordinates_lie!,
    get_coordinates_orthogonal,
    get_coordinates_orthonormal,
    get_coordinates_orthogonal!,
    get_coordinates_orthonormal!,
    get_coordinates_vee!,
    get_vector_diagonalizing!,
    get_vector_lie,
    get_vector_lie!,
    get_vector_orthogonal,
    get_vector_orthonormal,
    has_biinvariant_metric,
    has_invariant_metric,
    identity_element,
    identity_element!,
    inv,
    inv!,
    inverse_apply,
    inverse_apply!,
    inverse_apply_diff,
    inverse_apply_diff!,
    inverse_translate,
    inverse_translate!,
    inverse_translate_diff,
    inverse_translate_diff!,
    lie_bracket,
    lie_bracket!,
    log_lie,
    log_lie!,
    optimal_alignment,
    optimal_alignment!,
    screw_matrix,
    switch_direction,
    translate,
    translate!,
    translate_diff,
    translate_diff!
# Orthonormal bases
export AbstractBasis,
    AbstractOrthonormalBasis,
    DefaultOrthonormalBasis,
    DiagonalizingOrthonormalBasis,
    ProjectedOrthonormalBasis,
    CachedBasis,
    DiagonalizingBasisData,
    ProductBasisData,
    PowerBasisData
export OutOfInjectivityRadiusError
export get_basis,
    get_coordinates, get_coordinates!, get_vector, get_vector!, get_vectors, number_system

# atlases and charts
export get_point, get_point!, get_parameters, get_parameters!

end # module
