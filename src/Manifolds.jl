module Manifolds

import ManifoldsBase:
    _access_nested,
    _read,
    _write,
    allocate,
    allocate_result,
    allocate_result_type,
    allocation_promotion_function,
    array_value,
    base_manifold,
    check_point,
    check_point__transparent,
    check_vector,
    decorated_manifold,
    decorator_transparent_dispatch,
    default_decorator_dispatch,
    distance,
    dual_basis,
    embed,
    embed!,
    exp!,
    exp!__intransparent,
    get_basis,
    get_component,
    get_coordinates,
    get_coordinates!,
    get_embedding,
    get_iterator,
    get_vector,
    get_vector!,
    get_vectors,
    gram_schmidt,
    injectivity_radius,
    inner,
    inner__intransparent,
    is_point,
    is_vector,
    inverse_retract,
    inverse_retract!,
    log,
    log!,
    manifold_dimension,
    mid_point,
    mid_point!,
    number_eltype,
    number_of_coordinates,
    power_dimensions,
    project,
    project!,
    representation_size,
    retract,
    retract!,
    set_component!,
    vector_space_dimension,
    vector_transport_direction,
    vector_transport_direction!,
    vector_transport_to,
    vector_transport_to!,
    zero_vector,
    zero_vector!,
    CotangentSpace,
    TangentSpace
import Base:
    copyto!, convert, foreach, in, isapprox, isempty, length, ndims, showerror, size

using Base.Iterators: repeated
using Distributions
using Einsum: @einsum
using FiniteDifferences
using HybridArrays
using Kronecker
using LightGraphs
using LinearAlgebra
using ManifoldsBase
using ManifoldsBase:
    ℝ,
    ℂ,
    ℍ,
    AbstractBasis,
    AbstractDecoratorType,
    AbstractIsometricEmbeddingType,
    AbstractNumbers,
    AbstractOrthogonalBasis,
    AbstractOrthonormalBasis,
    AbstractPowerManifold,
    AbstractPowerRepresentation,
    AbstractVectorTransportMethod,
    AbstractLinearVectorTransportMethod,
    ApproximateInverseRetraction,
    ApproximateRetraction,
    DifferentiatedRetractionVectorTransport,
    ComponentManifoldError,
    CompositeManifoldError,
    CotangentSpaceType,
    CoTFVector,
    DefaultManifold,
    DefaultOrDiagonalizingBasis,
    DiagonalizingBasisData,
    DifferentiatedRetractionVectorTransport,
    FVector,
    NLsolveInverseRetraction,
    InversePowerRetraction,
    PowerManifold,
    PowerManifoldNested,
    PowerManifoldNestedReplacing,
    PowerRetraction,
    PowerVectorTransport,
    TangentSpaceType,
    TCoTSpaceType,
    TFVector,
    VectorSpaceType,
    VeeOrthogonalBasis,
    @decorator_transparent_fallback,
    @decorator_transparent_function,
    @decorator_transparent_signature,
    @invoke_maker,
    _euclidean_basis_vector,
    _extract_val,
    combine_allocation_promotion_functions,
    is_decorator_transparent,
    is_default_decorator,
    manifold_function_not_implemented_message,
    rep_size_to_colons,
    size_to_tuple
using Markdown: @doc_str
using Random
using RecipesBase
using RecipesBase: @recipe, @series
using Colors: RGBA
using Requires
using SimpleWeightedGraphs: AbstractSimpleWeightedGraph, get_weight
using StaticArrays
using Statistics
using StatsBase
using StatsBase: AbstractWeights

include("utils.jl")

include("product_representations.jl")
include("differentiation.jl")
include("riemannian_diff.jl")

# Main Meta Manifolds
include("manifolds/ConnectionManifold.jl")
include("manifolds/MetricManifold.jl")
include("manifolds/VectorBundle.jl")

# It's included early to ensure visibility of `Identity`
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
include("manifolds/FixedRankMatrices.jl")
include("manifolds/GeneralizedGrassmann.jl")
include("manifolds/GeneralizedStiefel.jl")
include("manifolds/Grassmann.jl")
include("manifolds/Hyperbolic.jl")
include("manifolds/MultinomialDoublyStochastic.jl")
include("manifolds/MultinomialSymmetric.jl")
include("manifolds/ProbabilitySimplex.jl")
include("manifolds/PositiveNumbers.jl")
include("manifolds/ProjectiveSpace.jl")
include("manifolds/Rotations.jl")
include("manifolds/SkewHermitian.jl")
include("manifolds/Spectrahedron.jl")
include("manifolds/Stiefel.jl")
include("manifolds/StiefelEuclideanMetric.jl")
include("manifolds/StiefelCanonicalMetric.jl")
include("manifolds/Sphere.jl")
include("manifolds/SphereSymmetricMatrices.jl")
include("manifolds/Symmetric.jl")
include("manifolds/SymmetricPositiveDefinite.jl")
include("manifolds/SymmetricPositiveDefiniteLinearAffine.jl")
include("manifolds/SymmetricPositiveDefiniteLogCholesky.jl")
include("manifolds/SymmetricPositiveDefiniteLogEuclidean.jl")
include("manifolds/SymmetricPositiveSemidefiniteFixedRank.jl")
include("manifolds/Tucker.jl")

# Product or power based manifolds
include("manifolds/Torus.jl")
include("manifolds/Multinomial.jl")
include("manifolds/Oblique.jl")
include("manifolds/EssentialManifold.jl")

#
# Group Manifolds

# a) generics
include("groups/connections.jl")
include("groups/metric.jl")
include("groups/group_action.jl")
include("groups/group_operation_action.jl")
include("groups/array_manifold.jl")
include("groups/product_group.jl")
include("groups/semidirect_product_group.jl")

# Special Group Manifolds
include("groups/general_linear.jl")
include("groups/special_linear.jl")
include("groups/translation_group.jl")
include("groups/special_orthogonal.jl")
include("groups/circle_group.jl")

include("groups/translation_action.jl")
include("groups/rotation_action.jl")

include("groups/special_euclidean.jl")

@doc raw"""
    Base.in(p, M::AbstractManifold; kwargs...)
    p ∈ M

Check, whether a point `p` is a valid point (i.e. in) a [`AbstractManifold`](@ref) `M`.
This method employs [`is_point`](@ref) deaticating the error throwing option.
"""
Base.in(p, M::AbstractManifold; kwargs...) = is_point(M, p, false; kwargs...)

@doc raw"""
    Base.in(p, TpM::TangentSpaceAtPoint; kwargs...)
    X ∈ TangentSpaceAtPoint(M,p)

Check whether `X` is a tangent vector from (in) the tangent space $T_p\mathcal M$, i.e.
the [`TangentSpaceAtPoint`](@ref) at `p` on the [`AbstractManifold`](@ref) `M`.
This method uses [`is_vector`](@ref) deactivating the error throw option.
"""
function Base.in(X, TpM::TangentSpaceAtPoint; kwargs...)
    return is_vector(base_manifold(TpM), TpM.point, X, false; kwargs...)
end

function __init__()
    @require FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41" begin
        using .FiniteDiff
        include("finite_diff.jl")
    end

    @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
        using .ForwardDiff
        include("forward_diff.jl")
    end

    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
        using .OrdinaryDiffEq: ODEProblem, AutoVern9, Rodas5, solve
        include("ode.jl")
    end

    @require NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56" begin
        using .NLsolve: NLsolve
        include("nlsolve.jl")
    end

    @require Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40" begin
        using .Test: Test
        include("tests/tests_general.jl")
        export test_manifold
        include("tests/tests_group.jl")
        export test_group, test_action
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
            include("tests/tests_forwarddiff.jl")
            export test_forwarddiff
        end

        @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
            include("tests/tests_reversediff.jl")
            export test_reversediff
        end
    end

    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
        using RecipesBase: @recipe, @series
        using Colors: RGBA
        include("recipes.jl")
    end

    @require RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01" begin
        @require Colors = "5ae59095-9a9b-59fe-a467-6f913c188581" begin
            using .RecipesBase: @recipe, @series
            using Colors: RGBA
            include("recipes.jl")
        end
    end
    return nothing
end

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
    GeneralizedGrassmann,
    GeneralizedStiefel,
    Grassmann,
    Hyperbolic,
    Lorentz,
    MultinomialDoubleStochastic,
    MultinomialMatrices,
    MultinomialSymmetric,
    Oblique,
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
    SymmetricPositiveSemidefiniteFixedRank,
    Torus,
    Tucker
export HyperboloidPoint, PoincareBallPoint, PoincareHalfSpacePoint, SVDMPoint, TuckerPoint
export HyperboloidTVector,
    PoincareBallTVector, PoincareHalfSpaceTVector, UMVTVector, TuckerTVector
export AbstractNumbers, ℝ, ℂ, ℍ

# decorator manifolds
export AbstractDecoratorManifold, MetricDecoratorType
export AbstractGroupDecoratorType, DefaultGroupDecoratorType, TransparentGroupDecoratorType
export ValidationManifold, ValidationMPoint, ValidationTVector, ValidationCoTVector
export CotangentBundle,
    CotangentSpaceAtPoint, CotangentBundleFibers, CotangentSpace, FVector
export AbstractPowerManifold,
    AbstractPowerRepresentation,
    ArrayPowerRepresentation,
    NestedPowerRepresentation,
    NestedReplacingPowerRepresentation,
    PowerManifold
export ProductManifold
export GraphManifold, GraphManifoldType, VertexManifold, EdgeManifold
export ProjectedPointDistribution, ProductRepr, TangentBundle, TangentBundleFibers
export TangentSpace, TangentSpaceAtPoint, VectorSpaceAtPoint, VectorSpaceType, VectorBundle
export VectorBundleFibers
export AbstractVectorTransportMethod,
    DifferentiatedRetractionVectorTransport, ParallelTransport, ProjectedPointDistribution
export PoleLadderTransport, SchildsLadderTransport
export PowerVectorTransport, ProductVectorTransport
export AbstractEmbeddedManifold
export AbstractAffineConnection,
    AbstractConnectionManifold, ConnectionManifold, LeviCivitaConnection
export AbstractCartanSchoutenConnection,
    CartanSchoutenMinus, CartanSchoutenPlus, CartanSchoutenZero
export AbstractMetric,
    RiemannianMetric,
    LorentzMetric,
    EmbeddedManifold,
    EuclideanMetric,
    LinearAffineMetric,
    LogCholeskyMetric,
    LogEuclideanMetric,
    MinkowskiMetric,
    PowerMetric,
    ProductMetric,
    CanonicalMetric,
    MetricManifold
export AbstractAtlas, RetractionAtlas
export AbstractEmbeddingType, AbstractIsometricEmbeddingType
export DefaultEmbeddingType, DefaultIsometricEmbeddingType, TransparentIsometricEmbedding
export AbstractVectorTransportMethod, ParallelTransport, ProjectionTransport
export AbstractRetractionMethod,
    CayleyRetraction,
    ExponentialRetraction,
    QRRetraction,
    PolarRetraction,
    ProjectionRetraction,
    SoftmaxRetraction,
    PadeRetraction,
    ProductRetraction,
    PowerRetraction
export AbstractInverseRetractionMethod,
    ApproximateInverseRetraction,
    ApproximateLogarithmicMap,
    LogarithmicInverseRetraction,
    QRInverseRetraction,
    PolarInverseRetraction,
    ProjectionInverseRetraction,
    SoftmaxInverseRetraction
export AbstractEstimationMethod,
    GradientDescentEstimation,
    CyclicProximalPointEstimation,
    GeodesicInterpolation,
    GeodesicInterpolationWithinRadius,
    ExtrinsicEstimation
export CachedBasis,
    DefaultBasis,
    DefaultOrthogonalBasis,
    DefaultOrthonormalBasis,
    DiagonalizingOrthonormalBasis,
    InducedBasis,
    ProjectedOrthonormalBasis
export ComponentManifoldError, CompositeManifoldError
export ×,
    allocate,
    allocate_result,
    base_manifold,
    bundle_projection,
    check_point,
    check_vector,
    christoffel_symbols_first,
    christoffel_symbols_second,
    christoffel_symbols_second_jacobian,
    convert,
    complex_dot,
    decorated_manifold,
    det_local_metric,
    distance,
    dual_basis,
    einstein_tensor,
    embed,
    embed!,
    exp,
    exp!,
    flat,
    flat!,
    gaussian_curvature,
    geodesic,
    get_default_atlas,
    get_component,
    get_embedding,
    hat,
    hat!,
    induced_basis,
    incident_log,
    injectivity_radius,
    inner,
    inverse_local_metric,
    inverse_retract,
    inverse_retract!,
    isapprox,
    is_group_decorator,
    is_decorator_transparent,
    is_default_metric,
    is_default_decorator,
    is_point,
    is_vector,
    isapprox,
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
    mid_point,
    mid_point!,
    minkowski_metric,
    moment,
    norm,
    normal_tvector_distribution,
    number_eltype,
    one,
    power_dimensions,
    project,
    project!,
    projected_distribution,
    real_dimension,
    ricci_curvature,
    ricci_tensor,
    representation_size,
    retract,
    retract!,
    riemann_tensor,
    set_component!,
    sharp,
    sharp!,
    shortest_geodesic,
    skewness,
    std,
    sym_rem,
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
    zero_vector,
    zero_vector!
# Lie group types & functions
export AbstractGroupAction,
    AbstractGroupOperation,
    AbstractGroupManifold,
    ActionDirection,
    AdditionOperation,
    CircleGroup,
    GeneralLinear,
    GroupManifold,
    GroupOperationAction,
    Identity,
    InvariantMetric,
    LeftAction,
    LeftInvariantMetric,
    MultiplicationOperation,
    ProductGroup,
    ProductOperation,
    RightAction,
    RightInvariantMetric,
    RotationAction,
    SemidirectProductGroup,
    SpecialEuclidean,
    SpecialLinear,
    SpecialOrthogonal,
    TranslationGroup,
    TranslationAction
export adjoint_action,
    adjoint_action!,
    affine_matrix,
    apply,
    apply!,
    apply_diff,
    apply_diff!,
    base_group,
    center_of_orbit,
    has_approx_invariant_metric,
    compose,
    compose!,
    direction,
    g_manifold,
    group_exp,
    group_exp!,
    group_log,
    group_log!,
    has_biinvariant_metric,
    has_invariant_metric,
    inv,
    inv!,
    invariant_metric_dispatch,
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
# differentiation
export AbstractDiffBackend,
    AbstractRiemannianDiffBackend, FiniteDifferencesBackend, RiemannianONBDiffBackend
export diff_backend, diff_backend!, diff_backends
# atlases and charts
export get_point, get_point!, get_parameters, get_parameters!

end # module
