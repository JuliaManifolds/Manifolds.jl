include("utils.jl")

using HybridArrays, Random
using Manifolds: default_metric_dispatch

Random.seed!(42)

@testset "Power manifold" begin

    Ms = Sphere(2)
    Ms1 = PowerManifold(Ms, 5)
    @test power_dimensions(Ms1) == (5,)
    @test manifold_dimension(Ms1) == 10
    @test injectivity_radius(Ms1) == π
    Ms2 = PowerManifold(Ms, 5, 7)
    @test power_dimensions(Ms2) == (5,7)
    @test manifold_dimension(Ms2) == 70
    Mr = Manifolds.Rotations(3)
    Mr1 = PowerManifold(Mr, 5)
    Mrn1 = PowerManifold(Mr, Manifolds.NestedPowerRepresentation(), 5)
    @test manifold_dimension(Mr1) == 15
    @test manifold_dimension(Mrn1) == 15
    Mr2 = PowerManifold(Mr, 5, 7)
    Mrn2 = PowerManifold(Mr, Manifolds.NestedPowerRepresentation(), 5, 7)
    @test manifold_dimension(Mr2) == 105
    @test manifold_dimension(Mrn2) == 105

    @test repr(Ms1) == "PowerManifold(Sphere(2), 5)"
    @test repr(Mrn1) == "PowerManifold(Rotations(3), NestedPowerRepresentation(), 5)"

    @test Manifolds.allocation_promotion_function(Ms, exp, ([1],)) == Manifolds.allocation_promotion_function(Ms1, exp, (1,))

    @test Ms^5 === Oblique(3,5)
    @test Ms^(5,) === Ms1
    @test Mr^(5, 7) === Mr2

    @test is_default_metric(Ms1, PowerMetric())
    @test default_metric_dispatch(Ms1, PowerMetric()) === Val{true}()
    types_s1 = [Array{Float64,2},
                HybridArray{Tuple{3,StaticArrays.Dynamic()}, Float64, 2}]
    types_s2 = [Array{Float64,3},
                HybridArray{Tuple{3,StaticArrays.Dynamic(),StaticArrays.Dynamic()}, Float64, 3}]

    types_r1 = [Array{Float64,3},
                HybridArray{Tuple{3,3,StaticArrays.Dynamic()}, Float64, 3}]

    types_rn1 = [Vector{Matrix{Float64}}, ]
    TEST_STATIC_SIZED && push!(types_rn1, Vector{MMatrix{3,3,Float64}})

    types_r2 = [Array{Float64,4},
                HybridArray{Tuple{3,3,StaticArrays.Dynamic(),StaticArrays.Dynamic()}, Float64, 4}]
    types_rn2 = [Matrix{Matrix{Float64}}]

    retraction_methods = [Manifolds.PowerRetraction(ManifoldsBase.ExponentialRetraction())]
    inverse_retraction_methods = [Manifolds.InversePowerRetraction(ManifoldsBase.LogarithmicInverseRetraction())]

    sphere_dist = Manifolds.uniform_distribution(Ms, @SVector [1.0, 0.0, 0.0])
    power_s1_pt_dist = Manifolds.PowerPointDistribution(Ms1, sphere_dist, randn(Float64, 3, 5))
    power_s2_pt_dist = Manifolds.PowerPointDistribution(Ms2, sphere_dist, randn(Float64, 3, 5, 7))
    sphere_tv_dist = Manifolds.normal_tvector_distribution(Ms, (@MVector [1.0, 0.0, 0.0]), 1.0)
    power_s1_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Ms1), rand(power_s1_pt_dist), sphere_tv_dist)
    power_s2_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Ms2), rand(power_s2_pt_dist), sphere_tv_dist)

    id_rot = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    rotations_dist = Manifolds.normal_rotation_distribution(Mr, id_rot, 1.0)
    power_r1_pt_dist = Manifolds.PowerPointDistribution(Mr1, rotations_dist, randn(Float64, 3, 3, 5))
    power_rn1_pt_dist = Manifolds.PowerPointDistribution(Mrn1, rotations_dist, [randn(Float64, 3, 3) for i in 1:5])
    power_r2_pt_dist = Manifolds.PowerPointDistribution(Mr2, rotations_dist, randn(Float64, 3, 3, 5, 7))
    power_rn2_pt_dist = Manifolds.PowerPointDistribution(Mrn2, rotations_dist, [randn(Float64, 3, 3) for i in 1:5, j in 1:7])
    rotations_tv_dist = Manifolds.normal_tvector_distribution(Mr, MMatrix(id_rot), 1.0)
    power_r1_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Mr1), rand(power_r1_pt_dist), rotations_tv_dist)
    power_rn1_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Mrn1), rand(power_rn1_pt_dist), rotations_tv_dist)
    power_r2_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Mr2), rand(power_r2_pt_dist), rotations_tv_dist)
    power_rn2_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Mrn2), rand(power_rn2_pt_dist), rotations_tv_dist)

    trim(s::String) = s[1:min(length(s), 20)]

    basis_types = (DefaultOrthonormalBasis(),
        ProjectedOrthonormalBasis(:svd)
    )
    for T in types_s1
        @testset "Type $(trim(string(T)))..." begin
            pts1 = [convert(T, rand(power_s1_pt_dist)) for _ in 1:3]
            @test injectivity_radius(Ms1,pts1[1]) == π
            basis_diag = DiagonalizingOrthonormalBasis(log(Ms1, pts1[1], pts1[2]))
            basis_arb = get_basis(Ms1, pts1[1], DefaultOrthonormalBasis())
            test_manifold(Ms1,
                pts1;
                test_reverse_diff = true,
                test_musical_isomorphisms = true,
                test_injectivity_radius = false,
                test_vee_hat = true,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                point_distributions = [power_s1_pt_dist],
                tvector_distributions = [power_s1_tv_dist],
                basis_types_to_from = (basis_diag, basis_arb, basis_types...),
                rand_tvector_atol_multiplier = 6.0,
                retraction_atol_multiplier = 12.0,
                is_tangent_atol_multiplier = 12.0,
            )
        end
    end
    for T in types_s2
        @testset "Type $(trim(string(T)))..." begin
            pts2 = [convert(T, rand(power_s2_pt_dist)) for _ in 1:3]
            test_manifold(
                Ms2,
                pts2;
                test_reverse_diff = true,
                test_musical_isomorphisms = true,
                test_injectivity_radius = false,
                test_vee_hat = true,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                point_distributions = [power_s2_pt_dist],
                tvector_distributions = [power_s2_tv_dist],
                rand_tvector_atol_multiplier = 6.0,
                retraction_atol_multiplier = 12,
                is_tangent_atol_multiplier = 12.0,
            )
        end
    end

    for T in types_r1
        @testset "Type $(trim(string(T)))..." begin
            pts1 = [convert(T, rand(power_r1_pt_dist)) for _ in 1:3]
            test_manifold(
                Mr1,
                pts1;
                test_reverse_diff = false,
                test_injectivity_radius = false,
                test_musical_isomorphisms = true,
                test_vee_hat = true,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                point_distributions = [power_r1_pt_dist],
                tvector_distributions = [power_r1_tv_dist],
                basis_types_to_from = basis_types,
                rand_tvector_atol_multiplier = 5.0,
                retraction_atol_multiplier = 12,
                is_tangent_atol_multiplier = 12.0
            )
        end
    end

    for T in types_rn1
        @testset "Type $(trim(string(T)))..." begin
            pts1 = [convert(T, rand(power_rn1_pt_dist)) for _ in 1:3]
            test_manifold(
                Mrn1,
                pts1;
                test_reverse_diff = false,
                test_injectivity_radius = false,
                test_musical_isomorphisms = true,
                test_vee_hat = true,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                point_distributions = [power_rn1_pt_dist],
                tvector_distributions = [power_rn1_tv_dist],
                basis_types_to_from = basis_types,
                rand_tvector_atol_multiplier = 5.0,
                retraction_atol_multiplier = 12,
                is_tangent_atol_multiplier = 12.0
            )
        end
    end
    for T in types_r2
        @testset "Type $(trim(string(T)))..." begin
            pts2 = [convert(T, rand(power_r2_pt_dist)) for _ in 1:3]
            test_manifold(
                Mr2,
                pts2;
                test_reverse_diff = false,
                test_injectivity_radius = false,
                test_musical_isomorphisms = true,
                test_vee_hat = true,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                point_distributions = [power_r2_pt_dist],
                tvector_distributions = [power_r2_tv_dist],
                rand_tvector_atol_multiplier = 5.0,
                retraction_atol_multiplier = 12,
                is_tangent_atol_multiplier = 12.0,
            )
        end
    end
    for T in types_rn2
        @testset "Type $(trim(string(T)))..." begin
            pts2 = [convert(T, rand(power_rn2_pt_dist)) for _ in 1:3]
            test_manifold(
                Mrn2,
                pts2;
                test_reverse_diff = false,
                test_injectivity_radius = false,
                test_musical_isomorphisms = true,
                test_vee_hat = true,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                point_distributions = [power_rn2_pt_dist],
                tvector_distributions = [power_rn2_tv_dist],
                rand_tvector_atol_multiplier = 5.0,
                retraction_atol_multiplier = 12,
                is_tangent_atol_multiplier = 12.0,
            )
        end
    end

    @testset "Power manifold of Circle" begin
        pts_t = [[0.0, 1.0, 2.0], [1.0, 1.0, 2.4], [0.0, 2.0, 1.0]]
        MT = PowerManifold(Circle(), 3)
        @test representation_size(MT) == (3,)
        test_manifold(
            MT,
            pts_t;
            test_reverse_diff = false,
            test_forward_diff = false,
            test_injectivity_radius = false,
            test_musical_isomorphisms = true,
            test_vee_hat = true,
            retraction_methods = retraction_methods,
            inverse_retraction_methods = inverse_retraction_methods,
            rand_tvector_atol_multiplier = 5.0,
            retraction_atol_multiplier = 12,
            is_tangent_atol_multiplier = 12.0,
        )
    end

    @testset "Basis printing" begin
        p = hcat([[1.0, 0.0, 0.0] for i in 1:5]...)
        Bc = get_basis(Ms1, p, DefaultOrthonormalBasis())
        @test sprint(show, "text/plain", Bc) == """
        DefaultOrthonormalBasis(ℝ) for a power manifold with coordinates in ℝ
        Basis for component (1,):
        DefaultOrthonormalBasis(ℝ) with coordinates in ℝ and 2 basis vectors:
         E1 =
          3-element Array{Int64,1}:
           0
           1
           0
         E2 =
          3-element Array{Int64,1}:
           0
           0
           1
        Basis for component (2,):
        DefaultOrthonormalBasis(ℝ) with coordinates in ℝ and 2 basis vectors:
         E1 =
          3-element Array{Int64,1}:
           0
           1
           0
         E2 =
          3-element Array{Int64,1}:
           0
           0
           1
        Basis for component (3,):
        DefaultOrthonormalBasis(ℝ) with coordinates in ℝ and 2 basis vectors:
         E1 =
          3-element Array{Int64,1}:
           0
           1
           0
         E2 =
          3-element Array{Int64,1}:
           0
           0
           1
        Basis for component (4,):
        DefaultOrthonormalBasis(ℝ) with coordinates in ℝ and 2 basis vectors:
         E1 =
          3-element Array{Int64,1}:
           0
           1
           0
         E2 =
          3-element Array{Int64,1}:
           0
           0
           1
        Basis for component (5,):
        DefaultOrthonormalBasis(ℝ) with coordinates in ℝ and 2 basis vectors:
         E1 =
          3-element Array{Int64,1}:
           0
           1
           0
         E2 =
          3-element Array{Int64,1}:
           0
           0
           1
        """
    end

    @testset "Power manifold of Circle" begin
        pts_t = [[0.0, 1.0, 2.0], [1.0, 1.0, 2.4], [0.0, 2.0, 1.0]]
        MT = PowerManifold(Circle(), 3)
        @test representation_size(MT) == (3,)
        test_manifold(
            MT,
            pts_t;
            test_reverse_diff = false,
            test_forward_diff = false,
            test_injectivity_radius = false,
            test_musical_isomorphisms = true,
            retraction_methods = retraction_methods,
            inverse_retraction_methods = inverse_retraction_methods,
            rand_tvector_atol_multiplier = 5.0,
            retraction_atol_multiplier = 12,
            is_tangent_atol_multiplier = 12.0,
        )
    end

end
