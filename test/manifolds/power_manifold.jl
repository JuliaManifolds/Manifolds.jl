include("../utils.jl")

using HybridArrays, Random
using StaticArrays: Dynamic

Random.seed!(42)

struct TestExponentialAtlas <: Manifolds.AbstractAtlas{ℝ} end

function Manifolds.get_point!(M::AbstractManifold, p, ::TestExponentialAtlas, i, a)
    return exp!(M, p, i, get_vector(M, i, a, DefaultOrthonormalBasis()))
end

function Manifolds.get_parameters!(M::AbstractManifold, a, ::TestExponentialAtlas, i, p)
    return get_coordinates!(M, a, i, log(M, i, p), DefaultOrthonormalBasis())
end

@testset "Power manifold" begin
    Ms = Sphere(2)
    Ms1 = PowerManifold(Ms, 5)
    @test power_dimensions(Ms1) == (5,)
    @test manifold_dimension(Ms1) == 10
    @test injectivity_radius(Ms1) == π
    Ms2 = PowerManifold(Ms, 5, 7)
    @test power_dimensions(Ms2) == (5, 7)
    @test manifold_dimension(Ms2) == 70
    Ms2n = PowerManifold(Ms1, NestedPowerRepresentation(), 7)
    @test power_dimensions(Ms2n) == (7,)
    @test manifold_dimension(Ms2n) == 70

    Mr = Manifolds.Rotations(3)
    Mr1 = PowerManifold(Mr, 5)
    Mrn1 = PowerManifold(Mr, Manifolds.NestedPowerRepresentation(), 5)
    @test PowerManifold(PowerManifold(Mr, 2), 3) == PowerManifold(Mr, 2, 3)
    @test PowerManifold(Torus(2), 3) isa PowerManifold{ℝ,Torus{2}}
    @test PowerManifold(Mrn1, 3) ==
          PowerManifold(Mr, Manifolds.NestedPowerRepresentation(), 5, 3)
    @test PowerManifold(Mrn1, Manifolds.ArrayPowerRepresentation(), 3) ==
          PowerManifold(Mr, Manifolds.ArrayPowerRepresentation(), 5, 3)
    @test manifold_dimension(Mr1) == 15
    @test manifold_dimension(Mrn1) == 15
    Mr2 = PowerManifold(Mr, 5, 7)
    Mrn2 = PowerManifold(Mr, Manifolds.NestedPowerRepresentation(), 5, 7)
    @test manifold_dimension(Mr2) == 105
    @test manifold_dimension(Mrn2) == 105

    @test repr(Ms1) == "PowerManifold(Sphere(2, ℝ), 5)"
    @test repr(Mrn1) == "PowerManifold(Rotations(3), NestedPowerRepresentation(), 5)"

    @test Manifolds.allocation_promotion_function(Ms, exp, ([1],)) ==
          Manifolds.allocation_promotion_function(Ms1, exp, (1,))

    @test Ms^5 === Oblique(3, 5)
    @test Ms^(5,) === Ms1
    @test Mr^(5, 7) === Mr2

    types_s1 = [Array{Float64,2}, HybridArray{Tuple{3,Dynamic()},Float64,2}]
    types_s2 = [Array{Float64,3}, HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64,3}]

    types_r1 = [Array{Float64,3}, HybridArray{Tuple{3,3,Dynamic()},Float64,3}]

    types_rn1 = [Vector{Matrix{Float64}}]
    TEST_STATIC_SIZED && push!(types_rn1, Vector{MMatrix{3,3,Float64,9}})

    types_r2 = [Array{Float64,4}, HybridArray{Tuple{3,3,Dynamic(),Dynamic()},Float64,4}]
    types_rn2 = [Matrix{Matrix{Float64}}]

    retraction_methods = [ManifoldsBase.ExponentialRetraction()]
    inverse_retraction_methods = [ManifoldsBase.LogarithmicInverseRetraction()]

    sphere_dist = Manifolds.uniform_distribution(Ms, @SVector [1.0, 0.0, 0.0])
    power_s1_pt_dist =
        Manifolds.PowerPointDistribution(Ms1, sphere_dist, randn(Float64, 3, 5))
    power_s2_pt_dist =
        Manifolds.PowerPointDistribution(Ms2, sphere_dist, randn(Float64, 3, 5, 7))
    sphere_tv_dist =
        Manifolds.normal_tvector_distribution(Ms, (@MVector [1.0, 0.0, 0.0]), 1.0)
    power_s1_tv_dist = Manifolds.PowerFVectorDistribution(
        TangentBundleFibers(Ms1),
        rand(power_s1_pt_dist),
        sphere_tv_dist,
    )
    power_s2_tv_dist = Manifolds.PowerFVectorDistribution(
        TangentBundleFibers(Ms2),
        rand(power_s2_pt_dist),
        sphere_tv_dist,
    )

    id_rot = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    rotations_dist = Manifolds.normal_rotation_distribution(Mr, id_rot, 1.0)
    power_r1_pt_dist =
        Manifolds.PowerPointDistribution(Mr1, rotations_dist, randn(Float64, 3, 3, 5))
    power_rn1_pt_dist = Manifolds.PowerPointDistribution(
        Mrn1,
        rotations_dist,
        [randn(Float64, 3, 3) for i in 1:5],
    )
    power_r2_pt_dist =
        Manifolds.PowerPointDistribution(Mr2, rotations_dist, randn(Float64, 3, 3, 5, 7))
    power_rn2_pt_dist = Manifolds.PowerPointDistribution(
        Mrn2,
        rotations_dist,
        [randn(Float64, 3, 3) for i in 1:5, j in 1:7],
    )
    rotations_tv_dist = Manifolds.normal_tvector_distribution(Mr, MMatrix(id_rot), 1.0)
    power_r1_tv_dist = Manifolds.PowerFVectorDistribution(
        TangentBundleFibers(Mr1),
        rand(power_r1_pt_dist),
        rotations_tv_dist,
    )
    power_rn1_tv_dist = Manifolds.PowerFVectorDistribution(
        TangentBundleFibers(Mrn1),
        rand(power_rn1_pt_dist),
        rotations_tv_dist,
    )
    power_r2_tv_dist = Manifolds.PowerFVectorDistribution(
        TangentBundleFibers(Mr2),
        rand(power_r2_pt_dist),
        rotations_tv_dist,
    )
    power_rn2_tv_dist = Manifolds.PowerFVectorDistribution(
        TangentBundleFibers(Mrn2),
        rand(power_rn2_pt_dist),
        rotations_tv_dist,
    )

    @testset "get_component, set_component!, getindex and setindex!" begin
        p1 = randn(3, 5)
        @test get_component(Ms1, p1, 1) == p1[:, 1]
        @test p1[Ms1, 1] == p1[:, 1]
        @test p1[Ms1, 1] isa Vector
        p2 = [10.0, 11.0, 12.0]
        set_component!(Ms1, p1, p2, 2)
        @test get_component(Ms1, p1, 2) == p2
        p1[Ms1, 2] = 2 * p2
        @test p1[Ms1, 2] == 2 * p2
        p1[Ms1, 2] += p2
        @test p1[Ms1, 2] ≈ 3 * p2
        p1[Ms1, 2] .+= p2
        @test p1[Ms1, 2] ≈ 4 * p2
        @test view(p1, Ms1, 1) == p1[Ms1, 1]
        @test view(p1, Ms1, 1) isa SubArray

        Msn1 = PowerManifold(Ms, Manifolds.NestedPowerRepresentation(), 5)
        pn1 = [randn(3) for _ in 1:5]
        @test get_component(Msn1, pn1, 1) == pn1[1]
        @test pn1[Msn1, 1] == pn1[1]
        @test pn1[Msn1, 1] isa Vector
        set_component!(Msn1, pn1, p2, 2)
        @test get_component(Msn1, pn1, 2) == p2
        pn1[Msn1, 2] = 2 * p2
        @test pn1[Msn1, 2] == 2 * p2
        pn1[Msn1, 2] += p2
        @test pn1[Msn1, 2] ≈ 3 * p2
        pn1[Msn1, 2] .+= p2
        @test pn1[Msn1, 2] ≈ 4 * p2
        @test view(pn1, Msn1, 1) == pn1[Msn1, 1]
        @test view(pn1, Msn1, 1) isa SubArray
    end

    @testset "ComponenException" begin
        M = PowerManifold(Sphere(2), NestedPowerRepresentation(), 2)
        p = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        X = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        @test_throws ComponentManifoldError is_point(M, X, true)
        @test_throws ComponentManifoldError is_vector(M, p, X, true)
    end

    @testset "power vector transport" begin
        m = ParallelTransport()
        p = repeat([1.0, 0.0, 0.0], 1, 5)
        q = repeat([0.0, 1.0, 0.0], 1, 5)
        X = log(Ms1, p, q)
        Y = vector_transport_to(Ms1, p, X, q, m)
        Z = -log(Ms1, q, p)
        @test isapprox(Ms1, q, Y, Z)
    end
    trim(s::String) = s[1:min(length(s), 20)]

    basis_types = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))
    for T in types_s1
        @testset "Type $(trim(string(T)))..." begin
            pts1 = [convert(T, rand(power_s1_pt_dist)) for _ in 1:3]
            @test injectivity_radius(Ms1, pts1[1]) == π
            basis_diag = get_basis(
                Ms1,
                pts1[1],
                DiagonalizingOrthonormalBasis(log(Ms1, pts1[1], pts1[2])),
            )
            basis_arb = get_basis(Ms1, pts1[1], DefaultOrthonormalBasis())
            test_manifold(
                Ms1,
                pts1;
                test_reverse_diff=true,
                test_musical_isomorphisms=true,
                test_injectivity_radius=false,
                test_default_vector_transport=true,
                test_project_point=true,
                test_project_tangent=true,
                vector_transport_methods=[
                    ParallelTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                test_vee_hat=true,
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                point_distributions=[power_s1_pt_dist],
                tvector_distributions=[power_s1_tv_dist],
                basis_types_to_from=(basis_diag, basis_arb, basis_types...),
                rand_tvector_atol_multiplier=600.0,
                retraction_atol_multiplier=12.0,
                is_tangent_atol_multiplier=12.0,
                exp_log_atol_multiplier=20 * prod(power_dimensions(Ms1)),
                test_inplace=true,
                test_rand_point=true,
                test_rand_tvector=true,
            )
        end
    end
    for T in types_s2
        @testset "Type $(trim(string(T)))..." begin
            pts2 = [convert(T, rand(power_s2_pt_dist)) for _ in 1:3]
            test_manifold(
                Ms2,
                pts2;
                test_reverse_diff=true,
                test_musical_isomorphisms=true,
                test_injectivity_radius=false,
                test_vee_hat=true,
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                point_distributions=[power_s2_pt_dist],
                tvector_distributions=[power_s2_tv_dist],
                rand_tvector_atol_multiplier=6.0,
                retraction_atol_multiplier=12,
                is_tangent_atol_multiplier=12.0,
                exp_log_atol_multiplier=3 * prod(power_dimensions(Ms2)),
                test_inplace=true,
            )
        end
    end

    for T in types_r1
        @testset "Type $(trim(string(T)))..." begin
            pts1 = [convert(T, rand(power_r1_pt_dist)) for _ in 1:3]
            test_manifold(
                Mr1,
                pts1;
                test_reverse_diff=false,
                test_injectivity_radius=false,
                test_musical_isomorphisms=true,
                test_vee_hat=true,
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                point_distributions=[power_r1_pt_dist],
                tvector_distributions=[power_r1_tv_dist],
                basis_types_to_from=basis_types,
                rand_tvector_atol_multiplier=8.0,
                retraction_atol_multiplier=12,
                is_tangent_atol_multiplier=12.0,
                exp_log_atol_multiplier=2e2 * prod(power_dimensions(Mr2)),
                test_inplace=true,
            )
        end
    end

    for T in types_rn1
        @testset "Type $(trim(string(T)))..." begin
            pts1 = [convert(T, rand(power_rn1_pt_dist)) for _ in 1:3]
            test_manifold(
                Mrn1,
                pts1;
                test_reverse_diff=false,
                test_injectivity_radius=false,
                test_musical_isomorphisms=true,
                test_vee_hat=true,
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                point_distributions=[power_rn1_pt_dist],
                tvector_distributions=[power_rn1_tv_dist],
                basis_types_to_from=basis_types,
                rand_tvector_atol_multiplier=500.0,
                retraction_atol_multiplier=12,
                is_tangent_atol_multiplier=12.0,
                exp_log_atol_multiplier=4e2 * prod(power_dimensions(Mrn1)),
                test_inplace=true,
                test_rand_point=true,
                test_rand_tvector=true,
            )
        end
    end
    for T in types_r2
        @testset "Type $(trim(string(T)))..." begin
            pts2 = [convert(T, rand(power_r2_pt_dist)) for _ in 1:3]
            test_manifold(
                Mr2,
                pts2;
                test_reverse_diff=false,
                test_injectivity_radius=false,
                test_musical_isomorphisms=true,
                test_vee_hat=true,
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                point_distributions=[power_r2_pt_dist],
                tvector_distributions=[power_r2_tv_dist],
                rand_tvector_atol_multiplier=8.0,
                retraction_atol_multiplier=12,
                is_tangent_atol_multiplier=12.0,
                exp_log_atol_multiplier=4e3 * prod(power_dimensions(Mr2)),
                test_inplace=true,
            )
        end
    end
    for T in types_rn2
        @testset "Type $(trim(string(T)))..." begin
            pts2 = [convert(T, rand(power_rn2_pt_dist)) for _ in 1:3]
            test_manifold(
                Mrn2,
                pts2;
                test_reverse_diff=false,
                test_injectivity_radius=false,
                test_musical_isomorphisms=true,
                test_vee_hat=true,
                retraction_methods=retraction_methods,
                inverse_retraction_methods=inverse_retraction_methods,
                point_distributions=[power_rn2_pt_dist],
                tvector_distributions=[power_rn2_tv_dist],
                rand_tvector_atol_multiplier=8.0,
                retraction_atol_multiplier=12,
                is_tangent_atol_multiplier=12.0,
                exp_log_atol_multiplier=4e3 * prod(power_dimensions(Mrn2)),
                test_inplace=true,
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
            test_reverse_diff=false,
            test_forward_diff=false,
            test_injectivity_radius=false,
            test_musical_isomorphisms=true,
            test_vee_hat=true,
            retraction_methods=retraction_methods,
            inverse_retraction_methods=inverse_retraction_methods,
            rand_tvector_atol_multiplier=5.0,
            retraction_atol_multiplier=12,
            is_tangent_atol_multiplier=12.0,
            exp_log_atol_multiplier=1.0,
            test_inplace=true,
            test_rand_point=true,
            test_rand_tvector=true,
        )
    end

    @testset "Basis printing" begin
        p = hcat([[1.0, 0.0, 0.0] for i in 1:5]...)
        Bc = get_basis(Ms1, p, DefaultOrthonormalBasis())
        @test sprint(show, "text/plain", Bc) == """
        DefaultOrthonormalBasis(ℝ) for a power manifold
        Basis for component (1,):
        $(sprint(show, "text/plain", Bc.data.bases[1]))
        Basis for component (2,):
        $(sprint(show, "text/plain", Bc.data.bases[2]))
        Basis for component (3,):
        $(sprint(show, "text/plain", Bc.data.bases[3]))
        Basis for component (4,):
        $(sprint(show, "text/plain", Bc.data.bases[4]))
        Basis for component (5,):
        $(sprint(show, "text/plain", Bc.data.bases[5]))
        """
    end

    @testset "Power manifold of Circle" begin
        pts_t = [[0.0, 1.0, 2.0], [1.0, 1.0, 2.4], [0.0, 2.0, 1.0]]
        MT = PowerManifold(Circle(), 3)
        @test representation_size(MT) == (3,)
        @test pts_t[1][MT, 2] == 1.0
        @test HybridVector{3}(pts_t[1])[MT, 2] == 1.0
        test_manifold(
            MT,
            pts_t;
            test_reverse_diff=false,
            test_forward_diff=false,
            test_injectivity_radius=false,
            test_musical_isomorphisms=true,
            retraction_methods=retraction_methods,
            inverse_retraction_methods=inverse_retraction_methods,
            rand_tvector_atol_multiplier=5.0,
            retraction_atol_multiplier=12,
            is_tangent_atol_multiplier=12.0,
            test_inplace=true,
        )
    end

    @testset "Atlas & Induced Basis" begin
        M = PowerManifold(Euclidean(2), NestedPowerRepresentation(), 2)
        p = [zeros(2), ones(2)]
        X = [ones(2), 2 .* ones(2)]
        A = RetractionAtlas()
        a = get_parameters(M, A, p, p)
        p2 = get_point(M, A, p, a)
        @test all(p2 .== p)
        A2 = TestExponentialAtlas()
        a2 = get_parameters(M, A2, p, p)
        @test isapprox(a, a2)
        @test_throws ErrorException get_point(M, A2, p, a2)
    end

    @testset "metric conversion" begin
        M = SymmetricPositiveDefinite(3)
        N = PowerManifold(M, NestedPowerRepresentation(), 2)
        e = EuclideanMetric()
        p = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1]
        q = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1]
        P = [p, q]
        X = [log(M, p, q), log(M, q, p)]
        Y = change_metric(N, e, P, X)
        Yc = [change_metric(M, e, p, log(M, p, q)), change_metric(M, e, q, log(M, q, p))]
        @test norm(N, P, Y .- Yc) ≈ 0
        Z = change_representer(N, e, P, X)
        Zc = [
            change_representer(M, e, p, log(M, p, q)),
            change_representer(M, e, q, log(M, q, p)),
        ]
        @test norm(N, P, Z .- Zc) ≈ 0
    end

    @testset "Nested replacing RNG" begin
        M = PowerManifold(Ms, NestedReplacingPowerRepresentation(), 2)
        @test is_point(M, rand(M))
        @test is_point(M, rand(MersenneTwister(123), M))
        @test rand(MersenneTwister(123), M) == rand(MersenneTwister(123), M)
        p = rand(M)
        @test is_vector(M, p, rand(M; vector_at=p); atol=1e-15)
        @test is_vector(M, p, rand(MersenneTwister(123), M; vector_at=p); atol=1e-15)
        @test rand(MersenneTwister(123), M; vector_at=p) ==
              rand(MersenneTwister(123), M; vector_at=p)
    end
end
