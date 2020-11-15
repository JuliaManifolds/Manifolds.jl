include("../utils.jl")
include("group_utils.jl")
using NLsolve

@testset "Special Linear group" begin
    @testset "basic properties" begin
        G = SpecialLinear(3)
        @test G === SpecialLinear(3, ℝ)
        @test repr(G) == "SpecialLinear(3, ℝ)"
        @test base_manifold(G) === SpecialLinear(3)
        @test number_system(G) === ℝ
        @test manifold_dimension(G) == 8
        @test representation_size(G) == (3, 3)
        Gc = SpecialLinear(2, ℂ)
        @test repr(Gc) == "SpecialLinear(2, ℂ)"
        @test number_system(Gc) == ℂ
        @test manifold_dimension(Gc) == 6
        @test representation_size(Gc) == (2, 2)
        Gh = SpecialLinear(4, ℍ)
        @test repr(Gh) == "SpecialLinear(4, ℍ)"
        @test number_system(Gh) == ℍ
        @test manifold_dimension(Gh) == 4 * 15
        @test representation_size(Gh) == (4, 4)

        @test (@inferred invariant_metric_dispatch(G, LeftAction())) === Val(true)
        @test (@inferred invariant_metric_dispatch(G, RightAction())) === Val(false)
        @test is_default_metric(MetricManifold(
            G,
            InvariantMetric(EuclideanMetric(), LeftAction()),
        )) === true
        @test Manifolds.default_metric_dispatch(MetricManifold(
            G,
            InvariantMetric(EuclideanMetric(), LeftAction()),
        )) === Val{true}()
    end

    @testset "Real" begin
        G = SpecialLinear(3)
        types = [Matrix{Float64}]
        pts =
            [[2 -1 -3; 4 -1 -6; -1 1 2], [0 2 1; 0 -3 -1; 1 0 2], [-2 0 -1; 1 0 0; -1 -1 2]]
        vpts = [[0 -1 -5; 1 2 0; 1 2 -2], [0 -2 1; -2 1 2; -4 2 -1]]

        retraction_methods = [
            Manifolds.GroupExponentialRetraction(LeftAction()),
            Manifolds.GroupExponentialRetraction(RightAction()),
        ]

        inverse_retraction_methods = [
            Manifolds.GroupLogarithmicInverseRetraction(LeftAction()),
            Manifolds.GroupLogarithmicInverseRetraction(RightAction()),
        ]

        for T in types
            gpts = convert.(T, pts)
            vgpts = convert.(T, vpts)
            test_group(G, gpts, vgpts, vgpts; test_diff = true, test_invariance = true)
            test_manifold(
                G,
                gpts;
                test_reverse_diff = false,
                test_forward_diff = false,
                # test_project_point = true,
                test_injectivity_radius = false,
                # test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_default_vector_transport = true,
                vector_transport_methods = [
                    ParallelTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                exp_log_atol_multiplier = 1e8,
                retraction_atol_multiplier = 1e8,
                is_tangent_atol_multiplier = 1e8,
            )
        end
    end

    @testset "Complex" begin
        G = SpecialLinear(2, ℂ)
        types = [Matrix{ComplexF64}]
        pts = [
            [0+1im 0+2im; 0+1im 0+1im],
            [-2+0im 0+1im; 0-1im -1+0im],
            [0+0im 0-1im; 0-1im -1+3im],
        ]
        vpts = [[0-1im -1+1im; -1+0im 0+1im], [1+1im 0+0im; 0+1im -1-1im]]

        retraction_methods = [
            Manifolds.GroupExponentialRetraction(LeftAction()),
            Manifolds.GroupExponentialRetraction(RightAction()),
        ]

        inverse_retraction_methods = [
            Manifolds.GroupLogarithmicInverseRetraction(LeftAction()),
            Manifolds.GroupLogarithmicInverseRetraction(RightAction()),
        ]

        for T in types
            gpts = convert.(T, pts)
            vgpts = convert.(T, vpts)
            test_group(G, gpts, vgpts, vgpts; test_diff = true, test_invariance = true)
            test_manifold(
                G,
                gpts;
                test_reverse_diff = false,
                test_forward_diff = false,
                test_injectivity_radius = false,
                test_musical_isomorphisms = true,
                test_default_vector_transport = true,
                vector_transport_methods = [
                    ParallelTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                exp_log_atol_multiplier = 1e8,
                retraction_atol_multiplier = 1e8,
                is_tangent_atol_multiplier = 1e8,
            )
        end
    end
end
