include("../utils.jl")

@testset "Grassmann" begin
    @testset "Real" begin
        M = Grassmann(3, 2)
        @testset "Basics" begin
            @test repr(M) == "Grassmann(3, 2, ℝ)"
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 2
            @test !is_point(M, [1.0, 0.0, 0.0, 0.0])
            @test !is_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0.0, 0.0, 1.0, 0.0])
            @test_throws ManifoldDomainError is_point(M, [2.0 0.0; 0.0 1.0; 0.0 0.0], true)
            @test_throws ManifoldDomainError is_vector(
                M,
                [2.0 0.0; 0.0 1.0; 0.0 0.0],
                zeros(3, 2),
                true,
            )
            @test_throws ManifoldDomainError is_vector(
                M,
                [1.0 0.0; 0.0 1.0; 0.0 0.0],
                ones(3, 2),
                true,
            )
            @test is_point(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], true)
            @test_throws ManifoldDomainError is_point(
                M,
                1im * [1.0 0.0; 0.0 1.0; 0.0 0.0],
                true,
            )
            @test is_vector(
                M,
                [1.0 0.0; 0.0 1.0; 0.0 0.0],
                zero_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0]),
                true,
            )
            @test_throws ManifoldDomainError is_vector(
                M,
                [1.0 0.0; 0.0 1.0; 0.0 0.0],
                1im * zero_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0]),
                true,
            )
            @test injectivity_radius(M) == π / 2
            @test injectivity_radius(M, ExponentialRetraction()) == π / 2
            @test injectivity_radius(M, [1.0 0.0; 0.0 1.0; 0.0 0.0]) == π / 2
            @test injectivity_radius(
                M,
                [1.0 0.0; 0.0 1.0; 0.0 0.0],
                ExponentialRetraction(),
            ) == π / 2
        end
        types = [Matrix{Float64}]
        TEST_STATIC_SIZED && push!(types, MMatrix{3,2,Float64,6})

        TEST_FLOAT32 && push!(types, Matrix{Float32})
        basis_types = (ProjectedOrthonormalBasis(:gram_schmidt),)
        @testset "Type $T" for T in types
            p1 = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            X = [0.0 0.0; 0.0 0.0; 0.0 1.0]
            p2 = exp(M, p1, X)
            Y = [0.0 1.0; -1.0 0.0; 1.0 0.0]
            p3 = exp(M, p1, Y)
            pts = convert.(T, [p1, p2, p3])
            test_manifold(
                M,
                pts,
                test_exp_log=true,
                test_injectivity_radius=false,
                test_project_tangent=true,
                test_default_vector_transport=false,
                point_distributions=[Manifolds.uniform_distribution(M, pts[1])],
                test_vee_hat=false,
                test_rand_point=true,
                test_rand_tvector=true,
                retraction_methods=[PolarRetraction(), QRRetraction()],
                inverse_retraction_methods=[
                    PolarInverseRetraction(),
                    QRInverseRetraction(),
                ],
                #basis_types_vecs = basis_types,
                # investigate why this is so large on dev
                exp_log_atol_multiplier=10.0 * (VERSION >= v"1.6-DEV" ? 10.0^8 : 1.0),
                is_tangent_atol_multiplier=20.0,
            )

            @testset "inner/norm" begin
                X1 = inverse_retract(M, pts[1], pts[2], PolarInverseRetraction())
                X2 = inverse_retract(M, pts[1], pts[3], PolarInverseRetraction())

                @test real(inner(M, pts[1], X1, X2)) ≈ real(inner(M, pts[1], X2, X1))
                @test imag(inner(M, pts[1], X1, X2)) ≈ -imag(inner(M, pts[1], X2, X1))
                @test imag(inner(M, pts[1], X1, X1)) ≈ 0

                @test norm(M, pts[1], X1) isa Real
                @test norm(M, pts[1], X1) ≈ sqrt(inner(M, pts[1], X1, X1))
            end
        end

        @testset "Distribution tests" begin
            ugd_mmatrix = Manifolds.uniform_distribution(M, @MMatrix [
                1.0 0.0
                0.0 1.0
                0.0 0.0
            ])
            @test isa(rand(ugd_mmatrix), MMatrix)
        end

        @testset "vector transport" begin
            p1 = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            X = [0.0 0.0; 0.0 0.0; 0.0 1.0]
            p2 = exp(M, p1, X)
            @test vector_transport_to(M, p1, X, p2, ProjectionTransport()) ==
                  project(M, p2, X)
            @test is_vector(
                M,
                p2,
                vector_transport_to(M, p1, X, p2, ProjectionTransport()),
                true;
                atol=10^-15,
            )
        end
    end

    @testset "Complex" begin
        M = Grassmann(3, 2, ℂ)
        @testset "Basics" begin
            @test repr(M) == "Grassmann(3, 2, ℂ)"
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 4
            @test !is_point(M, [1.0, 0.0, 0.0, 0.0])
            @test !is_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0.0, 0.0, 1.0, 0.0])
            @test Manifolds.allocation_promotion_function(M, exp!, (1,)) == complex
            @test_throws ManifoldDomainError is_point(M, [2.0 0.0; 0.0 1.0; 0.0 0.0], true)
            @test_throws ManifoldDomainError is_vector(
                M,
                [2.0 0.0; 0.0 1.0; 0.0 0.0],
                zeros(3, 2),
                true,
            )
            @test_throws ManifoldDomainError is_vector(
                M,
                [1.0 0.0; 0.0 1.0; 0.0 0.0],
                ones(3, 2),
                true,
            )
            @test is_vector(
                M,
                [1.0 0.0; 0.0 1.0; 0.0 0.0],
                1im * zero_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0]),
            )
            @test is_point(M, [1.0 0.0; 0.0 1.0; 0.0 0.0])
            @test injectivity_radius(M) == π / 2
        end
        types = [Matrix{ComplexF64}]
        @testset "Type $T" for T in types
            p1 = [0.5+0.5im 0.5+0.5im; 0.5+0.5im -0.5-0.5im; 0.0 0.0]
            X = [0.0 0.0; 0.0 0.0; 0.0 1.0]
            p2 = exp(M, p1, X)
            Y = [0.0 1.0; -1.0 0.0; 1.0 0.0]
            p3 = exp(M, p1, Y)
            pts = convert.(T, [p1, p2, p3])
            test_manifold(
                M,
                pts,
                test_exp_log=true,
                test_injectivity_radius=false,
                test_project_tangent=true,
                test_default_vector_transport=false,
                test_vee_hat=false,
                retraction_methods=[PolarRetraction(), QRRetraction()],
                inverse_retraction_methods=[
                    PolarInverseRetraction(),
                    QRInverseRetraction(),
                ],
                exp_log_atol_multiplier=10.0^3,
                is_tangent_atol_multiplier=20.0,
                test_inplace=true,
                test_rand_point=true,
            )

            @testset "inner/norm" begin
                X1 = inverse_retract(M, pts[1], pts[2], PolarInverseRetraction())
                X2 = inverse_retract(M, pts[1], pts[3], PolarInverseRetraction())

                @test real(inner(M, pts[1], X1, X2)) ≈ real(inner(M, pts[1], X2, X1))
                @test imag(inner(M, pts[1], X1, X2)) ≈ -imag(inner(M, pts[1], X2, X1))
                @test isapprox(imag(inner(M, pts[1], X1, X1)), 0; atol=1e-30)

                @test norm(M, pts[1], X1) isa Real
                @test norm(M, pts[1], X1) ≈ sqrt(inner(M, pts[1], X1, X1))
            end
        end
    end

    @testset "Complex and conjugate" begin
        G = Grassmann(3, 1, ℂ)
        p = reshape([im, 0.0, 0.0], 3, 1)
        @test is_point(G, p)
        X = reshape([-0.5; 0.5; 0], 3, 1)
        @test_throws ManifoldDomainError is_vector(G, p, X, true)
        Y = project(G, p, X)
        @test is_vector(G, p, Y)
    end

    @testset "Projector representation" begin end

    @testset "is_point & convert & show" begin
        M = Grassmann(3, 2)
        p = StiefelPoint([1.0 0.0; 0.0 1.0; 0.0 0.0])
        X = StiefelTVector([0.0 1.0; -1.0 0.0; 0.0 0.0])
        @test is_point(M, p, true)
        @test is_vector(M, p, X, true)
        @test repr(p) == "StiefelPoint($(p.value))"
        @test repr(X) == "StiefelTVector($(X.value))"
        M2 = Stiefel(3, 2)
        @test is_point(M2, p, true)
        @test is_vector(M2, p, X, true)

        p2 = convert(ProjectorPoint, p)
        @test is_point(M, p2, true)
        p3 = convert(ProjectorPoint, p.value)
        @test p2.value == p3.value
        X2 = ProjectorTVector([0.0 0.0 1.0; 0.0 0.0 1.0; 1.0 1.0 0.0])
        @test is_vector(M, p2, X2)
        @test repr(p2) == "ProjectorPoint($(p2.value))"
        @test repr(X2) == "ProjectorTVector($(X2.value))"

        # rank just 1
        pF1 = ProjectorPoint([1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
        @test_throws DomainError is_point(M, pF1, true)
        # not equal to its square
        pF2 = ProjectorPoint([1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 1.0 0.0])
        @test_throws DomainError is_point(M, pF2, true)
        # not symmetric
        pF3 = ProjectorPoint([0.0 1.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0])
        @test_throws DomainError is_point(M, pF3, true)

        # not symmetric
        XF1 = ProjectorTVector([0.0 0.0 1.0; 0.0 0.0 1.0; 1.0 0.0 0.0])
        @test_throws DomainError is_vector(M, p2, XF1, true)
        # XF2 is not p2*XF2 + XF2*p2
        XF2 = ProjectorTVector(ones(3, 3))
        @test_throws DomainError is_vector(M, p2, XF2, true)
    end
end
