include("../utils.jl")

@testset "Stiefel" begin
    @testset "Real" begin
        M = Stiefel(3, 2)
        M2 = MetricManifold(M, EuclideanMetric())
        @testset "Basics" begin
            @test repr(M) == "Stiefel(3, 2, ℝ)"
            x = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            @test is_default_metric(M, EuclideanMetric())
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 3
            base_manifold(M) === M
            @test_throws ManifoldDomainError is_point(M, [1.0, 0.0, 0.0, 0.0], true)
            @test_throws ManifoldDomainError is_point(
                M,
                1im * [1.0 0.0; 0.0 1.0; 0.0 0.0],
                true,
            )
            @test !is_vector(M, x, [0.0, 0.0, 1.0, 0.0])
            @test_throws ManifoldDomainError is_vector(
                M,
                x,
                1 * im * zero_vector(M, x),
                true,
            )
        end
        @testset "Embedding and Projection" begin
            x = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            y = similar(x)
            z = embed(M, x)
            @test z == x
            embed!(M, y, x)
            @test y == z
            a = [1.0 0.0; 0.0 2.0; 0.0 0.0]
            @test !is_point(M, a)
            b = similar(a)
            c = project(M, a)
            @test c == x
            project!(M, b, a)
            @test b == x
            X = [0.0 0.0; 0.0 0.0; -1.0 1.0]
            Y = similar(X)
            Z = embed(M, x, X)
            embed!(M, Y, x, X)
            @test Y == X
            @test Z == X
        end

        types = [Matrix{Float64}]
        TEST_FLOAT32 && push!(types, Matrix{Float32})
        TEST_STATIC_SIZED && push!(types, MMatrix{3,2,Float64,6})

        @testset "Stiefel(2, 1) special case" begin
            M21 = Stiefel(2, 1)
            w = inverse_retract(
                M21,
                SMatrix{2,1}([0.0, 1.0]),
                SMatrix{2,1}([sqrt(2), sqrt(2)]),
                QRInverseRetraction(),
            )
            @test isapprox(M21, w, SMatrix{2,1}([1.0, 0.0]))
        end

        @testset "inverse QR retraction cases" begin
            M43 = Stiefel(4, 3)
            p = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0; 0.0 0.0 0.0]
            Xinit = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 1.0 1.0]
            q = retract(M43, p, Xinit, QRRetraction())
            X1 = inverse_retract(
                M43,
                SMatrix{4,3}(p),
                SMatrix{4,3}(q),
                QRInverseRetraction(),
            )
            X2 = inverse_retract(M43, p, q, QRInverseRetraction())
            @test isapprox(M43, p, X1, X2)
            @test isapprox(M43, p, X1, Xinit)

            p2 = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            q2 = exp(M, p2, [0.0 0.0; 0.0 0.0; 1.0 1.0])

            X1 = inverse_retract(
                M,
                SMatrix{3,2}(p2),
                SMatrix{3,2}(q2),
                QRInverseRetraction(),
            )
            X2 = inverse_retract(M, p2, q2, QRInverseRetraction())
            @test isapprox(M43, p2, X1, X2)
        end

        @testset "Type $T" for T in types
            x = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            y = exp(M, x, [0.0 0.0; 0.0 0.0; 1.0 1.0])
            z = exp(M, x, [0.0 0.0; 0.0 0.0; -1.0 1.0])
            @test_throws MethodError distance(M, x, y)
            @test isapprox(
                M,
                retract(
                    M,
                    SMatrix{3,2}(x),
                    SA[0.0 0.0; 0.0 0.0; -1.0 1.0],
                    PolarRetraction(),
                ),
                retract(M, x, [0.0 0.0; 0.0 0.0; -1.0 1.0], PolarRetraction()),
                atol=1e-15,
            )
            pts = convert.(T, [x, y, z])
            v = inverse_retract(M, x, y, PolarInverseRetraction())
            @test !is_point(M, 2 * x)
            @test_throws DomainError !is_point(M, 2 * x, true)
            @test !is_vector(M, 2 * x, v)
            @test_throws ManifoldDomainError !is_vector(M, 2 * x, v, true)
            @test !is_vector(M, x, y)
            @test_throws DomainError is_vector(M, x, y, true)
            test_manifold(
                M,
                pts,
                basis_types_to_from=(DefaultOrthonormalBasis(),),
                basis_types_vecs=(DefaultOrthonormalBasis(),),
                test_exp_log=false,
                default_inverse_retraction_method=PolarInverseRetraction(),
                test_injectivity_radius=false,
                test_is_tangent=true,
                test_project_tangent=true,
                test_default_vector_transport=false,
                point_distributions=[Manifolds.uniform_distribution(M, pts[1])],
                test_vee_hat=false,
                projection_atol_multiplier=15.0,
                retraction_atol_multiplier=10.0,
                is_tangent_atol_multiplier=4 * 10.0^2,
                retraction_methods=[
                    PolarRetraction(),
                    QRRetraction(),
                    CayleyRetraction(),
                    PadeRetraction(2),
                ],
                inverse_retraction_methods=[
                    PolarInverseRetraction(),
                    QRInverseRetraction(),
                ],
                vector_transport_methods=[
                    DifferentiatedRetractionVectorTransport(PolarRetraction()),
                    DifferentiatedRetractionVectorTransport(QRRetraction()),
                    ProjectionTransport(),
                ],
                vector_transport_retractions=[
                    PolarRetraction(),
                    QRRetraction(),
                    PolarRetraction(),
                ],
                vector_transport_inverse_retractions=[
                    PolarInverseRetraction(),
                    QRInverseRetraction(),
                    PolarInverseRetraction(),
                ],
                test_vector_transport_direction=[true, true, false],
                mid_point12=nothing,
                test_inplace=true,
                test_rand_point=true,
                test_rand_tvector=true,
            )

            @testset "inner/norm" begin
                v1 = inverse_retract(M, pts[1], pts[2], PolarInverseRetraction())
                v2 = inverse_retract(M, pts[1], pts[3], PolarInverseRetraction())

                @test real(inner(M, pts[1], v1, v2)) ≈ real(inner(M, pts[1], v2, v1))
                @test imag(inner(M, pts[1], v1, v2)) ≈ -imag(inner(M, pts[1], v2, v1))
                @test imag(inner(M, pts[1], v1, v1)) ≈ 0

                @test norm(M, pts[1], v1) isa Real
                @test norm(M, pts[1], v1) ≈ sqrt(inner(M, pts[1], v1, v1))
            end
        end

        @testset "Distribution tests" begin
            usd_mmatrix = Manifolds.uniform_distribution(M, @MMatrix [
                1.0 0.0
                0.0 1.0
                0.0 0.0
            ])
            @test isa(rand(usd_mmatrix), MMatrix)
        end
    end

    @testset "Complex" begin
        M = Stiefel(3, 2, ℂ)
        @testset "Basics" begin
            @test repr(M) == "Stiefel(3, 2, ℂ)"
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 8
            @test Manifolds.allocation_promotion_function(M, exp!, (1,)) == complex
            @test !is_point(M, [1.0, 0.0, 0.0, 0.0])
            @test !is_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0.0, 0.0, 1.0, 0.0])
            x = [1.0 0.0; 0.0 1.0; 0.0 0.0]
        end
        types = [Matrix{ComplexF64}]
        @testset "Type $T" for T in types
            x = [0.5+0.5im 0.5+0.5im; 0.5+0.5im -0.5-0.5im; 0.0 0.0]
            y = exp(M, x, [0.0 0.0; 0.0 0.0; 1.0 1.0])
            z = exp(M, x, [0.0 0.0; 0.0 0.0; -1.0 1.0])
            pts = convert.(T, [x, y, z])
            v = inverse_retract(M, x, y, PolarInverseRetraction())
            @test !is_point(M, 2 * x)
            @test_throws DomainError !is_point(M, 2 * x, true)
            @test !is_vector(M, 2 * x, v)
            @test_throws ManifoldDomainError !is_vector(M, 2 * x, v, true)
            @test !is_vector(M, x, y)
            @test_throws DomainError is_vector(M, x, y, true)
            test_manifold(
                M,
                pts,
                test_exp_log=false,
                default_inverse_retraction_method=PolarInverseRetraction(),
                test_injectivity_radius=false,
                test_is_tangent=true,
                test_project_tangent=true,
                test_default_vector_transport=false,
                test_vee_hat=false,
                projection_atol_multiplier=15.0,
                retraction_atol_multiplier=10.0,
                is_tangent_atol_multiplier=4 * 10.0^2,
                retraction_methods=[PolarRetraction(), QRRetraction()],
                inverse_retraction_methods=[
                    PolarInverseRetraction(),
                    QRInverseRetraction(),
                ],
                vector_transport_methods=[
                    DifferentiatedRetractionVectorTransport(PolarRetraction()),
                    DifferentiatedRetractionVectorTransport(QRRetraction()),
                    ProjectionTransport(),
                ],
                vector_transport_retractions=[
                    PolarRetraction(),
                    QRRetraction(),
                    PolarRetraction(),
                ],
                vector_transport_inverse_retractions=[
                    PolarInverseRetraction(),
                    QRInverseRetraction(),
                    PolarInverseRetraction(),
                ],
                test_vector_transport_direction=[true, true, false],
                mid_point12=nothing,
                test_inplace=true,
                test_rand_point=true,
            )

            @testset "inner/norm" begin
                v1 = inverse_retract(M, pts[1], pts[2], PolarInverseRetraction())
                v2 = inverse_retract(M, pts[1], pts[3], PolarInverseRetraction())

                @test real(inner(M, pts[1], v1, v2)) ≈ real(inner(M, pts[1], v2, v1))
                @test imag(inner(M, pts[1], v1, v2)) ≈ -imag(inner(M, pts[1], v2, v1))
                @test imag(inner(M, pts[1], v1, v1)) ≈ 0

                @test norm(M, pts[1], v1) isa Real
                @test norm(M, pts[1], v1) ≈ sqrt(inner(M, pts[1], v1, v1))
            end
        end
    end

    @testset "Quaternion" begin
        M = Stiefel(3, 2, ℍ)
        @testset "Basics" begin
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 18
        end
    end

    @testset "Padé & Caley retractions and Caley based transport" begin
        M = Stiefel(3, 2)
        p = [1.0 0.0; 0.0 1.0; 0.0 0.0]
        X = [0.0 0.0; 0.0 0.0; 1.0 1.0]
        r1 = CayleyRetraction()
        @test r1 == PadeRetraction(1)
        @test repr(r1) == "CayleyRetraction()"
        q1 = retract(M, p, X, r1)
        @test is_point(M, q1)
        Y = vector_transport_direction(
            M,
            p,
            X,
            X,
            DifferentiatedRetractionVectorTransport(CayleyRetraction()),
        )
        @test is_vector(M, q1, Y; atol=10^-15)
        Y2 = vector_transport_direction(
            M,
            p,
            X,
            X,
            DifferentiatedRetractionVectorTransport(CayleyRetraction()),
        )
        @test is_vector(M, q1, Y2; atol=10^-15)
        r2 = PadeRetraction(2)
        @test repr(r2) == "PadeRetraction(2)"
        q2 = retract(M, p, X, r2)
        @test is_point(M, q2)
    end

    @testset "Canonical Metric" begin
        M3 = MetricManifold(Stiefel(3, 2), CanonicalMetric())
        p = [1.0 0.0; 0.0 1.0; 0.0 0.0]
        X = [0.0 0.0; 0.0 0.0; 1.0 1.0]
        q = exp(M3, p, X)
        Y = [0.0 0.0; 0.0 0.0; -1.0 1.0]
        r = exp(M3, p, Y)
        @test isapprox(M3, p, log(M3, p, q), X)
        @test isapprox(M3, p, log(M3, p, r), Y)
        @test inner(M3, p, X, Y) == 0
        @test inner(M3, p, X, 2 * X + 3 * Y) == 2 * inner(M3, p, X, X)
        @test norm(M3, p, X) ≈ distance(M3, p, q)
        # check on a higher dimensional manifold, that the iterations are actually used
        M4 = MetricManifold(Stiefel(10, 2), CanonicalMetric())
        p = Matrix{Float64}(I, 10, 2)
        Random.seed!(42)
        Z = project(base_manifold(M4), p, 0.2 .* randn(size(p)))
        s = exp(M4, p, Z)
        Z2 = log(M4, p, s)
        @test isapprox(M4, p, Z, Z2)
        Z3 = similar(Z2)
        log!(M4, Z3, p, s)
        @test isapprox(M4, p, Z2, Z3)
    end
end
