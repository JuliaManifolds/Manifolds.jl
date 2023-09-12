include("../utils.jl")

@testset "Stiefel" begin
    @testset "Real" begin
        M = Stiefel(3, 2)
        M2 = MetricManifold(M, EuclideanMetric())
        @testset "Basics" begin
            @test repr(M) == "Stiefel(3, 2, ℝ)"
            p = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            @test is_default_metric(M, EuclideanMetric())
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 3
            @test !is_flat(M)
            @test !is_flat(M2)
            @test is_flat(Stiefel(2, 1))
            base_manifold(M) === M
            @test_throws ManifoldDomainError is_point(M, [1.0, 0.0, 0.0, 0.0], true)
            @test_throws ManifoldDomainError is_point(
                M,
                1im * [1.0 0.0; 0.0 1.0; 0.0 0.0],
                true,
            )
            @test !is_vector(M, p, [0.0, 0.0, 1.0, 0.0])
            @test_throws ManifoldDomainError is_vector(
                M,
                p,
                1 * im * zero_vector(M, p),
                true,
            )
            @test default_retraction_method(M) === PolarRetraction()
            @test default_inverse_retraction_method(M) === PolarInverseRetraction()
            vtm = DifferentiatedRetractionVectorTransport(PolarRetraction())
            @test default_vector_transport_method(M) === vtm
        end
        @testset "Embedding and Projection" begin
            p = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            q = similar(p)
            r = embed(M, p)
            @test r == p
            embed!(M, q, p)
            @test q == r
            a = [1.0 0.0; 0.0 2.0; 0.0 0.0]
            @test !is_point(M, a)
            b = similar(a)
            c = project(M, a)
            @test c == p
            project!(M, b, a)
            @test b == p
            X = [0.0 0.0; 0.0 0.0; -1.0 1.0]
            Y = similar(X)
            Z = embed(M, p, X)
            embed!(M, Y, p, X)
            @test Y == X
            @test Z == X
        end
        @testset "gradient and metric conversion" begin
            p = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            X = [0.0 0.0; 0.0 0.0; -1.0 1.0]
            Y = change_metric(M, EuclideanMetric(), p, X)
            @test Y == X
            Z = change_representer(M, EuclideanMetric(), p, X)
            @test Z == X
            # In this case it stays as is
            @test riemannian_Hessian(M, p, Y, Z, X) == Z
            V = [1.0 1.0; 0.0 0.0; 0.0 0.0] # From T\bot_pM.
            @test Weingarten(M, p, X, V) == [0.0 0.0; 0.0 0.0; 1.0 1.0]
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
                test_exp_log=true,
                default_inverse_retraction_method=PolarInverseRetraction(),
                test_injectivity_radius=false,
                test_is_tangent=true,
                test_project_tangent=true,
                test_default_vector_transport=false,
                point_distributions=[Manifolds.uniform_distribution(M, pts[1])],
                test_vee_hat=false,
                projection_atol_multiplier=200.0,
                exp_log_atol_multiplier=10.0,
                retraction_atol_multiplier=10.0,
                is_tangent_atol_multiplier=4 * 10.0^2,
                retraction_methods=[
                    PolarRetraction(),
                    QRRetraction(),
                    CayleyRetraction(),
                    PadeRetraction(2),
                    ProjectionRetraction(),
                ],
                inverse_retraction_methods=[
                    PolarInverseRetraction(),
                    QRInverseRetraction(),
                    ProjectionInverseRetraction(),
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
            @test !is_flat(M)
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
                projection_atol_multiplier=200.0,
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
        @test !is_flat(M3)
        @test isapprox(M3, p, log(M3, p, q), X)
        @test isapprox(M3, p, log(M3, p, r), Y)
        @test inner(M3, p, X, Y) == 0
        @test inner(M3, p, X, 2 * X + 3 * Y) == 2 * inner(M3, p, X, X)
        @test norm(M3, p, X) ≈ distance(M3, p, q)
        Z = [0.0 0.0; 0.0 0.0; -1.0 -1.0]
        @test riemannian_Hessian(M3, p, Y, Z, X) == [0.0 0.5; -0.5 0.0; -1.0 -1.0]
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

        M4 = MetricManifold(Stiefel(3, 3), CanonicalMetric())
        p = project(M4, randn(3, 3))
        X = project(M4, p, randn(3, 3))
        Y = project(M4, p, randn(3, 3))
        @test inner(M4, p, X, Y) ≈ tr(X' * (I - p * p' / 2) * Y)
    end

    @testset "StiefelSubmersionMetric" begin
        @testset "StiefelFactorization" begin
            n = 6
            @testset for k in [2, 3]
                M = MetricManifold(Stiefel(n, k), StiefelSubmersionMetric(rand()))
                p = project(M, randn(representation_size(M)))
                X = project(M, p, randn(representation_size(M)))
                X /= norm(M, p, X)
                q = exp(M, p, X)
                qfact = Manifolds.stiefel_factorization(p, q)
                @testset "basic properties" begin
                    @test qfact isa Manifolds.StiefelFactorization
                    @test qfact.U[1:n, 1:k] ≈ p
                    @test qfact.U'qfact.U ≈ I
                    @test qfact.U * qfact.Z ≈ q
                    @test is_point(Stiefel(2k, k), qfact.Z)

                    Xfact = Manifolds.stiefel_factorization(p, X)
                    @test Xfact isa Manifolds.StiefelFactorization
                    @test Xfact.U[1:n, 1:k] ≈ p
                    @test Xfact.U'Xfact.U ≈ I
                    @test Xfact.U * Xfact.Z ≈ X
                    @test is_vector(Rotations(k), I(k), Xfact.Z[1:k, 1:k])

                    pfact2 = Manifolds.stiefel_factorization(p, p)
                    @test pfact2 isa Manifolds.StiefelFactorization
                    @test pfact2.U[1:n, 1:k] ≈ p
                    @test pfact2.U'pfact2.U ≈ I
                    @test pfact2.Z ≈ [I(k); zeros(k, k)] atol = 1e-6
                end
                @testset "basic functions" begin
                    @test size(qfact) == (n, k)
                    @test eltype(qfact) === eltype(q)
                end
                @testset "similar" begin
                    qfact2 = similar(qfact)
                    @test qfact2.U === qfact.U
                    @test size(qfact2.Z) == size(qfact.Z)
                    @test eltype(qfact2.Z) === eltype(qfact.Z)

                    qfact3 = similar(qfact, Float32)
                    @test eltype(qfact3) === Float32
                    @test eltype(qfact3.U) === Float32
                    @test eltype(qfact3.Z) === Float32
                    @test qfact3.U ≈ qfact.U
                    @test size(qfact3.Z) == size(qfact.Z)

                    qfact4 = similar(qfact, Float32, (n, k))
                    @test eltype(qfact4) === Float32
                    @test eltype(qfact4.U) === Float32
                    @test eltype(qfact4.Z) === Float32
                    @test qfact4.U ≈ qfact.U
                    @test size(qfact4.Z) == size(qfact.Z)

                    @test_throws Exception similar(qfact, Float32, (n, k + 1))
                end
                @testset "copyto!" begin
                    qfact2 = similar(qfact)
                    copyto!(qfact2, qfact)
                    @test qfact2.U === qfact.U
                    @test qfact2.Z ≈ qfact.Z

                    q2 = similar(q)
                    copyto!(q2, qfact)
                    @test q2 ≈ q

                    qfact3 = similar(qfact)
                    copyto!(qfact3, q)
                    @test qfact3.U === qfact.U
                    @test qfact3.Z ≈ qfact.Z
                end
                @testset "dot" begin
                    Afact = similar(qfact)
                    Afact.Z .= randn.()
                    A = copyto!(similar(q), Afact)
                    Bfact = similar(qfact)
                    Bfact.Z .= randn.()
                    B = copyto!(similar(q), Bfact)
                    @test dot(Afact, Bfact) ≈ dot(A, B)
                end
                @testset "broadcast!" begin
                    rfact = similar(qfact)
                    @testset for f in [*, +, -]
                        rfact .= f.(qfact, 2.5)
                        @test rfact.U === qfact.U
                        @test rfact.Z ≈ f.(qfact.Z, 2.5)
                    end
                end
                @testset "project" begin
                    rfact = similar(qfact)
                    rfact.Z .= randn.()
                    r = copyto!(similar(q), rfact)
                    rfactproj = project(M, rfact)
                    @test rfactproj isa Manifolds.StiefelFactorization
                    @test copyto!(similar(r), rfactproj) ≈ project(M, r)

                    Yfact = similar(qfact)
                    Yfact.Z .= randn.()
                    Y = copyto!(similar(q), Yfact)
                    Yfactproj = project(M, rfact, Yfact)
                    @test Yfactproj isa Manifolds.StiefelFactorization
                    @test copyto!(similar(Y), Yfactproj) ≈ project(M, r, Y)
                end
                @testset "inner" begin
                    rfact = similar(qfact)
                    rfact.Z .= randn.()
                    rfact = project(M, rfact)

                    Yfact = similar(qfact)
                    Yfact.Z .= randn.()
                    Yfact = project(M, rfact, Yfact)

                    Zfact = similar(qfact)
                    Zfact.Z .= randn.()
                    Zfact = project(M, rfact, Zfact)

                    r, Z, Y = map(x -> copyto!(similar(q), x), (rfact, Zfact, Yfact))
                    @test inner(M, rfact, Yfact, Zfact) ≈ inner(M, r, Y, Z)
                end
                @testset "exp" begin
                    pfact = copyto!(similar(qfact), p)
                    Xfact = copyto!(similar(qfact), X)
                    rfact = exp(M, pfact, Xfact)
                    r = exp(M, p, X)
                    @test rfact isa Manifolds.StiefelFactorization
                    @test copyto!(similar(r), rfact) ≈ r
                end
            end
        end

        g = StiefelSubmersionMetric(1)
        @test g isa StiefelSubmersionMetric{Int}

        @testset for M in [Stiefel(3, 3), Stiefel(4, 3), Stiefel(4, 2)]
            Mcan = MetricManifold(M, CanonicalMetric())
            Meu = MetricManifold(M, EuclideanMetric())
            @testset "α=$α" for (α, Mcomp) in [(0, Mcan), (-1 // 2, Meu)]
                p = project(M, randn(representation_size(M)))
                X = project(M, p, randn(representation_size(M)))
                X ./= norm(Mcomp, p, X)
                Y = project(M, p, randn(representation_size(M)))
                MM = MetricManifold(M, StiefelSubmersionMetric(α))
                @test inner(MM, p, X, Y) ≈ inner(Mcomp, p, X, Y)
                q = exp(Mcomp, p, X)
                @test isapprox(MM, q, exp(Mcomp, p, X))
                Mcomp === Mcan && isapprox(MM, p, log(MM, p, q), log(Mcomp, p, q))
                @test isapprox(MM, exp(MM, p, 0 * X), p)
                @test isapprox(MM, p, log(MM, p, p), zero_vector(MM, p); atol=1e-6)
            end
            @testset "α=$α" for α in [-0.75, -0.25, 0.5]
                MM = MetricManifold(M, StiefelSubmersionMetric(α))
                p = project(MM, randn(representation_size(M)))
                X = project(MM, p, randn(representation_size(M)))
                X ./= norm(MM, p, X)
                q = exp(MM, p, X)
                @test is_point(MM, q)
                @test isapprox(MM, p, log(MM, p, q), X)
                @test isapprox(MM, exp(MM, p, 0 * X), p)
                @test isapprox(MM, p, log(MM, p, p), zero_vector(MM, p); atol=1e-6)
            end
        end

        @testset "Hessian Conversion" begin
            M1 = MetricManifold(Stiefel(3, 2), StiefelSubmersionMetric(-0.5))
            M2 = Stiefel(3, 2)
            M2b = MetricManifold(Stiefel(3, 2), EuclideanMetric())
            M3 = MetricManifold(Stiefel(3, 2), StiefelSubmersionMetric(0.0))
            M4 = MetricManifold(Stiefel(3, 2), CanonicalMetric())
            p = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            X = [0.0 0.0; 0.0 0.0; 1.0 1.0]
            Y = [0.0 0.0; 0.0 0.0; -1.0 1.0]
            Z = [0.0 0.0; 0.0 0.0; -1.0 -1.0]
            rH = riemannian_Hessian(M2, p, Y, Z, X)
            @test riemannian_Hessian(M1, p, Y, Z, X) == rH #Special case of submersion metric
            @test riemannian_Hessian(M2b, p, Y, Z, X) == rH # metric is default
            @test riemannian_Hessian(M3, p, Y, Z, X) == riemannian_Hessian(M4, p, Y, Z, X)

            V = [0.0 -1.0; 1.0 0.0; 0.0 0.0]
            W = zero_vector(M2, p)
            Weingarten!(M2, W, p, X, V)
            Wb = zero_vector(M2b, p)
            Weingarten!(M2b, Wb, p, X, V)
            @test W == Wb
        end
    end
end
