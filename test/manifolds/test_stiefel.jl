using Distributions, LinearAlgebra, Manifolds, RecursiveArrayTools, StaticArrays, Test

@testset "The Stiefel manifolds" begin
    @testset "Real Stiefel Manifold" begin
        M = Stiefel(3, 2)
        p = [1.0 0.0; 0.0 1.0; 0.0 0.0]
        q = 1 ./ sqrt(2) .* [ 1.0 -1.0; 1.0 1.0; 0.0 0.0]
        r = 1 ./ sqrt(2) .* [ 0.0 0.0; 1.0 -1.0; 1.0 1.0]
        X = [0.0 0.0; 0.0 0.0; -0.1 0.2]
        X2 = [0.0 1.0; -1.0 0.0; 0.0 0.0]
        Y = [0.0 0.0; 0.0 0.0; -0.3 0.4]
        Z = [0.5 -0.6; 0.0 0.0; 0.0 0.0]
        # Invalids
        pf = [1.0, 0.0, 0.0, 0.0]
        Xf = [0.0, 0.0, 1.0, 0.0]
        # Embedded
        ep = [1.0 0.0; 0.0 2.0; 0.0 0.0]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :EmbeddedPoints => [ep],
                :Functions => [
                    copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                    default_vector_transport_method, distance, embed, embed_project, exp,
                    geodesic, get_basis, get_coordinates, get_embedding, get_vector, get_vectors,
                    injectivity_radius,
                    inner, inverse_retract, is_default_metric, is_flat, is_point, is_vector, log,
                    manifold_dimension, mid_point, norm, project, rand, repr, representation_size, retract,
                    shortest_geodesic, vector_transport_direction, vector_transport_to, zero_vector,
                ],
                :InvalidPoints => [pf],
                :InvalidVectors => [Xf],
                :InverseRetractionMethods => [PolarInverseRetraction(), PolarLightInverseRetraction(), ProjectionInverseRetraction(), QRInverseRetraction()],
                :Points => [p, q, r],
                :Vectors => [X, Y, Z],
                :Vector => X2,
                :RetractionMethod => [CayleyRetraction(), PadeRetraction(2), PolarRetraction(), PolarLightRetraction(), ProjectionRetraction(), QRRetraction()],
                :VectorTransportMethods => [
                    DifferentiatedRetractionVectorTransport(PolarRetraction()),
                    DifferentiatedRetractionVectorTransport(QRRetraction()),
                    # errors on alias test
                    # ProjectionTransport(),
                ],
            ),
            # Expectations
            Dict(
                :atol => 1.0e-12,
                default_inverse_retraction_method => PolarInverseRetraction(),
                default_retraction_method => PolarRetraction(),
                default_vector_transport_method => DifferentiatedRetractionVectorTransport(PolarRetraction()),
                injectivity_radius => π,
                manifold_dimension => 3,
                repr => "Stiefel(3, 2, ℝ)",
            )
        )
        @testset "Distribution tests" begin
            usd_mmatrix = Manifolds.uniform_distribution(M, @MMatrix [1.0 0.0; 0.0 1.0; 0.0 0.0])
            @test isa(rand(usd_mmatrix), MMatrix)
        end
        @testset "Basics" begin
            @test base_manifold(M) === M
        end
        @testset "gradient and metric conversion" begin
            Xm = change_metric(M, EuclideanMetric(), p, X)
            @test Xm == X
            Xr = change_representer(M, EuclideanMetric(), p, X)
            @test Xr == X
            # In this case it stays as is
            @test riemannian_Hessian(M, p, Xm, Xr, X) == Xr
            V = [1.0 1.0; 0.0 0.0; 0.0 0.0] # From T\bot_pM.
            @test Weingarten(M, p, X, V) == [0.0 0.0; 0.0 0.0; 0.1 0.1]
        end
        @testset "Different Metrics" begin
            @testset "StiefelSubmersionMetric" begin
                @testset "StiefelFactorization" begin
                    n = 6
                    @testset for k in [2, 3]
                        M = MetricManifold(Stiefel(n, k), StiefelSubmersionMetric(rand()))
                        pl = project(M, randn(representation_size(M)))
                        Xl = project(M, pl, randn(representation_size(M)))
                        Xl /= norm(M, pl, Xl)
                        ql = exp(M, pl, Xl)
                        qfact = Manifolds.stiefel_factorization(pl, ql)
                        @testset "basic properties" begin
                            @test qfact isa Manifolds.StiefelFactorization
                            @test qfact.U[1:n, 1:k] ≈ pl
                            @test qfact.U'qfact.U ≈ I
                            @test qfact.U * qfact.Z ≈ ql
                            @test is_point(Stiefel(2k, k), qfact.Z)

                            Xfact = Manifolds.stiefel_factorization(pl, Xl)
                            @test Xfact isa Manifolds.StiefelFactorization
                            @test Xfact.U[1:n, 1:k] ≈ pl
                            @test Xfact.U'Xfact.U ≈ I
                            @test Xfact.U * Xfact.Z ≈ Xl
                            @test is_vector(Rotations(k), I(k), Xfact.Z[1:k, 1:k])

                            pfact2 = Manifolds.stiefel_factorization(pl, pl)
                            @test pfact2 isa Manifolds.StiefelFactorization
                            @test pfact2.U[1:n, 1:k] ≈ pl
                            @test pfact2.U'pfact2.U ≈ I
                            @test pfact2.Z ≈ [I(k); zeros(k, k)] atol = 1.0e-6
                        end
                        @testset "basic functions" begin
                            @test size(qfact) == (n, k)
                            @test eltype(qfact) === eltype(ql)
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

                            q2 = similar(ql)
                            copyto!(q2, qfact)
                            @test q2 ≈ ql

                            qfact3 = similar(qfact)
                            copyto!(qfact3, ql)
                            @test qfact3.U === qfact.U
                            @test qfact3.Z ≈ qfact.Z
                        end
                        @testset "dot" begin
                            Afact = similar(qfact)
                            Afact.Z .= randn.()
                            A = copyto!(similar(ql), Afact)
                            Bfact = similar(qfact)
                            Bfact.Z .= randn.()
                            B = copyto!(similar(ql), Bfact)
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
                            r = copyto!(similar(ql), rfact)
                            rfactproj = project(M, rfact)
                            @test rfactproj isa Manifolds.StiefelFactorization
                            @test copyto!(similar(r), rfactproj) ≈ project(M, r)

                            Yfact = similar(qfact)
                            Yfact.Z .= randn.()
                            Y = copyto!(similar(ql), Yfact)
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

                            r, Z, Y = map(x -> copyto!(similar(ql), x), (rfact, Zfact, Yfact))
                            @test inner(M, rfact, Yfact, Zfact) ≈ inner(M, r, Y, Z)
                        end
                        @testset "exp" begin
                            pfact = copyto!(similar(qfact), pl)
                            Xfact = copyto!(similar(qfact), Xl)
                            rfact = exp(M, pfact, Xfact)
                            r = exp(M, pl, Xl)
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
                        @test isapprox(MM, q, exp(Mcomp, p, X); error = :error)
                        Mcomp === Mcan && isapprox(MM, p, log(MM, p, q), log(Mcomp, p, q))
                        @test isapprox(MM, exp(MM, p, 0 * X), p; error = :error)
                        @test isapprox(MM, p, log(MM, p, p), zero_vector(MM, p); error = :error, atol = 1.0e-6)
                    end
                    @testset "α=$α" for α in [-0.75, -0.25, 0.5]
                        MM = MetricManifold(M, StiefelSubmersionMetric(α))
                        p = project(MM, randn(representation_size(M)))
                        X = project(MM, p, randn(representation_size(M)))
                        X ./= norm(MM, p, X)
                        q = exp(MM, p, X)
                        @test is_point(MM, q; error = :error)
                        @test isapprox(MM, p, log(MM, p, q), X; error = :error)
                        @test isapprox(MM, exp(MM, p, 0 * X), p; error = :error)
                        @test isapprox(MM, p, log(MM, p, p), zero_vector(MM, p); error = :error, atol = 1.0e-6)
                    end
                end

                @testset "Hessian Conversion" begin
                    M1 = MetricManifold(Stiefel(3, 2), StiefelSubmersionMetric(-0.5))
                    M2 = Stiefel(3, 2)
                    M2b = MetricManifold(Stiefel(3, 2), EuclideanMetric())
                    M3 = MetricManifold(Stiefel(3, 2), StiefelSubmersionMetric(0.0))
                    M4 = MetricManifold(Stiefel(3, 2), CanonicalMetric())
                    pH = [1.0 0.0; 0.0 1.0; 0.0 0.0]
                    XH = [0.0 0.0; 0.0 0.0; 1.0 1.0]
                    YH = [0.0 0.0; 0.0 0.0; -1.0 1.0]
                    ZH = [0.0 0.0; 0.0 0.0; -1.0 -1.0]
                    rH = riemannian_Hessian(M2, pH, YH, ZH, XH)
                    @test riemannian_Hessian(M1, pH, YH, ZH, XH) == rH #Special case of submersion metric
                    @test riemannian_Hessian(M2b, pH, YH, ZH, XH) == rH # metric is default
                    @test riemannian_Hessian(M3, pH, YH, ZH, XH) == riemannian_Hessian(M4, pH, YH, ZH, XH)
                    VH = [0.0 -1.0; 1.0 0.0; 0.0 0.0]
                    WH = zero_vector(M2, pH)
                    Weingarten!(M2, WH, pH, XH, VH)
                    WHb = zero_vector(M2b, pH)
                    Weingarten!(M2b, WHb, pH, XH, VH)
                    @test WH == WHb
                end
            end
            @testset "field parameter" begin
                M = Stiefel(3, 2; parameter = :field)
                @test typeof(get_embedding(M)) === Euclidean{ℝ, Tuple{Int, Int}}
                @test repr(M) == "Stiefel(3, 2, ℝ; parameter=:field)"
            end
        end
        @testset "Stiefel(2, 1) special case" begin
            M21 = Stiefel(2, 1)
            X21 = inverse_retract(
                M21, SMatrix{2, 1}([0.0, 1.0]), SMatrix{2, 1}([sqrt(2), sqrt(2)]),
                QRInverseRetraction(),
            )
            @test isapprox(M21, X21, SMatrix{2, 1}([1.0, 0.0]))
        end
        @testset "Stiefel(4,3) inverse QR retraction cases" begin
            M43 = Stiefel(4, 3)
            p43 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0; 0.0 0.0 0.0]
            X43 = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 1.0 1.0]
            q43 = retract(M43, p43, X43, QRRetraction())
            Y243 = inverse_retract(
                M43,
                SMatrix{4, 3}(p43),
                SMatrix{4, 3}(q43),
                QRInverseRetraction(),
            )
            Z243 = inverse_retract(M43, p43, q43, QRInverseRetraction())
            @test isapprox(M43, p43, Y243, Z243)
            @test isapprox(M43, p43, Y243, X43)
            p243 = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            q243 = exp(M, p243, [0.0 0.0; 0.0 0.0; 1.0 1.0])

            Y243 = inverse_retract(
                M,
                SMatrix{3, 2}(p243),
                SMatrix{3, 2}(q243),
                QRInverseRetraction(),
            )
            Z243 = inverse_retract(M, p243, q243, QRInverseRetraction())
            @test isapprox(M43, p243, Y243, Z243)
        end
    end
    @testset "Complex Stiefel Manifold" begin
        Mc = Stiefel(3, 2, ℂ)
        pc = [0.5 + 0.5im 0.5 + 0.5im; 0.5 + 0.5im -0.5 - 0.5im; 0.0 0.0]
        Xc = [0.0 0.0; 0.0 0.0; 0.1 -0.1]
        qc = exp(Mc, pc, Xc)
        rc = exp(Mc, pc, Xc)
        Yc = inverse_retract(Mc, qc, pc, PolarInverseRetraction())
        Zc = inverse_retract(Mc, rc, pc, PolarInverseRetraction())
        # Invalid ones
        pcf = [1.0, 0.0, 0.0, 0.0]
        Xcf = [0.0, 0.0, 1.0, 0.0]
        # Embedded
        ep = [1.0 0.0; 0.0 2.0; 0.0 0.0]
        Manifolds.Test.test_manifold(
            Mc,
            Dict(
                :EmbeddedPoints => [ep],
                :Functions => [
                    copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                    default_vector_transport_method, embed, exp,
                    geodesic, get_basis, get_coordinates, get_embedding, get_vector, get_vectors,
                    injectivity_radius,
                    inner, inverse_retract, is_default_metric, is_flat, is_point, is_vector,
                    manifold_dimension, norm, project, repr, representation_size, retract,
                    vector_transport_direction, vector_transport_to, zero_vector,
                ],
                :InvalidPoints => [pcf, 2 .* pc],
                :InvalidVectors => [Xcf],
                :InverseRetractionMethods => [PolarInverseRetraction(), QRInverseRetraction()],
                :Points => [pc, qc, rc],
                :Vectors => [Xc, Yc, Zc],
                :RetractionMethod => [PolarRetraction(), QRRetraction()],
                :VectorTransportMethods => [
                    DifferentiatedRetractionVectorTransport(PolarRetraction()),
                ],
            ),
            # Expectations
            Dict(
                :atol => 1.0e-12,
                default_inverse_retraction_method => PolarInverseRetraction(),
                default_retraction_method => PolarRetraction(),
                default_vector_transport_method => DifferentiatedRetractionVectorTransport(PolarRetraction()),
                injectivity_radius => π,
                manifold_dimension => 3,
                repr => "Stiefel(3, 2, ℂ)",
                representation_size => (3, 2),
                manifold_dimension => 8,
                is_flat => false,
            )
        )
    end
    @testset "Quaternion Stiefel" begin
        M = Stiefel(3, 2, ℍ)
        @testset "Basics" begin
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 18
        end
    end
end
