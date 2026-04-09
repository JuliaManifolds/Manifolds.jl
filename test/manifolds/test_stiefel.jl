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
                :Bases => [DefaultOrthogonalBasis(), DefaultOrthonormalBasis()],
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
                # ProjectionRetraction and its inverse are tested separately for now since they do not seem inverses
                :InverseRetractionMethods => [missing, missing, PolarInverseRetraction(), PolarLightInverseRetraction(), ProjectionInverseRetraction(), missing, QRInverseRetraction()],
                :Points => [p, q, r],
                :Vectors => [X, Y, Z],
                :SecondVector => X2,
                :RetractionMethods => [CayleyRetraction(), PadeRetraction(2), PolarRetraction(), PolarLightRetraction(), missing, ProjectionRetraction(), QRRetraction()],
                :VectorTransportMethods => [
                    DifferentiatedRetractionVectorTransport(PolarRetraction()),
                    DifferentiatedRetractionVectorTransport(QRRetraction()),
                    ProjectionTransport(),
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
        @testset "Padé & Caley retractions and Caley based transport (direction only)" begin
            rca = CayleyRetraction()
            @test rca == PadeRetraction(1)
            @test repr(rca) == "CayleyRetraction()"
            qca = retract(M, p, X, rca)
            @test is_point(M, qca; error = :error)
            Yca = vector_transport_direction(
                M, p, X, X, DifferentiatedRetractionVectorTransport(CayleyRetraction()),
            )
            @test is_vector(M, qca, Yca; atol = 10^-15, error = :error)
            Zca = zero_vector(M, p)
            vector_transport_direction!(
                M, Zca, p, X, X, DifferentiatedRetractionVectorTransport(CayleyRetraction()),
            )
            @test is_vector(M, qca, Yca; atol = 10^-15)
            @test isapprox(M, qca, Yca, Zca)
            rpa = PadeRetraction(2)
            @test repr(rpa) == "PadeRetraction(2)"
            qpa = retract(M, p, X, rpa)
            @test is_point(M, qpa; error = :error)
        end


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
                        Ml = MetricManifold(Stiefel(n, k), StiefelSubmersionMetric(rand()))
                        pl = project(Ml, randn(representation_size(Ml)))
                        Xl = project(Ml, pl, randn(representation_size(Ml)))
                        Xl /= norm(Ml, pl, Xl)
                        ql = exp(Ml, pl, Xl)
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
                            rfactproj = project(Ml, rfact)
                            @test rfactproj isa Manifolds.StiefelFactorization
                            @test copyto!(similar(r), rfactproj) ≈ project(Ml, r)

                            Yfact = similar(qfact)
                            Yfact.Z .= randn.()
                            Y = copyto!(similar(ql), Yfact)
                            Yfactproj = project(Ml, rfact, Yfact)
                            @test Yfactproj isa Manifolds.StiefelFactorization
                            @test copyto!(similar(Y), Yfactproj) ≈ project(Ml, r, Y)
                        end
                        @testset "inner" begin
                            rfact = similar(qfact)
                            rfact.Z .= randn.()
                            rfact = project(Ml, rfact)

                            Yfact = similar(qfact)
                            Yfact.Z .= randn.()
                            Yfact = project(Ml, rfact, Yfact)

                            Zfact = similar(qfact)
                            Zfact.Z .= randn.()
                            Zfact = project(Ml, rfact, Zfact)

                            r, Z, Y = map(x -> copyto!(similar(ql), x), (rfact, Zfact, Yfact))
                            @test inner(Ml, rfact, Yfact, Zfact) ≈ inner(Ml, r, Y, Z)
                        end
                        @testset "exp" begin
                            pfact = copyto!(similar(qfact), pl)
                            Xfact = copyto!(similar(qfact), Xl)
                            rfact = exp(Ml, pfact, Xfact)
                            r = exp(Ml, pl, Xl)
                            @test rfact isa Manifolds.StiefelFactorization
                            @test copyto!(similar(r), rfact) ≈ r
                        end
                    end
                end
                g = StiefelSubmersionMetric(1)
                @test g isa StiefelSubmersionMetric{Int}
                @testset for Ml in [Stiefel(3, 3), Stiefel(4, 3), Stiefel(4, 2)]
                    Mcan = MetricManifold(Ml, CanonicalMetric())
                    Meu = MetricManifold(Ml, EuclideanMetric())
                    @testset "α=$α" for (α, Mcomp) in [(0, Mcan), (-1 // 2, Meu)]
                        pl = project(Ml, randn(representation_size(Ml)))
                        Xl = project(Ml, pl, randn(representation_size(Ml)))
                        Xl ./= norm(Mcomp, pl, Xl)
                        Yl = project(Ml, pl, randn(representation_size(Ml)))
                        MMl = MetricManifold(Ml, StiefelSubmersionMetric(α))
                        @test inner(MMl, pl, Xl, Yl) ≈ inner(Mcomp, pl, Xl, Yl)
                        ql = exp(Mcomp, pl, Xl)
                        @test isapprox(MMl, ql, exp(Mcomp, pl, Xl); error = :error)
                        if Mcomp === Mcan
                            @test !is_flat(Mcomp)
                            Zl1 = log(Mcomp, pl, ql)
                            isapprox(MMl, pl, log(MMl, pl, ql), Zl1)
                            Zl2 = similar(Zl1)
                            log!(Mcomp, Zl2, pl, ql)
                            isapprox(MMl, pl, log(MMl, pl, ql), Zl1)
                            @test distance(Mcomp, pl, ql) ≈ norm(Mcomp, pl, Zl2)
                        end
                        @test isapprox(MMl, exp(MMl, pl, 0 * Xl), pl; error = :error)
                        @test isapprox(MMl, pl, log(MMl, pl, pl), zero_vector(MMl, pl); error = :error, atol = 1.0e-6)
                    end
                    @testset "α=$α" for α in [-0.75, -0.25, 0.5]
                        MMl = MetricManifold(Ml, StiefelSubmersionMetric(α))
                        pl = project(MMl, randn(representation_size(Ml)))
                        Xl = project(MMl, pl, randn(representation_size(Ml)))
                        Xl ./= norm(MMl, pl, Xl)
                        ql = exp(MMl, pl, Xl)
                        @test is_point(MMl, ql; error = :error)
                        @test isapprox(MMl, pl, log(MMl, pl, ql), Xl; error = :error)
                        @test isapprox(MMl, exp(MMl, pl, 0 * Xl), pl; error = :error)
                        @test isapprox(MMl, pl, log(MMl, pl, pl), zero_vector(MMl, pl); error = :error, atol = 1.0e-6)
                    end
                end
                @testset "Hessian Conversion" begin
                    M1 = MetricManifold(M, StiefelSubmersionMetric(-0.5))
                    M2 = MetricManifold(M, EuclideanMetric())
                    M3 = MetricManifold(M, StiefelSubmersionMetric(0.0))
                    M4 = MetricManifold(M, CanonicalMetric())
                    pH = [1.0 0.0; 0.0 1.0; 0.0 0.0]
                    XH = [0.0 0.0; 0.0 0.0; 1.0 1.0]
                    YH = [0.0 0.0; 0.0 0.0; -1.0 1.0]
                    ZH = [0.0 0.0; 0.0 0.0; -1.0 -1.0]
                    rH = riemannian_Hessian(M, pH, YH, ZH, XH)
                    @test riemannian_Hessian(M1, pH, YH, ZH, XH) == rH #Special case of submersion metric
                    @test riemannian_Hessian(M2, pH, YH, ZH, XH) == rH # metric is default
                    @test riemannian_Hessian(M3, pH, YH, ZH, XH) == riemannian_Hessian(M4, pH, YH, ZH, XH)
                    VH = [0.0 -1.0; 1.0 0.0; 0.0 0.0]
                    WH = zero_vector(M, pH)
                    Weingarten!(M, WH, pH, XH, VH)
                    WHb = zero_vector(M, pH)
                    Weingarten!(M2, WHb, pH, XH, VH)
                    @test WH == WHb
                end
            end
        end
        @testset "field parameter" begin
            Mf = Stiefel(3, 2; parameter = :field)
            @test typeof(get_embedding(Mf)) === Euclidean{ℝ, Tuple{Int, Int}}
            @test repr(Mf) == "Stiefel(3, 2, ℝ; parameter=:field)"
        end
        @testset "Stiefel(2, 1) StaticArray inverse QR retraction cases" begin
            M21 = Stiefel(2, 1)
            p21 = SMatrix{2, 1}([0.0, 1.0])
            X21 = inverse_retract(
                M21, p21, SMatrix{2, 1}([sqrt(2), sqrt(2)]),
                QRInverseRetraction(),
            )
            @test isapprox(M21, p21, X21, SMatrix{2, 1}([1.0, 0.0]))
        end
        @testset "Stiefel(3, 2) StaticArray inverse QR retraction cases" begin
            pS = SMatrix{3, 2}(p)
            qS = SMatrix{3, 2}(q)
            XS = inverse_retract(
                M, pS, qS, QRInverseRetraction(),
            )
            q32 = Manifolds.retract(M, pS, XS, QRRetraction())
            @test isapprox(M, qS, q32)
            @test isapprox(M, pS, XS, SMatrix{3, 2}([0.0 -1.0; 1.0 0.0; 0.0 0.0]))
        end
        @testset "Stiefel(4,3) StaticArray inverse QR retraction cases" begin
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
        @testset "Allocation Promotion" begin
            @test Manifolds.allocation_promotion_function(Mc, get_vector, ()) === complex
        end
    end
    @testset "Quaternion Stiefel" begin
        M = Stiefel(3, 2, ℍ)
        @testset "Basics" begin
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 18
        end
    end

    @testset "Bug #871" begin
        M = Stiefel(3, 1)
        p = [-0.4760758523674722; 0.3378785657033835; -0.8119050792000313;;]
        Xs = get_vectors(M, p, DefaultOrthonormalBasis())
        for X in Xs
            @test is_vector(M, p, X; atol = 1.0e-12)
        end
    end
end
