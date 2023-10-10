include("../utils.jl")

using RecursiveArrayTools

struct TestVectorSpaceType <: VectorSpaceType end

@testset "Vector bundle" begin
    M = Sphere(2)
    TB = TangentBundle(M)
    m_prod_retr = Manifolds.FiberBundleProductRetraction()
    m_prod_invretr = Manifolds.FiberBundleInverseProductRetraction()
    m_sasaki = SasakiRetraction(5)

    @testset "Nice access to vector bundle components" begin
        @testset "ArrayPartition" begin
            p = ArrayPartition([1.0, 0.0, 0.0], [0.0, 2.0, 4.0])
            @test p[TB, :point] === p.x[1]
            @test p[TB, :vector] === p.x[2]
            p[TB, :vector] = [0.0, 3.0, 1.0]
            @test p.x[2] == [0.0, 3.0, 1.0]
            p[TB, :point] = [0.0, 1.0, 0.0]
            @test p.x[1] == [0.0, 1.0, 0.0]
            @test_throws DomainError p[TB, :error]
            @test_throws DomainError p[TB, :error] = [1, 2, 3]

            @test view(p, TB, :point) === p.x[1]
            @test view(p, TB, :vector) === p.x[2]
            view(p, TB, :point) .= [2.0, 3.0, 5.0]
            @test p.x[1] == [2.0, 3.0, 5.0]
            view(p, TB, :vector) .= [-2.0, -3.0, -5.0]
            @test p.x[2] == [-2.0, -3.0, -5.0]
            @test_throws DomainError view(p, TB, :error)
        end
    end

    @testset "basic tests" begin
        @test injectivity_radius(TB) == 0
        @test sprint(show, TB) == "TangentBundle(Sphere(2, ℝ))"
        @test base_manifold(TB) == M
        @test manifold_dimension(TB) == 2 * manifold_dimension(M)
        @test !is_flat(TB)
        @test representation_size(TB) === nothing
        @test default_inverse_retraction_method(TB) === m_prod_invretr
        @test default_retraction_method(TB) == m_prod_retr
        @test default_vector_transport_method(TB) isa
              Manifolds.FiberBundleProductVectorTransport
        CTB = CotangentBundle(M)
        @test sprint(show, CTB) == "CotangentBundle(Sphere(2, ℝ))"
        @test sprint(show, VectorBundle(TestVectorSpaceType(), M)) ==
              "VectorBundle(TestVectorSpaceType(), Sphere(2, ℝ))"

        @test Manifolds.fiber_dimension(M, ManifoldsBase.CotangentSpaceType()) == 2
        @test base_manifold(TangentBundle(M)) == M
    end

    types = [Vector{Float64}]
    TEST_FLOAT32 && push!(types, Vector{Float32})
    TEST_STATIC_SIZED && push!(types, MVector{3,Float64})

    for T in types
        p = convert(T, [1.0, 0.0, 0.0])
        TpM = TangentSpace(M, p)
        @test is_flat(TpM)

        @testset "Type $T" begin
            pts_tb = [
                ArrayPartition(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, -1.0, -1.0])),
                ArrayPartition(convert(T, [0.0, 1.0, 0.0]), convert(T, [2.0, 0.0, 1.0])),
                ArrayPartition(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, 2.0, -1.0])),
            ]
            for pt in pts_tb
                @test bundle_projection(TB, pt) ≈ pt.x[1]
            end
            X12_prod = inverse_retract(TB, pts_tb[1], pts_tb[2], m_prod_invretr)
            X13_prod = inverse_retract(TB, pts_tb[1], pts_tb[3], m_prod_invretr)
            diag_basis = DiagonalizingOrthonormalBasis(X12_prod)
            basis_types = (
                DefaultOrthonormalBasis(),
                get_basis(TB, pts_tb[1], DefaultOrthonormalBasis()),
                diag_basis,
                get_basis(TB, pts_tb[1], diag_basis),
            )
            test_manifold(
                TB,
                pts_tb,
                default_inverse_retraction_method=m_prod_invretr,
                default_retraction_method=m_prod_retr,
                inverse_retraction_methods=[m_prod_invretr],
                retraction_methods=[m_prod_retr, m_sasaki],
                test_exp_log=false,
                test_injectivity_radius=false,
                test_tangent_vector_broadcasting=true,
                test_vee_hat=true,
                test_project_tangent=true,
                test_project_point=true,
                test_default_vector_transport=true,
                vector_transport_methods=[],
                basis_types_vecs=basis_types,
                projection_atol_multiplier=4,
                test_inplace=true,
                test_representation_size=false,
                test_rand_point=true,
                test_rand_tvector=true,
            )

            Xir = allocate(pts_tb[1])
            inverse_retract!(TB, Xir, pts_tb[1], pts_tb[2], m_prod_invretr)
            @test isapprox(TB, pts_tb[1], Xir, X12_prod)
            @test isapprox(
                norm(TB.fiber, pts_tb[1][TB, :point], pts_tb[1][TB, :vector]),
                sqrt(
                    inner(
                        TB.fiber,
                        pts_tb[1][TB, :point],
                        pts_tb[1][TB, :vector],
                        pts_tb[1][TB, :vector],
                    ),
                ),
            )
            @test isapprox(
                distance(
                    TB.fiber,
                    pts_tb[1][TB, :point],
                    pts_tb[1][TB, :vector],
                    [0.0, 2.0, 3.0],
                ),
                5.0,
            )
            Xir2 = allocate(pts_tb[1])
            vector_transport_to!(
                TB,
                Xir2,
                pts_tb[1],
                Xir,
                pts_tb[2],
                Manifolds.FiberBundleProductVectorTransport(),
            )
            @test is_vector(TB, pts_tb[2], Xir2)

            # tangent space at point
            pts_TpM = map(
                p -> convert(T, p),
                [[0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, -1.0, 1.0]],
            )
            basis_types = (
                DefaultOrthonormalBasis(),
                get_basis(TpM, pts_TpM[1], DefaultOrthonormalBasis()),
            )
            @test get_basis(TpM, pts_TpM[1], basis_types[2]) === basis_types[2]
            test_manifold(
                TpM,
                pts_TpM,
                test_injectivity_radius=true,
                test_tangent_vector_broadcasting=true,
                test_vee_hat=false,
                test_project_tangent=true,
                test_project_point=true,
                test_default_vector_transport=true,
                basis_types_vecs=basis_types,
                projection_atol_multiplier=4,
                test_inplace=true,
                test_rand_point=true,
                test_rand_tvector=true,
            )
        end
    end

    @test TangentBundle{ℝ,Sphere{2,ℝ}} ==
          VectorBundle{ℝ,Manifolds.TangentSpaceType,Sphere{2,ℝ}}
    @test CotangentBundle{ℝ,Sphere{2,ℝ}} ==
          VectorBundle{ℝ,Manifolds.CotangentSpaceType,Sphere{2,ℝ}}

    @testset "tensor product" begin
        TT = Manifolds.TensorProductType(TangentSpace, TangentSpace)
        @test sprint(show, TT) == "TensorProductType(TangentSpace, TangentSpace)"
        @test vector_space_dimension(VectorSpaceFiber(TT, Sphere(2), [1.0, 0.0, 0.0])) == 4
        @test vector_space_dimension(
            VectorSpaceFiber(TT, Sphere(3), [1.0, 0.0, 0.0, 0.0]),
        ) == 9
        @test base_manifold(VectorSpaceFiber(TT, Sphere(2), [1.0, 0.0, 0.0])) == M
        @test sprint(show, VectorSpaceFiber(TT, Sphere(2), [1.0, 0.0, 0.0])) ==
              "VectorSpaceFiber(TensorProductType(TangentSpace, TangentSpace), Sphere(2, ℝ))"
    end

    @testset "Error messages" begin
        vbf = VectorSpaceFiber(TestVectorSpaceType(), Euclidean(3), [1.0, 0.0, 0.0])
        @test_throws MethodError inner(vbf, [1, 2, 3], [1, 2, 3], [1, 2, 3])
        @test_throws MethodError project!(vbf, [1, 2, 3], [1, 2, 3], [1, 2, 3])
        @test_throws MethodError zero_vector!(vbf, [1, 2, 3], [1, 2, 3])
        @test_throws MethodError vector_space_dimension(vbf)
    end

    @testset "product retraction and inverse retraction on tangent bundle for power and product manifolds" begin
        M = PowerManifold(Circle(ℝ), 2)
        N = TangentBundle(M)
        p1 = ArrayPartition([0.0, 0.0], [0.0, 0.0])
        p2 = ArrayPartition([-1.047, -1.047], [0.0, 0.0])
        X1 = inverse_retract(N, p1, p2, m_prod_invretr)
        @test isapprox(N, p2, retract(N, p1, X1, m_prod_retr))
        @test is_vector(N, p2, vector_transport_to(N, p1, X1, p2))

        M2 = ProductManifold(Circle(ℝ), Euclidean(2))
        N2 = TangentBundle(M2)
        p1_2 = ArrayPartition(
            ArrayPartition([0.0], [0.0, 0.0]),
            ArrayPartition([0.0], [0.0, 0.0]),
        )
        p2_2 = ArrayPartition(
            ArrayPartition([-1.047], [1.0, 0.0]),
            ArrayPartition([-1.047], [0.0, 1.0]),
        )
        @test isapprox(
            N2,
            p2_2,
            retract(N2, p1_2, inverse_retract(N2, p1_2, p2_2, m_prod_invretr), m_prod_retr),
        )

        ppt = ParallelTransport()
        tbvt = Manifolds.FiberBundleProductVectorTransport(ppt, ppt)
        @test TangentBundle(M, tbvt).vector_transport === tbvt
        @test CotangentBundle(M, tbvt).vector_transport === tbvt
        @test VectorBundle(TangentSpace, M, tbvt).vector_transport === tbvt
    end

    @testset "Extended flatness tests" begin
        M = TangentBundle(Euclidean(3))
        @test is_flat(M)
        @test injectivity_radius(M) == Inf
    end
end
