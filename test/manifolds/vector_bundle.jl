include("../utils.jl")

using RecursiveArrayTools

struct TestVectorSpaceType <: VectorSpaceType end

@testset "Tangent bundle" begin
    M = Sphere(2)
    m_prod_retr = Manifolds.VectorBundleProductRetraction()
    m_prod_invretr = Manifolds.VectorBundleInverseProductRetraction()
    m_sasaki = SasakiRetraction(5)

    @testset "Nice access to vector bundle components" begin
        TB = TangentBundle(M)
        @testset "ProductRepr" begin
            p = ProductRepr([1.0, 0.0, 0.0], [0.0, 2.0, 4.0])
            @test p[TB, :point] === p.parts[1]
            @test p[TB, :vector] === p.parts[2]
            p[TB, :vector] = [0.0, 3.0, 1.0]
            @test p.parts[2] == [0.0, 3.0, 1.0]
            p[TB, :point] = [0.0, 1.0, 0.0]
            @test p.parts[1] == [0.0, 1.0, 0.0]
            @test_throws DomainError p[TB, :error]
            @test_throws DomainError p[TB, :error] = [1, 2, 3]
        end
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

    types = [Vector{Float64}]
    TEST_FLOAT32 && push!(types, Vector{Float32})
    TEST_STATIC_SIZED && push!(types, MVector{3,Float64})

    for T in types, prepr in [ProductRepr, ArrayPartition]
        p = convert(T, [1.0, 0.0, 0.0])
        TB = TangentBundle(M)
        @test injectivity_radius(TB) == 0
        TpM = TangentSpaceAtPoint(M, p)
        @test sprint(show, TB) == "TangentBundle(Sphere(2, ℝ))"
        @test base_manifold(TB) == M
        @test manifold_dimension(TB) == 2 * manifold_dimension(M)
        @test !is_flat(TB)
        @test is_flat(TpM)
        @test representation_size(TB) === nothing
        @test default_inverse_retraction_method(TB) === m_prod_invretr
        @test default_retraction_method(TB) == m_prod_retr
        @test default_vector_transport_method(TB) isa
              Manifolds.VectorBundleProductVectorTransport
        CTB = CotangentBundle(M)
        @test sprint(show, CTB) == "CotangentBundle(Sphere(2, ℝ))"
        @test sprint(show, VectorBundle(TestVectorSpaceType(), M)) ==
              "VectorBundle(TestVectorSpaceType(), Sphere(2, ℝ))"
        @testset "Type $T" begin
            pts_tb = [
                prepr(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, -1.0, -1.0])),
                prepr(convert(T, [0.0, 1.0, 0.0]), convert(T, [2.0, 0.0, 1.0])),
                prepr(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, 2.0, -1.0])),
            ]
            @inferred prepr(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, -1.0, -1.0]))
            if prepr === ProductRepr
                for pt in pts_tb
                    @test bundle_projection(TB, pt) ≈ pt.parts[1]
                end
            else
                for pt in pts_tb
                    @test bundle_projection(TB, pt) ≈ pt.x[1]
                end
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
                test_tangent_vector_broadcasting=false,
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
                Manifolds.VectorBundleProductVectorTransport(),
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

    @test base_manifold(TangentBundle(M)) == M
    @testset "spaces at point" begin
        p = [1.0, 0.0, 0.0]
        t_p = TangentSpaceAtPoint(M, p)
        t_p2 = TangentSpace(M, p)
        @test t_p == t_p2
        ct_p = CotangentSpaceAtPoint(M, p)
        t_ps = sprint(show, "text/plain", t_p)
        sp = sprint(show, "text/plain", p)
        sp = replace(sp, '\n' => "\n ")
        t_ps_test = "Tangent space to the manifold $(M) at point:\n $(sp)"
        @test t_ps == t_ps_test
        @test base_manifold(t_p) == M
        @test base_manifold(ct_p) == M
        @test t_p.fiber.manifold == M
        @test ct_p.fiber.manifold == M
        @test t_p.fiber.fiber == TangentSpace
        @test ct_p.fiber.fiber == CotangentSpace
        @test t_p.point == p
        @test ct_p.point == p
        @test injectivity_radius(t_p) == Inf
        @test representation_size(t_p) == representation_size(M)
        X = [0.0, 0.0, 1.0]
        @test embed(t_p, X) == X
        @test embed(t_p, X, X) == X
        # generic vector space at
        fiber = VectorBundleFibers(TestVectorSpaceType(), M)
        X_p = VectorSpaceAtPoint(fiber, p)
        X_ps = sprint(show, "text/plain", X_p)
        fiber_s = sprint(show, "text/plain", fiber)
        X_ps_test = "$(typeof(X_p))\nFiber:\n $(fiber_s)\nBase point:\n $(sp)"
        @test X_ps == X_ps_test
        @test_throws ErrorException project(fiber, p, X)
        @test_throws ErrorException norm(fiber, p, X)
        @test_throws ErrorException distance(fiber, p, X, X)
    end

    @testset "tensor product" begin
        TT = Manifolds.TensorProductType(TangentSpace, TangentSpace)
        @test sprint(show, TT) == "TensorProductType(TangentSpace, TangentSpace)"
        @test vector_space_dimension(VectorBundleFibers(TT, Sphere(2))) == 4
        @test vector_space_dimension(VectorBundleFibers(TT, Sphere(3))) == 9
        @test base_manifold(VectorBundleFibers(TT, Sphere(2))) == M
        @test sprint(show, VectorBundleFibers(TT, Sphere(2))) ==
              "VectorBundleFibers(TensorProductType(TangentSpace, TangentSpace), Sphere(2, ℝ))"
    end

    @testset "Error messages" begin
        vbf = VectorBundleFibers(TestVectorSpaceType(), Euclidean(3))
        @test_throws ErrorException inner(vbf, [1, 2, 3], [1, 2, 3], [1, 2, 3])
        @test_throws ErrorException Manifolds.project!(vbf, [1, 2, 3], [1, 2, 3], [1, 2, 3])
        @test_throws ErrorException zero_vector!(vbf, [1, 2, 3], [1, 2, 3])
        @test_throws MethodError vector_space_dimension(vbf)
    end

    @testset "product retraction and inverse retraction on tangent bundle for power and product manifolds" begin
        M = PowerManifold(Circle(ℝ), 2)
        N = TangentBundle(M)
        p1 = ProductRepr([0.0, 0.0], [0.0, 0.0])
        p2 = ProductRepr([-1.047, -1.047], [0.0, 0.0])
        X1 = inverse_retract(N, p1, p2, m_prod_invretr)
        @test isapprox(N, p2, retract(N, p1, X1, m_prod_retr))
        @test is_vector(N, p2, vector_transport_to(N, p1, X1, p2))

        M2 = ProductManifold(Circle(ℝ), Euclidean(2))
        N2 = TangentBundle(M2)
        p1_2 = ProductRepr(ProductRepr([0.0], [0.0, 0.0]), ProductRepr([0.0], [0.0, 0.0]))
        p2_2 = ProductRepr(
            ProductRepr([-1.047], [1.0, 0.0]),
            ProductRepr([-1.047], [0.0, 1.0]),
        )
        @test isapprox(
            N2,
            p2_2,
            retract(N2, p1_2, inverse_retract(N2, p1_2, p2_2, m_prod_invretr), m_prod_retr),
        )

        ppt = ParallelTransport()
        tbvt = Manifolds.VectorBundleProductVectorTransport(ppt, ppt)
        @test TangentBundle(M, tbvt).vector_transport === tbvt
        @test CotangentBundle(M, tbvt).vector_transport === tbvt
        @test VectorBundle(TangentSpace, M, tbvt).vector_transport === tbvt
    end

    @testset "Extended flatness tests" begin
        M = TangentBundle(Euclidean(3))
        @test is_flat(M)
        @test injectivity_radius(M) == Inf
    end

    @testset "Weingarten Map" begin
        p0 = [1.0, 0.0, 0.0]
        M = TangentSpaceAtPoint(Sphere(2), p0)
        p = [0.0, 1.0, 1.0]
        X = [0.0, 1.0, 0.0]
        V = [1.0, 0.0, 0.0]
        @test Weingarten(M, p, X, V) == zero_vector(M, p)
    end
end
