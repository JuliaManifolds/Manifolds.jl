include("utils.jl")

struct TestVectorSpaceType <: VectorSpaceType end


@testset "Tangent bundle" begin
    M = Sphere(2)

    @testset "FVector" begin
        @test sprint(show, TangentSpace) == "TangentSpace"
        @test sprint(show, CotangentSpace) == "CotangentSpace"
        tvs = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        fv_tvs = map(v -> FVector(TangentSpace, v), tvs)
        fv1 = fv_tvs[1]
        tv1s = allocate(fv_tvs[1])
        @test isa(tv1s, FVector)
        @test size(tv1s) == (3,)
        @test tv1s.type == TangentSpace
        @test size(tv1s.data) == size(tvs[1])
        @test number_eltype(tv1s) == number_eltype(tvs[1])
        @test number_eltype(tv1s) == number_eltype(typeof(tv1s))
        @test isa(fv1 + fv1, FVector)
        @test (fv1 + fv1).type == TangentSpace
        @test isa(fv1 - fv1, FVector)
        @test (fv1 - fv1).type == TangentSpace
        @test isa(-fv1, FVector)
        @test (-fv1).type == TangentSpace
        @test isa(2 * fv1, FVector)
        @test (2 * fv1).type == TangentSpace

        PM = ProductManifold(Sphere(2), Euclidean(2))
        @test_throws ErrorException flat(
            PM,
            ProductRepr([0.0], [0.0]),
            FVector(CotangentSpace, ProductRepr([0.0], [0.0])),
        )
        @test_throws ErrorException sharp(
            PM,
            ProductRepr([0.0], [0.0]),
            FVector(TangentSpace, ProductRepr([0.0], [0.0])),
        )

        fv2 = FVector(TangentSpace, ProductRepr([1, 0, 0], [1 2]))
        @test submanifold_component(fv2, 1) == [1, 0, 0]
        @test submanifold_component(fv2, 2) == [1 2]
        @test (@inferred submanifold_component(fv2, Val(1))) == [1, 0, 0]
        @test (@inferred submanifold_component(fv2, Val(2))) == [1 2]
        @test submanifold_component(PM, fv2, 1) == [1, 0, 0]
        @test submanifold_component(PM, fv2, 2) == [1 2]
        @test (@inferred submanifold_component(PM, fv2, Val(1))) == [1, 0, 0]
        @test (@inferred submanifold_component(PM, fv2, Val(2))) == [1 2]
        @test submanifold_components(PM, fv2) == ([1.0, 0.0, 0.0], [1.0 2.0])
    end

    types = [Vector{Float64}]
    TEST_FLOAT32 && push!(types, Vector{Float32})
    TEST_STATIC_SIZED && push!(types, MVector{3,Float64})

    for T in types
        p = convert(T, [1.0, 0.0, 0.0])
        TB = TangentBundle(M)
        TpM = TangentSpaceAtPoint(M, p)
        @test sprint(show, TB) == "TangentBundle(Sphere(2, ℝ))"
        @test base_manifold(TB) == M
        @test manifold_dimension(TB) == 2 * manifold_dimension(M)
        @test representation_size(TB) == (6,)
        CTB = CotangentBundle(M)
        @test sprint(show, CTB) == "CotangentBundle(Sphere(2, ℝ))"
        @test sprint(show, VectorBundle(TestVectorSpaceType(), M)) ==
              "VectorBundle(TestVectorSpaceType(), Sphere(2, ℝ))"
        @testset "Type $T" begin
            pts_tb = [
                ProductRepr(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, -1.0, -1.0])),
                ProductRepr(convert(T, [0.0, 1.0, 0.0]), convert(T, [2.0, 0.0, 1.0])),
                ProductRepr(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, 2.0, -1.0])),
            ]
            @inferred ProductRepr(
                convert(T, [1.0, 0.0, 0.0]),
                convert(T, [0.0, -1.0, -1.0]),
            )
            for pt in pts_tb
                @test bundle_projection(TB, pt) ≈ pt.parts[1]
            end
            diag_basis = DiagonalizingOrthonormalBasis(log(TB, pts_tb[1], pts_tb[2]))
            basis_types = (
                DefaultOrthonormalBasis(),
                get_basis(TB, pts_tb[1], DefaultOrthonormalBasis()),
                diag_basis,
                get_basis(TB, pts_tb[1], diag_basis),
            )
            test_manifold(
                TB,
                pts_tb,
                test_injectivity_radius = false,
                test_reverse_diff = isa(T, Vector),
                test_forward_diff = isa(T, Vector),
                test_tangent_vector_broadcasting = false,
                test_vee_hat = true,
                test_project_tangent = true,
                test_project_point = true,
                test_default_vector_transport = true,
                basis_types_vecs = basis_types,
                projection_atol_multiplier = 4,
            )

            # tangent space at point
            pts_TpM = map(
                p -> convert(T, p),
                [[0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, -1.0, 1.0]],
            )
            test_manifold(
                TpM,
                pts_TpM,
                test_injectivity_radius = true,
                test_reverse_diff = isa(T, Vector),
                test_forward_diff = isa(T, Vector),
                test_tangent_vector_broadcasting = true,
                test_vee_hat = false,
                test_project_tangent = true,
                test_project_point = true,
                test_default_vector_transport = true,
                basis_types_to_from = (DefaultOrthonormalBasis(),),
                projection_atol_multiplier = 4,
            )
        end
    end

    @test TangentBundle{ℝ,Sphere{2,ℝ}} ==
          VectorBundle{ℝ,Manifolds.TangentSpaceType,Sphere{2,ℝ}}
    @test CotangentBundle{ℝ,Sphere{2,ℝ}} ==
          VectorBundle{ℝ,Manifolds.CotangentSpaceType,Sphere{2,ℝ}}

    @test base_manifold(TangentBundle(M)) == M
    @testset "spaces at point" begin
        x = [1.0, 0.0, 0.0]
        t_x = TangentSpaceAtPoint(M, x)
        t_x2 = TangentSpace(M, x)
        @test t_x == t_x2
        ct_x = CotangentSpaceAtPoint(M, x)
        t_xs = sprint(show, "text/plain", t_x)
        sp = sprint(show, "text/plain", x)
        sp = replace(sp, '\n' => "\n ")
        t_xs_test = "Tangent space to the manifold $(M) at point:\n $(sp)"
        @test t_xs == t_xs_test
        @test base_manifold(t_x) == M
        @test base_manifold(ct_x) == M
        @test t_x.fiber.manifold == M
        @test ct_x.fiber.manifold == M
        @test t_x.fiber.fiber == TangentSpace
        @test ct_x.fiber.fiber == CotangentSpace
        @test t_x.point == x
        @test ct_x.point == x
        # generic vector space at
        fiber = VectorBundleFibers(TestVectorSpaceType(), M)
        v_x = VectorSpaceAtPoint(fiber, x)
        v_xs = sprint(show, "text/plain", v_x)
        fiber_s = sprint(show, "text/plain", fiber)
        v_xs_test = "$(typeof(v_x))\nFiber:\n $(fiber_s)\nBase point:\n $(sp)"
        @test v_xs == v_xs_test
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
        @test_throws ErrorException vector_space_dimension(vbf)
        a = fill(0.0, 6)
        @test_throws ErrorException get_coordinates!(
            TangentBundle(M),
            a,
            ProductRepr([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ProductRepr([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            CachedBasis(DefaultOrthonormalBasis(), []),
        )
    end

    @testset "log and exp on tangent bundle for power and product manifolds" begin
        M = PowerManifold(Circle(ℝ), 2)
        N = TangentBundle(M)
        p1 = ProductRepr([0.0, 0.0], [0.0, 0.0])
        p2 = ProductRepr([-1.047, -1.047], [0.0, 0.0])
        X1 = log(N, p1, p2)
        @test isapprox(N, p2, exp(N, p1, X1))
        @test is_tangent_vector(N, p2, vector_transport_to(N, p1, X1, p2))

        M2 = ProductManifold(Circle(ℝ), Euclidean(2))
        N2 = TangentBundle(M2)
        p1_2 = ProductRepr(ProductRepr([0.0], [0.0, 0.0]), ProductRepr([0.0], [0.0, 0.0]))
        p2_2 = ProductRepr(
            ProductRepr([-1.047], [1.0, 0.0]),
            ProductRepr([-1.047], [0.0, 1.0]),
        )
        @test isapprox(N2, p2_2, exp(N2, p1_2, log(N2, p1_2, p2_2)))

        ppt = PowerVectorTransport(ParallelTransport())
        tbvt = Manifolds.VectorBundleVectorTransport(ppt, ppt)
        @test TangentBundle(M, tbvt).vector_transport === tbvt
        @test CotangentBundle(M, tbvt).vector_transport === tbvt
        @test VectorBundle(TangentSpace, M, tbvt).vector_transport === tbvt
    end
end
