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
        @test isa(2*fv1, FVector)
        @test (2*fv1).type == TangentSpace

        PM = ProductManifold(Sphere(2), Euclidean(2))
        @test_throws ErrorException flat(PM,ProductRepr([0.0,],[0.0]),FVector(CotangentSpace, ProductRepr([0.0],[0.0])))
        @test_throws ErrorException sharp(PM,ProductRepr([0.0,],[0.0]),FVector(TangentSpace, ProductRepr([0.0],[0.0])))

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

    types = [ Vector{Float64}, ]
    TEST_FLOAT32 && push!(types, Vector{Float32})
    TEST_STATIC_SIZED && push!(types, MVector{3, Float64})

    for T in types
        x = convert(T, [1.0, 0.0, 0.0])
        TB = TangentBundle(M)
        @test sprint(show, TB) == "TangentBundle(Sphere(2; field = ℝ))"
        @test base_manifold(TB) == M
        @test manifold_dimension(TB) == 2*manifold_dimension(M)
        @test representation_size(TB) == (6,)
        CTB = CotangentBundle(M)
        @test sprint(show, CTB) == "CotangentBundle(Sphere(2; field = ℝ))"
        @test sprint(show, VectorBundle(TestVectorSpaceType(), M)) == "VectorBundle(TestVectorSpaceType(), Sphere(2; field = ℝ))"
        @testset "Type $T" begin
            pts_tb = [ProductRepr(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, -1.0, -1.0])),
                      ProductRepr(convert(T, [0.0, 1.0, 0.0]), convert(T, [2.0, 0.0, 1.0])),
                      ProductRepr(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, 2.0, -1.0]))]
            @inferred ProductRepr(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, -1.0, -1.0]))
            for pt ∈ pts_tb
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
                basis_types_vecs = basis_types,
                projection_atol_multiplier = 4
            )
        end
    end

    @test TangentBundle{Sphere{2}} == VectorBundle{Manifolds.TangentSpaceType, Sphere{2}}
    @test CotangentBundle{Sphere{2}} == VectorBundle{Manifolds.CotangentSpaceType, Sphere{2}}

    @test base_manifold(TangentBundle(M)) == M
    @testset "spaces at point" begin
        x = [1.0, 0.0, 0.0]
        t_x = TangentSpaceAtPoint(M, x)
        ct_x = CotangentSpaceAtPoint(M, x)
        @test sprint(show, "text/plain", t_x) == """
        VectorSpaceAtPoint{VectorBundleFibers{Manifolds.TangentSpaceType,GeneralizedSphere{Tuple{3},ℝ}},Array{Float64,1}}
        Fiber:
         VectorBundleFibers(TangentSpace, Sphere(2; field = ℝ))
        Base point:
         3-element Array{Float64,1}:
          1.0
          0.0
          0.0"""
        @test base_manifold(t_x) == M
        @test base_manifold(ct_x) == M
        @test t_x.fiber.manifold== M
        @test ct_x.fiber.manifold== M
        @test t_x.fiber.fiber == TangentSpace
        @test ct_x.fiber.fiber == CotangentSpace
        @test t_x.point == x
        @test ct_x.point == x
    end

    @testset "tensor product" begin
        TT = Manifolds.TensorProductType(TangentSpace, TangentSpace)
        @test sprint(show, TT) == "TensorProductType(TangentSpace, TangentSpace)"
        @test vector_space_dimension(VectorBundleFibers(TT, Sphere(2))) == 4
        @test vector_space_dimension(VectorBundleFibers(TT, Sphere(3))) == 9
        @test base_manifold(VectorBundleFibers(TT, Sphere(2))) == M
        @test sprint(show, VectorBundleFibers(TT, Sphere(2))) == "VectorBundleFibers(TensorProductType(TangentSpace, TangentSpace), Sphere(2; field = ℝ))"
    end

    @testset "Error messages" begin
        vbf = VectorBundleFibers(TestVectorSpaceType(), Euclidean(3))
        @test_throws ErrorException inner(vbf, [1, 2, 3], [1, 2, 3], [1, 2, 3])
        @test_throws ErrorException Manifolds.project_vector!(vbf, [1, 2, 3], [1, 2, 3], [1, 2, 3])
        @test_throws ErrorException zero_vector!(vbf, [1, 2, 3], [1, 2, 3])
        @test_throws ErrorException vector_space_dimension(vbf)
    end
end
