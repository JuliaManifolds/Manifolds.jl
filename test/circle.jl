include("utils.jl")

@testset "Circle" begin
    M = Circle()
    @testset "Real Circle Basics" begin
        @test repr(M) == "Circle(ℝ)"
        @test representation_size(M) == ()
        @test manifold_dimension(M) == 1
        @test !is_manifold_point(M, 9.0)
        @test_throws DomainError is_manifold_point(M, 9.0, true)
        @test !is_tangent_vector(M, 9.0, 0.0)
        @test_throws DomainError is_tangent_vector(M, 9.0, 0.0, true)
        @test is_tangent_vector(M, 0.0, 0.0; check_base_point = false)
        @test get_coordinates(M, Ref(0.0), Ref(2.0), DefaultOrthonormalBasis())[] ≈ 2.0
        @test get_coordinates(
            M,
            Ref(0.0),
            Ref(2.0),
            DiagonalizingOrthonormalBasis(Ref(1.0)),
        )[] ≈ 2.0
        @test get_coordinates(
            M,
            Ref(0.0),
            Ref(-2.0),
            DiagonalizingOrthonormalBasis(Ref(1.0)),
        )[] ≈ -2.0
        @test get_coordinates(
            M,
            Ref(0.0),
            Ref(2.0),
            DiagonalizingOrthonormalBasis(Ref(-1.0)),
        )[] ≈ -2.0
        @test get_coordinates(
            M,
            Ref(0.0),
            Ref(-2.0),
            DiagonalizingOrthonormalBasis(Ref(-1.0)),
        )[] ≈ 2.0
        y = [0.0]
        get_coordinates!(M, y, Ref(0.0), Ref(2.0), DiagonalizingOrthonormalBasis(Ref(1.0)))
        @test y ≈ [2.0]
        @test get_vector(M, Ref(0.0), Ref(2.0), DefaultOrthonormalBasis())[] ≈ 2.0
        @test get_vector(M, Ref(0.0), Ref(2.0), DiagonalizingOrthonormalBasis(Ref(1.0)))[] ≈
              2.0
        @test get_vector(M, Ref(0.0), Ref(-2.0), DiagonalizingOrthonormalBasis(Ref(1.0)))[] ≈
              -2.0
        @test get_vector(M, Ref(0.0), Ref(2.0), DiagonalizingOrthonormalBasis(Ref(-1.0)))[] ≈
              -2.0
        @test get_vector(
            M,
            Ref(0.0),
            Ref(-2.0),
            DiagonalizingOrthonormalBasis(Ref(-1.0)),
        )[] ≈ 2.0
        @test flat(M, 0.0, FVector(TangentSpace, 1.0)) == FVector(CotangentSpace, 1.0)
        @test sharp(M, 0.0, FVector(CotangentSpace, 1.0)) == FVector(TangentSpace, 1.0)
        @test vector_transport_to(M, 0.0, 1.0, 1.0, ParallelTransport()) == 1.0
        @test retract(M, 0.0, 1.0) == exp(M, 0.0, 1.0)
        @test injectivity_radius(M) ≈ π
        @test injectivity_radius(M, Ref(-2.0)) ≈ π
        @test injectivity_radius(M, Ref(-2.0), ExponentialRetraction()) ≈ π
        @test injectivity_radius(M, ExponentialRetraction()) ≈ π
        @test mean(M, [-π / 2, 0.0, π]) ≈ -π / 2
        @test mean(M, [-π / 2, 0.0, π], [1.0, 1.0, 1.0]) == -π / 2
        z = project(M, 1.5 * π)
        z2 = [0.0]
        project!(M, z2, 1.5 * π)
        @test z2[1] == z
        @test project(M, z) == z
        @test project(M, 1.0, 2.0) == 2.0
    end
    TEST_STATIC_SIZED && @testset "Real Circle and static sized arrays" begin
        v = MVector(0.0)
        x = SVector(0.0)
        log!(M, v, x, SVector(π / 4))
        @test norm(M, x, v) ≈ π / 4
        @test is_tangent_vector(M, x, v)
        @test is_tangent_vector(M, [], v; check_base_point = false)
        @test project(M, 1.0) == 1.0
        x = MVector(0.0)
        project!(M, x, x)
        @test x == MVector(0.0)
        x .+= 2 * π
        project!(M, x, x)
        @test x == MVector(0.0)
        @test project(M, 0.0, 1.0) == 1.0
    end
    types = [Float64]
    TEST_FLOAT32 && push!(types, Float32)

    basis_types = (DefaultOrthonormalBasis(),)
    basis_types_real = (
        DefaultOrthonormalBasis(),
        DiagonalizingOrthonormalBasis(Ref(-1.0)),
        DiagonalizingOrthonormalBasis(Ref(1.0)),
    )
    for T in types
        @testset "Type $T" begin
            pts = convert.(Ref(T), [-π / 4, 0.0, π / 4])
            test_manifold(
                M,
                pts,
                test_forward_diff = false,
                test_reverse_diff = false,
                test_vector_spaces = false,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                test_vee_hat = false,
                is_mutating = false,
            )
            ptsS = SVector.(pts)
            test_manifold(
                M,
                ptsS,
                test_forward_diff = false,
                test_reverse_diff = false,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                test_vee_hat = true,
                basis_types_vecs = basis_types_real,
                basis_types_to_from = basis_types_real,
            )
        end
    end
    Mc = Circle(ℂ)
    @testset "Complex Circle Basics" begin
        @test repr(Mc) == "Circle(ℂ)"
        @test representation_size(Mc) == ()
        @test manifold_dimension(Mc) == 1
        @test is_tangent_vector(Mc, 1im, 0.0)
        @test is_manifold_point(Mc, 1im)
        @test !is_manifold_point(Mc, 1 + 1im)
        @test_throws DomainError is_manifold_point(Mc, 1 + 1im, true)
        @test !is_tangent_vector(Mc, 1 + 1im, 0.0)
        @test_throws DomainError is_tangent_vector(Mc, 1 + 1im, 0.0, true)
        @test !is_tangent_vector(Mc, 1im, 2im)
        @test_throws DomainError is_tangent_vector(Mc, 1im, 2im, true)
        @test flat(Mc, 0.0 + 0.0im, FVector(TangentSpace, 1.0im)) ==
              FVector(CotangentSpace, 1.0im)
        @test sharp(Mc, 0.0 + 0.0im, FVector(CotangentSpace, 1.0im)) ==
              FVector(TangentSpace, 1.0im)
        @test norm(Mc, 1.0, log(Mc, 1.0, -1.0)) ≈ π
        @test is_tangent_vector(Mc, 1.0, log(Mc, 1.0, -1.0))
        v = MVector(0.0 + 0.0im)
        x = SVector(1.0 + 0.0im)
        log!(Mc, v, x, SVector(-1.0 + 0.0im))
        @test norm(Mc, SVector(1.0), v) ≈ π
        @test is_tangent_vector(Mc, x, v)
        @test project(Mc, 1.0) == 1.0
        project(Mc, 1 / sqrt(2.0) + 1 / sqrt(2.0) * im) ==
        1 / sqrt(2.0) + 1 / sqrt(2.0) * im
        x = MVector(1.0 + 0.0im)
        project!(Mc, x, x)
        @test x == MVector(1.0 + 0.0im)
        x .*= 2
        project!(Mc, x, x)
        @test x == MVector(1.0 + 0.0im)

        angles = map(x -> exp(x * im), [-π / 2, 0.0, π])
        @test mean(Mc, angles) ≈ exp(-π * im / 2)
        @test mean(Mc, angles, [1.0, 1.0, 1.0]) ≈ exp(-π * im / 2)
        @test_throws ErrorException mean(Mc, [-1.0 + 0im, 1.0 + 0im])
        @test_throws ErrorException mean(Mc, [-1.0 + 0im, 1.0 + 0im], [1.0, 1.0])
    end
    types = [Complex{Float64}]
    TEST_FLOAT32 && push!(types, Complex{Float32})

    for T in types
        @testset "Type $T" begin
            a = 1 / sqrt(2.0)
            pts = convert.(Ref(T), [a - a * im, 1 + 0im, a + a * im])
            test_manifold(
                Mc,
                pts,
                test_forward_diff = false,
                test_reverse_diff = false,
                test_vector_spaces = false,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                is_mutating = false,
                test_vee_hat = false,
                exp_log_atol_multiplier = 2.0,
                is_tangent_atol_multiplier = 2.0,
            )
            ptsS = SVector.(pts)
            test_manifold(
                Mc,
                ptsS,
                test_forward_diff = false,
                test_reverse_diff = false,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                test_vee_hat = true,
                exp_log_atol_multiplier = 2.0,
                is_tangent_atol_multiplier = 2.0,
                basis_types_vecs = basis_types,
                basis_types_to_from = basis_types,
            )
        end
    end
end
