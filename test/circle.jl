include("utils.jl")

@testset "Circle" begin
    M = Circle()
    @testset "Real Circle Basics" begin
        @test repr(M) == "Circle(ℝ)"
        @test representation_size(M) == ()
        @test manifold_dimension(M) == 1
        @test !is_manifold_point(M, 9.)
        @test_throws DomainError is_manifold_point(M, 9., true)
        @test !is_tangent_vector(M, 9., 0.)
        @test_throws DomainError is_tangent_vector(M, 9., 0., true)
        @test flat(M,0.0, FVector(TangentSpace,1.0)) == FVector(CotangentSpace,1.0)
        @test sharp(M,0.0, FVector(CotangentSpace,1.0)) == FVector(TangentSpace,1.0)
        @test vector_transport_to(M,0.0,1.0,1.0, ParallelTransport()) == 1.0
        @test retract(M,0.0,1.0) == exp(M,0.0,1.0)
        @test injectivity_radius(M) ≈ π
        @test mean(M, [-π/2,0.,π]) ≈ π/2
        @test mean(M, [-π/2,0.,π],[1., 1., 1.]) == π/2
        v = MVector(0.0)
        x = SVector(0.0)
        log!(M,v,x,SVector(π/4))
        @test norm(M,x,v) ≈ π/4
        @test is_tangent_vector(M,x,v)
        @test project_point(M,1.0) == 1.0
        x = MVector(0.0)
        project_point!(M,x)
        @test x == MVector(0.0)
        x .+= 2*π
        project_point!(M,x)
        @test x == MVector(0.0)
        @test project_tangent(M,0.0,1.) == 1.
    end
    types = [Float64, Float32]

    basis_types = (ArbitraryOrthonormalBasis(),)
    basis_types_real = (ArbitraryOrthonormalBasis(),
        DiagonalizingOrthonormalBasis([-1]),
        DiagonalizingOrthonormalBasis([1])
    )
    for T in types
        @testset "Type $T" begin
            pts = convert.(Ref(T), [-π/4,0.,π/4])
            test_manifold(
                M,
                pts,
                test_forward_diff = false,
                test_reverse_diff = false,
                test_vector_spaces = false,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                is_mutating = false
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
                basis_types_vecs = basis_types_real,
                basis_types_to_from = basis_types_real
            )
        end
    end
    Mc = Circle(ℂ)
    @testset "Complex Circle Basics" begin
        @test repr(Mc) == "Circle(ℂ)"
        @test representation_size(Mc) == ()
        @test manifold_dimension(Mc) == 1
        @test is_tangent_vector(Mc, 1im, 0.)
        @test is_manifold_point(Mc, 1im)
        @test !is_manifold_point(Mc, 1+1im)
        @test_throws DomainError is_manifold_point(Mc, 1+1im, true)
        @test !is_tangent_vector(Mc, 1+1im, 0.)
        @test_throws DomainError is_tangent_vector(Mc, 1+1im, 0., true)
        @test !is_tangent_vector(Mc, 1im, 2im)
        @test_throws DomainError is_tangent_vector(Mc, 1im, 2im, true)
        @test flat(Mc,0.0+0.0im, FVector(TangentSpace,1.0im)) == FVector(CotangentSpace,1.0im)
        @test sharp(Mc,0.0+0.0im, FVector(CotangentSpace,1.0im)) == FVector(TangentSpace,1.0im)
        @test norm(Mc,1.0,log(Mc, 1.0,-1.0)) ≈ π
        @test is_tangent_vector(Mc,1.0,log(Mc,1.0,-1.0))
        v = MVector(0.0+0.0im)
        x = SVector(1.0+0.0im)
        log!(Mc,v,x,SVector(-1.0+0.0im))
        @test norm(Mc,SVector(1.0),v) ≈ π
        @test is_tangent_vector(Mc,x,v)
        @test project_point(Mc,1.0) == 1.0
        project_point(Mc,1/sqrt(2.0) + 1/sqrt(2.0) * im) == 1/sqrt(2.0) + 1/sqrt(2.0) * im
        x = MVector(1.0+0.0im)
        project_point!(Mc,x)
        @test x == MVector(1.0+0.0im)
        x .*= 2
        project_point!(Mc,x)
        @test x == MVector(1.0+0.0im)

    end
    types = [Complex{Float64}, Complex{Float32}]
    for T in types
        @testset "Type $T" begin
            a = 1/sqrt(2.0)
            pts = convert.(Ref(T), [a-a*im, 1+0im, a+a*im])
            test_manifold(
                Mc,
                pts,
                test_forward_diff = false,
                test_reverse_diff = false,
                test_vector_spaces = false,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                is_mutating=false,
                exp_log_atol_multiplier = 2.0,
                is_tangent_atol_multiplier = 2.0
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
                exp_log_atol_multiplier = 2.0,
                is_tangent_atol_multiplier = 2.0,
                basis_types_vecs = basis_types,
                basis_types_to_from = basis_types
            )
        end
    end
end
