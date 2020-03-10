include("utils.jl")

@testset "Generalized Stiefel" begin
    @testset "Real" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedStiefel(3,2,B)
        @testset "Basics" begin
            @test repr(M) == "GeneralizedStiefel(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℝ)"
            x = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            @test representation_size(M) == (3,2)
            @test manifold_dimension(M) == 3
            @test_throws DomainError is_manifold_point(M, [1., 0., 0., 0.],true)
            @test_throws DomainError is_manifold_point(M, 1im*[1.0 0.0; 0.0 1.0; 0.0 0.0],true)
            @test !is_tangent_vector(M, x, [0., 0., 1., 0.])
            @test_throws DomainError is_tangent_vector(M, x, 1 * im * zero_tangent_vector(M,x), true)
        end
        @testset "Embedding and Projection" begin
            x = [1.0 0.0; 0.0 0.5; 0.0 0.0]
            y = similar(x)
            z = embed(M,x)
            @test z==x
            embed!(M,y,x)
            @test y==z
            a = [1.0 0.0; 0.0 2.0; 0.0 0.0]
            @test !is_manifold_point(M,a)
            b = similar(a)
            c = project_point(M,a)
            @test c==x
            project_point!(M,b,a)
            @test b==x
        end

        types = [
            Matrix{Float64},
            MMatrix{3, 2, Float64},
        ]
        @testset "Type $T" for T in types
            x = [1.0 0.0; 0.0 0.5; 0.0 0.0]
            y = retract(M, x, [0.0 0.0; 0.0 0.0; 1.0 1.0])
            z = retract(M, x, [ 0.0 0.0; 0.0 0.0; -1.0 1.0])
            @test is_manifold_point(M,y)
            @test is_manifold_point(M,z)
            pts = convert.(T, [x,y,z])
            @test !is_manifold_point(M,2*x)
            @test_throws DomainError !is_manifold_point(M,2*x,true)
            @test !is_tangent_vector(M,x,y)
            @test_throws DomainError is_tangent_vector(M,x,y,true)
            test_manifold(
                M,
                pts,
                test_exp_log = false,
                default_inverse_retraction_method = nothing,
                default_retraction_method = ProjectionRetraction(),
                test_injectivity_radius = false,
                test_is_tangent = true,
                test_project_tangent = true,
                test_vector_transport = false,
                test_forward_diff = false,
                test_reverse_diff = false,
                projection_atol_multiplier = 15.0,
                retraction_atol_multiplier = 10.0,
                is_tangent_atol_multiplier = 4*10.0^2,
                retraction_methods = [PolarRetraction(), ProjectionRetraction()],
            )
        end
    end

    @testset "Complex" begin
        M = Stiefel(3,2,ℂ)
        @testset "Basics" begin
            @test repr(M) == "Stiefel(3, 2, ℂ)"
            @test representation_size(M) == (3,2)
            @test manifold_dimension(M) == 8
            @test !is_manifold_point(M, [1., 0., 0., 0.])
            @test !is_tangent_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0., 0., 1., 0.])
            x = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            @test_throws DomainError is_manifold_point(M, [:a :b; :c :d; :e :f],true)
            @test_throws DomainError is_tangent_vector(M, x, [:a :b; :c :d; :e :f], true)
        end
    end

    @testset "Quaternion" begin
        M = Stiefel(3,2,ℍ)
        @testset "Basics" begin
            @test representation_size(M) == (3,2)
            @test manifold_dimension(M) == 18
        end
    end
end
