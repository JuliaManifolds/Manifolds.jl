include("utils.jl")

@testset "Stiefel" begin
    @testset "Real" begin
        M = Stiefel(3,2)
        @testset "Basics" begin
            @test representation_size(M) == (3,2)
            @test manifold_dimension(M) == 3
            @test !is_manifold_point(M, [1., 0., 0., 0.])
            @test !is_tangent_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0., 0., 1., 0.])
        end

        types = [
            Matrix{Float64},
            MMatrix{3, 2, Float64},
            Matrix{Float32},
        ]
        @testset "Type $T" for T in types
            x = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            y = exp(M,x, [0.0 0.0; 0.0 0.0; 1.0 1.0])
            z = exp(M,x,[ 0.0 0.0; 0.0 0.0; -1.0 1.0])
            pts = convert.(T, [x,y,z])
            v = inverse_retract(M,x,y,PolarInverseRetraction())
            @test !is_manifold_point(M,2*x)
            @test_throws DomainError !is_manifold_point(M,2*x,true)
            @test !is_tangent_vector(M,2*x,v)
            @test_throws DomainError !is_tangent_vector(M,2*x,v,true)
            @test !is_tangent_vector(M,x,y)
            @test_throws DomainError is_tangent_vector(M,x,y,true)
            test_manifold(
                M,
                pts,
                test_exp_log = false,
                default_inverse_retraction_method = PolarInverseRetraction(),
                test_injectivity_radius = false,
                test_is_tangent = true,
                test_project_tangent = true,
                test_vector_transport = false,
                test_forward_diff = false,
                test_reverse_diff = false,
                projection_atol_multiplier = 15.0,
                retraction_atol_multiplier = 10.0,
                is_tangent_atol_multiplier = 4*10.0^2,
                retraction_methods = [PolarRetraction(), QRRetraction()],
                inverse_retraction_methods = [PolarInverseRetraction(), QRInverseRetraction()]
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

    @testset "Complex" begin
        M = Stiefel(3,2,ℂ)
        @testset "Basics" begin
            @test representation_size(M) == (3,2)
            @test manifold_dimension(M) == 8
            @test !is_manifold_point(M, [1., 0., 0., 0.])
            @test !is_tangent_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0., 0., 1., 0.])
        end

        types = [
            Matrix{ComplexF64},
        ]
        @testset "Type $T" for T in types
            x = [0.5+0.5im 0.5+0.5im; 0.5+0.5im -0.5-0.5im; 0.0 0.0]
            y = exp(M, x, [0.0 0.0; 0.0 0.0; 1.0 1.0])
            z = exp(M, x, [0.0 0.0; 0.0 0.0; -1.0 1.0])
            pts = convert.(T, [x,y,z])
            v = inverse_retract(M,x,y,PolarInverseRetraction())
            @test !is_manifold_point(M,2*x)
            @test_throws DomainError !is_manifold_point(M,2*x,true)
            @test !is_tangent_vector(M,2*x,v)
            @test_throws DomainError !is_tangent_vector(M,2*x,v,true)
            @test !is_tangent_vector(M,x,y)
            @test_throws DomainError is_tangent_vector(M,x,y,true)
            test_manifold(
                M,
                pts,
                test_exp_log = false,
                default_inverse_retraction_method = PolarInverseRetraction(),
                test_injectivity_radius = false,
                test_is_tangent = true,
                test_project_tangent = true,
                test_vector_transport = false,
                test_forward_diff = false,
                test_reverse_diff = false,
                projection_atol_multiplier = 15.0,
                retraction_atol_multiplier = 10.0,
                is_tangent_atol_multiplier = 4*10.0^2,
                retraction_methods = [PolarRetraction(), QRRetraction()],
                inverse_retraction_methods = [PolarInverseRetraction(), QRInverseRetraction()]
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
        M = Stiefel(3,2,ℍ)
        @testset "Basics" begin
            @test representation_size(M) == (3,2)
            @test manifold_dimension(M) == 18
        end
    end
end
