include("utils.jl")

@testset "fixed Rank" begin
    M = FixedRankMatrices(3,2,2)
    Mc = FixedRankMatrices(3,2,2,Complex)
    x = SVDMPoint([1.0 0.0; 0.0 1.0; 0.0 0.0])
    v = UMVTVector([0. 0.; 0. 0.; 1. 1.], [1. 0.; 0. 1.],zeros(2,2))
    @testset "Fixed Rank Matrices – Basics" begin
        @test representation_size(M) == (3,2)
        @test representation_size(Mc) == (3,2)
        @test manifold_dimension(M) == 6
        @test manifold_dimension(Mc) == 12
        @test !is_manifold_point(M, SVDMPoint([1. 0.; 0. 0.],2))
        @test_throws DomainError is_manifold_point(M, SVDMPoint([1. 0.; 0. 0.],2), true)

        @test !is_tangent_vector(M, SVDMPoint( [1. 0.; 0. 1.; 0. 0.] ), UMVTVector( zeros(2,1), zeros(1,2), zeros(2,2) )  )
        @test !is_tangent_vector(M, SVDMPoint([1. 0.; 0. 0.],2),v)
        @test_throws DomainError is_tangent_vector(M, SVDMPoint([1. 0.; 0. 0.],2), v, true)
        @test !is_tangent_vector(M, x, UMVTVector(x.U, v.M, x.Vt,2))
        @test_throws DomainError is_tangent_vector(M, x, UMVTVector(x.U, v.M, x.Vt,2), true)
        @test !is_tangent_vector(M, x, UMVTVector(v.U, v.M, x.Vt,2))
        @test_throws DomainError is_tangent_vector(M, x, UMVTVector(v.U, v.M, x.Vt,2), true)

        @test is_manifold_point(M,x)
        @test is_tangent_vector(M,x,v)
    end
    @testset "SVD MPoint Basics" begin
        s = svd(x.U*Diagonal(x.S)*x.Vt)
        x2 = SVDMPoint(s)
        x3 = SVDMPoint(s.U, s.S, s.Vt)
        @test x.S == x2.S
        @test x.U == x2.U
        @test x.Vt == x2.Vt
        @test x2.U == x3.U
        @test x2.S == x3.S
        @test x2.Vt == x3.Vt
        y = SVDMPoint([1. 0.; 0. 0.; 0. 0.], 1)
        s2 = svd([1. 0.; 0. 0.; 0. 0.])
        y2 = SVDMPoint(s2,1)
        y3 = SVDMPoint(s2.U, s2.S, s2.Vt,1)
        @test y.S == y2.S
        @test y.U == y2.U
        @test y.Vt == y2.Vt
        @test y2.U == y3.U
        @test y2.S == y3.S
        @test y2.Vt == y3.Vt
    end
    types = [ [ Matrix{Float64}, Vector{Float64}, Matrix{Float64} ],
            #   MMatrix{3, 2, Float64},
            #   SizedMatrix{3, 2, Float64},
            #   Matrix{Float32},
            #   MMatrix{3, 2, Float32},
            #   SizedMatrix{3, 2, Float64},
            ]
    for T in types
        @testset "Type $T" begin
            y = retract(M, x, v, PolarRetraction())
            z = SVDMPoint( [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2); 0. 0.] )
            pts = []
            for p in [x,y,z]
                push!(pts, SVDMPoint( convert.( T , [ p.U, p.S, p.Vt ] )... )   )
            end
            test_manifold(M,
                          pts,
                          test_exp_log = false,
                          default_inverse_retraction_method = nothing,
                          test_log_yields_tangent = false,
                          test_project_tangent = true,
                          test_vector_transport = false,
                          test_forward_diff = false,
                          test_reverse_diff = false,
                          projection_atol_multiplier = 15,
                          retraction_methods = [PolarRetraction()]
            )
        end
    end
end
