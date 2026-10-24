using Manifolds
using Random: MersenneTwister, rand!
using Test: Test
using Manifolds: RieszRepresenterCotangentVector, get_chart_index

function Manifolds.test_atlases(M::AbstractManifold, pts, test_atlases; is_mutating::Bool = true)
    return Test.@testset "atlases" begin # COV_EXCL_LINE
        if !isempty(test_atlases)
            Test.@test get_default_atlas(M) isa AbstractAtlas{ℝ}
        end
        for A in test_atlases
            i = get_chart_index(M, A, pts[1])
            a = get_parameters(M, A, i, pts[1])
            Test.@test isa(a, AbstractVector)
            Test.@test length(a) == manifold_dimension(M)
            Test.@test isapprox(M, pts[1], get_point(M, A, i, a))
            if is_mutating
                get_parameters!(M, a, A, i, pts[2])
                Test.@test a ≈ get_parameters(M, A, i, pts[2])

                q = allocate(pts[1])
                get_point!(M, q, A, i, a)
                Test.@test isapprox(M, pts[2], q)
            end
        end
    end
end
