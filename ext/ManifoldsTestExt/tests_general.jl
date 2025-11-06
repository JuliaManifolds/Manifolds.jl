using Manifolds
using Random: MersenneTwister, rand!
using Test: Test

"""
    Manifolds.find_eps(x...)

Find an appropriate tolerance for given points or tangent vectors, or their types.

!!! note
    This function is soon deprecated and will be replaced by a more systematic way in `Manifolds.Test`
"""
Manifolds.find_eps(x...) = Manifolds.find_eps(Base.promote_type(map(number_eltype, x)...))
Manifolds.find_eps(x::Type{TN}) where {TN <: Number} = eps(real(TN))
Manifolds.find_eps(x) = Manifolds.find_eps(number_eltype(x))

"""
    Manifolds.test_manifold(M::AbstractManifold, pts::AbstractVector; kwargs...)

Test general properties of manifold `M`, given at least three different points
that lie on it (contained in `pts`).

!!! note

    This is an old an soon deprecated interface, the new one is `Manifolds.Test.test_manifold`.

# Arguments
- `basis_has_specialized_diagonalizing_get = false`: if true, assumes that
    `DiagonalizingOrthonormalBasis` given in `basis_types` has
    [`get_coordinates`](@ref) and [`get_vector`](@ref) that work without caching.
- `basis_types_to_from = ()`: basis types that will be tested based on
    [`get_coordinates`](@ref) and [`get_vector`](@ref).
- `basis_types_vecs = ()` : basis types that will be tested based on `get_vectors`
- `default_inverse_retraction_method = ManifoldsBase.LogarithmicInverseRetraction()`:
    default method for inverse retractions (`log`.
- `default_retraction_method = ManifoldsBase.ExponentialRetraction()`: default method for
    retractions (`exp`).
- `exp_log_atol_multiplier = 0`: change absolute tolerance of exp/log tests
    (0 use default, i.e. deactivate atol and use rtol).
- `exp_log_rtol_multiplier = 1`: change the relative tolerance of exp/log tests
    (1 use default). This is deactivated if the `exp_log_atol_multiplier` is nonzero.
- `expected_dimension_type = Integer`: expected type of value returned by
    `manifold_dimension`.
- `inverse_retraction_methods = []`: inverse retraction methods that will be tested.
- `is_mutating = true`: whether mutating variants of functions should be tested.
- `is_point_atol_multiplier = 0`: determines atol of `is_point` checks.
- `is_tangent_atol_multiplier = 0`: determines atol of `is_vector` checks.
- `has_get_embedding = false`: whether the manifold has a specialized
    [`get_embedding`](@ref)`(M, p)` method (to test mutating `embed!`). This is experimental.
- `mid_point12 = test_exp_log ? shortest_geodesic(M, pts[1], pts[2], 0.5) : nothing`: if not `nothing`, then check
    that `mid_point(M, pts[1], pts[2])` is approximately equal to `mid_point12`. This is
    by default set to `nothing` if `text_exp_log` is set to false.
- `point_distributions = []` : point distributions to test.
- `rand_tvector_atol_multiplier = 0` : chage absolute tolerance in testing random vectors
    (0 use default, i.e. deactivate atol and use rtol) random tangent vectors are tangent
    vectors.
- `retraction_atol_multiplier = 0`: change absolute tolerance of (inverse) retraction tests
    (0 use default, i.e. deactivate atol and use rtol).
- `retraction_rtol_multiplier = 1`: change the relative tolerance of (inverse) retraction
    tests (1 use default). This is deactivated if the `exp_log_atol_multiplier` is nonzero.
- `retraction_methods = []`: retraction methods that will be tested.
- `test_atlases = []`: Vector or tuple of atlases that should be tested.
- `test_exp_log = true`: if true, check that [`exp`](@ref) is the inverse of [`log`](@extref `Base.log-Tuple{AbstractManifold, Any, Any}`).
- `test_injectivity_radius = true`: whether implementation of [`injectivity_radius`](@ref)
    should be tested.
- `test_inplace = false` : if true check if inplace variants work if they are activated,
   e.g. check that `exp!(M, p, p, X)` work if `test_exp_log = true`.
   This in general requires `is_mutating` to be true.
- `test_is_tangent`: if true check that the `default_inverse_retraction_method`
    actually returns valid tangent vectors.
- `test_musical_isomorphisms = false` : test musical isomorphisms.
- `test_mutating_rand = false` : test the mutating random function for points on manifolds.
- `test_project_point = false`: test projections onto the manifold.
- `test_project_tangent = false` : test projections on tangent spaces.
- `test_representation_size = true` : test representation size of points/tvectprs.
- `test_tangent_vector_broadcasting = true` : test boradcasting operators on TangentSpace.
- `test_vector_spaces = true` : test Vector bundle of this manifold.
- `test_default_vector_transport = false` : test the default vector transport (usually
   parallel transport).
- `tvector_distributions = []` : tangent vector distributions to test.
- `vector_transport_methods = []`: vector transport methods that should be tested.
- `vector_transport_inverse_retractions = [default_inverse_retraction_method for _ in 1:length(vector_transport_methods)]``
  inverse retractions to use with the vector transport method (especially the differentiated ones)
- `vector_transport_to = [ true for _ in 1:length(vector_transport_methods)]`: whether
   to check the `to` variant of vector transport
- `vector_transport_direction = [ true for _ in 1:length(vector_transport_methods)]`: whether
   to check the `direction` variant of vector transport

!!! note
    This function is soon deprecated and will be replaced by a more systematic way in `Manifolds.Test`
"""
function Manifolds.test_manifold(
        M::AbstractManifold, pts::AbstractVector;
        basis_has_specialized_diagonalizing_get = false,
        basis_types_to_from = (),
        basis_types_vecs = (),
        default_inverse_retraction_method = LogarithmicInverseRetraction(),
        default_retraction_method = ExponentialRetraction(),
        exp_log_atol_multiplier = 0,
        exp_log_rtol_multiplier = 1,
        expected_dimension_type = Integer,
        inverse_retraction_methods = [],
        is_mutating = true,
        is_point_atol_multiplier = 0,
        is_tangent_atol_multiplier = 0,
        musical_isomorphism_bases = [],
        point_distributions = [],
        projection_atol_multiplier = 0,
        rand_tvector_atol_multiplier = 0,
        retraction_atol_multiplier = 0,
        retraction_methods = [],
        retraction_rtol_multiplier = 1,
        test_atlases = (),
        test_exp_log = true,
        test_is_tangent = true,
        test_injectivity_radius = true,
        test_inplace = false,
        has_get_embedding = false,
        test_musical_isomorphisms = false,
        test_mutating_rand = false,
        parallel_transport = false,
        parallel_transport_to = parallel_transport,
        parallel_transport_direction = parallel_transport,
        test_inner = true,
        test_norm = true,
        test_project_point = false,
        test_project_tangent = false,
        test_rand_point = false,
        test_rand_tvector = false,
        test_representation_size = true,
        test_riesz_representer = false,
        test_tangent_vector_broadcasting = true,
        test_default_vector_transport = false,
        test_vector_spaces = true,
        tvector_distributions = [],
        vector_transport_methods = [],
        vector_transport_inverse_retractions = [
            default_inverse_retraction_method for _ in 1:length(vector_transport_methods)
        ],
        vector_transport_retractions = [
            default_retraction_method for _ in 1:length(vector_transport_methods)
        ],
        test_vector_transport_to = [true for _ in 1:length(vector_transport_methods)],
        test_vector_transport_direction = [true for _ in 1:length(vector_transport_methods)],
        mid_point12 = test_exp_log ? shortest_geodesic(M, pts[1], pts[2], 0.5) : nothing,
    )
    length(pts) ≥ 3 || error("Not enough points (at least three expected)")
    isapprox(M, pts[1], pts[2]) && error("Points 1 and 2 are equal")
    isapprox(M, pts[1], pts[3]) && error("Points 1 and 3 are equal")

    # get a default tangent vector for every of the three tangent spaces
    n = length(pts)
    if default_inverse_retraction_method === nothing
        tv = [zero_vector(M, pts[i]) for i in 1:n] # no other available
    else
        tv = [
            inverse_retract(
                    M, pts[i], pts[((i + 1) % n) + 1], default_inverse_retraction_method
                ) for i in 1:n
        ]
    end
    Test.@testset "dimension" begin # COV_EXCL_LINE
        Test.@test isa(manifold_dimension(M), expected_dimension_type)
        Test.@test manifold_dimension(M) ≥ 0
    end

    test_representation_size && Test.@testset "representation" begin
        function test_repr(repr)
            Test.@test isa(repr, Tuple)
            for rs in repr
                Test.@test rs > 0
            end
            return nothing
        end

        test_repr(Manifolds.representation_size(M))
    end

    test_injectivity_radius && Test.@testset "injectivity radius" begin
        Test.@test injectivity_radius(M, pts[1]) > 0
        Test.@test injectivity_radius(M, pts[1]) ≥ injectivity_radius(M)
        for rm in retraction_methods
            Test.@test injectivity_radius(M, rm) > 0
            Test.@test injectivity_radius(M, pts[1], rm) ≥ injectivity_radius(M, rm)
            Test.@test injectivity_radius(M, pts[1], rm) ≤ injectivity_radius(M, pts[1])
        end
    end

    Test.@testset "is_point" begin # COV_EXCL_LINE
        for pt in pts
            atol = is_point_atol_multiplier * Manifolds.find_eps(pt)
            Test.@test is_point(M, pt; atol = atol)
            Test.@test check_point(M, pt; atol = atol) === nothing
        end
    end

    test_is_tangent && Test.@testset "is_vector" begin
        for (p, X) in zip(pts, tv)
            atol = is_tangent_atol_multiplier * Manifolds.find_eps(p)
            if !(check_vector(M, p, X; atol = atol) === nothing)
                print(check_vector(M, p, X; atol = atol))
            end
            Test.@test is_vector(M, p, X; atol = atol, error = :error)
            Test.@test check_vector(M, p, X; atol = atol) === nothing
        end
    end

    test_exp_log && Test.@testset "log/exp tests" begin
        epsp1p2 = Manifolds.find_eps(pts[1], pts[2])
        atolp1p2 = exp_log_atol_multiplier * epsp1p2
        rtolp1p2 =
            exp_log_atol_multiplier == 0.0 ? sqrt(epsp1p2) * exp_log_rtol_multiplier : 0
        X1 = log(M, pts[1], pts[2])
        X2 = log(M, pts[2], pts[1])
        Test.@test isapprox(M, pts[2], exp(M, pts[1], X1); atol = atolp1p2, rtol = rtolp1p2)
        Test.@test isapprox(
            M, pts[1], Manifolds.exp_fused(M, pts[1], X1, 0);
            atol = atolp1p2, rtol = rtolp1p2
        )
        Test.@test isapprox(
            M, pts[2], Manifolds.exp_fused(M, pts[1], X1, 1);
            atol = atolp1p2, rtol = rtolp1p2
        )
        if is_mutating
            q2 = allocate(pts[1])
            exp!(M, q2, pts[1], X1)
            Test.@test isapprox(M, pts[2], q2; atol = atolp1p2, rtol = rtolp1p2)
            Manifolds.exp_fused!(M, q2, pts[1], X1, 0)
            Test.@test isapprox(M, pts[1], q2; atol = atolp1p2, rtol = rtolp1p2)
        end
        if VERSION >= v"1.5" && isa(M, GeneralizedStiefel)
            # TODO: investigate why this is so imprecise on newer Julia versions on CI
            Test.@test isapprox(
                M, pts[1], exp(M, pts[2], X2);
                # yields 5*10^-8 for the usual 10^-13 we impose on earlier Julia versions
                atol = atolp1p2 * 5 * 10^5, rtol = rtolp1p2
            )
        else
            Test.@test isapprox(M, pts[1], exp(M, pts[2], X2); atol = atolp1p2, rtol = rtolp1p2)
        end
        Test.@test is_point(M, exp(M, pts[1], X1); atol = atolp1p2, rtol = rtolp1p2)
        Test.@test isapprox(
            M, pts[1], Manifolds.exp_fused(M, pts[1], X1, 0);
            atol = atolp1p2, rtol = rtolp1p2
        )
        for p in pts
            epsx = Manifolds.find_eps(p)
            Test.@test isapprox(
                M, p, zero_vector(M, p), log(M, p, p);
                atol = epsx * exp_log_atol_multiplier,
                rtol = exp_log_atol_multiplier == 0.0 ? sqrt(epsx) * exp_log_rtol_multiplier : 0
            )
            Test.@test isapprox(
                M, p, zero_vector(M, p), inverse_retract(M, p, p);
                atol = epsx * exp_log_atol_multiplier,
                rtol = exp_log_atol_multiplier == 0.0 ? sqrt(epsx) * exp_log_rtol_multiplier : 0.0,
            )
        end
        atolp1 = exp_log_atol_multiplier * Manifolds.find_eps(pts[1])
        if is_mutating
            zero_vector!(M, X1, pts[1])
        else
            X1 = zero_vector(M, pts[1])
        end
        Test.@test isapprox(M, pts[1], X1, zero_vector(M, pts[1]); atol = atolp1)
        if is_mutating
            log!(M, X1, pts[1], pts[2])
        else
            X1 = log(M, pts[1], pts[2])
        end

        Test.@test isapprox(M, Manifolds.exp_fused(M, pts[1], X1, 1), pts[2]; atol = atolp1)
        Test.@test isapprox(M, Manifolds.exp_fused(M, pts[1], X1, 0), pts[1]; atol = atolp1)

        if test_norm
            Test.@test distance(M, pts[1], pts[2]) ≈ norm(M, pts[1], X1)
        end

        if test_inner
            X3 = log(M, pts[1], pts[3])
            Test.@test inner(M, pts[1], X1, X3) ≈ conj(inner(M, pts[1], X3, X1))
            Test.@test inner(M, pts[1], X1, X1) ≈ real(inner(M, pts[1], X1, X1))

            Test.@test norm(M, pts[1], X1) ≈ sqrt(inner(M, pts[1], X1, X1))
        end
        if test_norm
            Test.@test norm(M, pts[1], X1) isa Real
        end

        (test_inplace && is_mutating) && Test.@testset "inplace test for exp!" begin
            p = copy(M, pts[1])
            X = copy(M, pts[1], X1)
            q = exp(M, p, X)
            exp!(M, p, p, X)
            Test.@test isapprox(M, p, q)
            # This test is not reasonable for `log!(M, X, p, q)`,
            # since X is of different type/concept than p,q
        end
    end

    parallel_transport && test_parallel_transport(
        M, pts;
        to = parallel_transport_to, direction = parallel_transport_direction, mutating = is_mutating,
    )

    Test.@testset "(inverse &) retraction tests" begin # COV_EXCL_LINE
        for (p, X) in zip(pts, tv)
            epsx = Manifolds.find_eps(p)
            point_atol = is_point_atol_multiplier * Manifolds.find_eps(p)
            for retr_method in retraction_methods
                Test.@test is_point(M, retract(M, p, X, retr_method); atol = point_atol)
                Test.@test isapprox(
                    M, p, Manifolds.retract_fused(M, p, X, 0, retr_method);
                    atol = epsx * retraction_atol_multiplier,
                    rtol = retraction_atol_multiplier == 0 ? sqrt(epsx) * retraction_rtol_multiplier : 0
                )
                if is_mutating
                    new_pt = allocate(p)
                    retract!(M, new_pt, p, X, retr_method)
                else
                    new_pt = retract(M, p, X, retr_method)
                end
                Test.@test is_point(M, new_pt; atol = point_atol)
                (test_inplace && is_mutating) &&
                    Test.@testset "inplace test for retract!" begin # COV_EXCL_LINE
                    p2 = copy(M, p)
                    X2 = copy(M, p, X)
                    q = retract(M, p2, X2, retr_method)
                    retract!(M, p2, p2, X, retr_method)
                    Test.@test isapprox(M, p2, q; atol = point_atol)
                    # This test is not reasonable for `inverse_retract!(M, X, p, q, m)`,
                    # since X is of different type/concept than p,q

                end
            end
        end
        for p in pts
            epsx = Manifolds.find_eps(p)
            for inv_retr_method in inverse_retraction_methods
                X = inverse_retract(M, p, p, inv_retr_method)
                Test.@test isapprox(
                    M, p, zero_vector(M, p), X;
                    atol = epsx * retraction_atol_multiplier,
                    rtol = retraction_atol_multiplier == 0 ? sqrt(epsx) * retraction_rtol_multiplier : 0
                )
                if (test_inplace && is_mutating)
                    Y = copy(M, p, X)
                    inverse_retract!(M, Y, p, p, inv_retr_method)
                    Test.@test isapprox(
                        M, p, zero_vector(M, p), Y;
                        atol = epsx * retraction_atol_multiplier,
                        rtol = retraction_atol_multiplier == 0 ? sqrt(epsx) * retraction_rtol_multiplier : 0
                    )
                end
            end
        end
    end

    Test.@testset "atlases" begin # COV_EXCL_LINE
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

    test_riesz_representer && Test.@testset "RieszRepresenterCotangentVector" begin
        rrcv = flat(M, pts[1], tv[1])
        Test.@test rrcv isa RieszRepresenterCotangentVector
        Test.@test rrcv.p === pts[1]
        Test.@test rrcv.X === tv[1]
        basis = dual_basis(M, pts[1], basis_types_to_from[1])
        coords = get_coordinates(M, pts[1], rrcv, basis)
        rrcv2 = get_vector(M, pts[1], coords, basis)
        Test.@test isapprox(M, pts[1], rrcv.X, rrcv2.X)
    end

    test_vector_spaces && Test.@testset "vector spaces tests" begin
        for p in pts
            X = zero_vector(M, p)
            mts = TangentSpace(M, p)
            Test.@test isapprox(M, p, X, zero_vector(mts, X))
            if is_mutating
                zero_vector!(mts, X, X)
                Test.@test isapprox(M, p, X, zero_vector(M, p))
            end
        end
    end

    Test.@testset "basic linear algebra in tangent space" begin # COV_EXCL_LINE
        for (p, X) in zip(pts, tv)
            Test.@test isapprox(M, p, 0 * X, zero_vector(M, p); atol = Manifolds.find_eps(pts[1]))
            Test.@test isapprox(M, p, 2 * X, X + X)
            Test.@test isapprox(M, p, 0 * X, X - X)
            Test.@test isapprox(M, p, (-1) * X, -X)
        end
    end

    test_tangent_vector_broadcasting &&
        Test.@testset "broadcasted linear algebra in tangent space" begin # COV_EXCL_LINE
        for (p, X) in zip(pts, tv)
            Test.@test isapprox(M, p, 3 * X, 2 .* X .+ X)
            Test.@test isapprox(M, p, -X, X .- 2 .* X)
            Test.@test isapprox(M, p, -X, .-X)
            if (isa(X, AbstractArray))
                Y = allocate(X)
                Y .= 2 .* X .+ X
            else
                Y = 2 * X + X
            end
            Test.@test isapprox(M, p, Y, 3 * X)
        end
    end

    test_project_tangent && Test.@testset "project tangent test" begin
        for (p, X) in zip(pts, tv)
            atol = Manifolds.find_eps(p) * projection_atol_multiplier
            X_emb = embed(M, p, X)
            p_emb = embed(M, p)
            Test.@test isapprox(M, p, X, project(M, p, X_emb); atol = atol)
            if is_mutating
                X2 = allocate(X)
                project!(M, X2, p, X_emb)
            else
                X2 = project(M, p, X_emb)
            end
            Test.@test isapprox(M, p, X2, X; atol = atol)
        end
    end

    test_project_point && Test.@testset "project point test" begin
        for p in pts
            atol = Manifolds.find_eps(p) * projection_atol_multiplier
            p_emb = embed(M, p)
            Test.@test isapprox(M, p, project(M, p_emb); atol = atol)
            if is_mutating
                p2 = allocate(p)
                project!(M, p2, p_emb)
            else
                p2 = project(M, p_emb)
            end
            Test.@test isapprox(M, p2, p; atol = atol)
        end
    end

    !(
        default_retraction_method === nothing ||
            default_inverse_retraction_method === nothing
    ) && Test.@testset "vector transport" begin # COV_EXCL_LINE
        tvatol = is_tangent_atol_multiplier * Manifolds.find_eps(pts[1])
        X1 = inverse_retract(M, pts[1], pts[2], default_inverse_retraction_method)
        X2 = inverse_retract(M, pts[1], pts[3], default_inverse_retraction_method)
        pts32 = retract(M, pts[1], X2, default_retraction_method)
        test_default_vector_transport && Test.@testset "default vector transport" begin
            v1t1 = vector_transport_to(M, pts[1], X1, pts32)
            v1t2 = vector_transport_direction(M, pts[1], X1, X2)
            Test.@test is_vector(M, pts32, v1t1; atol = tvatol, error = :warn)
            Test.@test is_vector(M, pts32, v1t2; atol = tvatol, error = :warn)
            Test.@test isapprox(M, pts32, v1t1, v1t2)
            Test.@test isapprox(M, pts[1], vector_transport_to(M, pts[1], X1, pts[1]), X1)

            is_mutating && Test.@testset "mutating variants" begin
                v1t1_m = allocate(v1t1)
                v1t2_m = allocate(v1t2)
                vector_transport_to!(M, v1t1_m, pts[1], X1, pts32)
                vector_transport_direction!(M, v1t2_m, pts[1], X1, X2)
                Test.@test isapprox(M, pts32, v1t1, v1t1_m)
                Test.@test isapprox(M, pts32, v1t2, v1t2_m)
            end
        end

        for (vtm, test_to, test_dir, rtr_m, irtr_m) in zip(
                vector_transport_methods,
                test_vector_transport_to,
                test_vector_transport_direction,
                vector_transport_retractions,
                vector_transport_inverse_retractions,
            )
            Test.@testset "vector transport method $(vtm)" begin # COV_EXCL_LINE
                tvatol = is_tangent_atol_multiplier * Manifolds.find_eps(pts[1])
                X1 = inverse_retract(M, pts[1], pts[2], irtr_m)
                X2 = inverse_retract(M, pts[1], pts[3], irtr_m)
                pts32 = retract(M, pts[1], X2, rtr_m)
                test_to && (v1t1 = vector_transport_to(M, pts[1], X1, pts32, vtm))
                test_dir && (v1t2 = vector_transport_direction(M, pts[1], X1, X2, vtm))
                test_to &&
                    Test.@test is_vector(M, pts32, v1t1; atol = tvatol, error = :warn)
                test_dir &&
                    Test.@test is_vector(M, pts32, v1t2; atol = tvatol, error = :warn)
                (test_to && test_dir) &&
                    Test.@test isapprox(M, pts32, v1t1, v1t2, atol = tvatol)
                test_to && Test.@test isapprox(
                    M, pts[1], vector_transport_to(M, pts[1], X1, pts[1], vtm), X1;
                    atol = tvatol
                )
                test_dir && Test.@test isapprox(
                    M, pts[1], vector_transport_direction(M, pts[1], X1, zero_vector(M, pts[1]), vtm), X1;
                    atol = tvatol
                )

                is_mutating && Test.@testset "mutating variants" begin
                    if test_to
                        v1t1_m = allocate(v1t1)
                        vector_transport_to!(M, v1t1_m, pts[1], X1, pts32, vtm)
                        Test.@test isapprox(M, pts32, v1t1, v1t1_m; atol = tvatol)
                        test_inplace &&
                            Test.@testset "inplace test for vector_transport_to!" begin # COV_EXCL_LINE
                            X1a = copy(M, pts[1], X1)
                            Xt = vector_transport_to(M, pts[1], X1, pts32, vtm)
                            vector_transport_to!(M, X1a, pts[1], X1a, pts32, vtm)
                            Test.@test isapprox(M, pts[1], X1a, Xt; atol = tvatol)
                        end
                    end
                    if test_dir
                        v1t2_m = allocate(v1t2)
                        vector_transport_direction!(M, v1t2_m, pts[1], X1, X2, vtm)
                        Test.@test isapprox(M, pts32, v1t2, v1t2_m; atol = tvatol)
                        test_inplace &&
                            Test.@testset "inplace test for vector_transport_direction!" begin
                            X1a = copy(M, pts[1], X1)
                            X2a = copy(M, pts[1], X2)
                            Xt = vector_transport_direction(M, pts[1], X1, X2, vtm)
                            vector_transport_direction!(M, X1a, pts[1], X1a, X2, vtm)
                            vector_transport_direction!(M, X2a, pts[1], X1, X2a, vtm)
                            Test.@test isapprox(M, pts[1], X1a, Xt; atol = tvatol)
                            Test.@test isapprox(M, pts[1], X2a, Xt; atol = tvatol)
                        end
                    end
                end
            end
        end
    end

    for btype in basis_types_vecs
        Test.@testset "Basis support for $(btype)" begin # COV_EXCL_LINE
            p = pts[1]
            b = get_basis(M, p, btype)
            Test.@test isa(b, CachedBasis)
            bvectors = get_vectors(M, p, b)
            N = length(bvectors)

            # test orthonormality
            for i in 1:N
                Test.@test norm(M, p, bvectors[i]) ≈ 1
                for j in (i + 1):N
                    Test.@test real(inner(M, p, bvectors[i], bvectors[j])) ≈ 0 atol =
                        sqrt(Manifolds.find_eps(p))
                end
            end
            if isa(btype, ProjectedOrthonormalBasis)
                # check projection idempotency
                for i in 1:N
                    Test.@test norm(M, p, bvectors[i]) ≈ 1
                    for j in (i + 1):N
                        Test.@test real(inner(M, p, bvectors[i], bvectors[j])) ≈ 0 atol =
                            sqrt(Manifolds.find_eps(p))
                    end
                end
                # check projection idempotency
                for i in 1:N
                    Test.@test isapprox(M, p, embed_project(M, p, bvectors[i]), bvectors[i])
                end
            end
            if !isa(btype, ProjectedOrthonormalBasis) && (
                    basis_has_specialized_diagonalizing_get ||
                        !isa(btype, DiagonalizingOrthonormalBasis)
                )
                X1 = inverse_retract(M, p, pts[2], default_inverse_retraction_method)
                Xb = get_coordinates(M, p, X1, btype)

                Test.@test get_coordinates(M, p, X1, b) ≈ Xb
                Test.@test isapprox(
                    M, p, get_vector(M, p, Xb, b), get_vector(M, p, Xb, btype)
                )
            end
        end
    end

    for btype in (basis_types_to_from..., basis_types_vecs...)
        p = pts[1]
        N = number_of_coordinates(M, btype)
        if !isa(btype, ProjectedOrthonormalBasis) && (
                basis_has_specialized_diagonalizing_get ||
                    !isa(btype, DiagonalizingOrthonormalBasis)
            )
            X1 = inverse_retract(M, p, pts[2], default_inverse_retraction_method)

            Xb = get_coordinates(M, p, X1, btype)
            #Test.@test isa(Xb, AbstractVector{<:Real})
            Test.@test length(Xb) == N
            Xbi = get_vector(M, p, Xb, btype)
            Test.@test isapprox(M, p, X1, Xbi)

            Xs = [[ifelse(i == j, 1, 0) for j in 1:N] for i in 1:N]
            Xs_invs = [get_vector(M, p, Xu, btype) for Xu in Xs]
            # check orthonormality of inverse representation
            for i in 1:N
                Test.@test norm(M, p, Xs_invs[i]) ≈ 1 atol = sqrt(Manifolds.find_eps(p))
                for j in (i + 1):N
                    Test.@test real(inner(M, p, Xs_invs[i], Xs_invs[j])) ≈ 0 atol =
                        sqrt(Manifolds.find_eps(p))
                end
            end

            if is_mutating
                Xb_s = allocate(Xb)
                Test.@test get_coordinates!(M, Xb_s, p, X1, btype) === Xb_s
                Test.@test isapprox(Xb_s, Xb; atol = Manifolds.find_eps(p))

                Xbi_s = allocate(Xbi)
                Test.@test get_vector!(M, Xbi_s, p, Xb, btype) === Xbi_s
                Test.@test isapprox(M, p, X1, Xbi_s)
            end
        end
    end

    mid_point12 !== nothing && Test.@testset "midpoint" begin
        epsp1p2 = Manifolds.find_eps(pts[1], pts[2])
        atolp1p2 = exp_log_atol_multiplier * epsp1p2
        rtolp1p2 =
            exp_log_atol_multiplier == 0.0 ? sqrt(epsp1p2) * exp_log_rtol_multiplier : 0
        mp = mid_point(M, pts[1], pts[2])
        Test.@test isapprox(M, mp, mid_point12; atol = atolp1p2, rtol = rtolp1p2)
        if is_mutating
            mpm = allocate(mp)
            mid_point!(M, mpm, pts[1], pts[2])
            Test.@test isapprox(M, mpm, mid_point12; atol = atolp1p2, rtol = rtolp1p2)
            test_inplace && Test.@testset "inplace test for midpoint!" begin
                p1 = copy(M, pts[1])
                p2 = copy(M, pts[2])
                p3 = mid_point(M, pts[1], pts[2])
                mid_point!(M, p1, p1, pts[2])
                mid_point!(M, p2, pts[1], p2)
                Test.@test isapprox(M, p3, p1)
                Test.@test isapprox(M, p3, p2)
            end
        end
    end

    test_musical_isomorphisms && Test.@testset "Musical isomorphisms" begin
        if default_inverse_retraction_method !== nothing
            tv_m = inverse_retract(M, pts[1], pts[2], default_inverse_retraction_method)
        else
            tv_m = zero_vector(M, pts[1])
        end
        ctv_m = flat(M, pts[1], tv_m)
        Test.@test ctv_m(tv_m) ≈ norm(M, pts[1], tv_m)^2
        tv_m_back = sharp(M, pts[1], ctv_m)
        Test.@test isapprox(M, pts[1], tv_m, tv_m_back)

        if is_mutating
            ctv_m_s = allocate(ctv_m)
            flat!(M, ctv_m_s, pts[1], tv_m)
            Test.@test ctv_m_s(tv_m) ≈ ctv_m(tv_m)
            tv_m_s_back = allocate(tv_m_back)
            sharp!(M, tv_m_s_back, pts[1], ctv_m_s)
            Test.@test isapprox(M, pts[1], tv_m, tv_m_s_back)
        end

        for basis in musical_isomorphism_bases
            tv_m_f = ManifoldsBase.TFVector(get_coordinates(M, pts[1], tv_m, basis), basis)
            ctv_m_f = flat(M, pts[1], tv_m_f)
            Test.@test isa(ctv_m_f, CoTFVector)
            tv_m_f_back = sharp(M, pts[1], ctv_m_f)
            Test.@test isapprox(tv_m_f.data, tv_m_f_back.data)
        end
    end

    Test.@testset "number_eltype" begin # COV_EXCL_LINE
        for (p, X) in zip(pts, tv)
            Test.@test number_eltype(X) == number_eltype(p)
            p = retract(M, p, X, default_retraction_method)
            Test.@test number_eltype(p) == number_eltype(p)
        end
    end

    is_mutating && Test.@testset "copyto!" begin
        for (p, X) in zip(pts, tv)
            p2 = allocate(p)
            copyto!(p2, p)
            Test.@test isapprox(M, p2, p)

            X2 = allocate(X)
            if default_inverse_retraction_method === nothing
                X3 = zero_vector(M, p)
                copyto!(X2, X3)
                Test.@test isapprox(M, p, X2, zero_vector(M, p))
            else
                q = retract(M, p, X, default_retraction_method)
                X3 = inverse_retract(M, p, q, default_inverse_retraction_method)
                copyto!(X2, X3)
                Test.@test isapprox(
                    M, p, X2, inverse_retract(M, p, q, default_inverse_retraction_method)
                )
            end
        end
    end

    is_mutating && Test.@testset "point distributions" begin
        for p in pts
            prand = allocate(p)
            for pd in point_distributions
                for _ in 1:10
                    Test.@test is_point(M, rand(pd))
                    if test_mutating_rand
                        rand!(pd, prand)
                        Test.@test is_point(M, prand)
                    end
                end
            end
        end
    end

    test_rand_point && Test.@testset "Base.rand point generation" begin
        rng_a = MersenneTwister(123)
        rng_b = MersenneTwister(123)
        Test.@test is_point(M, rand(M); error = :error)
        # ensure that the RNG source is actually used
        Test.@test rand(rng_a, M) == rand(rng_b, M)
        # generation of multiple points
        Test.@test all(p -> is_point(M, p; error = :error), rand(M, 3))
        Test.@test all(p -> is_point(M, p; error = :error), rand(rng_a, M, 3))

        if test_inplace && is_mutating
            rng_a = MersenneTwister(123)
            rng_b = MersenneTwister(123)

            p = allocate(pts[1])
            rand!(M, p)
            Test.@test is_point(M, p; error = :error)
            p = allocate(pts[1])
            rand!(rng_a, M, p)
            Test.@test is_point(M, p; error = :error)
            # ensure that the RNG source is actually used
            q = allocate(pts[1])
            rand!(rng_b, M, q)
            Test.@test p == q
        end
    end

    test_rand_tvector && Test.@testset "Base.rand tangent vector generation" begin
        p = pts[1]
        rng_a = MersenneTwister(123)
        rng_b = MersenneTwister(123)
        randX = rand(M; vector_at = p)
        atol = rand_tvector_atol_multiplier * Manifolds.find_eps(randX)
        Test.@test is_vector(M, p, randX, true; atol = atol)
        # ensure that the RNG source is actually used
        Test.@test rand(rng_a, M; vector_at = p) == rand(rng_b, M; vector_at = p)
        # generation of multiple tangent vectors
        Test.@test all(X -> is_vector(M, p, X, true; atol = atol), rand(M, 3; vector_at = p))
        Test.@test all(
            X -> is_vector(M, p, X, true; atol = atol), rand(rng_a, M, 3; vector_at = p)
        )

        if test_inplace && is_mutating
            rng_a = MersenneTwister(123)
            rng_b = MersenneTwister(123)

            X = allocate(tv[1])
            rand!(M, X; vector_at = p)
            Test.@test is_vector(M, p, X, true; atol = atol)
            X = allocate(tv[1])
            rand!(rng_a, M, X; vector_at = p)
            Test.@test is_point(M, p; error = :error)
            # ensure that the RNG source is actually used
            Y = allocate(tv[1])
            rand!(rng_b, M, Y; vector_at = p)
            Test.@test X == Y
        end
    end

    Test.@testset "tangent vector distributions" begin # COV_EXCL_LINE
        for tvd in tvector_distributions
            p = tvd.type.point
            for _ in 1:5
                randtv = rand(tvd)
                atol = rand_tvector_atol_multiplier * Manifolds.find_eps(randtv)
                Test.@test is_vector(M, p, randtv, true; atol = atol)
                if test_mutating_rand
                    X = allocate(randtv)
                    rand!(tvd, X)
                    Test.@test is_vector(M, p, X, true; atol = atol)
                end
            end
        end
    end
    return nothing
end

"""
    Manifolds.test_parallel_transport(M,P; to=true, direction=true)

Generic tests for parallel transport on `M`given at least two pointsin `P`.

The single functions to transport `to` (a point) or (in a) `direction`
are sub-tests that can be activated by the keywords arguments

!!! note
    This function is soon deprecated and will be replaced by a more systematic way in `Manifolds.Test`
"""
function Manifolds.test_parallel_transport(
        M::AbstractManifold, P,
        Ξ = inverse_retract.(
            Ref(M), P[1:(end - 1)], P[2:end], Ref(default_inverse_retraction_method(M))
        );
        to = true, direction = true, mutating = true,
    )
    length(P) < 2 &&
        error("The Parallel Transport test set requires at least 2 points in P")
    return Test.@testset "Test Parallel Transport" begin # COV_EXCL_LINE
        Test.@testset "To (a point)" begin # COV_EXCL_LINE
            # even with to =false this displays no tests
            if to
                for i in 1:(length(P) - 1)
                    p = P[i]
                    q = P[i + 1]
                    X = Ξ[i]
                    Y1 = parallel_transport_to(M, p, X, q)
                    if mutating
                        Y2 = similar(X)
                        parallel_transport_to!(M, Y2, p, X, q)
                        # test that mutating and allocating to the same
                        Test.@test isapprox(M, q, Y1, Y2)
                        parallel_transport_to!(M, Y2, q, Y1, p)
                        # Test that transporting there and back again yields the identity
                        Test.@test isapprox(M, q, X, Y2)
                        parallel_transport_to!(M, Y1, q, Y1, p)
                        # Test that inplace does not have side effects
                    else
                        Y1 = parallel_transport_to(M, q, Y1, p)
                    end
                    Test.@test isapprox(M, q, X, Y1)
                end
            end
        end
        Test.@testset "(Tangent Vector) Direction" begin # COV_EXCL_LINE
            if direction
                for i in 1:(length(P) - 1)
                    p = P[i]
                    X = Ξ[i]
                    Y1 = parallel_transport_direction(M, p, X, X)
                    q = exp(M, p, X)
                    if mutating
                        Y2 = similar(X)
                        parallel_transport_direction!(M, Y2, p, X, X)
                        # test that mutating and allocating to the same
                        Test.@test isapprox(M, q, Y1, Y2)
                    end
                    # Test that Y is a tangent vector at q
                    Test.@test is_vector(M, p, Y1; error = :error)
                end
            end
        end
    end
end
