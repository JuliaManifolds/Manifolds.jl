module ManifoldsTestExt

using Manifolds
using ManifoldsBase
using Test
using Manifolds.Test: AbstractExpectation, Expect, NoExpectation, isexpected, expect

get_expectation(g::Dict, key, default = NoExpectation()) = Expect(get(g, key, default))

"""
    test_manifold(G::AbstractManifold, properties::Dict, expectations::Dict)

Test the [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) ``\\mathcal M``
based on a `Dict` of properties and a `Dict` of `expectations`.

Three functions are expected to be defined (without explicitly being passed in the `properties`):
`is_point(M, p)`, `is_vector(M, p, X)`, and `isapprox(M, p, q)` / `isapprox(M, p, X, Y)`,
since these are essential for verifying results.

From the following properties, the two often expected to be defined are `:Points` and `:Vectors`,
which should contain at least two points and two tangent vectors, respectively.

Possible properties are

* `:Aliased` is a boolean (same as `:Mutating` by default) whether to test the mutating variants with aliased input
* `:Bases` is a vector of bases, which can be used to test basis related functions, one basis for each entry in `:Coordinates`
* `:Covectors` is a vector of covectors, which should be in the cotangent space of the correspondinig point entry in `:Points`
* `:Coordinates` is a vector of coordinates, which can be used to test coordinate related functions, one coordinate vector for each entry in `:Bases`
* `:EmbeddedPoints` is a vector of points in the embedding space of `M`, to test `project`
* `:EmbeddedVectors` is a vector of tangent vectors in the embedding space of `M`, to test `project`
* `:Functions` is a vector of all defined functions for `M`
  note a test is activated by the function (like `exp`), adding the mutating function (like `exp!`) overwrites the
  global default (see `:Mutating`) to true.
* `:GeodesicMaxTime` is a real number indicating the time parameter to use when testing `geodesic`
* `:GeodesicSamples` is an integer indicating the number of samples to use when testing `geodesic` (defaults to 10)
* `:InvalidPoints` is a vector of points that are not on `M`, e.g. to test `is_point`
* `:InvalidVectors` is a vector of tangent vectors that are not in the tangent space of the first point from `:Points`
* `:InverseRetractionMethods` is a vector of inverse retraction methods to test on `M`
  these should have the same order as `:RetractionMethods` (use `missing` for skipping one)
* `:Mutating` is a boolean (`true` by default) whether to test the mutating variants of functions or not.
  when setting this to false, you can still activate single functions mutation checks by
  adding the mutating function to `:Functions`
* `:Name` is a name of the test. If not provided, defaults to `"\$M"`
* `:NormalVectors` is a vector of normal vectors, where each should be in the normal space of the corresponding point entries in `:Points`
* `:Points` is a vector of at least 2 points on `M`, which should not be the same point
* `:RetractionMethods` is a vector of retraction methods to test on `M`
  these should have the same order as `:InverseRetractionMethods` (use `missing` for skipping one)
* `:Rng` is a random number generator to use for generating random points/vectors if needed
* `:Seed` is a seed to use for generating random points/vectors if needed
* `:Vectors` is a vector of at least 2 tangent vectors, which should be in the tangent space of the correspondinig point entries in `:Points`
* `:VectorTransportMethods` is a vector of vector transport methods to test on `M`
* `:TestMidpointSymmetry` is a boolean (`true` by default) whether to test the symmetry property of the midpoint function
* `:TestInfo` is a boolean (`true` by default) whether to test that whether `error=:info` in verification functions issues info messages.
* `:TestWarn` is a boolean (`true` by default) whether to test that whether `error=:warn` in verification functions issues warning.

Possible entries of the `expectations` dictionary are

* any function tested to provide their expected resulting value, e.g. `exp => p` for the result of `exp(M, p, X)`
* for retractions, inverse retractions, and vector transports, the key is a tuple of the function and the method, e.g. `(retract, method) => q`
* for `embed`, and `project`, the key is a tuple of the function and `:Point` or `:Vector`, e.g. `(embed, :Point) of expected (embedded) points or vectors,
  omitting that symbol is interpreted as the expected point.
* for `get_basis`, the key is a tuple of the function and the basis, e.g. `(get_basis, B) => ...` to the expexted basis
* for `get_coordinates` the key is a tuple of the function and the basis, e.g. `(get_coordinates, B) => c`
* for `get_vector` the key is a tuple of the function, the coordinate vector, and the basis, e.g. `(get_vector, c, B) => X`
* for `get_vectors` the key is a tuple of the function and the basis, e.g. `(get_vectors, B) => :Symbol` where
  * `:Orthogonal` tests for orthogonality
  * `:Orthonormal` tests additionally to the previous for unit length
  For any basis this test calls `get_basis` on any provided basis, if that function is available.
* for `is_default_metric`, the value is the default metric
* `:atol => 0.0` a global absolute tolerance
* `:atols -> Dict()` a dictionary `function -> atol` for tolerances of specific function tested.
* `:Types` -> Dict() a dictionary `function -> Type` for specifying expected types of results of specific functions, for example `manifold_dimension => Int`.
*  `:IsPointErrors` is a vector of expected error types for each invalid point provided in `:InvalidPoints`, use `missing` to skip testing for errors for a specific point.
*  `:IsVectorErrors` is a vector of expected error types for each invalid vector provided in `:InvalidVectors`, use `missing` to skip testing for errors for a specific vector.
*  `:IsVectorBasepointError` is an expected error type when the base point is invalid e.g. for `is_vector`
"""
function Manifolds.Test.test_manifold(M::AbstractManifold, properties::Dict, expectations::Dict = Dict())
    atol = get(expectations, :atol, 0.0)
    mutating = get(properties, :Mutating, true)
    aliased = get(properties, :Aliased, mutating)
    functions = get(properties, :Functions, Function[])
    points = get(properties, :Points, [])
    vectors = get(properties, :Vectors, [])
    covectors = get(properties, :Covectors, [])
    normals = get(properties, :NormalVectors, [])
    bases = get(properties, :Bases, [])
    coordinates = get(properties, :Coordinates, [])
    test_name = get(properties, :Name, "Manifolds.Test suite for $M")
    test_warn = get(properties, :TestWarn, true)
    test_info = get(properties, :TestInfo, true)
    function_atols = get(expectations, :atols, Dict())
    result_types = get(expectations, :Types, Dict())
    retraction_methods = get(properties, :RetractionMethods, [])
    inverse_retraction_methods = get(properties, :InverseRetractionMethods, [])
    vector_transport_methods = get(properties, :VectorTransportMethods, [])
    t = Test.@testset "$test_name" begin # COV_EXCL_LINE
        n_points = length(points)
        n_vectors = length(vectors)
        if (copy in functions)
            Manifolds.Test.test_copy(
                M, points[1], vectors[1];
                name = "copy(M, p) & copy(M, p, X)",
            )
        end
        if (copyto! in functions)
            Manifolds.Test.test_copyto(
                M, points[1], vectors[1];
                name = "copyto!(M, q, p) & copyto!(M, Y, p, X)",
            )
        end
        if (default_inverse_retraction_method in functions)
            Manifolds.Test.test_default_inverse_retraction(
                M;
                expected_value = get_expectation(expectations, default_inverse_retraction_method),
                name = "default_inverse_retraction_method(M)",
            )
            Manifolds.Test.test_default_inverse_retraction(
                M, typeof(points[1]);
                expected_value = get_expectation(expectations, default_inverse_retraction_method),
                name = "default_inverse_retraction_method(M, P)",
            )
        end
        if (default_retraction_method in functions)
            Manifolds.Test.test_default_retraction(
                M;
                expected_value = get_expectation(expectations, default_retraction_method),
                name = "default_retraction_method(M)",
            )
            Manifolds.Test.test_default_retraction(
                M, typeof(points[1]);
                expected_value = get_expectation(expectations, default_retraction_method),
                name = "default_retraction_method(M, P)",
            )
        end
        if (default_vector_transport_method in functions)
            Manifolds.Test.test_default_vector_transport_method(
                M;
                expected_value = get_expectation(expectations, default_vector_transport_method),
                name = "default_vector_transport_method(M)",
            )
            Manifolds.Test.test_default_vector_transport_method(
                M, typeof(points[1]);
                expected_value = get_expectation(expectations, default_vector_transport_method),
                name = "default_vector_transport_method(M, P)",
            )
        end
        if (distance in functions)
            Manifolds.Test.test_distance(
                M, points[1], points[2];
                available_functions = functions,
                expected_value = get_expectation(expectations, distance),
                name = "distance(M, p, q)", # shorten name within large suite
                atol = get(function_atols, distance, atol),
            )
        end
        if (embed in functions)
            ep = get_expectation(expectations, embed)
            ep = isexpected(ep) ? ep : get_expectation(expectations, (embed, :Point))
            Manifolds.Test.test_embed(
                M, points[1], n_vectors ≥ 1 ? vectors[1] : NoExpectation();
                available_functions = functions,
                expected_point = ep,
                expected_vector = get_expectation(expectations, (embed, :Vector)),
                test_aliased = aliased,
                test_mutating = (embed! in functions) ? true : mutating,
                name = "embed(M, p) & embed(M, p, X)", # shorten name within large suite
                atol = get(function_atols, embed, atol),
            )
        end
        if (embed_project in functions)
            approx_p = get(properties, :EmbeddedPoints, [points[1]])[1]
            approx_X = get(properties, :EmbeddedVectors, [vectors[1]])[1]
            Manifolds.Test.test_embed_project(
                M, approx_p, approx_X;
                available_functions = functions,
                expected_point = points[1],
                expected_vector = vectors[1],
                test_aliased = aliased,
                test_mutating = (embed_project! in functions) ? true : mutating,
                atol = get(function_atols, embed_project, atol),
                name = "embed_project(M, q) & embed_project(M, q, Y)", # shorten name within large suite
            )
        end
        if (exp in functions)
            Manifolds.Test.test_exp(
                M, points[1], vectors[1];
                available_functions = functions,
                expected_value = get_expectation(expectations, exp),
                test_aliased = aliased,
                test_mutating = (exp! in functions) ? true : mutating,
                atol = get(function_atols, exp, atol),
                name = "exp(M, p, X)", # shorten name within large suite
            )
        end
        if (flat in functions)
            expected_flat = get_expectation(expectations, flat)
            Manifolds.Test.test_flat(
                M, points[1], vectors[1];
                available_functions = functions,
                expected_value = expected_flat,
                name = "flat(M, p, X)", # shorten name within large suite
                atol = get(function_atols, flat, atol),
            )
        end
        if (get_basis in functions)
            for B in bases
                expected_basis = get_expectation(expectations, (get_basis, B))
                Manifolds.Test.test_get_basis(
                    M, points[1], B;
                    available_functions = functions,
                    expected_value = expected_basis,
                    name = "get_basis(M, p, $B)", # shorten name within large suite
                )
            end
        end
        if (get_coordinates in functions)
            expected_coordinates = get_expectation(expectations, get_coordinates)
            for B in bases
                expected_coordinates_B = get_expectation(expectations, (get_coordinates, B), expected_coordinates)
                Manifolds.Test.test_get_coordinates(
                    M, points[1], vectors[1], B;
                    available_functions = functions,
                    expected_value = expected_coordinates_B,
                    test_mutating = (get_coordinates! in functions) ? true : mutating,
                    name = "get_coordinates(M, p, X, $B)", # shorten name within large suite
                )
            end
        end
        if (get_embedding in functions)
            expected_embed = get_expectation(expectations, get_embedding)
            Manifolds.Test.test_get_embedding(
                # check the global one if this point type does not have an expected embedding
                M, missing;
                expected_value = expected_embed,
                name = "get_embedding(M)", # shorten name within large suite
            )
            if length(points) >= 1
                expected_embed_P = get_expectation(expectations, (get_embedding, typeof(points[1])))
                Manifolds.Test.test_get_embedding(
                    # check the global one if this point type does not have an expected embedding
                    M, typeof(points[1]);
                    expected_value = expected_embed,
                    expected_type = expected_embed_P,
                    name = "get_embedding(M, p)", # shorten name within large suite
                )
            end
        end
        if (get_vector in functions)
            for (c, B) in zip(coordinates, bases)
                expected_vector = get_expectation(expectations, (get_vector, c, B))
                Manifolds.Test.test_get_vector(
                    M, points[1], c, B;
                    available_functions = functions,
                    expected_value = expected_vector,
                    test_mutating = (get_vector! in functions) ? true : mutating,
                    name = "get_vector(M, p, c, $B)", # shorten name within large suite
                    atol = get(function_atols, get_vector, atol),
                )
            end
        end
        if (get_vectors in functions)
            for B in bases
                expected_vectors_symbol = get_expectation(expectations, (get_vectors, B), :default)
                Manifolds.Test.test_get_vectors(
                    M, points[1], (get_basis in functions) ? get_basis(M, points[1], B) : B;
                    available_functions = functions,
                    test_orthogonality = expected_vectors_symbol in (:Orthogonal, :Orthonormal),
                    test_normality = expected_vectors_symbol == :Orthonormal,
                    name = "get_vectors(M, p, $B)", # shorten name within large suite
                    atol = get(function_atols, get_vectors, atol),
                )
            end
        end
        if (geodesic in functions)
            expected_geod = get_expectation(expectations, geodesic)
            t = get(properties, :GeodesicMaxTime, 1.0)
            Manifolds.Test.test_geodesic(
                M, points[1], vectors[1], t;
                available_functions = functions,
                atol = get(function_atols, geodesic, atol),
                expected_value = expected_geod,
                N = get(properties, :GeodesicSamples, 100),
                name = "geodesic(M, p, X, $t)", # shorten name within large suite
            )
        end
        if (injectivity_radius in functions)
            expected = get_expectation(expectations, (injectivity_radius, points[1]))
            expected_global = get_expectation(expectations, injectivity_radius)
            Manifolds.Test.test_injectivity_radius(
                M, missing;
                expected_value = expected_global,
                name = "injectivity_radius(M, p)", # shorten name within large suite
            )
            Manifolds.Test.test_injectivity_radius(
                M, points[1];
                expected_value = expected,
                expected_global_value = expected_global,
                name = "injectivity_radius(M, p)", # shorten name within large suite
            )
            for rm in retraction_methods
                ismissing(rm) && continue
                expected_rm = get_expectation(expectations, (injectivity_radius, points[1], rm))
                expected_rm_global = get_expectation(expectations, (injectivity_radius, rm))
                Manifolds.Test.test_injectivity_radius(
                    M, points[1];
                    expected_value = expected_rm,
                    expected_global_value = expected_rm_global,
                    retraction_method = rm,
                    name = "injectivity_radius(M, p, $rm)", # shorten name within large suite
                )
            end

        end
        if (inner in functions)
            expected_inner = get_expectation(expectations, inner)
            Manifolds.Test.test_inner(
                M, points[1], vectors[1], vectors[2];
                available_functions = functions,
                expected_value = expected_inner,
                name = "inner(M, p, X, Y)", # shorten name within large suite
                atol = get(function_atols, inner, atol),
            )
        end
        if (inverse_retract in functions)
            for (irm, rm) in zip(inverse_retraction_methods, retraction_methods)
                ismissing(irm) && continue
                expected_inv_retract = get_expectation(expectations, (inverse_retract, irm))
                Manifolds.Test.test_inverse_retract(
                    M, points[1], points[2], irm;
                    available_functions = functions,
                    expected_value = expected_inv_retract,
                    retraction_method = rm,
                    test_mutating = (inverse_retract! in functions) ? true : mutating,
                    atol = get(function_atols, inverse_retract, atol),
                    name = "inverse_retract(M, p, q, $irm)", # shorten name within large suite
                )
            end
        end
        if (is_default_metric in functions)
            m = get_expectation(expectations, is_default_metric, NoExpectation())
            Manifolds.Test.test_is_default_metric(
                M, m;
                name = "is_default_metric(M$(isexpected(m) ? ", $(expect(m))" : ""))",
            )
        end
        if (is_flat in functions)
            Manifolds.Test.test_is_flat(
                M; expected_value = get_expectation(expectations, is_flat), name = "is_flat(M)",
            )
        end
        if (is_point in functions)
            qs = get(properties, :InvalidPoints, [])
            # lets not wrap these in expectations
            errs = get(expectations, :IsPointErrors, [])
            Manifolds.Test.test_is_point(
                M, points[1], qs...;
                errors = errs,
                name = "is_point on M for $(typeof(points[1])) points",
                test_warn = test_warn,
                test_info = test_info,
                atol = get(function_atols, is_point, atol),
            )
        end
        if (is_vector in functions)
            Ys = get(properties, :InvalidVectors, [])
            # lets not wrap these in expectations
            errs = get(expectations, :IsVectorErrors, [])
            Manifolds.Test.test_is_vector(
                M, points[1], vectors[1], Ys...;
                basepoint_error = get_expectation(expectations, :IsVectorBasepointError),
                errors = errs,
                name = "is_vector(M, p, X)",
                test_warn = test_warn,
                test_info = test_info,
                q = get(properties, :InvalidPoints, [NoExpectation()])[1],
                atol = get(function_atols, is_vector, atol),
            )
        end
        if (log in functions)
            expected_log = get_expectation(expectations, :log)
            Manifolds.Test.test_log(
                M, points[1], points[2];
                available_functions = functions,
                expected_value = expected_log,
                test_mutating = (log! in functions) ? true : mutating,
                atol = get(function_atols, log, atol),
                name = "log(M, p, q)", # shorten name within large suite
            )
        end
        if (manifold_dimension in functions)
            expected_dim = get_expectation(expectations, manifold_dimension)
            Manifolds.Test.test_manifold_dimension(
                M; expected_value = expected_dim, expected_type = get(result_types, manifold_dimension, Int),
                name = "manifold_dimension(M)",
            )
        end
        if (manifold_volume in functions)
            expected_vol = get_expectation(expectations, manifold_volume)
            Manifolds.Test.test_manifold_volume(
                M;
                expected_value = expected_vol,
                name = "manifold_volume(M)",
            )
        end
        if (mid_point in functions)
            expected_mid = get_expectation(expectations, mid_point)
            Manifolds.Test.test_mid_point(
                M, points[1], points[2];
                available_functions = functions,
                expected_value = expected_mid,
                test_aliased = aliased,
                test_mutating = (mid_point! in functions) ? true : mutating,
                test_symmetry = get(properties, :TestMidpointSymmetry, true),
                atol = get(function_atols, mid_point, atol),
                name = "mid_point(M, p, q)", # shorten name within large suite
            )
        end
        if (norm in functions)
            expected_norm = get_expectation(expectations, norm)
            Manifolds.Test.test_norm(
                M, points[1], vectors[1];
                available_functions = functions,
                expected_value = expected_norm,
                name = "norm(M, p, X)", # shorten name within large suite
                atol = get(function_atols, norm, atol),
            )
        end
        if (parallel_transport_to in functions)
            expected_pt = get_expectation(expectations, parallel_transport_to)
            expected_ptd = get_expectation(expectations, parallel_transport_direction)
            Manifolds.Test.test_parallel_transport(
                M, points[1], vectors[1], points[2];
                available_functions = functions,
                expected_value = expected_pt,
                expected_value_direction = expected_ptd,
                test_aliased = aliased,
                test_mutating = (parallel_transport_to! in functions) ? true : mutating,
                atol = get(function_atols, parallel_transport_to, atol),
                name = "parallel_transport_to(M, p, X, q)", # shorten name within large suite
            )
        end
        if (project in functions)
            Q = get(properties, :EmbeddedPoints, missing)
            q = ismissing(Q) ? missing : Q[1]
            Ys = get(properties, :EmbeddedVectors, missing)
            Y = ismissing(Ys) ? missing : Ys[1]
            ep = get_expectation(expectations, project)
            ep = isexpected(ep) ? ep : get_expectation(expectations, (project, :Point))
            eX = get_expectation(expectations, (project, :Vector))
            ismissing(q) && error("To test `project`, at least one `:EmbeddedPoints` must be provided.")
            Manifolds.Test.test_project(
                M, q, Y;
                available_functions = functions,
                expected_point = ep,
                expected_vector = eX,
                test_aliased = aliased,
                test_mutating = (project! in functions) ? true : mutating,
                atol = get(function_atols, project, atol),
                name = "project(M, q) & project(M, q, Y)", # shorten name within large suite
            )
        end
        if (rand in functions)
            rng = get(properties, :Rng, missing)
            seed = get(properties, :Seed, missing)
            Manifolds.Test.test_rand(
                M;
                seed = seed,
                rng = rng,
                vector_at = points[1],
                test_mutating = (rand! in functions) ? true : mutating,
                name = "rand(M)",
                atol = get(function_atols, rand, atol),
            )
        end
        if (repr in functions)
            expected_repr = get_expectation(expectations, repr)
            Manifolds.Test.test_repr(
                M;
                expected_value = expected_repr,
                name = "repr(M)",
            )
        end
        if (representation_size in functions)
            expected_repr_size = get_expectation(expectations, representation_size)
            Manifolds.Test.test_representation_size(
                M;
                expected_value = expected_repr_size,
                name = "representation_size(M)", # shorten name within large suite
            )
        end
        if (retract in functions)
            for (rm, irm) in zip(retraction_methods, inverse_retraction_methods)
                ismissing(rm) && continue
                expected_retract = get_expectation(expectations, (retract, rm))
                Manifolds.Test.test_retract(
                    M, points[1], vectors[1], rm;
                    available_functions = functions,
                    expected_value = expected_retract,
                    inverse_retraction_method = irm,
                    test_aliased = aliased,
                    test_mutating = (retract! in functions) ? true : mutating,
                    atol = get(function_atols, retract, atol),
                    name = "retract(M, p, X, $rm)", # shorten name within large suite
                )
            end
        end
        if (sectional_curvature in functions)
            expected_sec_curv = get_expectation(expectations, sectional_curvature)
            expected_sec_curv_min = get_expectation(expectations, sectional_curvature_min)
            expected_sec_curv_max = get_expectation(expectations, sectional_curvature_max)
            Manifolds.Test.test_sectional_curvature(
                M, points[1], vectors[1], vectors[2];
                available_functions = functions,
                expected_value = expected_sec_curv,
                expected_value_min = expected_sec_curv_min,
                expected_value_max = expected_sec_curv_max,
                name = "sectional_curvature(M, p, X, Y)", # shorten name within large suite
                atol = get(function_atols, sectional_curvature, atol),
            )
        end
        if (sharp in functions)
            expected_sharp = get_expectation(expectations, sharp)
            Manifolds.Test.test_sharp(
                M, points[1], covectors[1];
                available_functions = functions,
                expected_value = expected_sharp,
                name = "sharp(M, p, ξ)", # shorten name within large suite
                atol = get(function_atols, sharp, atol),
            )
        end
        if (shortest_geodesic in functions)
            expected_geod = get_expectation(expectations, shortest_geodesic)
            t = get(properties, :ShortestGeodesicTime, 1.0)
            Manifolds.Test.test_shortest_geodesic(
                M, points[1], points[2], t;
                available_functions = functions,
                atol = get(function_atols, shortest_geodesic, atol),
                expected_value = expected_geod,
                N = get(properties, :GeodesicSamples, 100),
                name = "shortest_geodesic(M, p, X, $t)", # shorten name within large suite
            )
        end
        if (vector_transport_to in functions)
            for vtm in vector_transport_methods
                expected_vt = get_expectation(expectations, (vector_transport_to, vtm))
                expected_vtd = get_expectation(expectations, (vector_transport_direction, vtm))
                Manifolds.Test.test_vector_transport(
                    M, points[1], vectors[1], points[2], vtm;
                    available_functions = functions,
                    expected_value = expected_vt,
                    expected_value_direction = expected_vtd,
                    test_aliased = aliased,
                    test_mutating = (vector_transport_to! in functions) ? true : mutating,
                    atol = get(function_atols, vector_transport_to, atol),
                    name = "vector_transport_to(M, p, X, q, $vtm)", # shorten name within large suite
                )
            end
        end
        if (volume_density in functions)
            expected_vol_density = get_expectation(expectations, volume_density)
            Manifolds.Test.test_volume_density(
                M, points[1], vectors[1];
                expected_value = expected_vol_density,
                name = "volume_density(M, p, X)", # shorten name within large suite
                atol = get(function_atols, volume_density, atol),
            )
        end
        if (Weingarten in functions)
            Manifolds.Test.test_Weingarten(
                M, points[1], vectors[1], normals[1];
                available_functions = functions,
                expected_value = get_expectation(expectations, Weingarten),
                test_mutating = (Weingarten! in functions) ? true : mutating,
                atol = get(function_atols, Weingarten, atol),
                name = "Weingarten(M, p, X, N)", # shorten name within large suite
            )
        end
        if (zero_vector in functions)
            Manifolds.Test.test_zero_vector(
                M, points[1];
                available_functions = functions,
                test_mutating = (zero_vector! in functions) ? true : mutating,
                atol = get(function_atols, zero_vector, atol),
                name = "zero_vector(M, p)", # shorten name within large suite
            )
        end
    end # end of test_manifold testset
    return t
end

# Single function tests
#
# ------------------------------------------------------------------------------------------
"""
    Manifolds.Test.test_copy(M, p, X;
        name = "copying on \$M for \$(typeof(p)) points",
        kwargs...
    )

Test `copy(M, p)` and `copy(M, p, X)` on a manifold `M`.

* that the copied point/vector is a valid point/vector on the manifold / tangent space
* that the copied point/vector matches the original point/vector but is new memory
"""
function Manifolds.Test.test_copy(
        M::AbstractManifold, p, X;
        name = "copying on $M for $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        q = copy(M, p)
        Test.@test is_point(M, q; error = :error, kwargs...)
        Test.@test q == p
        Test.@test q !== p
        Y = copy(M, p, X)
        Test.@test is_vector(M, p, Y; error = :error, kwargs...)
        Test.@test Y == X
        Test.@test Y !== X

    end
    return nothing
end # end of Manifolds.Test.test_copy

"""
    Manifolds.Test.test_copyto(M, p, X;
        name = "copying on \$M for \$(typeof(p)) points",
        kwargs...
    )

Test `copyto!(M, q, p)` and `copyto!(M, Y, p, X)` on a manifold `M`.

* that the copied point/vector is a valid point/vector on the manifold / tangent space
* that the copied point/vector matches the original point/vector and is the same memory

Note that since this function does not modify its input, is is called `test_copyto`.
"""
function Manifolds.Test.test_copyto(
        M::AbstractManifold, p, X;
        name = "copyto! on $M for $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        # Allocate memory for copyto!
        q = allocate_result(M, exp, p, X)
        q2 = copyto!(M, q, p)
        Test.@test q == p
        Test.@test q2 === q
        Y = allocate_result(M, exp, p, X)
        Y2 = copyto!(M, Y, p, X)
        Test.@test Y == X
        Test.@test Y2 === Y
    end
    return nothing
end

"""
    Manifolds.Test.test_default_inverse_retraction_method(
        M, T=missing;
        expected_value = NoExpectation(),
        expected_type = !isexpected(expected_value) ? NoExpectation() : typeof(expected_value),
        name = "default_inverse_retraction_method on \$M \$(ismissing(T) ? "" : "for type \$T")",
    )

Test the [`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`) on manifold `M`.
* that it returns an [`AbstractInverseRetractionMethod`](@extref `ManifoldsBase.AbstractInverseRetractionMethod`)
* that the result matches `expected_value`, if given
* that the result is of type `expected_type`, if given, defaults to the type of the value
"""
function Manifolds.Test.test_default_inverse_retraction(
        M::AbstractManifold, T = missing;
        expected_value = NoExpectation(),
        expected_type = !isexpected(expected_value) ? NoExpectation() : Expect(typeof(expect(expected_value))),
        name = "default_inverse_retraction_method on $M $(ismissing(T) ? "" : "for type $T")",
    )
    Test.@testset "$(name)" begin
        m = ismissing(T) ? default_inverse_retraction_method(M) : default_inverse_retraction_method(M, T)
        Test.@test m isa AbstractInverseRetractionMethod
        !isexpected(expected_value) || Test.@test m == expect(expected_value)
        !isexpected(expected_type) || Test.@test m isa expect(expected_type)
    end
    return nothing
end # Manifolds.Test.test_default_inverse_retraction

"""
    Manifolds.Test.test_default_retraction_method(
        M, T=missing;
        expected_value = NoExpectation(),
        expected_type = isexpected(expected_value) ? Expect(typeof(expect(expected_value))) : NoExpectation(),
        name = "default_retraction_method on \$M \$(ismissing(T) ? "" : "for type \$T")",
    )

Test the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) on manifold `M`.
* that it returns an [`AbstractRetractionMethod`](@extref `ManifoldsBase.AbstractRetractionMethod`)
* that the result matches `expected_value`, if given
* that the result is of type `expected_type`, if given, defaults to the type of the value
"""
function Manifolds.Test.test_default_retraction(
        M::AbstractManifold, T = missing;
        expected_value = NoExpectation(),
        expected_type = isexpected(expected_value) ? Expect(typeof(expect(expected_value))) : NoExpectation(),
        name = "default_retraction_method on $M $(ismissing(T) ? "" : "for type $T")",
    )
    Test.@testset "$(name)" begin
        m = ismissing(T) ? default_retraction_method(M) : default_retraction_method(M, T)
        Test.@test m isa AbstractRetractionMethod
        ismissing(expected_value) || Test.@test m == expect(expected_value)
        ismissing(expected_type) || Test.@test m isa expect(expected_type)
    end
    return nothing
end # Manifolds.Test.test_default_retraction

"""
    Manifolds.Test.test_default_vector_transport_method(
        M, T=missing;
        expected_value = NoExpectation(),
        expected_type = isexpected(expected_value) ? Expect(typeof(expect(expected_value))) : NoExpectation(),
        name = "default_vector_transport_method on \$M \$(ismissing(T) ? "" : "for type \$T")",
    )

Test the [`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`) on manifold `M`.
* that it returns an [`AbstractVectorTransportMethod`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`)
* that the result matches `expected_value`, if given
* that the result is of type `expected_type`, if given, defaults to the type of the value
"""
function Manifolds.Test.test_default_vector_transport_method(
        M::AbstractManifold, T = missing;
        expected_value = NoExpectation(),
        expected_type = isexpected(expected_value) ? Expect(typeof(expect(expected_value))) : NoExpectation(),
        name = "default_vector_transport_method on $M $(ismissing(T) ? "" : "for type $T")",
    )
    Test.@testset "$(name)" begin
        m = ismissing(T) ? default_vector_transport_method(M) : default_vector_transport_method(M, T)
        Test.@test m isa AbstractVectorTransportMethod
        ismissing(expected_value) || Test.@test m == expect(expected_value)
        ismissing(expected_type) || Test.@test m isa expect(expected_type)
    end
    return nothing
end # Manifolds.Test.test_default_vector_transport

"""
    Manifolds.Test.test_distance(
        M, p, q;
        available_functions=[], expected_value=NoExpectation(),
        name = "Distance on \$M between \$(typeof(p)) points",
        kwargs...
    )

Test the distance function on manifold `M` between points `p` and `q`.

* that the result is a nonnegative number
* that the distance from `p` to `p` is zero
* that the distance is symmetric
* that the result matches `expected_value`, if given
* that the distance is equal to the norm of the logarithmic map (if `log` and `norm` are available)
    (only performed if either `injectivity_radius` is not available or the points are within)
"""
function Manifolds.Test.test_distance(
        M::AbstractManifold, p, q;
        available_functions = Function[], expected_value = NoExpectation(),
        name = "Distance on $M between $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        d = distance(M, p, q)
        Test.@test d ≥ 0.0
        d_pp = distance(M, p, p)
        Test.@test isapprox(d_pp, 0.0; kwargs...)
        d_qp = distance(M, q, p)
        Test.@test isapprox(d, d_qp; kwargs...)
        !isexpected(expected_value) || Test.@test isapprox(d, expect(expected_value); kwargs...)
        if (log in available_functions) && (norm in available_functions)
            # Test only if inj is not available of points are within inj radius
            run_test = !((injectivity_radius in available_functions)) || (d ≤ injectivity_radius(M, p))
            run_test || (@warn("Skipping distance-norm-log test since norm of X ($(norm(M, p, X))) is outside injectivity radius ($(injectivity_radius(M, p)))"))
            Y = log(M, p, q)
            n = norm(M, p, Y)
            Test.@test isapprox(d, n; kwargs...) skip = !run_test
        end
    end
    return nothing
end # Manifolds.Test.test_distance

"""
    Manifolds.Test.test_embed(
        M, p, X=missing;
        available_functions=[],
        expected_point=NoExpectation(),
        expected_vector=NoExpectation(),
        test_aliased=true,
        test_mutating=true,
        name = "Embedding on \$M for \$(typeof(p)) points",
        kwargs...
    )

Test the [`embed`](@extref `ManifoldsBase.embed-Tuple{AbstractManifold, Any}`)`(M, p)` and [`embed`](@extref `ManifoldsBase.embed-Tuple{AbstractManifold, Any, Any}`)`(M, p, X)`
to embed points and tangent vectors (if not `missing`).

Besides a simple call of `embed` (for both variants) the following ones are prefoemd if [`get_embedding`](@extref `ManifoldsBase.get_embedding-Tuple{AbstractManifold}`) is available:

* that the embedded point is a valid point on the embedding manifold
* that the embedded vector is a valid tangent vector on the embedding manifold (if [`get_embedding`](@extref `ManifoldsBase.get_embedding-Tuple{AbstractManifold}`) is available and `X` is not `missing`)
* that the result matches `expected_point` and `expected_vector`, respectively, if given
* that the projection inverts the embedding (if `project` is available)
* that the mutating version `embed!` produces the same result(s) (if activated _and_ [`get_embedding`](@extref `ManifoldsBase.get_embedding-Tuple{AbstractManifold}`) is available)
* that `embed!` works on aliased input (`p=q` or `X=Y`) (if activated _and_ p/q or X/Y are of same type)
"""
function Manifolds.Test.test_embed(
        M::AbstractManifold, p, X = missing;
        available_functions = Function[],
        expected_point = NoExpectation(),
        expected_vector = NoExpectation(),
        test_aliased = true,
        test_mutating = true,
        name = "Embedding on $M for $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        E = get_embedding(M) isa AbstractManifold ? get_embedding(M) : missing
        # Test point embedding
        q = embed(M, p)
        if !ismissing(E)
            Test.@test is_point(E, q; error = :error, kwargs...)
            !isexpected(expected_point) || Test.@test isapprox(E, q, expect(expected_point); error = :error, kwargs...)
            if project in available_functions
                p2 = project(M, q)
                Test.@test isapprox(M, p, p2; error = :error, kwargs...)
            end
            if test_mutating
                q2 = copy(E, q)
                embed!(M, q2, p)
                Test.@test isapprox(E, q2, q; error = :error, kwargs...)
                if test_aliased && (typeof(p) == typeof(q))
                    q3 = copy(E, p)
                    embed!(M, q3, q3)  # aliased
                    Test.@test isapprox(E, q3, q; error = :error, kwargs...)
                end
                if project in available_functions
                    p3 = copy(M, p)
                    project!(M, p3, q2)
                    Test.@test isapprox(M, p, p3; error = :error, kwargs...)
                end
            end
        end
        # Test vector embedding
        if !ismissing(X)
            Y = embed(M, p, X)
            if !ismissing(E)
                Test.@test is_vector(E, q, Y; error = :error, kwargs...)
                !isexpected(expected_vector) || Test.@test isapprox(E, q, Y, expect(expected_vector); error = :error, kwargs...)
                if project in available_functions
                    X2 = project(M, q, Y)
                    Test.@test isapprox(M, p, X, X2; error = :error, kwargs...)
                end
                if test_mutating
                    Y2 = copy(E, q, Y)
                    embed!(M, Y2, p, X)
                    Test.@test isapprox(E, q, Y2, Y; error = :error, kwargs...)
                    if test_aliased
                        Y3 = copy(E, q, Y)
                        embed!(M, Y3, q, Y3)  # aliased
                        Test.@test isapprox(E, q, Y3, Y; error = :error, kwargs...)
                    end
                    if project in available_functions
                        X3 = copy(M, p, X)
                        project!(M, X3, q, Y2)
                        Test.@test isapprox(M, p, X, X3; error = :error, kwargs...)
                    end
                end
            end
        end
    end
    return nothing
end # Manifolds.Test.test_embed

"""
    Manifolds.Test.test_embed_project(
        M, ap, aX = missing;
        available_functions=[],
        expected_point=NoExpectation(),
        expected_vector=NoExpectation(),
        test_aliased=true,
        test_mutating=true,
        name = "Projection on \$M for \$(typeof(q)) points",
        kwargs...
    )

Test the `p=`[`embed_project`](@extref `ManifoldsBase.embed_project-Tuple{AbstractManifold, Any}`)`(M, ap)` and [`embed_project`](@extref `ManifoldsBase.embed_project-Tuple{AbstractManifold, Any, Any}`)`(M, p, aX)`
to project points and tangent vectors (if not `aX` is not `missing`) after embedding them.
Besides a simple call of `embed_project` (for both variants)  the following tests are performed

* that the projected point is a valid point on the manifold
* that the projected vector is a valid tangent vector on the manifold
* that the result matches `expected_point` and `expected_vector`, respectively, if given
* that the mutating version `embed_project!` produces the same result(s) (if activated)
* that `embed_project!` works on aliased input (`p=q` or `X=Y`) (if activated _and_ p/q or X/Y are of same type)
"""
function Manifolds.Test.test_embed_project(
        M::AbstractManifold, ap, aX = missing;
        available_functions = Function[],
        expected_point = NoExpectation(),
        expected_vector = NoExpectation(),
        test_aliased = true,
        test_mutating = true,
        name = "Embed-then-project on $M for $(typeof(ap)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        # Test point projection
        p = embed_project(M, ap)
        Test.@test is_point(M, p; error = :error, kwargs...)
        !isexpected(expected_point) || Test.@test isapprox(M, p, expect(expected_point); error = :error, kwargs...)
        if test_mutating
            p2 = copy(M, p)
            embed_project!(M, p2, ap)
            Test.@test isapprox(M, p2, p; error = :error, kwargs...)
            if test_aliased && (typeof(p) == typeof(ap))
                p3 = copy(M, ap)
                embed_project!(M, p3, p3)  # aliased
                Test.@test isapprox(M, p3, p; error = :error, kwargs...)
            end
        end
        # Test vector projection
        if !ismissing(aX)
            X = project(M, p, aX)
            Test.@test is_vector(M, p, X; error = :error, kwargs...)
            !isexpected(expected_vector) || Test.@test isapprox(M, p, X, expect(expected_vector); error = :error, kwargs...)
            if test_mutating
                X2 = copy(M, p, aX)
                project!(M, X2, p, aX)
                Test.@test isapprox(M, p, X2, X; error = :error, kwargs...)
                if test_aliased
                    X3 = copy(M, p, aX)
                    project!(M, X3, p, X3)  # aliased
                    Test.@test isapprox(M, p, X3, X; error = :error, kwargs...)
                end
            end
        end
    end
    return nothing
end # Manifolds.Test.test_embed_project

"""
    Manifolds.Test.test_exp(
        M, p, X, t=1.0;
        available_functions=[], expected_value=NoExpectation(), test_mutating=true,
        test_log = (log in available_functions),
        test_fused = true,
        test_injectivity_radius = (injectivity_radius in available_functions),
        name = "Exponential map on \$M for \$(typeof(p)) points",
        kwargs...
    )

Test the exponential map on manifold `M` at point `p` with tangent vector `X`.

* that the result is a valid point on the manifold
* that the result matches `expected_value`, if given
* that the mutating version `exp!` matches the non-mutating version, (if activated)
* that `exp!` works on aliased in put (`p=q`) (if activated for mutating)
* that the logarithmic map inverts the exponential map (if activated)
(only performed if either `injectivity_radius` is not available or `X` is within)
* that the fused version `exp_fused(M, p, t, X)` matches the non-fused version (if activated)
"""
function Manifolds.Test.test_exp(
        M::AbstractManifold, p, X, t = 1.0;
        available_functions = Function[],
        expected_value = NoExpectation(),
        test_aliased = true,
        test_fused = true,
        test_log = (log in available_functions),
        test_injectivity_radius = (injectivity_radius in available_functions),
        test_mutating = true,
        name = "Exponential map on $M for $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        q = exp(M, p, X)
        Test.@test is_point(M, q; error = :error, kwargs...)
        !isexpected(expected_value) || Test.@test isapprox(M, q, expect(expected_value); error = :error, kwargs...)
        if test_mutating
            q2 = copy(M, p)
            exp!(M, q2, p, X)
            Test.@test isapprox(M, q2, q; error = :error, kwargs...)
            if test_aliased
                q3 = copy(M, p)
                exp!(M, q3, q3, X)  # aliased
                Test.@test isapprox(M, q3, q; error = :error, kwargs...)
            end
        end
        if test_fused
            q4 = Manifolds.exp_fused(M, p, X, t)
            q5 = exp(M, p, t * X)
            Test.@test isapprox(M, q4, q5; error = :error, kwargs...)
            if test_mutating
                q6 = copy(M, p)
                Manifolds.exp_fused!(M, q6, p, X, t)
                Test.@test isapprox(M, q6, q5; error = :error, kwargs...)
                if test_aliased
                    q7 = copy(M, p)
                    Manifolds.exp_fused!(M, q7, q7, X, t)  # aliased
                    Test.@test isapprox(M, q7, q5; error = :error, kwargs...)
                end
            end
        end
        if test_log
            # Test only if inj is not available of X is within inj radius
            run_test = !test_injectivity_radius || norm(M, p, X) ≤ injectivity_radius(M, p)
            run_test || (@warn("Skipping log-exp test since norm of X ($(norm(M, p, X))) is outside injectivity radius ($(injectivity_radius(M, p)))"))
            Y = log(M, p, q)
            Test.@test isapprox(M, X, Y; error = :error, kwargs...) skip = !run_test
        end
    end
    return nothing
end # Manifolds.Test.test_exp

"""
    Manifolds.Test.test_flat(M, p, X;
        available_functions = [],
        expected_value= NoExpectation(),
        name = "Flat on \$M for \$(typeof(p)) points",
        kwargs...
    )

Test the flat operation on manifold `M` at point `p` with tangent vector `X`.
* (we can not yet test valid cotangent vectors)
* that the result matches `expected_value`, if given
* that [`sharp`](@ref) is the inverse
* that mutating version `flat!` matches non-mutating version (if activated)
"""
function Manifolds.Test.test_flat(
        M, p, X;
        available_functions = Function[],
        expected_value = NoExpectation(),
        test_mutating = true,
        name = "Flat on $M for $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        ξ = flat(M, p, X)
        !isexpected(expected_value) || Test.@test isapprox(M, ξ, expect(expected_value); error = :error, kwargs...)
        if (sharp in available_functions)
            X2 = sharp(M, p, ξ)
            Test.@test isapprox(M, X, X2; error = :error, kwargs...)
        end
        if test_mutating
            ξ2 = copy(M, p, ξ) # Improve once cotangents are first class citizens
            flat!(M, ξ2, p, X)
            Test.@test isapprox(M, ξ2, ξ; error = :error, kwargs...)
        end
    end
    return nothing
end

"""
    Manifolds.Test.test_geodesic(M, p, X, t=1.0;
        available_functions=[],
        expected_value = NoExpectation(),
        N = 10,
        name = "Geodesic on \$M for \$(typeof(p)) points",
        kwargs...
    )

Test the geodesic on manifold `M` at point `p` with tangent vector `X` at time `t`.
* that at time `0` the geodesic returns `p`
* that the function `γ = geodesic(M, p, X)` is consistent with evaluation at `0` and `t``
* that the result is a valid point on the manifold
* that the result matches `expected_value`, if given
* that the geodesic has constant speed (if activated) using `N` samples and each of the
  segments is of length equal to the average speed, i.e. `t*norm(M, p, X) / (N-1)`
"""
function Manifolds.Test.test_geodesic(
        M::AbstractManifold, p, X, t = 1.0;
        available_functions = Function[],
        expected_value = NoExpectation(),
        N = 10,
        name = "Geodesic on $M for $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        q = geodesic(M, p, X, t)
        Test.@test is_point(M, q; error = :error, kwargs...)
        !isexpected(expected_value) || Test.@test isapprox(M, q, expect(expected_value); error = :error, kwargs...)
        p0 = geodesic(M, p, X, 0.0)
        Test.@test isapprox(M, p0, p; error = :error, kwargs...)
        γ = geodesic(M, p, X)
        qt = γ(t)
        Test.@test isapprox(M, qt, q; error = :error, kwargs...)
        # Since this test might exit early, it should always be the last test of this function
        if N > 0
            ts = range(0.0, t; length = N)
            points = [geodesic(M, p, X, ti) for ti in ts]
            if distance in available_functions
                dists = [distance(M, points[i], points[i + 1]) for i in 1:(length(points) - 1)]
                speeds = [d / (ts[i + 1] - ts[i]) for (i, d) in enumerate(dists)]
                avg_speed = sum(speeds) / length(speeds)
                for s in speeds
                    Test.@test isapprox(s, avg_speed; kwargs...)
                end
                Test.@test isapprox(avg_speed, t * norm(M, p, X); kwargs...)
            end
        end
    end
    return nothing
end # Manifolds.Test.test_geodesic

"""
    Manifolds.Test.test_get_coordinates(
        M, p, X, B;
        available_functions = [],
        expected_value = NoExpectation(),
        name = "get_coordinates on \$M at point \$(typeof(p))",
        test_mutating = true,
    )

Test the [`get_coordinates`](@extref `ManifoldsBase.get_coordinates`) on manifold `M` at point `p`
for a tangent vector `X` and a basis `B`.

* that the result is a valid coordinate vector, that is of correct length
* that the result matches `expected_value`, if given
* that the mutating version `get_coordinates!` matches the non-mutating version, (if activated)
* that [`get_vector`](@extref `ManifoldsBase.get_vector`) inverts `get_coordinates` (if activated)
"""
function Manifolds.Test.test_get_coordinates(
        M::AbstractManifold, p, X, B;
        available_functions = Function[],
        expected_value = NoExpectation(),
        test_mutating = true,
        name = "get_coordinates on $M at point $(typeof(p))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        c = get_coordinates(M, p, X, B)
        # TODO: Improve for comlex manifolds and check for real of complex coordinates
        Test.@test length(c) == manifold_dimension(M)
        !isexpected(expected_value) || Test.@test isapprox(c, expect(expected_value); kwargs...)
        if test_mutating
            c2 = similar(c)
            get_coordinates!(M, c2, p, X, B)
            Test.@test isapprox(c2, c; kwargs...)
        end
        if (get_vector in available_functions)
            X2 = get_vector(M, p, c, B)
            Test.@test isapprox(M, p, X2, X; kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_get_coordinates

"""
    Manifolds.Test.test_get_basis(
        M, p, b::AbstractBasis;
        expected_value = NoExpectation(),
        expected_type = isexpected(expected_value) ? Expect(typeof(expect(expected_value))) : Expect(CachedBasis),
        name = "get_basis on \$M at point \$(typeof(p)) for basis \$(typeof(b))",
    )

Test the [`get_basis`](@extref `ManifoldsBase.get_basis`) on manifold `M` at point `p` for basis `b`.

* that it returns a basis of type `expected_type`, which defaults to [`CachedBasis`](@extref ManifoldsBase.CachedBasis).
* that the result matches `expected_value`, if given
"""
function Manifolds.Test.test_get_basis(
        M::AbstractManifold, p, b::AbstractBasis;
        available_functions = Function[],
        expected_value = NoExpectation(),
        expected_type = isexpected(expected_value) ? Expect(typeof(expect(expected_value))) : Expect(CachedBasis),
        name = "get_basis on $M at point $(typeof(p)) for basis $(typeof(b))",
    )
    Test.@testset "$(name)" begin
        B = get_basis(M, p, b)
        Test.@test B isa AbstractBasis
        !isexpected(expected_type) || Test.@test B isa expect(expected_type)
        !isexpected(expected_value) || Test.@test B == expect(expected_value)
    end
    return nothing
end # Manifolds.Test.test_get_basis

"""
    Manifolds.Test.test_get_embedding(M, P=missing;
        expected_value = NoExpectation(),
        expected_type = isexpected(expected_value) ? Expect(typeof(expect(expected_value))) : NoExpectation(),
        name = "get_embedding on \$M \$(ismissing(P) ? "" : "for type \$P")",
    )

Test the [`get_embedding`](@extref `ManifoldsBase.get_embedding-Tuple{AbstractManifold}`) on manifold `M`.

* that it returns an `AbstractManifold`.
* that its type matches `expected_type`, if given, defaults to the type of the expected value
* that the result matches `expected_value`, if given
"""
function Manifolds.Test.test_get_embedding(
        M::AbstractManifold, P::Union{Type, Missing} = missing;
        expected_value = NoExpectation(),
        expected_type = isexpected(expected_value) ? Expect(typeof(expect(expected_value))) : NoExpectation(),
        name = "get_embedding on $M $(ismissing(P) ? "" : "for type $P")",
    )
    Test.@testset "$(name)" begin
        E = ismissing(P) ? get_embedding(M) : get_embedding(M, P)
        Test.@test E isa AbstractManifold
        !isexpected(expected_type) || Test.@test E isa expect(expected_type)
        !isexpected(expected_value) || Test.@test E == expect(expected_value)
    end
    return nothing
end # Manifolds.Test.test_get_embedding

"""
    Manifolds.Test.test_get_vector(
        M, p, c, B;
        available_functions=[],
        expected_value = NoExpectation(),
        name = "get_vector on \$M at point \$(typeof(p))",
        test_mutating = true,
        kwargs...
    )

Test the [`get_vector`](@extref `ManifoldsBase.get_vector`) on manifold `M` at point `p`
for a vector of coordinates `c` in basis `B`.

* that the result is a valid tangent vector at `p` on the manifold
* that the result matches `expected_value`, if given
* that the mutating version `get_vector!` matches the non-mutating version, (if activated)
* that [`get_coordinates`](@extref `ManifoldsBase.get_coordinates`) inverts `get_vector` (if activated)
"""
function Manifolds.Test.test_get_vector(
        M::AbstractManifold, p, c, B;
        available_functions = Function[],
        expected_value = NoExpectation(),
        test_mutating = true,
        name = "get_vector on $M at point $(typeof(p))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        X = get_vector(M, p, c, B)
        Test.@test is_vector(M, p, X; error = :error)
        !isexpected(expected_value) || Test.@test isapprox(M, p, X, expect(expected_value); error = :error)
        if test_mutating
            Y = copy(M, p, X)
            get_vector!(M, Y, p, c, B)
            Test.@test isapprox(M, p, Y, X; error = :error, kwargs...)
        end
        if (get_coordinates in available_functions)
            c2 = get_coordinates(M, p, X, B)
            Test.@test isapprox(c2, c; kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_get_vector

"""
    Manifolds.Test.test__get_vectors(
        M, p, b;
        available_functions=[],
        name = "get_vectors on \$M at point \$(typeof(p))",
        test_orthogonality = false,
        test_normality = false,
        kwargs...
    )

Test the [`get_vectors`](@extref `ManifoldsBase.get_vectors`) on manifold `M` at point `p` for basis `b`,
where the basis is assumed to come from a call to [`get_basis`](@extref `ManifoldsBase.get_basis`).

* that there are as many vectors as the manifold dimension (if available)
* that each vector is a valid tangent vector at `p` on the manifold
* that the vectors are orthogonal (if activated)
* that the vectors are normal (if activated)
"""
function Manifolds.Test.test_get_vectors(
        M::AbstractManifold, p, b::AbstractBasis;
        available_functions = Function[],
        test_orthogonality = false,
        test_normality = false,
        name = "get_vectors on $M at point $(typeof(p)) and $(typeof(b))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        Vs = get_vectors(M, p, b)
        n = length(Vs)
        if manifold_dimension in available_functions
            md = manifold_dimension(M)
            Test.@test n == md
        end
        for i in 1:n
            Test.@test is_vector(M, p, Vs[i]; error = :error, kwargs...)
            if test_orthogonality
                for j in (n + 1):n
                    ip = inner(M, p, Vs[i], Vs[j])
                    Test.@test isapprox(ip, 0.0; kwargs...)
                end
            end
            if test_normality
                nn = norm(M, p, Vs[i])
                Test.@test isapprox(nn, 1.0; kwargs...)
            end
        end
    end
    return nothing
end # Manifolds.Test.test_get_vectors

"""
    Manifolds.Test.test_injectority_radius(M, p = missing;
        expected_value = NoExpectation(),
        expected_global_value = NoExpectation(),
        retraction_method = missing,
        name = "Injectivity radius on \$M at point \$(ismissing(p) ? "" : "\$typeof(p)")) and \$(retraction_method)",
        kwargs...
    )

Test the injectivity radius on manifold `M` at point `p`.

* that the result is a nonnegative real number
* that the result matches `expected_value`, if given
* if a point `p` is given, that the result is larger or equal to the global injectivity radius
"""
function Manifolds.Test.test_injectivity_radius(
        M::AbstractManifold, p = missing;
        expected_value = NoExpectation(),
        expected_global_value = NoExpectation(),
        name = "Injectivity radius on $M at point $(ismissing(p) ? "" : "$(typeof(p))")",
        retraction_method = missing,
        kwargs...
    )
    Test.@testset "$(name)" begin
        r = if ismissing(p)
            ismissing(retraction_method) ? injectivity_radius(M) : injectivity_radius(M, retraction_method)
        else
            ismissing(retraction_method) ? injectivity_radius(M, p) : injectivity_radius(M, p, retraction_method)
        end
        Test.@test r ≥ 0.0
        !isexpected(expected_value) || Test.@test isapprox(r, expect(expected_value); kwargs...)
        if !ismissing(p)
            r_global = ismissing(retraction_method) ? injectivity_radius(M) : injectivity_radius(M, retraction_method)
            Test.@test r ≥ r_global
            !isexpected(expected_global_value) || Test.@test isapprox(r_global, expect(expected_global_value); kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_injectivity_radius

"""
    Manifolds.Test.test_inner(M, p, X, Y;
        available_functions=[], expected_value = NoExpectation(),
        name = "Inner product on \$M at point \$(typeof(p))",
        test_norm = (norm in available_functions),
        kwargs...
    )

Test the inner product on the manifold `M` at point `p` for tangent vectors `X` and `Y`.

* that the result is a real number
* that the result matches `expected_value`, if given
* that the inner product of `X` with itself is non negative
* that the inner product of `X` with itself is the same as its norm squared (if activated)
"""
function Manifolds.Test.test_inner(
        M::AbstractManifold, p, X, Y;
        available_functions = Function[],
        expected_value = NoExpectation(),
        test_norm = (norm in available_functions),
        name = "Inner product on $M at point $(typeof(p))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        v = inner(M, p, X, Y)
        Test.@test v isa (Real)
        !isexpected(expected_value) || Test.@test isapprox(v, expect(expected_value); kwargs...)
        w = inner(M, p, X, X)
        Test.@test w ≥ 0.0
        if test_norm
            n = norm(M, p, X)
            Test.@test isapprox(w, n^2; kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_inner

"""
    Manifolds.Test.test_inverse_retraction(
        M, p, q, m::AbstractInverseRetractionMethod;
        available_functions=[],
        expected_value = NoExpectation(),
        name = "Inverse retraction \$m on \$M at point \$(typeof(p))",
        retraction_method = missing,
        test_mutating = true,
        test_retraction = (retract in available_functions) && !ismissing(retraction_method),
        kwargs...
    )

Test the inverse retraction method `m` on manifold `M` at point `p` towards point `q`.

* that the result is a valid tangent vector at `p` on the manifold
* that the result matches `expected_value`, if given
* that the mutating version `inverse_retract!` matches the non-mutating version, (if activated)
* that the retraction inverts the inverse retraction (if activated)
"""
function Manifolds.Test.test_inverse_retract(
        M, p, q, m::AbstractInverseRetractionMethod;
        available_functions = Function[],
        expected_value = NoExpectation(),
        name = "Inverse retraction $m on $M at point $(typeof(p))",
        retraction_method = missing,
        test_retraction = (retract in available_functions) && !ismissing(retraction_method),
        test_mutating = true,
        kwargs...
    )
    Test.@testset "$(name)" begin
        X = inverse_retract(M, p, q, m)
        Test.@test is_vector(M, p, X; error = :error, kwargs...)
        !isexpected(expected_value) || Test.@test isapprox(M, p, X, expect(expected_value); error = :error, kwargs...)
        if test_mutating
            Y = copy(M, p, X)
            inverse_retract!(M, Y, p, q, m)
            Test.@test isapprox(M, p, Y, X; error = :error, kwargs...)
        end
        if test_retraction
            q2 = retract(M, p, X, retraction_method)
            Test.@test isapprox(M, q2, q; error = :error, kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_inverse_retraction

"""
    Manifolds.Test.test_is_default_metric(M, metric::AbstractMetric;
    name = "is_default_metric on \$M",
    )

Test the [`is_default_metric`](@ref) on manifold `M`.

* that it returns true for the given `metric`
* that it returns false for all other provided metrics
"""
function Manifolds.Test.test_is_default_metric(
        M::AbstractManifold, metric::Union{AbstractMetric, AbstractExpectation};
        name = "is_default_metric on $M",
    )
    Test.@testset "$(name)" begin
        !isexpected(metric) || Test.@test is_default_metric(M, expect(metric))
    end
    return nothing
end # Manifolds.Test.test_default_metric

"""
    Manifolds.Test.test_is_flat(
    M;
    expected_value = NoExpectation(),
    name = "is_flat on \$M",
    )

Test the function [`is_flat`](@ref) on manifold `M`. Since it returns a boolean,
there is also only the check that it agrees with the `expected_value`, if given.
"""
function Manifolds.Test.test_is_flat(
        M::AbstractManifold;
        expected_value = NoExpectation(),
        name = "is_flat on $M",
    )
    Test.@testset "$(name)" begin
        v = is_flat(M)
        !isexpected(expected_value) || Test.@test v == expect(expected_value)
    end
    return nothing
end

"""
    Manifolds.Test.is_point(
        M, p qs...;
        errors = [],
        name = "is_point on \$M for \$(typeof(p)) points",
        test_warn = true,
        test_info = true,
        kwargs...
    )

Test the function [`is_point`](@ref) on  for point `p` on manifold `M`.

* that for `p` it returns `true`.
* that for each `q` in `qs` it
  * returns `false`
  * issues a warning (if activated)
  * isues an info message (if activated)
  * throws the corresponding error from `error_types` (if not `missing`)
"""
function Manifolds.Test.test_is_point(
        M::AbstractManifold, p, qs...;
        errors = [],
        name = "is_point on $M for $(typeof(p)) points",
        test_warn = true,
        test_info = true,
        kwargs...
    )
    Test.@testset "$(name)" begin
        Test.@test is_point(M, p; kwargs...)
        for (i, q) in enumerate(qs)
            Test.@test !is_point(M, q; kwargs...)
            (test_warn) && Test.@test_logs (:warn,) is_point(M, q; error = :warn, kwargs...)
            (test_info) && Test.@test_logs (:info,) is_point(M, q; error = :info, kwargs...)
            if length(errors) >= i && !ismissing(errors[i])
                Test.@test_throws (errors[i]) is_point(M, q; error = :error, kwargs...)
            end
        end
    end
    return nothing
end

"""
    Manifolds.Test.test_is_vector(
        M, p, X, Ys...;
        basepoint_error = missing,
        check_basepoint = true,
        errors = [],
        name = "is_vector on \$M for \$(typeof(p)) points",
        test_warn = true,
        test_info = true,
        q = missing,
        kwargs...
    )

Test the function [`is_vector`](@ref) on manifold `M` at point `p` for tangent vector `X`.

* that for `X` it returns `true`.
* that for each `Y` in `Ys` it
    * returns `false`
    * issues a warning (if activated)
    * isues an info message (if activated)
    * throws the corresponding error from `error_types` (if not `missing`)
* if `check_basepoint` is `true`, then it checks that
    * for `p` this still returns `true`
    * for the base point `q` it
      * returns `false`
      * issues a warning (if activated)
      * isues an info message (if activated)
      * throws the corresponding error from `error_basepoint` (if activated)
"""
function Manifolds.Test.test_is_vector(
        M::AbstractManifold, p, X, Ys...;
        check_basepoint = true,
        errors = [],
        basepoint_error = missing,
        name = "is_vector on $M for $(typeof(p)) points",
        test_warn = true,
        test_info = true,
        q = missing,
        kwargs...
    )
    Test.@testset "$(name)" begin
        Test.@test is_vector(M, p, X; kwargs...)
        for (i, Y) in enumerate(Ys)
            Test.@test !is_vector(M, p, Y; kwargs...)
            (test_warn) && Test.@test_logs (:warn,) is_vector(M, p, Y; error = :warn, kwargs...)
            (test_info) && Test.@test_logs (:info,) is_vector(M, p, Y; error = :info, kwargs...)
            if length(errors) >= i && !ismissing(errors[i])
                Test.@test_throws (errors[i]) is_vector(M, p, Y; error = :error, kwargs...)
            end
        end
        if check_basepoint
            Test.@test is_vector(M, p, X, true; kwargs...)
            if !ismissing(q) && isexpected(q)
                Test.@test !is_vector(M, q, X, true; kwargs...)
                (test_warn) && Test.@test_logs (:warn,) is_vector(M, q, X, true; error = :warn, kwargs...)
                (test_info) && Test.@test_logs (:info,) is_vector(M, q, X, true; error = :info, kwargs...)
                !isexpected(basepoint_error) || Test.@test_throws (basepoint_error) is_vector(M, q, X, true; error = :error, kwargs...)
            end
        end
    end
    return nothing
end

"""
    Manifolds.Test.test_log(
        M, p, q;
        available_functions=[], expected_value=NoExpectation(), test_mutating=true,
        test_exp = (exp in available_functions),
        test_injectivity_radius = (injectivity_radius in available_functions),
        name = "Logarithmic map on \$M for \$(typeof(p)) points",
        kwargs...
    )
    Test the logarithmic map on manifold `M` at point `p` towards q

    * that the result is a valid tangent vector at `p` on the manifold
    * that the result matches `expected_value`, if given
    * that the mutating version `log!` matches the non-mutating version, (if activated)
    * that the exponential map inverts the logarithmic map (if activated)
    (only performed if either `injectivity_radius` is not available or `X` is within)
"""
function Manifolds.Test.test_log(
        M::AbstractManifold, p, q;
        available_functions = Function[],
        expected_value = NoExpectation(),
        test_exp = (exp in available_functions),
        test_injectivity_radius = (injectivity_radius in available_functions),
        test_mutating = true,
        name = "Logarithmic map on $M for $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        X = log(M, p, q)
        Test.@test is_vector(M, p, X; error = :error, kwargs...)
        Z = log(M, p, p)
        Test.@test is_vector(M, p, Z; error = :error, kwargs...)
        Test.@test norm(M, p, Z) ≈ 0.0   # log
        !isexpected(expected_value) || Test.@test isapprox(M, p, X, expect(expected_value); error = :error, kwargs...)
        if test_mutating
            Y = copy(M, p, X)
            log!(M, Y, p, q)
            Test.@test isapprox(M, Y, X; error = :error, kwargs...)
        end
        if test_exp
            # Test only if inj is not available of X is within inj radius
            run_test = !test_injectivity_radius || norm(M, p, X) ≤ injectivity_radius(M, p)
            run_test || (@warn("Skipping exp-log test since norm of X ($(norm(M, p, X))) is outside injectivity radius ($(injectivity_radius(M, p)))"))
            q2 = exp(M, p, X)
            Test.@test isapprox(M, q2, q; error = :error, kwargs...) skip = !run_test
        end
    end
    return nothing
end # Manifolds.Test.test_log

"""
    test_manifold_dimension(
        M;
        expected_value = NoExpectation(),
        expected_type = Int,
        name = "Manifold dimension for \$M",
    )

Test that the dimension of the manifold `M` is consistent.

* that the dimension is nonnegative
* that it is an integer (of type `expected_type`, `Int` by default)
* that it matches the expected value (if provided)
"""
function Manifolds.Test.test_manifold_dimension(
        M;
        expected_value = NoExpectation(),
        expected_type = Int,
        name = "Manifold dimension for $M",
    )
    Test.@testset "$(name)" begin
        d = manifold_dimension(M)
        Test.@test d ≥ 0
        Test.@test isa(d, expected_type)
        !isexpected(expected_value) || (@test d == expect(expected_value))
    end
    return nothing
end # Manifolds.Test.test_manifold_dimension

"""
    Manifolds.Test.test_manifold_volume(
        M;
        expected_value = NoExpectation(),
        name = "Manifold volume for \$M",
    )

Test the volume of the manifold `M`.

* that the volume is nonnegative
* that it matches the expected value (if provided)
"""
function Manifolds.Test.test_manifold_volume(
        M;
        expected_value = NoExpectation(),
        name = "Manifold volume for $M",
    )
    Test.@testset "$(name)" begin
        v = manifold_volume(M)
        Test.@test v ≥ 0.0
        !isexpected(expected_value) || Test.@test isapprox(v, expect(expected_value))
    end
    return nothing
end # Manifolds.Test.test_manifold_volume

"""
    Manifolds.Test.test_mid_point(M, p, q;
        available_functions=[],
        expected_value = NoExpectation(),
        test_aliased = true,
        test_mutating = true,
        test_symmetry = true,
        name = "Mid-point on \$M between \$(typeof(p)) points",
        kwargs...
    )

Test the mid-point function on manifold `M` between points `p` and `q`.

* that the result is a valid point on the manifold
* that the result matches `expected_value`, if given
* that the mid-point is symmetric (if activated)
* that the distance from `p` and `q` to the mid-point is half the distance from `p` to `q` (if distance is present)
* that the mutating version `mid_point!` matches the non-mutating version (if activated)
* that `mid_point!` works on aliased in put (`r=p` or `r=q`) (if activated for mutating)
"""
function Manifolds.Test.test_mid_point(
        M, p, q;
        available_functions = Function[],
        expected_value = NoExpectation(),
        test_aliased = true,
        test_mutating = true,
        test_symmetry = true,
        name = "Mid-point on $M between $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        r = mid_point(M, p, q)
        Test.@test is_point(M, r; error = :error, kwargs...)
        !isexpected(expected_value) || Test.@test isapprox(M, r, expect(expected_value); error = :error, kwargs...)
        r2 = mid_point(M, q, p)
        test_symmetry || Test.@test isapprox(M, r2, r; error = :error, kwargs...)
        if distance in available_functions
            d_pq = distance(M, p, q)
            d_pr = distance(M, p, r)
            d_qr = distance(M, q, r)
            Test.@test isapprox(d_pr, d_pq / 2; kwargs...)
            Test.@test isapprox(d_qr, d_pq / 2; kwargs...)
        end
        if test_mutating
            r3 = copy(M, p)
            mid_point!(M, r3, p, q)
            Test.@test isapprox(M, r3, r; error = :error, kwargs...)
            if test_aliased
                r4 = copy(M, p)
                mid_point!(M, r4, r4, q)  # aliased #1
                Test.@test isapprox(M, r4, r; error = :error, kwargs...)
                r5 = copy(M, q)
                mid_point!(M, r5, p, r5)  # aliased #2
                Test.@test isapprox(M, r5, r; error = :error, kwargs...)
            end
        end
    end
    return nothing
end # Manifolds.Test.test_mid_point

"""
    Manifolds.Test.test_norm(M, p, X;
        available_functions = [], expected_value = NoExpectation(),
        name = "Norm on \$M at point \$(typeof(p))",
        test_inner = (inner in available_functions),
        kwargs...
    )

Test the norm on the manifold `M` at point `p` for tangent vector `X`.

* that the result is a real number
* that the result matches `expected_value`, if given
* that the norm of `X` with itself is non negative
* that the inner product of `X` with itself is the same as its norm squared (if activated)
"""
function Manifolds.Test.test_norm(
        M::AbstractManifold, p, X;
        available_functions = Function[],
        expected_value = NoExpectation(),
        test_inner = (inner in available_functions),
        name = "Norm on $M at point $(typeof(p))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        v = norm(M, p, X)
        Test.@test v isa (Real)
        !isexpected(expected_value) || Test.@test isapprox(v, expect(expected_value); kwargs...)
        Test.@test v ≥ 0.0
        if test_inner
            w = inner(M, p, X, X)
            Test.@test isapprox(w, v^2; kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_norm

"""
    Manifolds.Test.test_parallel_transport(
        M, p, X, q;
        available_functions=[],
        direction = (log in available_functions) ? log(M, p, q) : missing,
        expected_value=NoExpectation(),
        expected_value_direction=NoExpectation(),
        name = "Parallel transport on \$M at point \$(typeof(p))",
        test_aliased = true,
        test_mutating = true,
        kwargs...
    )

Test the parallel transport on manifold `M` at point `p` with tangent vector `X` to a point `q`.

this defaults to testing `parallel_transport_to`, but can also be used to test `parallel_transport_direction`,
by passing that function to the `available_functions`, using the `direction` argument.
The `direction` has to be the one that transports `X` also to `q`.

* that the result is a valid tangent vector at `q` on the manifold
* that the result matches `expected_value`, if given
* that the mutating version `parallel_transport!` matches the non-mutating version
* that `parallel_transport_to` works on aliased in put (`Y=X`) (if activated)
* that the direction of the transport is consistent with `expected_direction`, if given
* that the mutating version `parallel_transport_direction!` matches the non-mutating version (if activated)
* that `parallel_transport_direction` works on aliased in put (`Y=X` or `Y=d`) (if activated for mutating)
* that both functions are consistent
"""
function Manifolds.Test.test_parallel_transport(
        M, p, X, q;
        available_functions = [],
        direction = (log in available_functions) ? log(M, p, q) : missing,
        expected_value_direction = NoExpectation(),
        expected_value = NoExpectation(),
        name = "Parallel transport on $M at point $(typeof(p))",
        test_aliased = true,
        test_mutating = true,
        kwargs...
    )
    Test.@testset "$(name)" begin
        Y = parallel_transport_to(M, p, X, q)
        Test.@test is_vector(M, q, Y; error = :error, kwargs...)
        !isexpected(expected_value) || Test.@test isapprox(M, p, Y, expect(expected_value); error = :error, kwargs...)
        if test_mutating
            Y2 = copy(M, p, X)
            parallel_transport_to!(M, Y2, p, X, q)
            Test.@test isapprox(M, q, Y2, Y; error = :error, kwargs...)
            if test_aliased
                Y3 = copy(M, p, X)
                parallel_transport_to!(M, Y3, p, Y3, q)  # aliased
                Test.@test isapprox(M, q, Y3, Y; error = :error, kwargs...)
            end
        end
        if (parallel_transport_direction in available_functions) && !ismissing(direction)
            Y4 = parallel_transport_direction(M, p, X, direction)
            !isexpected(expected_value_direction) || Test.@test isapprox(M, q, Y4, expect(expected_value_direction); error = :error, kwargs...)
            if test_mutating
                Y5 = copy(M, p, X)
                parallel_transport_direction!(M, Y5, p, X, direction)
                Test.@test isapprox(M, p, Y5, Y4; error = :error, kwargs...)
                if test_aliased
                    Y6 = copy(M, p, X)
                    parallel_transport_direction!(M, Y6, p, Y6, direction)  # aliased #1
                    Test.@test isapprox(M, p, Y6, Y4; error = :error, kwargs...)
                    Y7 = copy(M, p, direction)
                    parallel_transport_direction!(M, Y7, p, X, direction)  # aliased #2
                    Test.@test isapprox(M, p, Y7, Y4; error = :error, kwargs...)
                end
            end
            # consistency check
            Test.@test is_vector(M, q, Y4; error = :error, kwargs...)
            Test.@test isapprox(M, q, Y, Y4; error = :error, kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_parallel_transport

"""
    Manifolds.Test.test_project(
        M, q, Y = missing;
        available_functions = [],
        expected_point = NoExpectation(),
        expected_vector = NoExpectation(),
        test_aliased = true,
        test_mutating = true,
        name = "Projection on \$M for \$(typeof(q)) points",
        kwargs...
    )

Test the [`project`](@extref `ManifoldsBase.project-Tuple{AbstractManifold, Any}`)`(M, q)` and [`project`](@extref `ManifoldsBase.project-Tuple{AbstractManifold, Any, Any}`)`(M, q, Y)`
to project points and tangent vectors (if not `missing`).

Besides a simple call of `project` (for both variants)  the following tests are performed

* that the projected point is a valid point on the manifold
* that the projected vector is a valid tangent vector on the manifold
* that the result matches `expected_point` and `expected_vector`, respectively, if given
* that the mutating version `project!` produces the same result(s) (if activated)
* that `project!` works on aliased input (`p=q` or `X=Y`) (if activated _and_ p/q or X/Y are of same type)
"""
function Manifolds.Test.test_project(
        M::AbstractManifold, q, Y = missing;
        available_functions = Function[],
        expected_point = NoExpectation(),
        expected_vector = NoExpectation(),
        test_aliased = true,
        test_mutating = true,
        name = "Projection on $M for $(typeof(q)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        # Test point projection
        p = project(M, q)
        Test.@test is_point(M, p; error = :error, kwargs...)
        !isexpected(expected_point) || Test.@test isapprox(M, p, expect(expected_point); error = :error, kwargs...)
        if test_mutating
            p2 = copy(M, p)
            project!(M, p2, q)
            Test.@test isapprox(M, p2, p; error = :error, kwargs...)
            if test_aliased && (typeof(p) == typeof(q))
                p3 = copy(M, q)
                project!(M, p3, p3)  # aliased
                Test.@test isapprox(M, p3, p; error = :error, kwargs...)
            end
        end
        # Test vector projection
        if !ismissing(Y)
            X = project(M, q, Y)
            Test.@test is_vector(M, p, X; error = :error, kwargs...)
            !isexpected(expected_vector) || Test.@test isapprox(M, p, X, expect(expected_vector); error = :error, kwargs...)
            if test_mutating
                X2 = copy(M, p, X)
                project!(M, X2, q, Y)
                Test.@test isapprox(M, p, X2, X; error = :error, kwargs...)
                if test_aliased
                    X3 = copy(M, p, X)
                    project!(M, X3, q, X3)  # aliased
                    Test.@test isapprox(M, p, X3, X; error = :error, kwargs...)
                end
            end
        end
    end
    return nothing
end # Manifolds.Test.test_project

"""
    Manifolds.Test.test_repr(
        M;
        expected_value = NoExpectation(),
        name = "(String) repr_esentation of \$M",
    )

Test that the default `show` method works as expected by calling `repr(M)`.
"""
function Manifolds.Test.test_repr(
        M; expected_value = NoExpectation(), name = "(String) repr_esentation of \$M",
    )
    Test.@testset "$(name)" begin
        s = repr(M)
        !isexpected(expected_value) || Test.@test s == expect(expected_value)
    end
    return nothing
end # Manifolds.Test.test_repr

"""
    Manifolds.Test.test_rand(M;
        vector_at = missing,
        seed = missing,
        test_mutating = true,
        rng = missing,
        name = "Random sampling on \$M",
        kwargs...
    )

Test the random sampling functions `rand(M)` and `rand(M; vector_at=p` (if `vector_at` is given) on manifold `M`.

* that the result of `rand(M)` is a valid point on the manifold
* that the result of `rand(M; vector_at=p)` is a valid tangent vector at `p` on the manifold (if `vector_at` is given)
* that the mutating versions `rand!` match the non-mutating versions (if activated)
* that the four mentioned functions also work with a seed upfront.
"""
function Manifolds.Test.test_rand(
        M::AbstractManifold;
        vector_at = missing,
        seed = missing,
        test_mutating = true,
        rng = missing,
        name = "Random sampling on $M $(ismissing(vector_at) ? "" : "at points of type $(typeof(vector_at))")",
        kwargs...
    )
    Test.@testset "$(name)" begin
        ismissing(seed) || (ismissing(rng) ? Random.seed!(seed) : Random.seed!(rng, seed))
        # Test rand(M)
        p = ismissing(rng) ? rand(M) : rand(rng, M)
        Test.@test is_point(M, p; error = :error, kwargs...)
        if test_mutating
            p2 = copy(M, p)
            ismissing(rng) ? rand!(M, p2) : rand!(rng, M, p2)
            Test.@test is_point(M, p2; error = :error, kwargs...)
        end
        # Test rand(M; vector_at=p)
        q = ismissing(vector_at) ? p : vector_at
        X = ismissing(rng) ? rand(M; vector_at = vector_at) : rand(rng, M; vector_at = vector_at)
        Test.@test is_vector(M, q, X; error = :error, kwargs...)
        if test_mutating
            X2 = copy(M, vector_at, X)
            ismissing(rng) ? rand!(M, X2; vector_at = q) : rand!(rng, M, X2; vector_at = q)
            Test.@test is_vector(M, q, X2; error = :error, kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_rand

"""
    Manifolds.Test.test_representation_size(
        M;
        expected_value = NoExpectation(),
        name = "Representation size of \$M",
    )

Test the representation size of the manifold `M`.

* that the result is tuple of nonnegative integers or nothing (if there is no reasonable representation)
* that it matches the expected value (if provided)
"""
function Manifolds.Test.test_representation_size(
        M::AbstractManifold;
        expected_value = NoExpectation(),
        name = "Representation size of $M",
    )
    Test.@testset "$(name)" begin
        s = representation_size(M)
        if !isnothing(s)
            Test.@test all(x -> x ≥ 0 && x isa Int, s)
        end
        !isexpected(expected_value) || Test.@test s == expect(expected_value)
    end
    return nothing
end # Manifolds.Test.test_representation_size

"""
    Manifolds.test.test_retract(
        M, p, X, m::AbstractRetractionMethod;
        available_functions=[],
        expected_value = NoExpectation(),
        inverse_retraction_method = missing,
    name = "Retraction \$m on \$M at point \$(typeof(p))",
        t = 1.0
        test_mutating = true,
        test_mutating = true,
        test_inverse_retraction = (inverse_retraction in available_functions) && !ismissing(inverse_retraction_method),
        kwargs...
    )

Test the retraction method `m` on manifold `M` at point `p` with tangent vector `X`.

* that the result is a valid point on the manifold
* that the result matches `expected_value`, if given
* that the mutating version `retract!` matches the non-mutating version, (if activated)
* that `retract!` works on aliased in put (`p=q`) (if activated for mutating)
* that the inverse retraction inverts the retraction (if activated)
* that the fused version `retract_fused(M, p, X, t, m)` matches the non-fused version (if activated)
"""
function Manifolds.Test.test_retract(
        M, p, X, m::AbstractRetractionMethod;
        t = 1.0,
        available_functions = Function[],
        expected_value = NoExpectation(),
        inverse_retraction_method = missing,
        name = "Retraction $m on $M at point $(typeof(p))",
        test_aliased = true,
        test_inverse_retraction = (inverse_retract in available_functions) && !ismissing(inverse_retraction_method),
        test_mutating = true,
        test_fused = true,
        kwargs...
    )
    Test.@testset "$(name)" begin
        q = retract(M, p, X, m)
        Test.@test is_point(M, q; error = :error, kwargs...)
        !isexpected(expected_value) || Test.@test isapprox(M, q, expected(expected_value); error = :error, kwargs...)
        if test_mutating
            q2 = copy(M, p)
            retract!(M, q2, p, X, m)
            Test.@test isapprox(M, q2, q; error = :error, kwargs...)
            if test_aliased
                q3 = copy(M, p)
                retract!(M, q3, q3, X, m)  # aliased
                Test.@test isapprox(M, q3, q; error = :error, kwargs...)
            end
        end
        if test_fused
            q4 = Manifolds.retract_fused(M, p, X, t, m)
            q5 = retract(M, p, t * X, m)
            Test.@test isapprox(M, q4, q5; error = :error, kwargs...)
            if test_mutating
                q6 = copy(M, p)
                Manifolds.retract_fused!(M, q6, p, X, t, m)
                Test.@test isapprox(M, q6, q5; error = :error, kwargs...)
                if test_aliased
                    q7 = copy(M, p)
                    Manifolds.retract_fused!(M, q7, q7, X, t, m)  # aliased
                    Test.@test isapprox(M, q7, q5; error = :error, kwargs...)
                end
            end
        end
        if test_inverse_retraction
            X2 = inverse_retract(M, p, q, inverse_retraction_method)
            Test.@test isapprox(M, p, X2, X; error = :error, kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_retract

"""
    Manifolds.Test.test_sectional_curvature(
        M, p, X, Y;
        expected_value = NoExpectation(),
        expected_min = NoExpectation(),
        expected_max = NoExpectation(),
        name = "Sectional curvature on \$M at point \$(typeof(p))",
        kwargs...
    )

Test the sectional curvature on manifold `M` at point `p` for tangent vectors `X` and `Y`.
* that the result is a real number
* that the result matches `expected_value`, if given
* that the minimum sectional curvature at `p` is less than or equal to the sectional curvature
* that the maximum sectional curvature at `p` is greater than or equal to the sectional curvature
* that the minimum and maximum sectional curvatures match `expected_min` and `expected_max`, if given
"""
function Manifolds.Test.test_sectional_curvature(
        M, p, X, Y;
        expected_value = NoExpectation(),
        expected_min = NoExpectation(),
        expected_max = NoExpectation(),
        name = "Sectional curvature on $M at point $(typeof(p))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        k = sectional_curvature(M, p, X, Y)
        Test.@test k isa (Real)
        !isexpected(expected_value) || Test.@test isapprox(k, expected(expected_value); kwargs...)
        K_min = sectional_curvature_min(M)
        K_max = sectional_curvature_max(M)
        Test.@test K_min ≤ k
        Test.@test K_max ≥ k
        !isexpected(expected_min) || Test.@test isapprox(K_min, expected(expected_min); kwargs...)
        !isexpected(expected_max) || Test.@test isapprox(K_max, expected(expected_max); kwargs...)
    end
    return nothing
end # Manifolds.Test.test_sectional_curvature

"""
    Manifolds.Test.test_sharp(M, p, ξ;
        available_functions=[],
        expected_value = NoExpectation(),
        name = "Sharp on \$M for \$(typeof(p)) points",
        kwargs...
    )

Test the sharp operation on manifold `M` at point `p` with cotangent vector `ξ`.
* test that the result is a valid tangent vector at `p` on the manifold
* that the result matches `expected_value`, if given
* that [`flat`](@ref) is the inverse
* that mutating version `sharp!` matches non-mutating version (if activated)
"""
function Manifolds.Test.test_sharp(
        M, p, ξ;
        available_functions = Function[],
        expected_value = NoExpectation(),
        test_mutating = true,
        name = "Sharp on $M for $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        X = sharp(M, p, ξ)
        Test.@test is_vector(M, p, X; error = :error, kwargs...)
        !isexpected(expected_value) || Test.@test isapprox(M, p, X, expected(expected_value); error = :error, kwargs...)
        if flat in available_functions
            ξ2 = flat(M, p, X)
            Test.@test isapprox(M, ξ2, ξ; error = :error, kwargs...)
        end
        if test_mutating
            X2 = copy(M, p, X)
            sharp!(M, X2, p, ξ)
            Test.@test isapprox(M, X2, X; error = :error, kwargs...)
        end
    end
    return nothing
end

"""
    Manifolds.Test.test_shortest_geodesic(M, p, q, t=1.0;
        available_functions=[],
        expected_value = NoExpectation(),
        N = 10,
        name = "Shortest geodesic on \$M for \$(typeof(p)) points",
        kwargs...
    )

Test the geodesic on manifold `M` at point `p` with tangent vector `X` at time `t`.
* that at time `0` the geodesic returns `p`, at time 2 it returns `q`
* that the function `γ = shortest_geodesic(M, p, X)` is consistent with evaluation at `0`, `1` and `t``
* that the result at `t` is a valid point on the manifold
* that the geodesic has constant speed (if activated) using `N` samples and the approximated
  derivative via finite differences
* that the geodesic is length minimizing, i.e. the sum of the segments is approximately equal to the distance from `p` to `q` (if activated)
"""
function Manifolds.Test.test_shortest_geodesic(
        M, p, q, t = 0.5;
        available_functions = Function[],
        expected_value = NoExpectation(),
        N = 10,
        name = "Shortest geodesic on $M for $(typeof(p)) points",
        kwargs...
    )
    Test.@testset "$(name)" begin
        p2 = shortest_geodesic(M, p, q, 0.0)
        Test.@test isapprox(M, p2, p; error = :error, kwargs...)
        q2 = shortest_geodesic(M, p, q, 1.0)
        Test.@test isapprox(M, q2, q; error = :error, kwargs...)
        qt = shortest_geodesic(M, p, q, t)
        Test.@test is_point(M, qt; error = :error, kwargs...)
        γ = shortest_geodesic(M, p, q)
        Test.@test isapprox(M, p2, γ(0.0); error = :error, kwargs...)
        Test.@test isapprox(M, q2, γ(1.0); error = :error, kwargs...)
        Test.@test isapprox(M, qt, γ(t); error = :error, kwargs...)
        # consistency
        @test isapprox(distance(M, p, q) * t, distance(M, p, qt); kwargs...)
        @test isapprox(distance(M, p, q) * (1 - t), distance(M, qt, q); kwargs...)
        !isexpected(expected_value) || Test.@test isapprox(M, qt, expected(expected_value); error = :error, kwargs...)
        # Test constant speed
        if (distance in available_functions) && (norm in available_functions)
            ts = range(0.0, 1.0; length = N)
            points = [shortest_geodesic(M, p, q, ti) for ti in ts]
            dists = [distance(M, points[i], points[i + 1]) for i in 1:(length(points) - 1)]
            speeds = [d / (ts[i + 1] - ts[i]) for (i, d) in enumerate(dists)]
            avg_speed = mean(speeds)
            for s in speeds
                Test.@test isapprox(s, avg_speed; kwargs...)
            end
            # unit speed
            Test.@test isapprox(avg_speed, distance(M, p, q); kwargs...)
            # Test length minimizing
            total_length = sum(dists)
            d_pq = distance(M, p, q)
            Test.@test isapprox(total_length, d_pq; kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_shortest_geodesic

"""
    Manifolds.Test.test_vector_transport(
        M, p, X, q, default_vector_transport_method(M);
        available_functions=[],
        direction = inverse_retract(M, p, q, default_inverse_retraction_method(M)),
        expected_value = NoExpectation(),
        expected_value_direction = NoExpectation(),
        test_aliased = true,
        test_mutating = true,
        name = "Vector transport method \$(vector_transport_method) on \$M at point \$(typeof(p))",
        kwargs...
    )

Test the vector transport on manifold `M` at point `p` with tangent vector `X` to a point `q`.

this defaults to testing `vector_transport_to`, but can also be used to test `vector_transport_direction`,
by passing that function to the `available_functions`, using the `direction` argument.
The `direction` has to be the one that transports `X` also to `q`.

* that the result is a valid tangent vector at `q` on the manifold
* that the result matches `expected_value`, if given
* that the mutating version `vector_transport!` matches the non-mutating version
* that `vector_transport_to` works on aliased in put (`Y=X`) (if activated)
* that the direction of the transport is consistent with `expected_direction`, if given
* that the mutating version `vector_transport_direction!` matches the non-mutating version (if activated)
* that `vector_transport_direction` works on aliased in put (`Y=X` or `Y=d`) (if activated for mutating)
* that both functions are consistent
"""
function Manifolds.Test.test_vector_transport(
        M, p, X, q, m = default_vector_transport_method(M);
        available_functions = [],
        direction = inverse_retract(M, p, q, default_inverse_retraction_method(M)),
        expected_value_direction = NoExpectation(),
        expected_value = NoExpectation(),
        test_aliased = true,
        test_mutating = true,
        name = "Vector transport method $(vector_transport_method) on $M at point $(typeof(p))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        Y = vector_transport_to(M, p, X, q, m)
        Test.@test is_vector(M, q, Y; error = :error, kwargs...)
        !isexpected(expected_value) || Test.@test isapprox(M, p, Y, expected(expected_value); error = :error, kwargs...)
        if test_mutating
            Y2 = copy(M, p, X)
            vector_transport_to!(M, Y2, p, X, q, m)
            Test.@test isapprox(M, q, Y2, Y; error = :error, kwargs...)
            if test_aliased
                Y3 = copy(M, p, X)
                vector_transport_to!(M, Y3, p, Y3, q)  # aliased
                Test.@test isapprox(M, p, Y3, Y; error = :error, kwargs...)
            end
        end
        if (vector_transport_direction in available_functions) && !isnothing(direction)
            Y4 = vector_transport_direction(M, p, X, direction, m)
            !isexpected(expected_value_direction) || Test.@test isapprox(M, p, Y4, expected(expected_value_direction); error = :error, kwargs...)
            if test_mutating
                Y5 = copy(M, p, X)
                vector_transport_direction!(M, Y5, p, X, direction, m)
                Test.@test isapprox(M, p, Y5, Y4; error = :error, kwargs...)
                if test_aliased
                    Y6 = copy(M, p, X)
                    vector_transport_direction!(M, Y6, p, Y6, direction, m)  # aliased #1
                    Test.@test isapprox(M, q, Y6, Y4; error = :error, kwargs...)
                    Y7 = copy(M, p, direction)
                    vector_transport_direction!(M, Y7, p, X, direction, m)  # aliased #2
                    Test.@test isapprox(M, q, Y7, Y4; error = :error, kwargs...)
                end
            end
            # consistency check
            Test.@test is_vector(M, q, Y4; error = :error, kwargs...)
            Test.@test isapprox(M, q, Y, Y4; error = :error, kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_vector_transport

"""
    Manifolds.Test.test_volume_density(
        M, p, X;
        expected_value = NoExpectation(),
        name = "Manifold volume density for \$M",
        kwargs...
    )

Test the [`volume_density`](@ref) of the manifold `M`.

* that it matches the expected value (if provided)
"""
function Manifolds.Test.test_volume_density(
        M, p, X;
        expected_value = NoExpectation(),
        name = "Manifold volume density for $M",
        kwargs...
    )
    Test.@testset "$(name)" begin
        v = volume_density(M, p, X)
        !isexpected(expected_value) || Test.@test isapprox(v, expect(expected_value); kwargs...)
    end
    return nothing
end # Manifolds.Test.test_volume_density

"""
    Manifolds.Test.Weingarten(
        M, p, X, V;
        expected_value = NoExpectation(),
        test_aliased = true,
        test_mutating = true,
        name = "Weingarten map on \$M at point \$(typeof(p))",
        kwargs...
    )

Test the Weingarten map on manifold `M` at point `p` for tangent vector `X` and normal vector `V`.

* that the result is a valid tangent vector at `p` on the manifold
* that the result matches `expected_value`, if given
* that the result is consistent with the mutating version `weingarten!` (if activated)
* that `Weingarten!` works on aliased in put (`Y=X`) (if activated for mutating)
"""
function Manifolds.Test.test_Weingarten(
        M, p, X, V;
        available_functions = [],
        expected_value = NoExpectation(),
        test_aliased = true,
        test_mutating = true,
        name = "Weingarten map on $M at point $(typeof(p))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        Y = Weingarten(M, p, X, V)
        Test.@test is_vector(M, p, Y; error = :error, kwargs...)
        !isexpected(expected_value) || Test.@test isapprox(M, p, Y, expect(expected_value); error = :error, kwargs...)
        if test_mutating
            Y2 = copy(M, p, X)
            Weingarten!(M, Y2, p, X, V)
            Test.@test isapprox(M, p, Y2, Y; error = :error, kwargs...)
            if test_aliased
                Y3 = copy(M, p, X)
                Weingarten!(M, Y3, p, Y3, V)  # aliased
                Test.@test isapprox(M, p, Y3, Y; error = :error, kwargs...)
            end
        end
    end
    return nothing
end # Manifolds.Test.test_Weingarten

"""
    Manifolds.Test.test_zero_vector(M, p;
        available_functions=[],
        test_mutating = true,
        test_norm = (norm in available_functions),
        name = "Zero vector test on \$M at point \$(typeof(p))",
    )

Test the zero vector on the manifold `M` at point `p`.
* verify that it is a valid tangent vector
* verify that its norm is zero (if provided)
* verify that the mutating version `zero_vector!` matches the non-mutating version (if activated)
"""
function Manifolds.Test.test_zero_vector(
        M::AbstractManifold, p;
        available_functions = Function[],
        test_mutating = true,
        test_norm = (norm in available_functions),
        name = "Zero vector on $M at point $(typeof(p))",
        kwargs...
    )
    Test.@testset "$(name)" begin
        Z = zero_vector(M, p)
        Test.@test is_vector(M, p, Z; error = :error, kwargs...)
        if test_norm
            n = norm(M, p, Z)
            Test.@test isapprox(n, 0.0; kwargs...)
        end
        if test_mutating
            Z2 = copy(M, p)
            zero_vector!(M, Z2, p)
            Test.@test isapprox(M, Z2, Z; error = :error, kwargs...)
        end
    end
    return nothing
end # Manifolds.Test.test_zero_vector

# Include the old tests for now as well
include("tests_general.jl")

end
