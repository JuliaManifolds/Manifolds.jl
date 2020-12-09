using RecipesBase, VisualRegressionTests, Plots, Colors
include("utils.jl")
# Note that the `false` in the test avoids to start Gtk
# I (kellertuer)  think it also avoids asking whether a new reference should be
# created but I am not 100% sure

@testset "Recipes Test" begin
    references_folder = joinpath(@__DIR__, "assets")
    function Hyp2PB_plot()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0, 0.0], [0.0, 1.0]])
        p2 = convert.(Ref(PoincareBallPoint), p)
        return plot(M, p2)
    end
    @plottest Hyp2PB_plot joinpath(references_folder, "Hyp2PBPlot.png") false

    function Hyp2PB_plot_geo()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0, 0.0], [0.0, 1.0]])
        p2 = convert.(Ref(PoincareBallPoint), p)
        return plot(M, p2; geodesic_interpolation = 80)
    end
    @plottest Hyp2PB_plot_geo joinpath(references_folder, "Hyp2PBPlotGeo.png") false

    function Hyp2PB_quiver()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        p2 = convert.(Ref(PoincareBallPoint), p)
        X = [log(M, p2[2], p2[1]), log(M, p2[1], p2[3])]
        return plot(M, [p2[2], p2[1]], X)
    end
    @plottest Hyp2PB_quiver joinpath(references_folder, "Hyp2PBQuiver.png") false

    function Hyp2PH_plot()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0, 0.0], [0.0, 1.0]])
        p2 = convert.(Ref(PoincareHalfSpacePoint), p)
        return plot(M, p2)
    end
    @plottest Hyp2PH_plot joinpath(references_folder, "Hyp2PHPlot.png") false

    function Hyp2PH_plot_geo()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0, 0.0], [0.0, 1.0]])
        p2 = convert.(Ref(PoincareHalfSpacePoint), p)
        return plot(M, p2; geodesic_interpolation = 80)
    end
    @plottest Hyp2PH_plot_geo joinpath(references_folder, "Hyp2PHPlotGeo.png") false

    function Hyp2PH_quiver()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        p2 = convert.(Ref(PoincareHalfSpacePoint), p)
        X = [log(M, p2[2], p2[1]), log(M, p2[1], p2[3])]
        return plot(M, [p2[2], p2[1]], X)
    end
    @plottest Hyp2PH_quiver joinpath(references_folder, "Hyp2PHQuiver.png") false

    function Hyp2_plot_geo()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0, 0.0], [0.0, 1.0]])
        return plot(M, p; geodesic_interpolation = 80)
    end
    @plottest Hyp2_plot_geo joinpath(references_folder, "Hyp2PlotGeo.png")

    function Hyp2_quiver()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        X = [log(M, p[2], p[1]), log(M, p[1], p[3])]
        return plot(M, [p[2], p[1]], X)
    end
    @plottest Hyp2_quiver joinpath(references_folder, "Hyp2Quiver.png")

    function Sphere2_plot_geo()
        pyplot()
        M = Sphere(2)
        pts = [ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0] ]
        return plot(M,pts; wireframe_color=colorant"#CCCCCC", markersize=10)
    end
    @plottest Sphere2_plot_geo joinpath(references_folder, "Sphere2PlotGeo.png")

    function Sphere2_quiver()
        pyplot()
        M = Sphere(2)
        pts2 =  [ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0] ]
        p3 = 1/sqrt(3) .* [1.0, -1.0, 1.0]
        vecs = log.(Ref(M), pts2, Ref(p3))
        return plot(M, pts2, vecs; wireframe_color=colorant"#CCCCCC", linewidth=1.5)
    end
    @plottest Sphere2_quiver joinpath(references_folder, "Sphere2Quiver.png")
end
