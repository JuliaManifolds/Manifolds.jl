using RecipesBase, VisualRegressionTests, Plots, Gtk
include("utils.jl")
# Note that the `false` in the test avoids to start Gtk
# I (kellertuer)  think it also avoids asking whether a new reference should be
# created but I am not 100% sure

@testset "Recipes Test" begin
    references_folder = joinpath(@__DIR__,"assets")
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
        return plot(M, p2; geodesic_interpolation=80)
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
        return plot(M, p2; geodesic_interpolation=80)
    end
    @plottest Hyp2PH_plot_geo joinpath(references_folder, "Hyp2PHPlotGeo.png")

    function Hyp2PH_quiver()
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize.(Ref(M), [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        p2 = convert.(Ref(PoincareHalfSpacePoint), p)
        X = [log(M, p2[2], p2[1]), log(M, p2[1], p2[3])]
        return plot(M, [p2[2], p2[1]], X)
    end
    @plottest Hyp2PH_quiver joinpath(references_folder, "Hyp2PHQuiver.png") false
end
