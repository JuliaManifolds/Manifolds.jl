using Manifolds
using CSV
using DataFrames
using Plots
using MultivariateStats

function load_hands()
    hands_url = "https://raw.githubusercontent.com/geomstats/geomstats/master/geomstats/datasets/data/hands/hands.txt"
    hand_labels_url = "https://raw.githubusercontent.com/geomstats/geomstats/master/geomstats/datasets/data/hands/labels.txt"

    hands = Matrix(CSV.read(download(hands_url), DataFrame, header=false))
    hands = reshape(hands, size(hands, 1), 3, 22)
    hand_labels = CSV.read(download(hand_labels_url), DataFrame, header=false).Column1
    return hands, hand_labels
end

function hand_analysis()
    hands, hand_labels = load_hands()

    scatter3d(hands[1,1,:], hands[1,2,:], hands[1,3,:])

    Mshape = KendallsShapeSpace(3, 22)
    # projecting hands
    hands_projected = [project(Mshape, hands[i, :, :]) for i in axes(hands, 1)]

    # doing tangent PCA
    mean_hand = mean(Mshape, hands_projected)
    B = get_basis(Mshape, mean_hand, ProjectedOrthonormalBasis(:svd))
    hand_logs = [log(Mshape, mean_hand, p) for p in hands_projected]
    red_coords = reduce(hcat, [get_coordinates(Mshape, mean_hand, X, B) for X in hand_logs])
    fp = fit(PCA, red_coords; mean=0)
    # show explained variance of each principal component
    plot(principalvars(fp), title="explained variance", label="Tangent PCA")

    fig = plot(; title="coordinates per gesture of the first two principal components")
    for label_num in [0, 1]
        mask = hand_labels .== label_num
        cur_hand_logs = red_coords[:, mask]
        cur_t = MultivariateStats.transform(fp, cur_hand_logs)
        scatter!(fig, cur_t[1,:], cur_t[2,:], label="gesture " * string(label_num))
    end
    xlabel!(fig, "principal component 1")
    ylabel!(fig, "principal component 2")
    fig
end
