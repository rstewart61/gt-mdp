#!/bin/bash

GW_SMALL=output/gridworld_small
GW_BIG=output/gridworld_big
TC_SMALL=output/treblecross_small
TC_BIG=output/treblecross_big

#convert +append "$GW_SMALL/vi_policy.png" "$GW_SMALL/pi_policy.png" report/gw_viz_vi_vs_pi.png
#cp "$GW_SMALL/qLearning_policy.png" report/gw_viz_ql.png

for dataset in treblecross gridworld; do
    for x_field in "Discount Factor" "Learning Rate" "Learning Rate Decay" QInit; do
        LEFT=output/"${dataset}_small/qlearning/${x_field}/avg_rewards.png"
        RIGHT=output/"${dataset}_big/qlearning/${x_field}/avg_rewards.png"
        convert +append "$LEFT" "$RIGHT" report/"$dataset $x_field avg rewards comparison.png"
    
        LEFT=output/"${dataset}_small/qlearning/${x_field}/avg_steps.png"
        RIGHT=output/"${dataset}_big/qlearning/${x_field}/avg_steps.png"
        convert +append "$LEFT" "$RIGHT" report/"$dataset $x_field avg steps comparison.png"
    done
    
    #LEFT=output/"${dataset}_small/qlearning/Num States/average_reward_by_num_steps_convergence.png"
    #cp "$LEFT" report/"$dataset num states avg rewards by num steps comparison.png"
    
    LEFT=output/"${dataset}_small/vi by state delta_rewards_convergence.png"
    MIDDLE=output/"${dataset}_small/pi by state delta_rewards_convergence.png"
    RIGHT=output/"${dataset}_small/qlearning/Num States/average_reward_by_num_steps_convergence.png"
    convert +append "$LEFT" "$MIDDLE" "$RIGHT" report/"$dataset comparison convergence by num states.png"

    #LEFT=output/"${dataset}_small/qlearning/Num States/average_rewards_convergence.png"
    #cp "$LEFT" report/"$dataset num states avg rewards by episodes comparison.png"
    
    LEFT=output/"${dataset}_small/Num States - Iterations.png"
    RIGHT=output/"${dataset}_small/Num States - CPU Time.png"
    convert +append "$LEFT" "$RIGHT" report/"$dataset comparison by Num States - Iterations vs CPU Time.png"

    LEFT=output/"${dataset}_small/Num States - CPU Time.png"
    RIGHT=output/"${dataset}_small/qlearning/Num States/convergence_cpu_time.png"
    convert +append "$LEFT" "$RIGHT" report/"$dataset comparison by Num States - Planning vs QL CPU Time.png"

    for plot_file in output/${dataset}_small/*.png; do
        plot_base=`basename "$plot_file"`
        
        LEFT=output/"${dataset}_small/${plot_base}"
        RIGHT=output/"${dataset}_big/${plot_base}"
        if test -f "$RIGHT"; then
            convert +append "$LEFT" "$RIGHT" report/"$dataset comparison ${plot_base}"
        else
            cp "$LEFT" report/"${dataset}_small ${plot_base}"
        fi
    done
done

rm report/*comparison\ Num\ States*.png
rm report/gridworld\ comparison*freq*.png
rm report/gridworld\ comparison*policy*.png

for size in small big; do
    p=output/gridworld_$size
    LEFT=$p/vi_policy_subimage.png
    RIGHT=$p/pi_policy_subimage.png
    convert "$LEFT" "$RIGHT" -background white -splice 10x0+0+0  +append -chop 10x0+0+0 report/"gridworld $size comparison planning policy subimage.png"

    LEFT=$p/qLearning_policy_subimage.png
    RIGHT=$p/qLearning_freq_subimage.png
    convert "$LEFT" "$RIGHT" -background white -splice 10x0+0+0  +append -chop 10x0+0+0 report/"gridworld $size comparison qlearning policy subimage.png"
done
    

echo Finished gluing plots into report directory

exit 0


LEFT=plots/"Dexter/DEFAULT/KM/elbow_silhouette.png"
RIGHT=plots/"Dexter Like Noise/DEFAULT/KM/elbow_silhouette.png"
convert +append "$LEFT" "$RIGHT" report/"Dexter_vs_Noise_KM.png"

for dr in DT ICA PCA RP; do
    for dataset in "Polish Bankruptcy" Dexter; do
        LEFT=plots/"${dataset}/${dr}/KM/KM[$dr] - Cluster labels_best_tsne.png"
        MIDDLE=plots/"${dataset}/${dr}/DEFAULT/Ground truth_best_tsne.png"
        RIGHT=plots/"${dataset}/${dr}/EM/EM[$dr] - Cluster labels_best_tsne.png"
        convert +append "$LEFT" "$MIDDLE" "$RIGHT" report/"$dataset $dr TSNE comparison.png"

        LEFT=plots/"Polish Bankruptcy/$dr/EM/ch_vs_db.png"
        RIGHT=plots/"Dexter/$dr/EM/ch_vs_db.png"
        convert +append "$LEFT" "$RIGHT" report/"Both_${dr}_EM_ch_vs_db.png"
        
        LEFT=plots/"${dataset}/$dr/KM/elbow_silhouette.png"
        RIGHT=plots/"${dataset}/$dr/KM/Largest Cluster Size % of Samples.png"
        convert +append "$LEFT" "$RIGHT" report/"${dataset}_${dr}_silhoutte_vs_cluster_size.png"
    done
done

for dataset in "Polish Bankruptcy" Dexter; do
    LEFT=plots/"${dataset}/DEFAULT/KM/KM - Cluster labels_best_tsne.png"
    MIDDLE=plots/"${dataset}/DEFAULT/DEFAULT/Ground truth_best_tsne.png"
    RIGHT=plots/"${dataset}/DEFAULT/EM/EM - Cluster labels_best_tsne.png"
    convert +append "$LEFT" "$MIDDLE" "$RIGHT" report/"$dataset DEFAULT TSNE comparison.png"
    
    ONE=plots/"${dataset}/RP/KM/elbow_silhouette.png"
    TWO=plots/"${dataset}/PCA/KM/elbow_silhouette.png"
    THREE=plots/"${dataset}/ICA/KM/elbow_silhouette.png"
    FOUR=plots/"${dataset}/DT/KM/elbow_silhouette.png"
    convert +append "$ONE" "$TWO" "$THREE" "$FOUR" report/"$dataset KM Elbow Silhouette comparison.png"

    ONE=plots/"${dataset}/RP/KM/Largest Cluster Size % of Samples.png"
    TWO=plots/"${dataset}/PCA/KM/Largest Cluster Size % of Samples.png"
    THREE=plots/"${dataset}/ICA/KM/Largest Cluster Size % of Samples.png"
    FOUR=plots/"${dataset}/DT/KM/Largest Cluster Size % of Samples.png"
    convert +append "$ONE" "$TWO" "$THREE" "$FOUR" report/"$dataset KM Largest Cluster Size % of Samples comparison.png"
done

LEFT=plots/"Polish Bankruptcy/DEFAULT/PCA/explained_variance_ratio.png"
RIGHT=plots/"Dexter/DEFAULT/PCA/explained_variance_ratio.png"
convert +append "$LEFT" "$RIGHT" report/"Both_PCA_explained_variance_ratio.png"

LEFT=plots/"Polish Bankruptcy/DEFAULT/RP/reconstruction_error.png"
RIGHT=plots/"Dexter/DEFAULT/RP/reconstruction_error.png"
convert +append "$LEFT" "$RIGHT" report/"Both_RP_reconstruction_error.png"

LEFT=plots/"Polish Bankruptcy/DEFAULT/RP/johnson-lindenstrauss.png"
RIGHT=plots/"Dexter/DEFAULT/RP/johnson-lindenstrauss.png"
convert +append "$LEFT" "$RIGHT" report/"Both_RP_johnson-lindenstrauss.png"

LEFT=plots/"Polish Bankruptcy/DEFAULT/ICA/kurtosis.png"
RIGHT=plots/"Dexter/DEFAULT/ICA/kurtosis.png"
convert +append "$LEFT" "$RIGHT" report/"Both_ICA_kurtosis.png"

LEFT=plots/"Polish Bankruptcy/DEFAULT/DT/feature_importances.png"
RIGHT=plots/"Dexter/DEFAULT/DT/feature_importances.png"
convert +append "$LEFT" "$RIGHT" report/"Both_DT_feature_importances.png"

for dataset in "Polish Bankruptcy" Dexter; do
    LEFT=plots/"${dataset}/DEFAULT/DEFAULT/nn_results_Balanced Accuracy.png"
    RIGHT=plots/"${dataset}/DEFAULT/DEFAULT/nn_results_Tuning Time (s).png"
    convert +append "$LEFT" "$RIGHT" report/"${dataset}_nn_results.png"

    LEFT=plots/"${dataset}/DEFAULT/DEFAULT/roc_auc.png"
    RIGHT=plots/"${dataset}/DEFAULT/DEFAULT/nn_results_Tuning Time (s).png"
    convert +append "$LEFT" "$RIGHT" report/"${dataset}_roc_auc.png"

    LEFT=plots/"${dataset}/DEFAULT/KM/DR comparison for KM - Silhouette.png"
    RIGHT=plots/"${dataset}/DEFAULT/KM/DR comparison for KM - Largest Cluster Size % of Samples.png"
    convert +append "$LEFT" "$RIGHT" report/"${dataset}_KM_Silhouette Comparison.png"

    PCA=plots/"${dataset}/DEFAULT/PCA/PCA Features 1, 2 for ${dataset}.png"
    ICA=plots/"${dataset}/DEFAULT/ICA/ICA Features 1, 2 for ${dataset}.png"
    RP=plots/"${dataset}/DEFAULT/RP/RP Features 1, 2 for ${dataset}.png"
    DT=plots/"${dataset}/DEFAULT/DT/DT Features 1, 2 for ${dataset}.png"
    convert +append "$PCA" "$ICA"  "$RP" "$DT" report/"${dataset}_dimensions_1_2.png"
done

LEFT=plots/"Polish Bankruptcy/DEFAULT/DEFAULT/cluster_results_ami.png"
RIGHT=plots/"Dexter/DEFAULT/DEFAULT/cluster_results_ami.png"
convert +append "$LEFT" "$RIGHT" report/"Both_cluster_results_ami.png"

LEFT=plots/"Polish Bankruptcy/DEFAULT/EM/aic_vs_bic.png"
RIGHT=plots/"Dexter/DEFAULT/EM/aic_vs_bic.png"
convert +append "$LEFT" "$RIGHT" report/"Both_EM_aic_vs_bic.png"

LEFT=plots/"Polish Bankruptcy/DEFAULT/EM/ch_vs_db.png"
RIGHT=plots/"Dexter/DEFAULT/EM/ch_vs_db.png"
convert +append "$LEFT" "$RIGHT" report/"Both_EM_ch_vs_db.png"



