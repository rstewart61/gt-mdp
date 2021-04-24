#!/bin/bash

GW_SMALL=output/gridworld_small
GW_BIG=output/gridworld_big
TC_SMALL=output/treblecross_small
TC_BIG=output/treblecross_big

#convert +append "$GW_SMALL/vi_policy.png" "$GW_SMALL/pi_policy.png" report/gw_viz_vi_vs_pi.png
#cp "$GW_SMALL/qLearning_policy.png" report/gw_viz_ql.png

for dataset in treblecross gridworld; do
    for x_field in "Discount Factor" "Learning Rate" "Learning Rate Decay" QInit Epsilon; do
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

