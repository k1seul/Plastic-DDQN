cd ../..

python run_parallel.py \
    --group_name ddqn \
    --exp_name ddqn_baseline \
    --config_name ddqn \
    --config_path ./configs \
    --seeds 1 \
    --num_games 26 \
    --num_devices 1 \
    --num_exp_per_device 3 \
    --games phoenix \

