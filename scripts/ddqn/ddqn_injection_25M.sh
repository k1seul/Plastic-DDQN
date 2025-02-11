cd ../..

python run_parallel.py \
    --group_name ddqn_injection_25M \
    --exp_name ddqn_injection \
    --config_name ddqn_injection \
    --config_path ./configs \
    --seeds 0 1 2 \
    --num_games 26 \
    --num_devices 1 \
    --num_exp_per_device 3 \
    --games phoenix assault\
    --injection_frame 6_250_000 