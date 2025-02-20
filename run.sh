# python3 main.py \
#     --root 'data' \
#     --batch_size 8 \
#     --epoch 10 \
#     --lr 0.001\
#     --mask_percentage 0.3 \
#     --resize_to 256 \
#     --data_aug False \

# Experiment 1 - mask percentage
# for mask_percentage in 0.1 0.3 0.7 1; do #
#         python3 main.py \
#             --root 'data' \
#             --batch_size 8 \
#             --epoch 10 \
#             --lr 0.001\
#             --mask_percentage $mask_percentage \
#             --resize_to 256 \
#             --data_aug 0 
#         done

# Experiment 2 - img size
# for resize in 128 256 512; do
#     python3 main.py \
#         --root 'data' \
#         --batch_size 8 \
#         --epoch 10 \
#         --lr 0.001\
#         --mask_percentage 0.3 \
#         --resize_to $resize \
#         --data_aug 0 
#     done

# # Experiment 3 - batch_size
# for batch in 8 16 32 64; do
#     python3 main.py \
#         --root 'data' \
#         --batch_size $batch \
#         --epoch 10 \
#         --lr 0.001\
#         --mask_percentage 0.3 \
#         --resize_to 128 \
#         --data_aug 0 
#     done

# Experiment 3 - data aug
# for aug in 1 0; do
#     python3 main.py \
#         --root 'data' \
#         --batch_size 8 \
#         --epoch 10 \
#         --lr 0.001\
#         --mask_percentage 0.3 \
#         --resize_to 256 \
#         --data_aug $aug 
#     done

# Experiment 5
for gamma in 1 3; do #
        python3 main.py \
            --root 'data' \
            --batch_size 8 \
            --epoch 10 \
            --lr 0.001\
            --mask_percentage 0.3 \
            --resize_to 256 \
            --data_aug 0  \
            --gamma $gamma 
        done
