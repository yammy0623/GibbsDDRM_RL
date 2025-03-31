export CUDA_VISIBLE_DEVICES=0
EXP="/tmp2/ICML2025/ffhq"

SEED=232
CONFIG="ffhq_256_deblur.yml"
PATHY="ffhq"
IMAGE_FOLDER="/tmp2/ICML2025/ddnm_rl/"$PATHY
DEG="deblur_arbitral"
# train ours (1st subtask)
STEP=10

# train ours (1st subtask)
python train.py --ni --config $CONFIG --exp $EXP --path_y $PATHY --eta 0.85 --deg $DEG --deg_scale 4 --sigma_y 0. -i $PATHY"_"$DEG"_""$STEP" --target_steps $STEP --seed $SEED
# # eval ours (1st subtask)
python eval.py --ni --config $CONFIG --exp $EXP --path_y $PATHY --eta 0.85 --deg $DEG --deg_scale 4 --sigma_y 0. -i $PATHY"_"$DEG"_""$STEP""eval" --target_steps $STEP --eval_model_name $DEG"_"$PATHY"_2_agents_A2C_""$STEP" --subtask1 >> "model/$DEG"_"$PATHY"_2_agents_A2C_""$STEP"/sub1.txt"
# train ours (2nd subtask)
python train.py --ni --config $CONFIG --exp $EXP --path_y $PATHY --eta 0.85 --deg $DEG --deg_scale 4 --sigma_y 0. -i $PATHY"_"$DEG"_""$STEP" --target_steps $STEP --second_stage --seed $SEED
# # eval ours
python eval.py --ni --config $CONFIG --exp $EXP --path_y $PATHY --eta 0.85 --deg $DEG --deg_scale 4 --sigma_y 0. -i $PATHY"_"$DEG"_""$STEP""eval" --target_steps $STEP --eval_model_name $DEG"_"$PATHY"_2_agents_A2C_""$STEP" >> "model/$DEG"_"$PATHY"_2_agents_A2C_""$STEP"/sub2.txt"
