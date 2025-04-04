export CUDA_VISIBLE_DEVICES=0
EXP="/tmp2/ICML2025/GoPro"

SEED=232
CONFIG="gopro_blur_gamma_256_deblur.yml"
PATHY="gopro"
IMAGE_FOLDER="/tmp2/ICML2025/gibbsddrm/"$PATHY
DEG="deblur_arbitral"
# train ours (1st subtask)
STEP=5
# train ours (1st subtask) (include train and eval)
python train_less.py --train --ni --config $CONFIG --exp $EXP --path_y $PATHY --eta 0.85 --deg $DEG --deg_scale 4 --sigma_y 0. -i $PATHY"_"$DEG"_""$STEP""_less" --target_steps $STEP --seed $SEED
# eval ours (1st subtask) (test)
python eval.py --ni --config $CONFIG --exp $EXP --path_y $PATHY --eta 0.85 --deg $DEG --deg_scale 4 --sigma_y 0. -i $PATHY"_"$DEG"_""$STEP""eval_less" --target_steps $STEP --eval_model_name$DEG"_${PATHY}_2_agents_A2C_${STEP}" --subtask1 >> "model/${DEG}_${PATHY}_2_agents_A2C_${STEP}_less/sub1.txt"
# train ours (2nd subtask)
python train_less.py --train --ni --config $CONFIG --exp $EXP --path_y $PATHY --eta 0.85 --deg $DEG --deg_scale 4 --sigma_y 0. -i $PATHY"_"$DEG"_""$STEP""_less" --target_steps $STEP --second_stage --seed $SEED
# # eval ours
python eval.py --ni --config $CONFIG --exp $EXP --path_y $PATHY --eta 0.85 --deg $DEG --deg_scale 4 --sigma_y 0. -i $PATHY"_"$DEG"_""$STEP""eval_less" --target_steps $STEP --eval_model_name $DEG"_${PATHY}_2_agents_A2C_${STEP}" >> "model/${DEG}_${PATHY}_2_agents_A2C_${STEP}_less/sub2.txt"
