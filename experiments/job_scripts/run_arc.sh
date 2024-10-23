cd ..

device=rtx_3090
memory=62
hours=16

# arcc

lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer svgd --lr 0.01    --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 1 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-3 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer svgd --lr 0.001   --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 1 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-4 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer svgd --lr 0.0001  --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 1 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-5 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer svgd --lr 0.00001 --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 1 --write_log 1 --save_model_every 5 --save_results 1'


lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-4_ADAM --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer adamw --lr 0.0001  --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-5_ADAM --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer adamw --lr 0.00001 --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --write_log 1 --save_model_every 5 --save_results 1'


lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-2_g-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer svgd --lr 0.01    --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.01 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-3_g-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer svgd --lr 0.001   --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.01 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-4_g-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer svgd --lr 0.0001  --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.01 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-5_g-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer svgd --lr 0.00001 --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.01 --write_log 1 --save_model_every 5 --save_results 1'


lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-4_g-3 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer svgd --lr 0.0001  --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.001 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-5_g-3 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer svgd --lr 0.00001 --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.001 --write_log 1 --save_model_every 5 --save_results 1'

lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-4_g-1 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer svgd --lr 0.0001  --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.1 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCC_lr-5_g-1 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arcc --optimizer svgd --lr 0.00001 --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.1 --write_log 1 --save_model_every 5 --save_results 1'



# arce

lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer svgd --lr 0.01    --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 1 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-3 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer svgd --lr 0.001   --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 1 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-4 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer svgd --lr 0.0001  --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 1 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-5 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer svgd --lr 0.00001 --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 1 --write_log 1 --save_model_every 5 --save_results 1'


lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-4_ADAM --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer adamw --lr 0.0001  --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-5_ADAM --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer adamw --lr 0.00001 --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --write_log 1 --save_model_every 5 --save_results 1'


lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-2_g-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer svgd --lr 0.01    --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.01 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-3_g-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer svgd --lr 0.001   --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.01 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-4_g-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer svgd --lr 0.0001  --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.01 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-5_g-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer svgd --lr 0.00001 --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.01 --write_log 1 --save_model_every 5 --save_results 1'


lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-4_g-3 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer svgd --lr 0.0001  --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.001 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-5_g-3 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer svgd --lr 0.00001 --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.001 --write_log 1 --save_model_every 5 --save_results 1'

lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-4_g-1 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer svgd --lr 0.0001  --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.1 --write_log 1 --save_model_every 5 --save_results 1'
lbatch -c 1 -g 1 --gputype $device -t $hours -m $memory -a $PROJ_CODE -n ARCE_lr-5_g-1 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --dataset_name arce --optimizer svgd --lr 0.00001 --model 'meta-llama/Llama-3.2-1B' --batch_size 4 --num_epochs 10 --K 8 --gamma 0.1 --write_log 1 --save_model_every 5 --save_results 1'


