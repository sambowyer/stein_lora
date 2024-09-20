# lbatch -c 1 -g 1 --gputype rtx_2080 -t 2 -m 11 -a $PROJ_CODE -n adamw3 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer adamw --lr 1e-3'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 2 -m 11 -a $PROJ_CODE -n adamw4 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer adamw --lr 1e-4'

# lbatch -c 1 -g 1 --gputype rtx_2080 -t 2 -m 11 -a $PROJ_CODE -n sgd3 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sgd --lr 1e-3'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 2 -m 11 -a $PROJ_CODE -n sgd4 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sgd --lr 1e-4'


# lbatch -c 1 -g 1 --gputype rtx_2080 -t 8 -m 11 -a $PROJ_CODE -n sadamw3_g-0 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sadamw --lr 1e-3 --gamma 1'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 8 -m 11 -a $PROJ_CODE -n sadamw4_g-0 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sadamw --lr 1e-4 --gamma 1'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 8 -m 11 -a $PROJ_CODE -n sadamw5_g-0 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sadamw --lr 1e-5 --gamma 1'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 8 -m 11 -a $PROJ_CODE -n sadamw3_g-1 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sadamw --lr 1e-3 --gamma 1e-1'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 8 -m 11 -a $PROJ_CODE -n sadamw4_g-1 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sadamw --lr 1e-4 --gamma 1e-1'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 8 -m 11 -a $PROJ_CODE -n sadamw5_g-1 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sadamw --lr 1e-5 --gamma 1e-1'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 8 -m 11 -a $PROJ_CODE -n sadamw3_g-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sadamw --lr 1e-3 --gamma 1e-2'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 8 -m 11 -a $PROJ_CODE -n sadamw4_g-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sadamw --lr 1e-4 --gamma 1e-2'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 8 -m 11 -a $PROJ_CODE -n sadamw5_g-2 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sadamw --lr 1e-5 --gamma 1e-2'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 8 -m 11 -a $PROJ_CODE -n sadamw3_g-3 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sadamw --lr 1e-3 --gamma 1e-3'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 8 -m 11 -a $PROJ_CODE -n sadamw4_g-3 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sadamw --lr 1e-4 --gamma 1e-3'
# lbatch -c 1 -g 1 --gputype rtx_2080 -t 8 -m 11 -a $PROJ_CODE -n sadamw5_g-3 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer sadamw --lr 1e-5 --gamma 1e-3'


lbatch -c 1 -g 1 --gputype A100 -t 8 -m 124 -a $PROJ_CODE -n L7b-4plainK8 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer svgd --lr 1e-4 --model 'meta-llama/Meta-Llama-3-8B' --batch_size 4 --num_epochs 10 --K 8 --write_job_logs 1'
lbatch -c 1 -g 1 --gputype A100 -t 8 -m 124 -a $PROJ_CODE -n L7b-4plainK8 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer svgd --lr 1e-4 --model 'meta-llama/Meta-Llama-3-8B' --batch_size 4 --num_epochs 10 --K 8 --write_job_logs 1'

lbatch -c 1 -g 1 --gputype A100 -t 8 -m 124 -a $PROJ_CODE -n L7b-4plainAdamWK8 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer adamw --lr 1e-4 --model 'meta-llama/Meta-Llama-3-8B' --batch_size 4 --num_epochs 10 --K 8 --write_job_logs 1'
lbatch -c 1 -g 1 --gputype A100 -t 8 -m 124 -a $PROJ_CODE -n L7b-5plainAdamWK8 --queue cnu --venv '../stein_env' --cmd 'python training_loop.py --optimizer adamw --lr 1e-5 --model 'meta-llama/Meta-Llama-3-8B' --batch_size 4 --num_epochs 10 --K 8 --write_job_logs 1'
