#!/bin/bash
. ~/miniconda3/etc/profile.d/conda.sh
conda activate wsc

#DER
MODEL_NAME=der_20
python main.py --config=./exps/der_memory/$MODEL_NAME.json

MODEL_NAME=der_80
python main.py --config=./exps/der_memory/$MODEL_NAME.json

MODEL_NAME=der_200
python main.py --config=./exps/der_memory/$MODEL_NAME.json

MODEL_NAME=der_400
python main.py --config=./exps/der_memory/$MODEL_NAME.json

#FOSTER
MODEL_NAME=foster_20
python main.py --config=./exps/foster_memory/$MODEL_NAME.json

MODEL_NAME=foster_80
python main.py --config=./exps/foster_memory/$MODEL_NAME.json

MODEL_NAME=foster_200
python main.py --config=./exps/foster_memory/$MODEL_NAME.json

MODEL_NAME=foster_400
python main.py --config=./exps/foster_memory/$MODEL_NAME.json

#MEMO
MODEL_NAME=memo_20
python main.py --config=./exps/memo_memory/$MODEL_NAME.json

MODEL_NAME=memo_80
python main.py --config=./exps/memo_memory/$MODEL_NAME.json

MODEL_NAME=memo_200
python main.py --config=./exps/memo_memory/$MODEL_NAME.json

MODEL_NAME=memo_400
python main.py --config=./exps/memo_memory/$MODEL_NAME.json

#Replay
MODEL_NAME=replay_20
python main.py --config=./exps/replay_memory/$MODEL_NAME.json

MODEL_NAME=replay_80
python main.py --config=./exps/replay_memory/$MODEL_NAME.json

MODEL_NAME=replay_200
python main.py --config=./exps/replay_memory/$MODEL_NAME.json

MODEL_NAME=replay_400
python main.py --config=./exps/replay_memory/$MODEL_NAME.json

#iCaRL
MODEL_NAME=icarl_20
python main.py --config=./exps/icarl_memory/$MODEL_NAME.json

MODEL_NAME=icarl_80
python main.py --config=./exps/icarl_memory/$MODEL_NAME.json

MODEL_NAME=icarl_200
python main.py --config=./exps/icarl_memory/$MODEL_NAME.json

MODEL_NAME=icarl_400
python main.py --config=./exps/icarl_memory/$MODEL_NAME.json

#BiC
MODEL_NAME=bic_20
python main.py --config=./exps/bic_memory/$MODEL_NAME.json

MODEL_NAME=bic_80
python main.py --config=./exps/bic_memory/$MODEL_NAME.json

MODEL_NAME=bic_200
python main.py --config=./exps/bic_memory/$MODEL_NAME.json

MODEL_NAME=bic_400
python main.py --config=./exps/bic_memory/$MODEL_NAME.json

#WA
MODEL_NAME=wa_20
python main.py --config=./exps/wa_memory/$MODEL_NAME.json

MODEL_NAME=wa_80
python main.py --config=./exps/wa_memory/$MODEL_NAME.json

MODEL_NAME=wa_200
python main.py --config=./exps/wa_memory/$MODEL_NAME.json

MODEL_NAME=wa_400
python main.py --config=./exps/wa_memory/$MODEL_NAME.json

#OURS
MODEL_NAME=wsc_20
python main.py --config=./exps/wsc_memory/$MODEL_NAME.json

MODEL_NAME=wsc_80
python main.py --config=./exps/wsc_memory/$MODEL_NAME.json

MODEL_NAME=wsc_200
python main.py --config=./exps/wsc_memory/$MODEL_NAME.json

MODEL_NAME=wsc_400
python main.py --config=./exps/wsc_memory/$MODEL_NAME.json

#end
