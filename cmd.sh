#t2t-datagen --t2t_usr_dir=script --problem=my_problem --data_dir=./data

NUM=0
TRAIN=./train/$NUM

if [ ! -d $TRAIN ];then
	mkdir $TRAIN
fi

#t2t-trainer --t2t_usr_dir=script --problem=my_problem --data_dir=./data --model=lstm_seq2seq_attention --hparams_set=lstm_attention --output_dir=$TRAIN

t2t-decoder --t2t_usr_dir=script --problem=my_problem --data_dir=./data --model=lstm_seq2seq_attention --hparams_set=lstm_attention --output_dir=$TRAIN --decode_hparams="beam_size=4,alpha=0.6" --decode_from_file=decoder/seq_crt_q.txt --decode_to_file=decoder/results/a$NUM.txt

#t2t-exporter --t2t_usr_dir=script --problem=my_problem --data_dir=./data --model=lstm_seq2seq_attention --hparams_set=lstm_attention --output_dir=$TRAIN
