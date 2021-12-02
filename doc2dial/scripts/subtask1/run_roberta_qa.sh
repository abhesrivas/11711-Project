python run_qa.py \
 --dataset_name '../datasets/doc2dial/doc2dial.py' \
 --dataset_config_name doc2dial_rc \
 --model_name_or_path ankur310794/roberta-base-squad2-nq \
 --do_train \
 --do_eval \
 --version_2_with_negative \
 --logging_steps 2000 \
 --save_steps 2000 \
 --learning_rate 3e-5  \
 --num_train_epochs 2 \
 --max_seq_length 512  \
 --max_answer_length 80 \
 --doc_stride 256  \
 --cache_dir roberta_nq \
 --output_dir roberta_nq \
 --overwrite_output_dir  \
 --per_device_train_batch_size 16 \
 --per_device_train_batch_size 16 \
 --gradient_accumulation_steps 2  \
 --warmup_steps 1000 \
 --weight_decay 0.01  \
 --fp16