## preprocessing train clean

cd preprocess_audio && python preprocess.py \
    --src-path /home/stud_vantuan/share_with_150/cache/cd92.93_95_with_5h_clean_and_5h_noisy/clean/train \
    --tgt-magnitude-path /home/stud_vantuan/share_with_150/train_uvcganv2/CD92_5h_5h/train/clean \
    --tgt-metadata-path /home/stud_vantuan/share_with_150/train_uvcganv2/CD92_5h_5h/metadata/train/clean \
    --threads 16

# preprocessing train noisy, it's test
cd preprocess_audio && python preprocess.py \
    --src-path /home/stud_vantuan/share_with_150/cache/cd92.93_95_with_5h_clean_and_5h_noisy/noisy/test \
    --tgt-magnitude-path /home/stud_vantuan/share_with_150/train_uvcganv2/CD92_5h_5h/train/noisy \
    --tgt-metadata-path /home/stud_vantuan/share_with_150/train_uvcganv2/CD92_5h_5h/metadata/train/noisy \
    --threads 16

# copy to pretrain pool
cp -t /home/stud_vantuan/share_with_150/train_uvcganv2/CD92_5h_5h/pretrain_pool /home/stud_vantuan/share_with_150/train_uvcganv2/CD92_5h_5h/train/clean/*
cp -t /home/stud_vantuan/share_with_150/train_uvcganv2/CD92_5h_5h/pretrain_pool /home/stud_vantuan/share_with_150/train_uvcganv2/CD92_5h_5h/train/noisy/*