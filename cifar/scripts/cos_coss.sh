# COS PRE
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill cos_pre --beta 0.5
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill cos_pre --beta 1.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill cos_pre --beta 5.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill cos_pre --beta 10.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill cos_pre --beta 25.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill cos_pre --beta 50.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill cos_pre --beta 100.0
#COS POST
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill cos_post --beta 1.0
#COSS PRE
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill coss_pre --w-coss 1.0 --beta 0.5
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill coss_pre --w-coss 1.0 --beta 1.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill coss_pre --w-coss 1.0 --beta 5.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill coss_pre --w-coss 1.0 --beta 10.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill coss_pre --w-coss 1.0 --beta 25.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill coss_pre --w-coss 1.0 --beta 50.0
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill coss_pre --w-coss 1.0 --beta 100.0
#COSS POST
python train_unsupervised_student.py --model_t wrn_40_2 --model_s wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill coss_post --w-cos--beta 1.0
