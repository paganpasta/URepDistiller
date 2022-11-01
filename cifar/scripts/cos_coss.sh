#cos
python train_unsupervised_student.py --model_t resnext32x4 --model_s  resnext8x4 --path_t ./save/models/resnext32x4_vanilla/ckpt_epoch_240.pth --distill cos_pre --beta 50.0

#coss
python train_unsupervised_student.py --model_t resnext32x4 --model_s  resnext8x4 --path_t ./save/models/resnext32x4_vanilla/ckpt_epoch_240.pth --distill coss_pre --w_cos 1.0 --beta 50.0

#sld
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --beta 1.0
python train_unsupervised_student.py --model_t resnext32x4 --model_s  resnext8x4 --path_t ./save/models/resnext32x4_vanilla/ckpt_epoch_240.pth --distill kd --beta 1.0

#AT
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill at --beta 1000.0
python train_unsupervised_student.py --model_t resnext32x4 --model_s  resnext8x4 --path_t ./save/models/resnext32x4_vanilla/ckpt_epoch_240.pth --distill at --beta 1000.0

#SP
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill sp --beta 3000.0
python train_unsupervised_student.py --model_t resnext32x4 --model_s  resnext8x4 --path_t ./save/models/resnext32x4_vanilla/ckpt_epoch_240.pth --distill sp --beta 3000.0

#VID
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill vid --beta 2.0
python train_unsupervised_student.py --model_t resnext32x4 --model_s  resnext8x4 --path_t ./save/models/resnext32x4_vanilla/ckpt_epoch_240.pth --distill vid --beta 2.0

#RKD
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill rkd --beta 1.0
python train_unsupervised_student.py --model_t resnext32x4 --model_s  resnext8x4 --path_t ./save/models/resnext32x4_vanilla/ckpt_epoch_240.pth --distill rkd --beta 1.0

#PKT
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill pkt --beta 30000.0
python train_unsupervised_student.py --model_t resnext32x4 --model_s  resnext8x4 --path_t ./save/models/resnext32x4_vanilla/ckpt_epoch_240.pth --distill pkt --beta 30000.0

#Factor
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill factor --beta 200.0
python train_unsupervised_student.py --model_t resnext32x4 --model_s  resnext8x4 --path_t ./save/models/resnext32x4_vanilla/ckpt_epoch_240.pth --distill factor --beta 200.0

#NST
python train_unsupervised_student.py --model_t wrn_40_2 --model_s  wrn_16_2 --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill nst --beta 100.0
python train_unsupervised_student.py --model_t resnext32x4 --model_s  resnext8x4 --path_t ./save/models/resnext32x4_vanilla/ckpt_epoch_240.pth --distill nst --beta 100.0

#DINO

#CRD
