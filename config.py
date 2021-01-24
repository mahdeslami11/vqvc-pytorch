import torch
from utils.path import get_path

class Arguments:

	""" 
		path configurations
	"""
	dataset_name = "korean_emotional_speech_dataset"
	dataset_path = get_path("/home/minsu/dataset/", dataset_name)

	converted_sample_dir = "results"	
	prepro_dir = "preprocessed"
	model_log_dir = "logs"
	model_checkpoint_dir = "ckpts"
	

	# path for loading audio(wav) samples to be preprocessed
	wav_dir = get_path(dataset_path, "wavs")

	# by default, preprocessed samples and metadata are stored in "prepro_path"
	prepro_path = get_path(prepro_dir, dataset_name)
	prepro_mel_dir = get_path(prepro_path, "mels")
	prepro_meta_dir = get_path(prepro_path, "metas")
	prepro_meta_train = get_path(prepro_meta_dir, "meta_train.csv")
	prepro_meta_eval = get_path(prepro_meta_dir, "meta_eval.csv")

	model_log_path = get_path(model_log_dir, dataset_name)
	model_checkpoint_path = get_path(model_checkpoint_dir, dataset_name)


	"""
		preprocessing hyperparams
	"""
	sr = 22050	# 22050kHz sampling rate
	sample_frame = 128
	n_mels = 80
	filter_length = 1024
	hop_length = 256
	win_length = 1024

	max_wav_value = 32768.0	# for other dataset
	mel_fmin = 0
	mel_fmax = 8000


	"""
		VQVC hyperparameters
	"""
	encoder_hidden= 256
	encoder_kernel_size = 3
	z_dim=32		# bottleneck dimension

	n_embeddings = 128
	commitment_cost = 0.01

	decoder_kernel_size = 3
	decoder_hidden = 256

	norm_epsilon = 1e-4
	speaker_emb_discount_factor=3

	warmup_steps = 1000
	init_lr = 1e-5
	max_lr = 4e-4
	gamma = 0.25
	milestones = [20000]


	"""
        	data & training setting
	"""
	grad_clip_thresh=2.0
	seed = 999
	n_workers = 0

	#scheduler setting

	use_cuda = True
	mem_mode = True

	data_split_ratio = [0.95, 0.05]		# [train, evaluation] in 0 ~ 1 range
	train_batch_size = 64
	eval_batch_size = 100 
	eval_step = 1000
	eval_path = "eval_results"
	save_checkpoint_step = 10000

	log_tensorboard = True
	max_training_step = 600000

	# vocoder setting
	vocoder = "vocgan"
	vocoder_pretrained_model_name = "vocgan_kss_pretrained_model_epoch_4500.pt"
	vocoder_pretrained_model_path = get_path("./vocoder", "{}", "pretrained_models", vocoder_pretrained_model_name).format(vocoder)

