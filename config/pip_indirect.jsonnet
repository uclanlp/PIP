local dataset = "paranmt";
local model_type = "pip";
local pretrained_model = "facebook/bart-base";
local data_dir = "./dataset/data_300k";
local use_enc_src_parse = false;
local use_dec_tgt_parse = false;
local prefix_type = "pip_indirect";
local prefix_learning_rate = "3e-04"; 

local model_name_map = {
    "facebook/bart-base": "bart-base", 
    "gpt2": "gpt2",
};

{
    "dataset": dataset, 
	"seed": 0, 
    "gpu_num": 4,
	"model_type": model_type, 
	"pretrained_model": pretrained_model,
    "prefix_type": prefix_type,
    "cache_dir": "./cache/", 
    "train_src_sent_file": "%s/para/train/src.txt" % [data_dir], 
    "train_src_synt_file": "%s/para/train/src.parse" % [data_dir], 
    "train_tgt_sent_file": "%s/para/train/tgt.txt" % [data_dir], 
    "train_tgt_synt_file": "%s/para/train/tgt.parse" % [data_dir], 
    "dev_src_sent_file": "%s/para/test/src.txt" % [data_dir], 
    "dev_src_synt_file": "%s/para/test/src.parse" % [data_dir], 
    "dev_tgt_sent_file": "%s/para/test/tgt.txt" % [data_dir], 
    "dev_tgt_synt_file": "%s/para/test/tgt.parse" % [data_dir], 
    "pan_src_sent_file": "%s/pan/dev/src.txt" % [data_dir], 
    "pan_src_synt_file": "%s/pan/dev/src.parse" % [data_dir], 
    "pan_tgt_sent_file": "%s/pan/dev/tgt.txt" % [data_dir], 
    "pan_tgt_synt_file": "%s/pan/dev/tgt.parse" % [data_dir], 
    "mrpc_src_sent_file": "%s/mrpc/dev/src.txt" % [data_dir], 
    "mrpc_src_synt_file": "%s/mrpc/dev/src.parse" % [data_dir], 
    "mrpc_tgt_sent_file": "%s/mrpc/dev/tgt.txt" % [data_dir], 
    "mrpc_tgt_synt_file": "%s/mrpc/dev/tgt.parse" % [data_dir], 
    "quora_src_sent_file": "%s/quora/dev/src.txt" % [data_dir], 
    "quora_src_synt_file": "%s/quora/dev/src.parse" % [data_dir], 
    "quora_tgt_sent_file": "%s/quora/dev/tgt.txt" % [data_dir], 
    "quora_tgt_synt_file": "%s/quora/dev/tgt.parse" % [data_dir], 
    "max_src_sent_len": 60, 
    "max_src_synt_len": 260, 
    "max_tgt_sent_len": 60, 
    "max_tgt_synt_len": 200, 
    "use_enc_src_parse": use_enc_src_parse, 
    "use_dec_tgt_parse": use_dec_tgt_parse, 
    "warmup_epoch": 0, 
    "max_epoch": 0, 
    "prefix_max_epoch": 10, 
    "prefix_length": 200,
    "prefix_threshold": 4,
    "train_batch_size": 64, 
    "eval_batch_size": 64, 
    "prefix_learning_rate": prefix_learning_rate, 
    "output_dir": "./outputs/%s_%s_%s_%s_%s%s%s/" % [model_type, prefix_type, prefix_learning_rate, model_name_map[pretrained_model], dataset, 
                                             if use_enc_src_parse then "_use-enc-src-parse" else "", 
                                             if use_dec_tgt_parse then "_use-dec-tgt-parse" else ""], 
    "weight_decay": 0.0, 
    "grad_clipping": 1.0, 
    "num_beams": 4, 
    "sep_token": "<sep>", 
}
