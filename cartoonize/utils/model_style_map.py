model_dict = {}

styles = ["Type-0", "Type-1", "Type-2", "Type-3"]
checkpoints = ["models/whitebox/whitebox/version_0/checkpoints/epoch=4.ckpt", "models/whitebox-v2/whitebox/version_0/checkpoints/epoch=4.ckpt", "models/whitebox-v2/whitebox/version_1/checkpoints/epoch=4.ckpt", "models/whitebox-v2/whitebox/version_2/checkpoints/epoch=6.ckpt"]

for i, style in enumerate(styles):
    model_dict[style] = checkpoints[i]

input_dir_wb = "wb_cartoonization/asset"
