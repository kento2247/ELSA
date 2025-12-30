mkdir -p data
cd data

# Download RELATE_wave dataset
aria2c -x10 -s10 -k1M https://sarulab.sakura.ne.jp/kanamori/RELATE_open_dataset/RELATE_wave.zip
unzip RELATE_wave.zip
rm RELATE_wave.zip
git clone git@github.com:sarulab-speech/RELATE.git

# Download HumanEval dataset
aria2c -x10 -s10 -k1M https://zenodo.org/records/10737388/files/human_eval.zip?download=1
unzip human_eval.zip
rm human_eval.zip

# Download XACLE dataset
aria2c -x10 -s10 -k1M https://y-okamoto.sakura.ne.jp/XACLE_Challenge/2025/dataset/XACLE_dataset_train_val.zip
aria2c -x10 -s10 -k1M https://y-okamoto.sakura.ne.jp/XACLE_Challenge/2025/dataset/XACLE_test_data_with_score.zip
unzip XACLE_dataset_train_val.zip
unzip XACLE_test_data_with_score.zip
rm XACLE_dataset_train_val.zip
rm XACLE_test_data_with_score.zip
rm -rf __MACOSX
cd ..


mkdir -p models
cd models

# Download pre-trained model
aria2c -x10 -s10 -k1M -o 630k-audioset-best.pt https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt?download=true

cd ..