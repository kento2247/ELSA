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

# # Download XACLE dataset
# aria2c -x10 -s10 -k1M https://y-okamoto.sakura.ne.jp/XACLE_Challenge/2025/dataset/XACLE_dataset_train_val.zip
# aria2c -x10 -s10 -k1M https://y-okamoto.sakura.ne.jp/XACLE_Challenge/2025/dataset/XACLE_test_data_with_score.zip
# unzip XACLE_dataset_train_val.zip
# unzip XACLE_test_data_with_score.zip
# rm XACLE_dataset_train_val.zip
# rm XACLE_test_data_with_score.zip
# rm -rf __MACOSX

# Download AISHELL-7A dataset
gdown 1KjrZAzmd3k3BWZ0XofwvOG-0jvsiRjCQ
unzip MusicEval-full.zip
rm MusicEval-full.zip

# Download Clotho dataset
git clone git@github.com:lourson1091/audiobertscore.git
mkdir clotho
mv audiobertscore/wave_all_16k clotho
mv audiobertscore/clotho_ovl_rel_test_set.csv clotho
rm -rf audiobertscore

# Download COMPA dataset
gdown 1A_HDH0sO6Pp-kvdcTJrAA6MJZiItHZTQ
gdown 1vWpq2fTcT8T7ec8pZ_EG2v29PwJPfcJm
unzip CompA-attribute.zip
unzip CompA-order.zip
rm CompA-attribute.zip
rm CompA-order.zip
rm -rf __MACOSX
mv CompA\ Attribute CompA_attribute
mkdir CompA_order
mv CompA_order_files CompA_order/
mv CompA_order_benchmark.csv CompA_order/


cd ..
mkdir -p models
cd models

# Download pre-trained model
aria2c -x10 -s10 -k1M -o 630k-audioset-best.pt https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt?download=true
aria2c -x10 -s10 -k1M -o clap-sep-best.pt https://huggingface.co/spaces/AisakaMikoto/CLAPSep/resolve/main/model/best_model.ckpt?download=true
aria2c -x10 -s10 -k1M -o clapsep-clap.ckpt https://huggingface.co/spaces/AisakaMikoto/CLAPSep/blob/main/model/music_audioset_epoch_15_esc_90.14.pt

aria2c -x10 -s10 -k1M -o soloaudio_vae.pt https://huggingface.co/westbrook/SoloAudio/resolve/main/audio-vae.pt?download=true 
aria2c -x10 -s10 -k1M -o soloaudio.pt https://huggingface.co/westbrook/SoloAudio/resolve/main/soloaudio_v2.pt?download=true

cd ..