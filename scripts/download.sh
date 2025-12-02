mkdir -p data
cd data

# Download RELATE_wave dataset
aria2c -x10 -s10 -k1M https://sarulab.sakura.ne.jp/kanamori/RELATE_open_dataset/RELATE_wave.zip
unzip RELATE_wave.zip
rm RELATE_wave.zip
git clone git@github.com:sarulab-speech/RELATE.git

cd ..