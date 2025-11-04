mkdir ./data
wget https://downloads.wortschatz-leipzig.de/corpora/hye_wikipedia_2021_1M.tar.gz -O ./data/wiki_2021.tar.gz
tar -xvzf ./data/wiki_2021.tar.gz -C ./data
rm ./data/wiki_2021.tar.gz

# wget https://downloads.wortschatz-leipzig.de/corpora/hye_community_2017.tar.gz -O ./data/community_2017.tar.gz
# tar -xvzf ./data/community_2017.tar.gz  -C ./data
# rm ./data/community_2017.tar.gz