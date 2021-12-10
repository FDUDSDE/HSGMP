# heterogeneous-scene-graph

Since the data is too large, we do not provide the data, including visual & textual scene graph, please generate the data by yourself using the code in [data generation](#data generation).

## Requirement
```
python>=3.6
pytorch
```

```
git submodule update --init --recursive
```

## Install parser enviroment
```
pip install spacy
python -m spacy download en
```

## Install faster R-CNN
follow the repo's instruction to install [Faster R-CNN](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome.git)
```
git clone https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome.git
cd lib
python setup.py build develop
```
download pre-trained model


## data generation
The code of image scene graph generation is from [Unbiased Scene Graph Generation from Biased Training](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

The code of text scene graph generation is from [SceneGraphParser](https://github.com/vacancy/SceneGraphParser)

The code of image feature extraction is from [Faster-R-CNN-with-model-pretrained-on-Visual-Genome](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome)
