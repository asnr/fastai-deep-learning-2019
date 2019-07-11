## Clean up instance

Commit and push code changes


```sh
ls *.ipynb | xargs -n 1 /opt/anaconda3/bin/jupytext --to py
git add *.py
git commit -m ...
git push
```

## Figuring out spectrograms

https://dsp.stackexchange.com/questions/45670/self-studying-getting-a-quality-spectrogram?newreg=ea77ab8a239c4be0b1434f8c9d65b8eb
https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html
https://towardsdatascience.com/audio-classification-using-fastai-and-on-the-fly-frequency-transforms-4dbe1b540f89


## Papers

Visualizing and Understanding Convolutional Networks, Zeiler, Fergus
