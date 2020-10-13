# VI-Glow
PyTorch Code for our paper: "Color Visual Illusions: A Statistics-based Computational Model", NeurIPS 2020

(( Very soon... ))

# Usage

**Install Requirements**
```
pip3 install -r requirements.txt
```

**Prepare your data**
Download an external dataset, i.e. [Places](http://places2.csail.mit.edu/download.html).

This implementation uses torchvision.datasets.ImageFolder. Therefore, the images should be arranges as follows:

```
.
+-- <dataset_folder>
|   +-- <class folder>
|.     +-- *.png
+-- _drafts
|   +-- begin-with-the-crazy-ideas.textile
|   +-- on-simplicity-in-technology.markdown
+-- _includes
|   +-- footer.html
|   +-- header.html
+-- _layouts
|   +-- default.html
|   +-- post.html
+-- _posts
|   +-- 2007-10-29-why-every-programmer-should-play-nethack.textile
|   +-- 2009-04-26-barcamp-boston-4-roundup.textile
+-- _data
|   +-- members.yml
+-- _site
+-- index.html
```


