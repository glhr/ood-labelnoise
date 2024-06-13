Download ``fgvc-aircraft-2013b.tar.gz`` from the official project page [https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) and extract it here.

Download ``aircraft_osr_splits.pkl`` from [https://github.com/sgvaze/osr_closed_set_all_you_need/tree/main/data/open_set_splits](https://github.com/sgvaze/osr_closed_set_all_you_need/tree/main/data/open_set_splits)

Expected directory structure:
```
.
├── data
├── aircraft_osr_splits.pkl
├── create_txt_files_aircraft.ipynb
├── evaluation.m
├── example_evaluation.m
├── README.html
├── README.md
├── vl_argparse.m
├── vl_pr.m
├── vl_roc.m
└── vl_tpfp.m
```

Below is the official README.md.

-----------

# FGVC-Aircraft Benchmark

**Fine-Grained Visual Classification of Aircraft (FGVC-Aircraft)** is
a benchmark dataset for the fine grained visual categorization of
aircraft.

* [Data, annotations, and evaluation code](archives/fgvc-aircraft-2013b.tar.gz) [2.75 GB | [MD5 Sum](archives/fgvc-aircraft-2013b.html)].
* [Annotations and evaluation code only](archives/fgvc-aircraft-2013b-annotations.tar.gz) [375 KB | [MD5 Sum](archives/fgvc-aircraft-2013b-annotations.html)].
* Project [home page](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/).
* This data was used as part of the fine-grained recognition challenge
  [FGComp 2013](https://sites.google.com/site/fgcomp2013/) which ran
  jointly with the ImageNet Challenge 2013
  ([results](https://sites.google.com/site/fgcomp2013/results)). Please
  note that *the evaluation code provided here may differ* from the
  one used in the challenge.

Please use the following citation when referring to this dataset:

*Fine-Grained Visual Classification of Aircraft*, S. Maji, J. Kannala,
E. Rahtu, M. Blaschko, A. Vedaldi, [arXiv.org](http://arxiv.org/abs/1306.5151), 2013

    @techreport{maji13fine-grained,
       title         = {Fine-Grained Visual Classification of Aircraft},
       author        = {S. Maji and J. Kannala and E. Rahtu
                        and M. Blaschko and A. Vedaldi},
       year          = {2013},
       archivePrefix = {arXiv},
       eprint        = {1306.5151},
       primaryClass  = "cs-cv",
    }

For further information see:

* [Quick start](#quick)
  * [About aircraft](#aircraft)
* [Data and annotation format](#format)
* [Evaluation](#evaluation)
  * [Evaluation metric](#metric)
  * [Evaluation code](#code)
* [Ackwonledgments](#ack)
* [Release notes](#release)

**Note.** This data has been used as part of the *ImageNet FGVC
challenge in conjuction with the International Conference on Computer
Vision (ICCV) 2013*. Test labels were not made available until the
challenge due to the ImageNet challenge policy. They have now been
released as part of the download above. If you arelady downloaded the
iamge archive and want to have access to the test labels, simply
download the annotations archive again.

**Note.** Images in the benchmark are generously made available **for
non-commercial research purposes only** by a number of *airplane
spotters*. Please note that the original authors retain the copyright
of the respective photographs and should be contacted for any other
use. For further details see the [copyright note](#ack) below.

# <a id=quick></a> Quick start

The dataset contains 10,200 images of aircraft, with 100 images for
each of 102 different aircraft model variants, most of which are
airplanes. The (main) aircraft in each image is annotated with a tight
bounding box and a hierarchical airplane model label.

Aircraft models are organized in a four-levels hierarchy. The four
levels, from finer to coarser, are:

* **Model**, e.g. *Boeing 737-76J*. Since certain models are nearly visually
  indistinguishable, this level is not used in the evaluation.
* **Variant**, e.g. *Boeing 737-700*. A variant collapses all the
  models that are visually indistinguishable into one class. The
  dataset comprises 102 different variants.
* **Family**, e.g. *Boeing 737*. The dataset comprises 70 different
  families.
* **Manufacturer**, e.g. *Boeing*. The dataset comprises 41
  different manufacturers.

The data is divided into three equally-sized *training*, *validation*
and *test* subsets. The first two sets can be used for development,
and the latter should be used for final evaluation only. The format of
the data is described [next](#format).

The performance of a fine-grained classification algorithm is
evaluated in term of average class-prediction accuracy. This is
defined as the average of the diagonal of the row-normalized confusion
matrix, as used for example in Caltech-101. Three classification
challenges are considered: variant, family, and manufacturer. An
[evaluation script](#software) in MATLAB is provided.

## <a href=aircraft></a> About aircraft

Aircraft, and in particular airplanes, are alternative to objects
typically considered for fine-grained categorization such as birds and
pets. There are several aspects that make aircraft model recognition
particularly interesting. Firstly, aircraft designs span a hundred
years, including many thousand different models and hundreds of
different makes and airlines. Secondly, aircraft designs vary
significantly depending on the size (from home-built to large
carriers), destination (private, civil, military), purpose
(transporter, carrier, training, sport, fighter, etc.), propulsion
(glider, propeller, jet), and many other factors including
technology. One particular axis of variation, which is is not shared
with categories such as animals, is the fact that the *structure* of
the aircraft changes with their design (number of wings,
undercarriages, wheel per undercarriage, engines, etc.). Thirdly, any
given aircraft model can be re-purposed or used by different
companies, which causes further variations in appearance
(livery). These, depending on the identification task, may be consider
as noise or as useful information to be extracted. Finally, aircraft
are largely rigid objects, which simplifies certain aspects of their
modeling (compared to highly-deformable animals such as cats),
allowing one to focus on the core aspects of the fine-grained
recognition problem.

# <a id=format></a> Data format

The directory `data` contains the images as well as a number of text
files with the data annotations.

Images are contained in the `data/images` sub-directory. They are in
JPEG format and have a name composed of seven digits and the `.jpg`
suffix (e.g. `data/images/1187707.jpg`). The image resolution is about
1-2MP. Each image has at the bottom a banner 20 pixels high containing
[copyright](#ack) information. Please make sure to remove this banner
when using the images to train and evaluate algorithms.

The annotations come in a number of text files. Each line of these
files contains an image name optionally followed by an image
annotation, either a textual label or a sequence of numbers.

`data/images_train.txt` contains the list of training images:
<pre>
0787226
1481091
1548899
0674300
...
</pre>
Similar files `data/images_val.txt` and `data/images_test.txt` contain the list
of validation and test images.

`data/images_variant_train.txt`, `data/images_family_train.txt`, and
`data/images_manufacturer_train.txt` contain the list of training
images annotated with the model variant, family, and manufacturer
names respectively:
<pre>
0787226 Abingdon Spherical Free Balloon
1481091 AEG Wagner Eule
1548899 Aeris Naviter AN-2 Enara
0674300 Aeritalia F-104S Starfighter
...
</pre>
Similar files are provided for the validation and test subsets.

Finally, `data/images_box.txt` contains the aircraft bounding
boxes, one per image. The bounding box is specified by four numbers:
*xmin*, *ymin*, *xmax* and *ymax*. The top-left pixel of an image has
coordinate (1,1).

# <a id=evaluation></a> Evaluation

The performance of a classifier is measured in term of its average
classification accuracy, as detailed next.

## <a id=metric></a> Evaluation metric

The output of a classification algorithm must be a list of triplets of
the type (*image*,*label*,*score*), where

* *image* is an image label, i.e. a seven-digit number,
* *label* is an image label, i.e.. an aircraft model variant, family, or manufacturer, and
* *score* is a real number expressing the belief in the judgment.

When computing the classification accuracy, an image is assigned the
label contained in its highest-scoring triplet. An image that has no
triplets is considered unclassified and always count as a
classification error (therefore it is better to guess at least one
label for each image rather than leaving it unclassified).

The quality of the predictions is measured in term of *average
accuracy*, obtained as follows:

* The confusion matrix is square, with one row per class.
* Each element of the confusion matrix is the number of time aircraft
  of a given class (specified by the row) are classified as a second
  class (column). Ideally, the confusion matrix should be diagonal.
* The confusion matrix is row-normalized by the number of images of
  the corresponding aircraft class (each row therefore sums to one if
  there are no unclassified images).
* The average accuracy is computed as the average of the diagonal of
  the confusion matrix.

There are three challenges: classifying the aircraft variant, family, and manufacturer.

## <a id=code></a> Evaluation code

The evaluation protocol has been implemented in the MATLAB m-file
`evaluation.m`. This function takes the path to the `data` folder, a
composite name indicating the evaluation subset and challenge
(e.g. `'manufacturer_test'` or `'family_val'`), and the list of
triplets, and returns the confusion matrix. For example

<pre>
images = {'2074164'} ;
labels = {'McDonnell Douglas MD-90-30'} ;
scores = 1 ;
confusion = evaluate('/path/fgcv-aircraft/data', 'test', images, labels, scores) ;
accuracy = mean(diag(confusion)) ;
</pre>

evaluates a classifier output containing exactly one triplet (image,
label, score), where the image is `'2074164'`, its predicted class is
`'McDonnell Douglas MD-90-30'`, and the score of the prediction is
`1`. In practice, a complete set of predictions (one for each
image-class pair) is usually evaluated.

See the builtin help of the `evaluation` MATLAB functions for further
practical details. See also `example_evaluation.m` for examples on how
to use this function.

# <a id=ack></a> Acknowledgments

The creation of this dataset started during the *Johns Hopkins CLSP
Summer Workshop 2012*
[Towards a Detailed Understanding of Objects and Scenes in Natural Images](http://www.clsp.jhu.edu/workshops/archive/ws-12/groups/tduosn/)
with, in alphabetical order, Matthew B. Blaschko, Ross B. Girshick,
Juho Kannala, Iasonas Kokkinos, Siddharth Mahendran, Subhransu Maji,
Sammy Mohamed, Esa Rahtu, Naomi Saphra, Karen Simonyan, Ben Taskar,
Andrea Vedaldi, and David Weiss.

The CLSP workshop was supported by the National Science Foundation via
Grant No 1005411, the Office of the Director of National Intelligence
via the JHU Human Language Technology Center of Excellence; and Google
Inc.

A special thanks goes to Pekka Rantalankila for helping with the
creation of the airplane hieararchy.

Many thanks to the photographers that kindly made available their
images for research purposes. Each photographer is listed below, along
with a link to his/her [airlners.net](http://airliners.net) page:

* [Mick Bajcar](http://www.airliners.net/profile/dendrobatid)
* [Aldo Bidini](http://www.airliners.net/profile/aldobid)
* [Wim Callaert](http://www.airliners.net/profile/minoeke)
* [Tommy Desmet](http://www.airliners.net/profile/tommypilot)
* [Thomas Posch](http://www.airliners.net/profile/snorre)
* [James Richard Covington](http://www.airliners.net/profile/lemonkitty)
* [Gerry Stegmeier](http://www.airliners.net/profile/stegi)
* [Ben Wang](http://www.airliners.net/profile/aal151heavy)
* [Darren Wilson](http://www.airliners.net/profile/dazbo5)
* [Konstantin von Wedelstaedt](http://www.airliners.net/profile/fly-k)

Please note that the images are made available **exclusively for
non-commercial research purposes**. The original authors retain the
copyright on the respective pictures and should be contacted for any
other usage of them.

# <a id=release></a> Release notes

* *FGVC-Aircraft 2013b* - The same as 2013a, but with test annotations included.
* *FGVC-Aircraft 2013a* - First public release of the data.
