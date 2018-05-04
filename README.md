# Final Project

## Disclaimer!
This project was submitted as a final project for a college computer vision course.  I do not plan to continue development on this repository, and as such a few points need to be made clear up front:

* The full dataset is not contained in this repository. Not only does the data contain several GB of satellite imagery, but it contains raw latitude and longitude coordinates for farmland which can be traced back to the farmer who volunteered harvest/yield data related to their crop.  As such, the intermediate product these scripts create have been provided in an anonymized form.  This means the dataset -- as it was used in my analysis and reporting -- is accessible and usable from this repo, but I will not be performing any "advanced" methods to enrich the dataset further (for example, providing anonymized pixel information to retain locality and adjacency).
* There are undocumented steps in preparing files (such as the imagery geoTIFF files) which are necessary before the ``setup.py`` program can process the files. I will discuss what the script expects, but will make no effort to detail all the work I did to get them to that point.
* If you are anyone other than my professor or TA, I probably won't respond to requests for further details or explanation not already outlined in this README.

## Getting Started

All files should be ran from the directory in which they are located.

* Dependencies:
  * Python 2.7
  * scikit-learn (sklearn) -- any version prior to 0.20
  * numpy
  * geoio
  * gdal/osgeo

### Working File Structure

* FinalProject
  * Data
    * proc - contains intermediate csv files produced by setup.py
      * csv
        * 2016 - folder(s) containing yield measurements collected from combine harvesting (see ``setup.py`` for details), each file being data for each field
        * 2017
      * img
        * 2016 - folder(s) containing transformed satellite imagery (see ``setup.py`` for details), each image being its own band.
        * 2017 ...
    * raw
      * img - folder containing ``tar.gz`` archives of satellite imagery, and informational txt file describing each band designation
      * yield
        * 2016 - folder(s) containing original zip files sent from farmer. These are a combination of ``.dat`` and ``.txt`` files readable by ASF View by Chase IH.
        * 2017 ...
  * paper - tex, pdf, and image files related to my final report
  * src - python code that does all the heavy lifting

### setup.py

__This file shouldn't ever need to be ran if you want to use the farm yield dataset.__ The purpose of this program was to reading in satellite imagery (already transformed from UTM to lat/long coordinate system, a.k.a, WGS 84) found in ``/data/proc/img/YYYY``. The global variable ``IMGS`` contains the file names of the geoTIFF files it expects (and can be modified to include more if necessary)

This program also requires harvest data in ``.csv`` format, with at least the following values per record:

* __Longitude__
* __Latitude__
* __Product__ (referring to crop name)
* __Yld Vol(Dry)(bu/ac)__

The pipleline of this program is as follows:

Bin each record (measured in lat/long) by corresponding pixel (x,y) by yield ->
Grab the pixel value for each pixel that has been identified, across all bands ->
Write out to the intermediate file all the new features

### main.py

Running this program from terminal will not do what you expect -- it returns a class object of my own definition which loosely follows the sklearn.datasets objects.  It is recommended that you use this in combination with a driver program, or the python interpreter (or interactive notebook), and simply call the ``load_data()`` function, which will return the dataset you probably want.

I __have__ taken the time to document the ``Dataset`` class object which is returned, which you can check out in the docstring either in the code or by using ``.__doc__`` on the object.

#### Helpful Functions you may want to know about

* ``load_data()`` -- __THE__ function that makes and returns the dataset. This reads in the intermediate files and builds the Dataset class every time it is ran, keep this in mind.
* ``std_transform(DATA)`` -- This transforms the data into a standard normal curve using ``StandardScalar`` in numpy, and returns the transformed result as an ``np.array``
* ``do_pca(DATA, LABELS=None, COMPONENTS=None)`` -- this performs Principal Component Analysis on the dataset. This, too, has a docstring if you want to find out more.
* ``random_forrest(X, Y)`` -- given x and y as data and targets respectively, this will build a random forrest classifier, split the data into training and testing sets, train the RF and then perform k-fold cross validation -- printing the output scores for accuracy.
* ``svr_est(X,Y)`` -- given x and y as data and targets respectively, this will build a support vector machine for regression (SVR), using a linear kernel, split the data into training and testing sets, train the SVR and then perform k-fold cross validation -- printing the output scores for r2.
