
<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![MIT License][license-shield]][license-url]
![tensorflow badge](https://img.shields.io/badge/tensorflow-2.2.0-blue)
![keras badge](https://img.shields.io/badge/keras-2.4.0-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-1.2.0-brightgreen)
![scipy](https://img.shields.io/badge/scipy-1.2.0-brightgreen)


<!-- PROJECT LOGO -->
<br />
<p align="center">
<!--   <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80"> -->
  </a>

  <h1 align="center">Spot the difference</h1>

  <p align="center">
    Graduation project using image conversion & GAN
    <br />
    <a href="https://github.com/doljae/image2emage"><strong>Explore the docs »</strong></a>
    <br />
    <br />
<!--     <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a> -->
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Motivation](#motivation)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [About GAN Model...](#About-GAN-Model)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project
When you enter an image, it creates a transformed image with some modifications to the objects inside the original image.<br>

![캡처](https://user-images.githubusercontent.com/37795866/85731582-9938af00-b735-11ea-8d78-994b6ef147bd.JPG)

Brief description
* When you input an image, objects inside the image are detected through a suitable algorithm.
* Various conversion methods (rotation, inversion, substitution with other objects, GAN conversion, etc.) are added to objects detected from the original image.
* Create a new image by combining the original image and the converted object.

### Motivation
The project started with creating the images needed for the `spot the difference` game. The `spot the difference` game is a game that finds different parts by comparing two similar images at a given time. In general, the game is played using a fixed image set in the local environment. With this problem in mind, I started a project that analyzes the input image and creates a similar transform image.

This project provides the ability to detect small objects inside the input image and convert the detected objects by applying various image processing techniques.

### Built With

* [Python 3.7](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [Tensorflow](https://www.tensorflow.org/?hl=ko)
* [Keras](https://keras.io/)


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* OpenCV with contrib modules
```sh
pip install opencv-contrib-python
```
* Scipy 1.2.0(Important)
```sh
pip install scipy==1.2.0
```
* Pillow
```sh
pip install pillow
```
* Tensorflow
```sh
pip install tensorflow==2.2.0
```
* Keras
```sh
pip install Keras==2.2.0
```
* Numpy
```sh
pip install numpy
```
Or you can install the libraries you need to use at once. It is suitable when using a new virtual environment (interpreter).
```sh
pip install -r requirements.txt
```

### Installation

1. Complete the prerequisite settings

2. Clone the repo
```sh
git clone https://github.com/doljae/image2emage.git
```
3. Unzip the repo and put the folder in the path to use the library.

4. Please refer to the manual to `import` and use the library.

5. For detailed usage instructions, see `user_guide.pdf`

<!-- USAGE EXAMPLES -->
## Usage
Please refer to the [Wiki page](https://github.com/doljae/convert_image_maker/wiki)

## About GAN Model...
The GAN model is required for image conversion using the GAN model. Please put the GAN model in the directory below.
```sh
./pix2pix/your_gan_model.h5
```
We attach a [link](https://drive.google.com/file/d/1Qhoa712WZGNe0QfIPoHS5sAfJBD1PQqd/view?usp=sharing) to download the GAN model for users who are in a condition where the model is difficult to learn, or who need quick testing.<br><br>

(Note) The artificial intelligence model depends on the data used for learning.
If you build a model with a lot of high quality data, you will get much better results.<br><br>
For information on GAN, please refer to the link below.
<br>
[Pix2Pix main](https://phillipi.github.io/pix2pix/)<br>
[Pix2Pix paper](https://arxiv.org/abs/1611.07004)<br>
[Pix2Pix repo](https://github.com/phillipi/pix2pix)<br>
[Pix2Pix Tensorflow exercise](https://www.tensorflow.org/tutorials/generative/pix2pix)<br>






<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/doljae/image2emage/issues) for a list of proposed features (and known issues).




## Release History
* 2.1.5
    * CHAGNE: Updated guide for image conversion with GAN model (2/3)
    * CHAGNE: Directory & structure description
* 2.1.0
    * CHAGNE: Updated guide for image conversion with GAN model (1/3)
    * CHAGNE: Upload the created GAN model file (.h5) to an external drive.
* 2.0.0
    * CHAGNE: Code refactoring(2/2)
    * CHANGE: Update docs (module code remains unchanged)
* 1.5.0
    * CHAGNE: Code refactoring(1/2)
* 1.0.0
    * The first release
    * FIX: Crash when calling GAN model file(.h5) with absolute directory
* 0.5.0
    * The first function test
* 0.2.0
    * Work in progress


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project(<https://github.com/doljae/image2emage/fork>)
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the `GPLv3` License. See [`LICENSE`](https://github.com/doljae/image2emage/blob/master/LICENSE.md) for more information.



<!-- CONTACT -->
## Contact
* Gwangryoun Kim - gais0726@gmail.com
* Seokjae Lee - seok9211@naver.com
* Jinkyung Choi - twin7014@naver.com


Project Link: [https://github.com/doljae/image2emage/](https://github.com/doljae/image2emage/)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Tensorflow](https://github.com/tensorflow/tensorflow)
* [Scipy](https://github.com/scipy/scipy)
* [Keras](https://github.com/keras-team/keras)
* [OpenCV](https://github.com/opencv/opencv)
* [pix2pix](https://github.com/phillipi/pix2pix)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
