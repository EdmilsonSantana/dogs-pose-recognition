# Dogs Pose Recognition


<img src="./assets/banner.jpg" alt="Dog laying down" width="300" height="300">

The main goal of this project is to utilize object detection to identify my dogs in three different positions: squatting, sitting, and lying down. As I don't have enough photos of them, I have utilized various data sources to create a comprehensive dataset:

- **Squatting**: I obtained the images from a public Instagram account called ["dogspoopinginprettyplaces"](https://www.instagram.com/dogspoopinginprettyplaces/?hl=en) using the [instaloader](https://instaloader.github.io/) library.
- **Sitting**: I manually selected similar dog images from the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) that were in a sitting position.
- **Lying Down**: While most of the images are from my own dogs, I also added a few more from the internet using the ["bing_image_downloader"](https://pypi.org/project/bing-image-downloader/) library.

By incorporating data from different sources, I aim to create a diverse and robust dataset for accurate object detection of my dogs in different positions.

## TODO

- Include additional images of my own dogs for the Squatting and Lying Down classes.

