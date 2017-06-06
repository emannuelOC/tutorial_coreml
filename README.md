# Core ML

_Integrating Machine Learning models into your app._

After the WWDC17 Keynote, my friend [Aleph](https://github.com/alaphao) and I 
were super excited about the new Core ML framework. We then started playing 
around and decided to share some of what we did here.

With *Core ML* you can provide your apps with machine learning that runs locally 
and is optimized for on-device performance, minimizing memory footprint and power 
consumption.

Core ML supports many machine learning models (neural networks, tree ensembles, 
support vector machines, and generalized linear models). The model should be in the 
Core ML model format (models with a .mlmodel file extension)<sup>1</sup>.

In our example, we will use a famous model, the [VGG16](https://arxiv.org/abs/1409.1556), 
which is used to classify images, and luckily for us it is one of the pre-trained models 
that are available in [Core ML](https://developer.apple.com/machine-learning/).

## Downloading the model

In our example, we will use a model that is already available in the `.mlmodel` format. 
However if you have your model trained in a different framework such as Keras, for example, 
you can use %%%% to convert it to the appropriate format.
For our tutorial, you can download the VGG model 
[here](https://docs-assets.developer.apple.com/coreml/models/VGG16.mlmodel).<sup>2</sup>

After downloading the `.mlmodel` file, you can add it to your project simply by dropping 
it along side your files.

## App setup

The app we will create in our example is quite simple. It will capture the image from the 
camera and show it in an image view. When the model finishes classifying what's being captured 
by the camera, it will show the class in a label down at the bottom of the screen.

<< image >>

The code for the app is available [here](https://github.com/alaphao). In this post I will focus in 
the part that uses Core ML.

## Pre-processing

Basically, once you have a UIImage, there is some preprocessing needed before we can feed the image 
data into the model. Firstly, we need to crop a square in the image, since the VGG model we're 
using can only take 224x224 images. After cropping the square you might need to resize it to 224x224. 

<< code for cropping and resizing >>

Finally, you will need to create a `CVPixelBuffer` from the image data. The `CVPixelBuffer` is the 
type our model expect as an argument.

<< code for creating the CVPixelBuffer >> 

## The "Core ML" part 

Once we have the image data in a `CVPixelBuffer` we can pass it to our model and the code could 
not be simpler.

```swift

var vgg: VGG16?

override func viewDidLoad() {
    super.viewDidLoad()
    vgg = VGG16()
}

func classify(image: UIImage) {
    let buffer = pixelBufferFromImage(image: image)
    do {
        let answer = try vgg?.prediction(image: buffer)
        self.title = answer?.classLabel ?? ""
    } catch {
        // handle error
    }
}

```

The inferred classification for the image will be available in the `classLabel` property in 
the `answer`.

## Finally

Well, as you can see, the Core ML framework is a very powerful and easy to use tool to make our 
apps more intelligent. You can think of all the crazy ideas that Natural Language Processing, 
Computer Vision and other amazing areas can allow your apps to do.

If you have any suggestions, please let us know.

Also, if you wanna share how you've been planning to use Core ML in your app, we'd be glad to hear 
your ideas!

----------

1 See [Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating_a_core_ml_model_into_your_app)

2 There will be a post on that soon! WWDC is crazy right now so when we find some time to play 
around with keras and Core ML together we will most certainly share the results here.
