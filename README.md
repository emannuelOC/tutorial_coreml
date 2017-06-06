# Core ML

_Integrating Machine Learning models into your app._

After the WWDC17 Keynote, my friend [Aleph](https://github.com/alaphao) and I were super excited about the new Core ML framework. We then started playing around and decided to share some of what we did here.

With *Core ML* you can provide your apps with machine learning that runs locally and is optimized for on-device performance, minimizing memory footprint and power consumption.

Core ML supports many machine learning models (neural networks, tree ensembles, support vector machines, and generalized linear models). The model should be in the Core ML model format (models with a .mlmodel file extension)<sup>1</sup>.

In our example, we will use a famous model, the [VGG16](https://arxiv.org/abs/1409.1556), which is used to classify images, and luckily for us it is one of the pre-trained models that are available in [Core ML](https://developer.apple.com/machine-learning/).

## Downloading the model

In our example, we will use a model that is already available in the `.mlmodel` format. However if you have your model trained in a different framework such as Keras, for example, you can use %%%% to convert it to the appropriate format.
For our tutorial, you can download the VGG model [here](https://docs-assets.developer.apple.com/coreml/models/VGG16.mlmodel).<sup>2</sup>

After downloading the `.mlmodel` file, you can add it to your project simply by dropping it along side your files.

## App setup

The app we will create in our example is quite simple. It has an image view that will display the image to be classified, a bar button that the user can tap in order to choose another picture and a label that will show the classification given by the model.

When the user taps the "+" button, we present an Action Sheet with the "Library" and "Camera" options. After choosing one of them, the image picker controller is presented. 

```swift 
@IBAction func addNewPicture(_ sender: UIBarButtonItem) {
    let actionSheet = UIAlertController(title: nil, message: nil, preferredStyle: .actionSheet)
    let cameraAction = UIAlertAction(title: "Take a picture", style: .default) { (_) in
        self.takePicture()
    }
    let libraryAction = UIAlertAction(title: "Choose from library", style: .default) { (_) in
        self.choosePicture()
    }
    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
    actionSheet.addAction(cameraAction)
    actionSheet.addAction(libraryAction)
    actionSheet.addAction(cancelAction)
    present(actionSheet, animated: true, completion: nil)
}
    
func choosePicture() {
    let imagePicker = UIImagePickerController()
    imagePicker.sourceType = .photoLibrary
    imagePicker.delegate = self
    imagePicker.allowsEditing = true
    present(imagePicker, animated: true, completion: nil)
}
    
func takePicture() {
    let imagePicker = UIImagePickerController()
    imagePicker.sourceType = .camera
    imagePicker.delegate = self
    imagePicker.allowsEditing = true
    present(imagePicker, animated: true, completion: nil)
}
```

The image picker delegate methods are really straightforward. In the `didCancel` we simply call `dismiss()`. In the `didFinishWithMediaType`, we get the image from the `info` dictionary and pass it to our `classify(image: UIImage)` method.

```swift
func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
    dismiss(animated: true, completion: nil)
}
    
func imagePickerController(_ picker: UIImagePickerController,
                           didFinishPickingMediaWithInfo info: [String : Any]) {
    dismiss(animated: true, completion: nil)
    if let image = info[UIImagePickerControllerEditedImage] as? UIImage {
        pictureImageView.image = image
        classify(image: image)
    }
}
```

## Pre-processing

Firstly, we need to scale the image, since our model receives as input that has to be 224x224. After resizing it, we need to create a `CVPixelBuffer` from the image data. The `CVPixelBuffer` is the type our model expect as an argument.

```swift
func resize(image: UIImage, newSize: CGSize) -> UIImage? {
    UIGraphicsBeginImageContextWithOptions(newSize, false, 0.0)
    image.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
    let newImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    return newImage
}
    
func pixelBufferFromImage(image: UIImage) -> CVPixelBuffer {
    
    let newImage = resize(image: image, newSize: CGSize(width: 224/3.0, height: 224/3.0))
    
    let ciimage = CIImage(image: newImage!)
    let tmpcontext = CIContext(options: nil)
    let cgimage =  tmpcontext.createCGImage(ciimage!, from: ciimage!.extent)
    
    let cfnumPointer = UnsafeMutablePointer<UnsafeRawPointer>.allocate(capacity: 1)
    let cfnum = CFNumberCreate(kCFAllocatorDefault, .intType, cfnumPointer)
    let keys: [CFString] = [kCVPixelBufferCGImageCompatibilityKey, kCVPixelBufferCGBitmapContextCompatibilityKey, kCVPixelBufferBytesPerRowAlignmentKey]
    let values: [CFTypeRef] = [kCFBooleanTrue, kCFBooleanTrue, cfnum!]
    let keysPointer = UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 1)
    let valuesPointer =  UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 1)
    keysPointer.initialize(to: keys)
    valuesPointer.initialize(to: values)
    
    let options = CFDictionaryCreate(kCFAllocatorDefault, keysPointer, valuesPointer, keys.count, nil, nil)
    
    let width = cgimage!.width
    let height = cgimage!.height
    
    var pxbuffer: CVPixelBuffer?
    var status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                     kCVPixelFormatType_32BGRA, options, &pxbuffer)
    status = CVPixelBufferLockBaseAddress(pxbuffer!, CVPixelBufferLockFlags(rawValue: 0));
    
    let bufferAddress = CVPixelBufferGetBaseAddress(pxbuffer!);
    
    
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    let bytesperrow = CVPixelBufferGetBytesPerRow(pxbuffer!)
    let context = CGContext(data: bufferAddress,
                            width: width,
                            height: height,
                            bitsPerComponent: 8,
                            bytesPerRow: bytesperrow,
                            space: rgbColorSpace,
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue);
    context?.concatenate(CGAffineTransform(rotationAngle: 0))
    context?.concatenate(__CGAffineTransformMake( 1, 0, 0, -1, 0, CGFloat(height) )) //Flip Vertical
    
    
    
    context?.draw(cgimage!, in: CGRect(x:0, y:0, width:CGFloat(width), height:CGFloat(height)));
    status = CVPixelBufferUnlockBaseAddress(pxbuffer!, CVPixelBufferLockFlags(rawValue: 0));
    return pxbuffer!;
    
}
```

## The "Core ML" part 

Once we have the image data in a `CVPixelBuffer` we can pass it to our model and the code could not be simpler.

In order to do that, we will create our `VGG16` model in `viewDidLoad()` and in our `classify()` method, all we need to do is call `prediction()` on our model. The method will return a `VGG16Output` from which you can take a `classLabel` property that is a string. 

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

## Finally

Well, as you can see, the Core ML framework is a very powerful and easy to use tool to make our apps more intelligent. You can think of all the crazy ideas that Natural Language Processing, Computer Vision and other amazing areas can allow your apps to do.

If you have any suggestions, please let us know.

Also, if you wanna share how you've been planning to use Core ML in your app, we'd be glad to hear your ideas!

-

1 See [Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating_a_core_ml_model_into_your_app)

2 There will be a post on that soon! WWDC is crazy right now so when we find some time to play around with keras and Core ML together we will most certainly share the results here.
