# realistic-vision-v5-inpainting Cog model

This is an implementation of inpainting using the model [Realistic Vision V5.1](https://civitai.com/models/4201?modelVersionId=130090) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights.py

Then, you can run predictions:

    cog predict -i image=@demo.png -i mask=@mask.png

## Example:

Input - "a tabby cat, high resolution, sitting on a park bench"

![alt text](demo.png)

![alt text](mask.png)

Output:

![alt text](output.png)
